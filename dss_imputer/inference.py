from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .diffusion import MixedDiffusion
from .model import MaskConditionedDenoiser
from .schema import TablePreprocessor


def load_imputer(
    checkpoint_path: Path, device: torch.device
) -> tuple[MixedDiffusion, TablePreprocessor, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    preprocessor = TablePreprocessor.from_dict(checkpoint["preprocessor"])
    denoiser = MaskConditionedDenoiser(**checkpoint["model_config"])
    diffusion = MixedDiffusion(denoiser, **checkpoint["diffusion_config"])
    diffusion.load_state_dict(checkpoint["model_state"])
    diffusion.to(device).eval()
    return diffusion, preprocessor, checkpoint


@torch.no_grad()
def impute_dataframe(
    frame: pd.DataFrame,
    diffusion: MixedDiffusion,
    preprocessor: TablePreprocessor,
    device: torch.device,
    samples: int = 5,
    batch_size: int = 2_048,
    temperature: float = 1.0,
    seed: int = 42,
    verbose: bool = False,
) -> list[pd.DataFrame]:
    if samples < 1:
        raise ValueError("samples must be positive.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    encoded = preprocessor.transform(frame)
    numerical_target = ~encoded.numerical_available
    categorical_target = ~encoded.categorical_available
    generated_frames: list[pd.DataFrame] = []
    batch_count = max(1, math.ceil(len(frame) / batch_size))
    progress_interval = max(1, math.ceil(batch_count / 5))

    for sample_index in range(samples):
        if verbose:
            print(f"Sampling draw {sample_index + 1}/{samples}", flush=True)
        sample_seed = seed + sample_index
        np.random.seed(sample_seed)
        torch.manual_seed(sample_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(sample_seed)

        numerical_parts: list[np.ndarray] = []
        categorical_parts: list[np.ndarray] = []
        for batch_index, start in enumerate(range(0, len(frame), batch_size)):
            stop = min(start + batch_size, len(frame))
            numerical = torch.from_numpy(encoded.numerical[start:stop]).to(device)
            categorical = torch.from_numpy(encoded.categorical[start:stop]).to(device)
            numerical_mask = torch.from_numpy(numerical_target[start:stop]).to(device)
            categorical_mask = torch.from_numpy(categorical_target[start:stop]).to(
                device
            )
            sampled_num, sampled_cat = diffusion.sample(
                numerical,
                categorical,
                numerical_mask,
                categorical_mask,
                temperature=temperature,
            )
            numerical_parts.append(sampled_num.cpu().numpy())
            categorical_parts.append(sampled_cat.cpu().numpy())
            completed = batch_index + 1
            if verbose and (
                completed == batch_count or completed % progress_interval == 0
            ):
                print(
                    f"  batches {completed}/{batch_count}",
                    flush=True,
                )

        decoded = preprocessor.inverse_transform(
            np.concatenate(numerical_parts, axis=0),
            np.concatenate(categorical_parts, axis=0),
        )
        result = preprocessor.fill_constants(frame)
        for column in preprocessor.categorical_columns:
            if frame[column].isna().any():
                result[column] = result[column].astype(object)
        for column in preprocessor.model_columns:
            missing = frame[column].isna()
            result.loc[missing, column] = decoded.loc[missing, column]
        generated_frames.append(result.loc[:, frame.columns])
    return generated_frames


def point_estimate(
    original: pd.DataFrame,
    samples: list[pd.DataFrame],
    preprocessor: TablePreprocessor,
) -> pd.DataFrame:
    estimate = preprocessor.fill_constants(original)
    for column in preprocessor.numerical_columns:
        missing = original[column].isna()
        if not missing.any():
            continue
        values = np.stack(
            [
                pd.to_numeric(sample.loc[missing, column]).to_numpy()
                for sample in samples
            ]
        )
        prediction = values.mean(axis=0)
        if preprocessor.numerical_integer[column]:
            prediction = np.rint(prediction)
        estimate.loc[missing, column] = prediction

    for column in preprocessor.categorical_columns:
        missing = original[column].isna()
        if not missing.any():
            continue
        estimate[column] = estimate[column].astype(object)
        values = pd.concat(
            [sample.loc[missing, column].reset_index(drop=True) for sample in samples],
            axis=1,
        )
        estimate.loc[missing, column] = (
            values.mode(axis=1, dropna=False).iloc[:, 0].to_numpy()
        )
    return estimate.loc[:, original.columns]


def uncertainty_frame(
    original: pd.DataFrame,
    samples: list[pd.DataFrame],
    preprocessor: TablePreprocessor,
) -> pd.DataFrame:
    uncertainty = pd.DataFrame(index=original.index)
    for column in preprocessor.numerical_columns:
        output = np.full(len(original), np.nan, dtype=np.float64)
        missing = original[column].isna().to_numpy()
        if missing.any():
            values = np.stack(
                [
                    pd.to_numeric(sample.loc[missing, column]).to_numpy()
                    for sample in samples
                ]
            )
            output[missing] = values.std(axis=0)
        uncertainty[f"{column}__std"] = output

    for column in preprocessor.categorical_columns:
        output = np.full(len(original), np.nan, dtype=np.float64)
        missing_indices = np.flatnonzero(original[column].isna().to_numpy())
        for row in missing_indices:
            counts = Counter(sample.iloc[row][column] for sample in samples)
            probabilities = np.asarray(list(counts.values()), dtype=np.float64)
            probabilities /= probabilities.sum()
            output[row] = float(
                -(probabilities * np.log(probabilities.clip(min=1e-12))).sum()
            )
        uncertainty[f"{column}__entropy"] = output
    return uncertainty
