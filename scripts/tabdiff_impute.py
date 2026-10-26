#!/usr/bin/env python3
"""Conditional column imputation with a trained TabDiff checkpoint.

This is an inference-only baseline. It keeps observed columns fixed during
reverse diffusion and resamples the requested columns from their conditional
distribution. The upstream TabDiff model is not modified or retrained.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Impute arbitrary columns with an already trained TabDiff model."
    )
    parser.add_argument(
        "--dataname",
        required=True,
        help="Processed TabDiff dataset name, for example dss_hdd_sequential_iops.",
    )
    parser.add_argument(
        "--mask-cols",
        nargs="+",
        required=True,
        help="Original column names to hide and reconstruct.",
    )
    parser.add_argument(
        "--exp-name",
        default=None,
        help="TabDiff experiment name. Used to infer the checkpoint directory.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Explicit TabDiff checkpoint. Defaults to best_ema_model* in the run.",
    )
    parser.add_argument(
        "--tabdiff-repo",
        type=Path,
        default=Path("external/tabdiff"),
        help="Path to the cloned official TabDiff repository.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for imputed tables, aggregate prediction, and metrics.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Number of independent conditional samples.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Rows processed in one sampling batch.",
    )
    parser.add_argument(
        "--resample-rounds",
        type=int,
        default=1,
        help="RePaint-style resampling rounds at each diffusion step.",
    )
    parser.add_argument(
        "--impute-condition",
        choices=["x_t", "x_0"],
        default="x_t",
        help="Condition on forward-noised observed values or directly on x_0.",
    )
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of test rows for a quick experiment.",
    )
    return parser.parse_args()


def resolve_checkpoint(
    repo: Path, dataname: str, exp_name: str | None, checkpoint: Path | None
) -> Path:
    if checkpoint is not None:
        path = checkpoint if checkpoint.is_absolute() else repo / checkpoint
        if not path.is_file():
            raise FileNotFoundError(f"TabDiff checkpoint does not exist: {path}")
        return path

    dataset_ckpt_dir = repo / "tabdiff" / "ckpt" / dataname
    if not exp_name:
        experiment_dirs = sorted(
            path
            for path in dataset_ckpt_dir.iterdir()
            if path.is_dir()
            and not path.name.endswith("_y_only")
            and (
                list(path.glob("best_ema_model*.pt"))
                or list(path.glob("best_model*.pt"))
                or list(path.glob("model_*.pt"))
            )
        ) if dataset_ckpt_dir.is_dir() else []
        if len(experiment_dirs) != 1:
            names = [path.name for path in experiment_dirs]
            raise ValueError(
                "Cannot infer a unique TabDiff experiment for "
                f"{dataname}: {names}. Pass --exp-name explicitly."
            )
        ckpt_dir = experiment_dirs[0]
    else:
        ckpt_dir = dataset_ckpt_dir / exp_name
    candidates = sorted(ckpt_dir.glob("best_ema_model*.pt"))
    if not candidates:
        candidates = sorted(ckpt_dir.glob("best_model*.pt"))
    if not candidates:
        periodic = list(ckpt_dir.glob("model_*.pt"))
        candidates = sorted(
            periodic,
            key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
            reverse=True,
        )
    if not candidates:
        raise FileNotFoundError(f"No usable checkpoint found in {ckpt_dir}.")
    return candidates[0]


def transformed_column_order(info: dict[str, Any]) -> tuple[list[str], list[str]]:
    names = info["column_names"]
    targets = [names[index] for index in info["target_col_idx"]]
    numerical = [names[index] for index in info["num_col_idx"]]
    categorical = [names[index] for index in info["cat_col_idx"]]

    if info["task_type"] == "regression":
        return targets + numerical, categorical
    return numerical, targets + categorical


def mask_indices(
    info: dict[str, Any], mask_cols: list[str]
) -> tuple[list[int], list[int]]:
    numerical, categorical = transformed_column_order(info)
    available = set(numerical) | set(categorical)
    unknown = sorted(set(mask_cols) - available)
    if unknown:
        raise ValueError(
            f"Unknown mask columns: {unknown}. Available columns: {sorted(available)}"
        )
    return (
        [numerical.index(column) for column in mask_cols if column in numerical],
        [categorical.index(column) for column in mask_cols if column in categorical],
    )


def load_model(
    repo: Path,
    dataname: str,
    checkpoint: Path,
    device: Any,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    sys.path.insert(0, str(repo.resolve()))

    import torch
    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from tabdiff.modules.main_modules import Model, UniModMLP
    from utils_train import TabDiffDataset

    data_dir = repo / "data" / dataname
    info_path = data_dir / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(
            f"Processed TabDiff metadata is missing: {info_path}. Run prepare first."
        )
    info = json.loads(info_path.read_text())

    config_path = checkpoint.parent / "config.pkl"
    if not config_path.is_file():
        raise FileNotFoundError(f"Checkpoint config is missing: {config_path}")
    with config_path.open("rb") as stream:
        config = pickle.load(stream)

    data_config = config.get("data", {})
    dataset = TabDiffDataset(
        dataname,
        str(data_dir),
        info,
        isTrain=True,
        dequant_dist=data_config.get("dequant_dist", "none"),
        int_dequant_factor=data_config.get("int_dequant_factor", 0.0),
    )
    test_dataset = TabDiffDataset(
        dataname,
        str(data_dir),
        info,
        isTrain=False,
        dequant_dist=data_config.get("dequant_dist", "none"),
        int_dequant_factor=data_config.get("int_dequant_factor", 0.0),
    )

    model_config = dict(config["unimodmlp_params"])
    model_config["d_numerical"] = dataset.d_numerical
    model_config["categories"] = (dataset.categories + 1).tolist()
    denoiser = Model(
        UniModMLP(**model_config),
        **config["diffusion_params"]["edm_params"],
    ).to(device)

    diffusion = UnifiedCtimeDiffusion(
        num_classes=dataset.categories,
        num_numerical_features=dataset.d_numerical,
        denoise_fn=denoiser,
        y_only_model=None,
        **config["diffusion_params"],
        device=device,
    ).to(device)

    state = torch.load(checkpoint, map_location=device)
    diffusion._denoise_fn.load_state_dict(state["denoise_fn"])
    diffusion.num_schedule.load_state_dict(state["num_schedule"])
    diffusion.cat_schedule.load_state_dict(state["cat_schedule"])
    diffusion.eval()
    return diffusion, dataset, test_dataset, info


def recover_frame(
    sample: Any, dataset: Any, info: dict[str, Any]
) -> pd.DataFrame:
    num_count = len(info["num_col_idx"])
    cat_count = len(info["cat_col_idx"])
    target_count = len(info["target_col_idx"])
    transformed_num_count = num_count
    if info["task_type"] == "regression":
        transformed_num_count += target_count

    syn_num = sample[:, :transformed_num_count]
    syn_cat = sample[:, transformed_num_count:]
    syn_num = dataset.num_inverse(syn_num).astype(np.float32)
    syn_num = dataset.int_inverse(syn_num).astype(np.float32)
    syn_cat = dataset.cat_inverse(syn_cat)

    if info["task_type"] == "regression":
        syn_target = syn_num[:, :target_count]
        syn_num = syn_num[:, target_count:]
    else:
        syn_target = syn_cat[:, :target_count]
        syn_cat = syn_cat[:, target_count:]

    num_indices = set(info["num_col_idx"])
    cat_indices = set(info["cat_col_idx"])
    idx_mapping = {int(key): value for key, value in info["idx_mapping"].items()}
    frame = pd.DataFrame()
    for index in range(num_count + cat_count + target_count):
        if index in num_indices:
            frame[index] = syn_num[:, idx_mapping[index]]
        elif index in cat_indices:
            frame[index] = syn_cat[:, idx_mapping[index] - num_count]
        else:
            frame[index] = syn_target[
                :, idx_mapping[index] - num_count - cat_count
            ]

    mapping = {int(key): value for key, value in info["idx_name_mapping"].items()}
    return frame.rename(columns=mapping)


def impute_trial(
    diffusion: Any,
    train_dataset: Any,
    test_dataset: Any,
    info: dict[str, Any],
    real_frame: pd.DataFrame,
    mask_cols: list[str],
    num_mask_idx: list[int],
    cat_mask_idx: list[int],
    batch_size: int,
    resample_rounds: int,
    impute_condition: str,
    device: Any,
) -> pd.DataFrame:
    import torch

    d_numerical = test_dataset.d_numerical
    transformed = test_dataset.X[: len(real_frame)]
    output_parts: list[pd.DataFrame] = []

    for start in range(0, len(transformed), batch_size):
        batch = transformed[start : start + batch_size].clone().to(device)
        x_num = batch[:, :d_numerical]
        x_cat = batch[:, d_numerical:].long()

        if num_mask_idx:
            train_num = train_dataset.X[:, :d_numerical]
            fill = train_num[:, num_mask_idx].mean(dim=0).to(device)
            x_num[:, num_mask_idx] = fill
        if cat_mask_idx:
            mask_values = torch.as_tensor(
                test_dataset.categories, dtype=x_cat.dtype, device=device
            )
            x_cat[:, cat_mask_idx] = mask_values[cat_mask_idx]

        with torch.no_grad():
            sample = diffusion.sample_impute(
                x_num,
                x_cat,
                num_mask_idx,
                cat_mask_idx,
                resample_rounds,
                impute_condition,
                0.0,
                0.0,
            )
        output_parts.append(recover_frame(sample, test_dataset, info))

    result = pd.concat(output_parts, ignore_index=True)
    observed = [column for column in real_frame.columns if column not in mask_cols]
    result.loc[:, observed] = real_frame.loc[:, observed].to_numpy()
    return result.loc[:, real_frame.columns]


def point_estimate(
    real_frame: pd.DataFrame,
    samples: list[pd.DataFrame],
    mask_cols: list[str],
) -> pd.DataFrame:
    estimate = real_frame.copy()
    for column in mask_cols:
        values = pd.concat([sample[column] for sample in samples], axis=1)
        if pd.api.types.is_numeric_dtype(real_frame[column]):
            estimate[column] = values.apply(pd.to_numeric, errors="coerce").mean(axis=1)
        else:
            estimate[column] = values.mode(axis=1, dropna=False).iloc[:, 0]
    return estimate


def evaluate_point_estimate(
    real_frame: pd.DataFrame,
    estimate: pd.DataFrame,
    mask_cols: list[str],
) -> dict[str, dict[str, float]]:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        recall_score,
    )

    metrics: dict[str, dict[str, float]] = {}
    for column in mask_cols:
        truth = real_frame[column]
        prediction = estimate[column]
        if pd.api.types.is_numeric_dtype(truth):
            metrics[column] = {
                "r2": float(r2_score(truth, prediction)),
                "rmse": float(mean_squared_error(truth, prediction) ** 0.5),
                "mae": float(mean_absolute_error(truth, prediction)),
            }
        else:
            metrics[column] = {
                "accuracy": float(accuracy_score(truth, prediction)),
                "macro_f1": float(
                    f1_score(truth, prediction, average="macro", zero_division=0)
                ),
                "macro_recall": float(
                    recall_score(truth, prediction, average="macro", zero_division=0)
                ),
            }
    return metrics


def main() -> None:
    args = parse_args()
    if args.trials < 1:
        raise ValueError("--trials must be positive.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")

    repo = args.tabdiff_repo.resolve()
    checkpoint = resolve_checkpoint(
        repo, args.dataname, args.exp_name, args.checkpoint
    )

    import torch

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    )
    diffusion, train_dataset, test_dataset, info = load_model(
        repo, args.dataname, checkpoint, device
    )
    num_mask_idx, cat_mask_idx = mask_indices(info, args.mask_cols)

    source_dataset = train_dataset if args.split == "train" else test_dataset
    real_path = repo / "data" / args.dataname / f"{args.split}.csv"
    real_frame = pd.read_csv(real_path)
    if len(real_frame) != len(source_dataset.X):
        raise ValueError(f"Prepared {args.split} CSV and transformed arrays differ in length.")
    if args.limit is not None:
        real_frame = real_frame.iloc[: args.limit].reset_index(drop=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples: list[pd.DataFrame] = []
    for trial in range(args.trials):
        trial_seed = args.seed + trial
        np.random.seed(trial_seed)
        torch.manual_seed(trial_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(trial_seed)

        sample = impute_trial(
            diffusion,
            train_dataset,
            source_dataset,
            info,
            real_frame,
            args.mask_cols,
            num_mask_idx,
            cat_mask_idx,
            args.batch_size,
            args.resample_rounds,
            args.impute_condition,
            device,
        )
        sample_path = args.output_dir / f"trial_{trial:03d}.csv"
        sample.to_csv(sample_path, index=False)
        samples.append(sample)
        print(f"Saved conditional sample {trial + 1}/{args.trials}: {sample_path}")

    estimate = point_estimate(real_frame, samples, args.mask_cols)
    estimate_path = args.output_dir / "point_estimate.csv"
    estimate.to_csv(estimate_path, index=False)

    metrics = evaluate_point_estimate(real_frame, estimate, args.mask_cols)
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    run_info = {
        "dataname": args.dataname,
        "split": args.split,
        "data_csv": str(real_path),
        "cfg": False,
        "w_num": 0.0,
        "w_cat": 0.0,
        "checkpoint": str(checkpoint),
        "mask_cols": args.mask_cols,
        "num_mask_idx": num_mask_idx,
        "cat_mask_idx": cat_mask_idx,
        "trials": args.trials,
        "rows": len(real_frame),
        "device": str(device),
        "impute_condition": args.impute_condition,
        "resample_rounds": args.resample_rounds,
    }
    (args.output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))

    print(f"Point estimate: {estimate_path}")
    print(f"Metrics: {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
