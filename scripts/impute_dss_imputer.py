#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dss_imputer.inference import (  # noqa: E402
    impute_dataframe,
    load_imputer,
    point_estimate,
    uncertainty_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill arbitrary missing cells with a trained diffusion imputer."
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mask-cols",
        nargs="*",
        default=[],
        help="Optionally hide complete columns before imputation.",
    )
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    diffusion, preprocessor, checkpoint = load_imputer(args.checkpoint, device)
    frame = pd.read_csv(args.input_csv, low_memory=False, na_values=["?"])
    if args.limit is not None:
        frame = frame.iloc[: args.limit].reset_index(drop=True)

    unknown = sorted(set(args.mask_cols) - set(frame.columns))
    if unknown:
        raise ValueError(f"Unknown mask columns: {unknown}")
    excluded = sorted(set(args.mask_cols) & set(preprocessor.excluded_columns))
    if excluded:
        raise ValueError(f"Excluded columns cannot be generated: {excluded}")

    masked = frame.copy()
    for column in args.mask_cols:
        masked[column] = np.nan
    generated = impute_dataframe(
        masked,
        diffusion,
        preprocessor,
        device,
        samples=args.samples,
        batch_size=args.batch_size,
        temperature=args.temperature,
        seed=args.seed,
        verbose=True,
    )
    estimate = point_estimate(masked, generated, preprocessor)
    uncertainty = uncertainty_frame(masked, generated, preprocessor)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    masked.to_csv(args.output_dir / "input_masked.csv", index=False)
    for index, sample in enumerate(generated):
        sample.to_csv(args.output_dir / f"sample_{index:03d}.csv", index=False)
    estimate.to_csv(args.output_dir / "point_estimate.csv", index=False)
    uncertainty.to_csv(args.output_dir / "uncertainty.csv", index=False)
    info = {
        "input_csv": str(args.input_csv),
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "rows": len(frame),
        "mask_cols": args.mask_cols,
        "input_missing_cells": int(masked.isna().sum().sum()),
        "samples": args.samples,
        "device": str(device),
    }
    (args.output_dir / "run_info.json").write_text(json.dumps(info, indent=2))
    print(f"Point estimate: {args.output_dir / 'point_estimate.csv'}")
    print(f"Uncertainty: {args.output_dir / 'uncertainty.csv'}")


if __name__ == "__main__":
    main()
