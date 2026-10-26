#!/usr/bin/env python3
"""Recompute generative metrics from saved stochastic trials without sampling."""

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

from dss_imputer.generative_metrics import (  # noqa: E402
    METRIC_PROTOCOL_VERSION,
    aggregate_metric_trials,
    evaluate_generative_quality,
    generative_metric_seed,
)
from dss_imputer.schema import TablePreprocessor  # noqa: E402
from scripts.eval_dss_imputer import (  # noqa: E402
    context_integrity_records,
    load_training_reference,
)


def load_target_mask(directory: Path, truth: pd.DataFrame) -> np.ndarray:
    path = directory / "target_mask.csv"
    if path.is_file():
        frame = pd.read_csv(path)
        if list(frame.columns) != list(truth.columns) or frame.shape != truth.shape:
            raise ValueError(f"Mask columns/rows do not match truth: {path}")
        if not frame.isin([True, False, 0, 1]).all().all():
            raise ValueError(f"Mask must contain only boolean values: {path}")
        return frame.to_numpy(dtype=bool)
    if directory.name.startswith("loco_"):
        target = directory.name.removeprefix("loco_")
        if target not in truth:
            raise ValueError(f"Unknown LOCO target {target!r}")
        mask = np.zeros(truth.shape, dtype=bool)
        mask[:, truth.columns.get_loc(target)] = truth[target].notna().to_numpy()
        return mask
    raise FileNotFoundError(
        f"{path} is required for MCAR/block metrics. The original mask cannot "
        "be inferred reliably from reconstructed values. Select LOCO scenarios "
        "with --scenarios loco_* or rerun eval to save masks."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--train-reference-csv",
        type=Path,
        help="Exact fitted training subset for DCR; otherwise rebuilt from checkpoint.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["loco_*"],
        help="Scenario names or glob patterns.",
    )
    parser.add_argument("--metric-sample-size", type=int, default=5000)
    parser.add_argument("--dcr-sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.metric_sample_size < 0 or args.dcr_sample_size < 2:
        raise ValueError(
            "Metric sample size must be nonnegative and DCR sample size >= 2."
        )

    info = json.loads((args.results_dir / "run_info.json").read_text())
    columns = info["model_columns"]
    numerical = info["numerical_columns"]
    categorical = info["categorical_columns"]
    source_truth = pd.read_csv(
        args.results_dir / "evaluation_truth.csv", low_memory=False
    )
    if len(source_truth) != info["rows"]:
        raise ValueError(
            "Saved truth row count differs from run_info.json; refusing to align incomplete CSVs."
        )
    truth = source_truth.loc[:, columns]
    checkpoint_path = args.checkpoint or Path(info["checkpoint"])
    if args.train_reference_csv is not None:
        train = pd.read_csv(args.train_reference_csv, low_memory=False).loc[:, columns]
        reference_source = str(args.train_reference_csv)
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        preprocessor = TablePreprocessor.from_dict(checkpoint["preprocessor"])
        train = load_training_reference(checkpoint, preprocessor)
        if list(train.columns) != columns:
            raise ValueError("Checkpoint columns differ from saved evaluation schema.")
        reference_source = f"fitted subset from {checkpoint_path}"

    directories = sorted(
        {
            directory
            for pattern in args.scenarios
            for directory in args.results_dir.glob(pattern)
            if directory.is_dir() and list(directory.glob("trial_*.csv"))
        }
    )
    if not directories:
        raise FileNotFoundError(
            "No saved trial_*.csv files for the requested scenarios. Point estimates are not stochastic samples."
        )
    output_dir = args.output_dir or args.results_dir / "metrics_corrected"
    if output_dir.resolve() == args.results_dir.resolve():
        raise ValueError(
            "Use a separate output directory to preserve the original report."
        )
    seed = args.seed if args.seed is not None else int(info.get("seed", 42))
    trial_rows, summary_rows, error_rows = [], [], []
    for directory in directories:
        mask = load_target_mask(directory, truth)
        trials = []
        for trial_path in sorted(directory.glob("trial_*.csv")):
            generated = pd.read_csv(trial_path, low_memory=False)
            if len(generated) != len(truth):
                raise ValueError(
                    f"Prediction row count differs from truth: {trial_path}"
                )
            context_checks = context_integrity_records(
                source_truth, generated, columns, mask, directory.name, trial_path.stem
            )
            if any(row["mismatches"] for row in context_checks):
                raise ValueError(
                    f"Known context differs from saved truth: {trial_path}"
                )
            print(f"Metrics: {directory.name} / {trial_path.name}", flush=True)
            scores, errors = evaluate_generative_quality(
                truth,
                generated.loc[:, columns],
                train,
                sample_size=args.metric_sample_size,
                dcr_sample_size=args.dcr_sample_size,
                seed=generative_metric_seed(seed + 100_000, directory.name),
                c2st_seed=generative_metric_seed(seed + 900_000, directory.name),
                numerical_columns=numerical,
                categorical_columns=categorical,
                target_mask=mask,
            )
            trial = int(trial_path.stem.removeprefix("trial_"))
            trials.append(scores)
            trial_rows.append({"scenario": directory.name, "trial": trial, **scores})
            error_rows.extend(
                {"scenario": directory.name, "trial": trial, "error": error}
                for error in errors
            )
            for error in errors:
                print(
                    f"WARNING: {directory.name}/{trial_path.stem}: {error}", flush=True
                )
        summary_rows.append(
            {
                "scenario": directory.name,
                "trials": len(trials),
                **aggregate_metric_trials(trials),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trial_rows).to_csv(
        output_dir / "generative_metrics_by_trial.csv", index=False
    )
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "generative_metrics_summary.csv", index=False
    )
    pd.DataFrame(error_rows, columns=["scenario", "trial", "error"]).to_csv(
        output_dir / "generative_metric_errors.csv", index=False
    )
    (output_dir / "run_info.json").write_text(
        json.dumps(
            {
                "metric_protocol": METRIC_PROTOCOL_VERSION,
                "results_dir": str(args.results_dir),
                "training_reference": reference_source,
                "model_columns": columns,
                "numerical_columns": numerical,
                "categorical_columns": categorical,
                "rows": len(truth),
                "seed": seed,
                "scenarios": [directory.name for directory in directories],
                "metric_sample_size": args.metric_sample_size,
                "dcr_sample_size": args.dcr_sample_size,
                "sampling_performed": False,
                "trial_policy": "Only saved stochastic trials; no point estimates or regenerated draws.",
            },
            indent=2,
        )
    )
    print(f"Saved corrected metrics: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
