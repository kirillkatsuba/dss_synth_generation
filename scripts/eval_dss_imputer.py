#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dss_imputer.inference import (  # noqa: E402
    impute_dataframe,
    load_imputer,
    point_estimate,
)
from dss_imputer.generative_metrics import (  # noqa: E402
    METRIC_PROTOCOL_VERSION,
    aggregate_metric_trials,
    evaluate_generative_quality,
    generative_metric_seed,
)
from dss_imputer.splits import split_indices  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate diffusion imputation on artificial missing values."
    )
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        "--data-csv",
        type=Path,
        help="CSV split to evaluate, for example the training or test table.",
    )
    data_group.add_argument(
        "--test-csv",
        type=Path,
        help="Deprecated alias for --data-csv kept for existing commands.",
    )
    parser.add_argument(
        "--split-label",
        default="evaluation",
        help="Human-readable split name stored in run_info.json.",
    )
    parser.add_argument(
        "--training-subset",
        choices=["all", "fit", "validation"],
        default="all",
        help="Select the checkpoint's full training or validation table (legacy runs use an internal split).",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--patterns",
        nargs="+",
        choices=["mcar", "block", "loco"],
        default=["mcar", "block", "loco"],
    )
    parser.add_argument("--rates", nargs="+", type=float, default=[0.1, 0.3, 0.5])
    parser.add_argument(
        "--loco-cols", nargs="*", default=None, help="Defaults to all model columns."
    )
    parser.add_argument(
        "--exclude-loco-cols",
        nargs="*",
        default=[],
        help="Exclude these model columns from LOCO scenarios.",
    )
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument(
        "--generative-metrics",
        action="store_true",
        help=(
            "Compute SDMetrics quality, C2ST, alpha-precision, beta-recall, "
            "authenticity, and DCR for every stochastic trial."
        ),
    )
    parser.add_argument("--metric-sample-size", type=int, default=5000)
    parser.add_argument("--dcr-sample-size", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--save-samples",
        action="store_true",
        help="Save every stochastic reconstruction as trial_*.csv.",
    )
    parser.add_argument(
        "--save-first-sample",
        action="store_true",
        help="Save trial_000.csv and its per-column metrics without storing all draws.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def random_mask(
    frame: pd.DataFrame,
    columns: list[str],
    rate: float,
    pattern: str,
    seed: int,
) -> np.ndarray:
    if not 0 < rate < 1:
        raise ValueError("Evaluation rates must be between zero and one.")
    generator = np.random.default_rng(seed)
    available = frame[columns].notna().to_numpy()
    result = np.zeros_like(available, dtype=bool)
    for row in range(len(frame)):
        eligible = np.flatnonzero(available[row])
        if len(eligible) < 2:
            continue
        if pattern == "mcar":
            selected = eligible[generator.random(len(eligible)) < rate]
            if len(selected) == 0:
                selected = generator.choice(eligible, size=1, replace=False)
            if len(selected) >= len(eligible):
                selected = generator.choice(
                    eligible, size=len(eligible) - 1, replace=False
                )
        elif pattern == "block":
            count = min(len(eligible) - 1, max(1, round(rate * len(eligible))))
            start = int(generator.integers(0, len(eligible)))
            selected = np.roll(eligible, -start)[:count]
        else:
            raise ValueError(f"Unsupported random pattern: {pattern}")
        result[row, selected] = True
    return result


def apply_mask(
    frame: pd.DataFrame, columns: list[str], mask: np.ndarray
) -> pd.DataFrame:
    masked = frame.copy()
    for index, column in enumerate(columns):
        if mask[:, index].any():
            masked[column] = masked[column].mask(mask[:, index])
    return masked


def context_integrity_records(
    reference: pd.DataFrame,
    reconstructed: pd.DataFrame,
    model_columns: list[str],
    target_mask: np.ndarray,
    scenario: str,
    output: str,
) -> list[dict[str, object]]:
    """Count changes outside the artificial target mask."""
    if len(reference) != len(reconstructed):
        raise ValueError(
            f"Context check row mismatch: {len(reference)} != {len(reconstructed)}"
        )
    if target_mask.shape != (len(reference), len(model_columns)):
        raise ValueError("Context check target mask has an invalid shape.")

    target_by_column = {
        column: target_mask[:, index] for index, column in enumerate(model_columns)
    }
    records: list[dict[str, object]] = []
    for column in reference.columns:
        if column not in reconstructed:
            raise ValueError(f"Reconstruction is missing column {column!r}.")
        artificial_target = target_by_column.get(
            column, np.zeros(len(reference), dtype=bool)
        )
        known_context = reference[column].notna().to_numpy() & ~artificial_target
        context_cells = int(known_context.sum())
        if pd.api.types.is_numeric_dtype(reference[column]):
            expected = pd.to_numeric(
                reference.loc[known_context, column], errors="coerce"
            ).to_numpy(dtype=float)
            actual = pd.to_numeric(
                reconstructed.loc[known_context, column], errors="coerce"
            ).to_numpy(dtype=float)
            equal = np.isclose(expected, actual, rtol=0.0, atol=0.0, equal_nan=True)
        else:
            expected = reference.loc[known_context, column].astype(str).to_numpy()
            actual = reconstructed.loc[known_context, column].astype(str).to_numpy()
            equal = expected == actual
        records.append(
            {
                "scenario": scenario,
                "output": output,
                "column": column,
                "context_cells": context_cells,
                "mismatches": int((~equal).sum()),
            }
        )
    return records


def evaluate_scenario(
    truth: pd.DataFrame,
    estimate: pd.DataFrame,
    columns: list[str],
    mask: np.ndarray,
    numerical_columns: set[str],
    scenario: str,
    pattern: str,
    rate: float | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, column in enumerate(columns):
        selected = mask[:, index]
        count = int(selected.sum())
        if count == 0:
            continue
        actual = truth.loc[selected, column]
        predicted = estimate.loc[selected, column]
        row: dict[str, object] = {
            "scenario": scenario,
            "pattern": pattern,
            "rate": rate,
            "column": column,
            "masked_cells": count,
        }
        if column in numerical_columns:
            actual_num = pd.to_numeric(actual).to_numpy(dtype=float)
            predicted_num = pd.to_numeric(predicted).to_numpy(dtype=float)
            row.update(
                {
                    "type": "numerical",
                    "r2": float(r2_score(actual_num, predicted_num))
                    if count > 1
                    else np.nan,
                    "mse": float(mean_squared_error(actual_num, predicted_num)),
                    "rmse": float(mean_squared_error(actual_num, predicted_num) ** 0.5),
                    "mae": float(mean_absolute_error(actual_num, predicted_num)),
                    "accuracy": np.nan,
                    "macro_f1": np.nan,
                }
            )
        else:
            actual_labels = actual.map(str)
            predicted_labels = predicted.map(str)
            row.update(
                {
                    "type": "categorical",
                    "r2": np.nan,
                    "mse": np.nan,
                    "rmse": np.nan,
                    "mae": np.nan,
                    "accuracy": float(accuracy_score(actual_labels, predicted_labels)),
                    "macro_f1": float(
                        f1_score(
                            actual_labels,
                            predicted_labels,
                            average="macro",
                            zero_division=0,
                        )
                    ),
                }
            )
        rows.append(row)
    return rows


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def aggregate_deterministic_trials(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cell-prediction metrics across stochastic reconstructions."""
    if frame.empty:
        return pd.DataFrame()
    groups = ["scenario", "pattern", "rate", "column", "type"]
    values = [
        "r2",
        "mse",
        "rmse",
        "mae",
        "accuracy",
        "macro_f1",
    ]
    aggregate = frame.groupby(groups, dropna=False)[values].agg(["mean", "std"])
    aggregate.columns = [f"{metric}_{stat}" for metric, stat in aggregate.columns]
    aggregate = aggregate.reset_index()
    aggregate["trials"] = frame.groupby(groups, dropna=False)["trial"].nunique().values
    std_columns = [column for column in aggregate if column.endswith("_std")]
    aggregate[std_columns] = aggregate[std_columns].fillna(0.0)
    return aggregate


def load_training_reference(
    checkpoint: dict[str, object],
    preprocessor: object,
) -> pd.DataFrame:
    """Rebuild the exact training reference used by DCR."""
    config = checkpoint.get("run_config")
    if not isinstance(config, dict) or not config.get("train_csv"):
        raise ValueError(
            "Generative metrics require run_config.train_csv in the checkpoint."
        )
    train_csv = Path(str(config["train_csv"]))
    if not train_csv.is_file():
        raise FileNotFoundError(f"Training reference is unavailable: {train_csv}")
    frame = pd.read_csv(train_csv, low_memory=False, na_values=["?"])
    frame, _ = select_training_subset(frame, checkpoint, train_csv, "fit")
    columns = list(preprocessor.model_columns)
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Training reference is missing model columns: {missing}")
    return frame.loc[:, columns].reset_index(drop=True)


def select_training_subset(
    frame: pd.DataFrame,
    checkpoint: dict[str, object],
    data_csv: Path,
    subset: str,
) -> tuple[pd.DataFrame, str]:
    if subset == "all":
        return frame, "all"
    config = checkpoint.get("run_config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint has no run_config for the training split.")
    training_csv = config.get("train_csv")
    validation_csv = config.get("validation_csv")
    if validation_csv:
        expected_csv = training_csv if subset == "fit" else validation_csv
        if not expected_csv or Path(expected_csv).resolve() != data_csv.resolve():
            raise ValueError(
                f"{subset} evaluation requires the checkpoint's CSV: {expected_csv!r}"
            )
        if subset == "fit" and config.get("max_rows") is not None:
            frame = frame.sample(
                n=min(int(config["max_rows"]), len(frame)),
                random_state=int(config["seed"]),
            ).reset_index(drop=True)
        return frame.reset_index(drop=True), "external_csv"
    if not training_csv or Path(training_csv).resolve() != data_csv.resolve():
        raise ValueError(
            "Training subset requires the original train CSV from the checkpoint: "
            f"{training_csv!r}"
        )
    max_rows = config.get("max_rows")
    if max_rows is not None:
        frame = frame.sample(
            n=min(int(max_rows), len(frame)), random_state=int(config["seed"])
        ).reset_index(drop=True)
    fit_indices, validation_indices, method = split_indices(
        frame,
        float(config["validation_fraction"]),
        str(config["group_column"]),
        int(config["seed"]),
    )
    selected = fit_indices if subset == "fit" else validation_indices
    return frame.iloc[selected].reset_index(drop=True), method


def main() -> None:
    args = parse_args()
    if args.samples < 1 or args.batch_size < 1:
        raise ValueError("samples and batch-size must be positive.")
    if args.metric_sample_size < 0 or args.dcr_sample_size < 2:
        raise ValueError(
            "metric-sample-size must be non-negative and dcr-sample-size at least two."
        )
    device = resolve_device(args.device)
    diffusion, preprocessor, checkpoint = load_imputer(args.checkpoint, device)
    data_csv = args.data_csv or args.test_csv
    if data_csv is None:
        raise ValueError("One of --data-csv or --test-csv is required.")
    config = checkpoint.get("run_config")
    validation_csv = config.get("validation_csv") if isinstance(config, dict) else None
    evaluating_selection_data = bool(
        validation_csv and Path(validation_csv).resolve() == data_csv.resolve()
    )
    if evaluating_selection_data:
        print(
            "WARNING: this CSV selected best.pt during training; these are "
            "validation results, not an independent test.",
            flush=True,
        )
    truth = pd.read_csv(data_csv, low_memory=False, na_values=["?"])
    truth, split_method = select_training_subset(
        truth, checkpoint, data_csv, args.training_subset
    )
    if args.limit is not None:
        truth = truth.iloc[: args.limit].reset_index(drop=True)

    columns = preprocessor.model_columns
    numerical = set(preprocessor.numerical_columns)
    loco_columns = args.loco_cols or columns
    unknown_loco = sorted(
        (set(loco_columns) | set(args.exclude_loco_cols)) - set(columns)
    )
    if unknown_loco:
        raise ValueError(f"LOCO columns are not modeled: {unknown_loco}")
    loco_columns = [
        column for column in loco_columns if column not in args.exclude_loco_cols
    ]
    if "loco" in args.patterns and not loco_columns:
        raise ValueError("No LOCO columns remain after exclusions.")

    scenarios: list[tuple[str, str, float | None, np.ndarray]] = []
    for pattern in args.patterns:
        if pattern in {"mcar", "block"}:
            for rate_index, rate in enumerate(args.rates):
                mask = random_mask(
                    truth,
                    columns,
                    rate,
                    pattern,
                    args.seed + rate_index + (10_000 if pattern == "block" else 0),
                )
                scenarios.append((f"{pattern}_{rate:.2f}", pattern, rate, mask))
        else:
            for column in loco_columns:
                mask = np.zeros((len(truth), len(columns)), dtype=bool)
                column_index = columns.index(column)
                mask[:, column_index] = truth[column].notna().to_numpy()
                scenarios.append((f"loco_{column}", "loco", None, mask))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    truth.to_csv(args.output_dir / "evaluation_truth.csv", index=False)
    metrics: list[dict[str, object]] = []
    trial_metrics: list[dict[str, object]] = []
    first_sample_metrics: list[dict[str, object]] = []
    generative_trial_metrics: list[dict[str, object]] = []
    generative_errors: list[dict[str, object]] = []
    context_checks: list[dict[str, object]] = []
    train_reference = (
        load_training_reference(checkpoint, preprocessor)
        if args.generative_metrics
        else None
    )
    for scenario_index, (name, pattern, rate, mask) in enumerate(scenarios):
        print(f"Evaluating {name} ({scenario_index + 1}/{len(scenarios)})")
        masked = apply_mask(truth, columns, mask)
        samples = impute_dataframe(
            masked,
            diffusion,
            preprocessor,
            device,
            samples=args.samples,
            batch_size=args.batch_size,
            temperature=args.temperature,
            seed=args.seed + scenario_index * 1_000,
            verbose=True,
        )
        estimate = point_estimate(masked, samples, preprocessor)
        scenario_dir = args.output_dir / safe_name(name)
        scenario_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(mask, columns=columns).to_csv(
            scenario_dir / "target_mask.csv", index=False
        )
        for sample_index, sample in enumerate(samples):
            context_checks.extend(
                context_integrity_records(
                    truth,
                    sample,
                    columns,
                    mask,
                    name,
                    f"trial_{sample_index:03d}",
                )
            )
            if args.save_samples or (args.save_first_sample and sample_index == 0):
                sample.to_csv(
                    scenario_dir / f"trial_{sample_index:03d}.csv",
                    index=False,
                )
            current_trial_metrics = evaluate_scenario(
                truth,
                sample,
                columns,
                mask,
                numerical,
                name,
                pattern,
                rate,
            )
            for metric_row in current_trial_metrics:
                metric_row["trial"] = sample_index
            trial_metrics.extend(current_trial_metrics)
            if args.generative_metrics:
                current_generative, current_errors = evaluate_generative_quality(
                    truth.loc[:, columns],
                    sample.loc[:, columns],
                    train_reference,
                    sample_size=args.metric_sample_size,
                    dcr_sample_size=args.dcr_sample_size,
                    seed=generative_metric_seed(args.seed + 100_000, name),
                    c2st_seed=generative_metric_seed(args.seed + 900_000, name),
                    numerical_columns=preprocessor.numerical_columns,
                    categorical_columns=preprocessor.categorical_columns,
                    target_mask=mask,
                )
                generative_trial_metrics.append(
                    {
                        "scenario": name,
                        "pattern": pattern,
                        "rate": rate,
                        "trial": sample_index,
                        **current_generative,
                    }
                )
                generative_errors.extend(
                    {
                        "scenario": name,
                        "trial": sample_index,
                        "error": error,
                    }
                    for error in current_errors
                )
        context_checks.extend(
            context_integrity_records(
                truth,
                estimate,
                columns,
                mask,
                name,
                "point_estimate",
            )
        )
        current_context = pd.DataFrame(context_checks)
        mismatches = int(current_context["mismatches"].sum())
        if mismatches:
            current_context.to_csv(
                args.output_dir / "context_integrity.csv", index=False
            )
            raise RuntimeError(
                f"Known context changed in {mismatches} cells. "
                f"See {args.output_dir / 'context_integrity.csv'}"
            )
        estimate.to_csv(scenario_dir / "point_estimate.csv", index=False)
        metrics.extend(
            evaluate_scenario(
                truth,
                estimate,
                columns,
                mask,
                numerical,
                name,
                pattern,
                rate,
            )
        )
        if args.save_first_sample:
            first_sample_metrics.extend(
                evaluate_scenario(
                    truth, samples[0], columns, mask, numerical, name, pattern, rate
                )
            )

    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(args.output_dir / "metrics_by_column.csv", index=False)
    trial_metrics_frame = pd.DataFrame(trial_metrics)
    trial_metrics_frame.to_csv(
        args.output_dir / "deterministic_metrics_by_trial.csv", index=False
    )
    aggregate_deterministic_trials(trial_metrics_frame).to_csv(
        args.output_dir / "deterministic_metrics_trial_summary.csv", index=False
    )
    if args.save_first_sample:
        pd.DataFrame(first_sample_metrics).to_csv(
            args.output_dir / "metrics_first_sample.csv", index=False
        )
    if not metrics_frame.empty:
        aggregate = (
            metrics_frame.groupby(["scenario", "pattern", "type"], dropna=False)[
                ["r2", "mse", "rmse", "mae", "accuracy", "macro_f1"]
            ]
            .mean(numeric_only=True)
            .reset_index()
        )
    else:
        aggregate = pd.DataFrame()
    aggregate.to_csv(args.output_dir / "metrics_summary.csv", index=False)
    if args.generative_metrics:
        generative_frame = pd.DataFrame(generative_trial_metrics)
        generative_frame.to_csv(
            args.output_dir / "generative_metrics_by_trial.csv", index=False
        )
        summary_rows: list[dict[str, object]] = []
        for (scenario, pattern, rate), group in generative_frame.groupby(
            ["scenario", "pattern", "rate"], dropna=False, sort=False
        ):
            values = group.drop(
                columns=["scenario", "pattern", "rate", "trial"]
            ).to_dict("records")
            summary_rows.append(
                {
                    "scenario": scenario,
                    "pattern": pattern,
                    "rate": rate,
                    "trials": len(group),
                    **aggregate_metric_trials(values),
                }
            )
        pd.DataFrame(summary_rows).to_csv(
            args.output_dir / "generative_metrics_summary.csv", index=False
        )
        pd.DataFrame(
            generative_errors, columns=["scenario", "trial", "error"]
        ).to_csv(args.output_dir / "generative_metric_errors.csv", index=False)
    context_frame = pd.DataFrame(context_checks)
    context_frame.to_csv(args.output_dir / "context_integrity.csv", index=False)
    run_info = {
        "data_csv": str(data_csv),
        "test_csv": str(data_csv),
        "split_label": args.split_label,
        "training_subset": args.training_subset,
        "split_method": split_method,
        "used_for_checkpoint_selection": evaluating_selection_data,
        "loco_columns": loco_columns if "loco" in args.patterns else [],
        "model_columns": columns,
        "numerical_columns": preprocessor.numerical_columns,
        "categorical_columns": preprocessor.categorical_columns,
        "evaluation_truth_csv": str(args.output_dir / "evaluation_truth.csv"),
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "rows": len(truth),
        "patterns": args.patterns,
        "rates": args.rates,
        "samples": args.samples,
        "seed": args.seed,
        "generative_metrics": args.generative_metrics,
        "metric_protocol": METRIC_PROTOCOL_VERSION,
        "metric_sample_size": args.metric_sample_size,
        "dcr_sample_size": args.dcr_sample_size,
        "save_samples": args.save_samples,
        "save_first_sample": args.save_first_sample,
        "scenarios": len(scenarios),
        "device": str(device),
        "context_cells_checked": int(context_frame["context_cells"].sum()),
        "context_mismatches": int(context_frame["mismatches"].sum()),
    }
    (args.output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))
    print(f"Metrics: {args.output_dir / 'metrics_by_column.csv'}")
    print(f"Summary: {args.output_dir / 'metrics_summary.csv'}")


if __name__ == "__main__":
    main()
