#!/usr/bin/env python3
"""Compare the raw target distributions in the imputer's fit, validation and test splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dss_imputer.schema import TablePreprocessor  # noqa: E402
from dss_imputer.splits import split_indices  # noqa: E402


COLORS = {"Fit train": "#007f86", "Validation": "#d07432", "Test": "#7a4da0"}
FOCUS_LIMITS = {"iops": (0, 14_000), "lat": (0, 100)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=Path("dataset/pools/train_hdd_sequential.csv"))
    parser.add_argument("--test-csv", type=Path, default=Path("dataset/pools/test_hdd_sequential.csv"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("reports/dss_imputer/hdd_split_distributions"),
    )
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--group-column", default="id")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--continuous-cols", nargs="+", default=["iops", "lat", "n_jobs", "iodepth"]
    )
    parser.add_argument("--exclude-cols", nargs="*", default=["id"])
    return parser.parse_args()


def display_values(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    if column == "lat":
        return values / 1e6
    if column == "iops":
        return values / 1e3
    return values


def axis_label(column: str) -> str:
    return {"iops": "IOPS (10^3)", "lat": "Latency (us)"}.get(column, column)


def numerical_histogram(
    ax: plt.Axes, splits: dict[str, pd.DataFrame], column: str, limits: tuple[float, float]
) -> None:
    lower, upper = limits
    bins = np.linspace(lower, upper, 51)
    for name, frame in splits.items():
        values = display_values(frame, column).dropna().to_numpy()
        visible = values[(values >= lower) & (values <= upper)]
        if not len(visible):
            continue
        weights = np.full(len(visible), 1 / (len(values) * (bins[1] - bins[0])))
        ax.hist(
            visible, bins=bins, weights=weights, histtype="stepfilled", linewidth=1.1,
            alpha=0.22, color=COLORS[name], edgecolor=COLORS[name],
            label=f"{name} (n={len(values):,}, shown={len(visible) / len(values):.1%})",
        )
    ax.set(xlim=limits, xlabel=axis_label(column), ylabel="Density (all rows)")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)


def discrete_bars(
    ax: plt.Axes, splits: dict[str, pd.DataFrame], column: str, numeric: bool
) -> None:
    values_by_split = {
        name: (display_values(frame, column) if numeric else frame[column]).dropna()
        for name, frame in splits.items()
    }
    ordered_as_number = numeric or pd.api.types.is_numeric_dtype(next(iter(splits.values()))[column])
    labels = sorted(
        set().union(*(set(values.unique()) for values in values_by_split.values())),
        key=lambda value: (float(value) if ordered_as_number else str(value)),
    )
    x = np.arange(len(labels))
    width = 0.26
    for index, (name, values) in enumerate(values_by_split.items()):
        probability = values.value_counts(normalize=True).reindex(labels, fill_value=0)
        ax.bar(
            x + (index - 1) * width, probability, width=width,
            color=COLORS[name], alpha=0.82, label=f"{name} (n={len(values):,})",
        )
    ax.set_xticks(x, [str(int(value)) if numeric else str(value) for value in labels])
    ax.tick_params(axis="x", labelrotation=45 if len(labels) > 6 else 0, labelsize=8)
    ax.set(xlim=(-0.7, len(labels) - 0.3), ylabel="Proportion", xlabel=axis_label(column))
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.25)


def distribution_distance(
    first: pd.Series, second: pd.Series, numerical: bool
) -> float:
    first = first.dropna()
    second = second.dropna()
    if numerical:
        return float(ks_2samp(first, second).statistic)
    first_frequency = first.value_counts(normalize=True)
    second_frequency = second.value_counts(normalize=True)
    labels = first_frequency.index.union(second_frequency.index)
    return float(
        (first_frequency.reindex(labels, fill_value=0) - second_frequency.reindex(labels, fill_value=0))
        .abs().sum() / 2
    )


def main() -> None:
    args = parse_args()
    source_train = pd.read_csv(args.train_csv, low_memory=False)
    source_test = pd.read_csv(args.test_csv, low_memory=False)
    fit_indices, validation_indices, method = split_indices(
        source_train, args.validation_fraction, args.group_column, args.seed
    )
    splits = {
        "Fit train": source_train.iloc[fit_indices].reset_index(drop=True),
        "Validation": source_train.iloc[validation_indices].reset_index(drop=True),
        "Test": source_test.reset_index(drop=True),
    }
    schema = TablePreprocessor(args.continuous_cols, args.exclude_cols).fit(splits["Fit train"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    columns = schema.model_columns
    summary = []
    overview, overview_axes = plt.subplots(3, 3, figsize=(18, 13), constrained_layout=True)
    for column, overview_ax in zip(columns, overview_axes.flat):
        numerical = column in schema.numerical_columns
        focused = column in FOCUS_LIMITS
        if focused:
            focused_limit = FOCUS_LIMITS[column]
            focused_limit = (
                focused_limit[0] / (1e3 if column == "iops" else 1),
                focused_limit[1] / (1e3 if column == "iops" else 1),
            )
            full_upper = max(display_values(frame, column).max() for frame in splits.values())
            full_limit = (0, float(full_upper))
            figure, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
            numerical_histogram(axes[0], splits, column, focused_limit)
            axes[0].set_title("Focused range")
            numerical_histogram(axes[1], splits, column, full_limit)
            axes[1].set_title("Full range")
            numerical_histogram(overview_ax, splits, column, focused_limit)
        else:
            figure, axes = plt.subplots(figsize=(11, 4), constrained_layout=True)
            discrete_bars(axes, splits, column, numerical)
            discrete_bars(overview_ax, splits, column, numerical)
        figure.suptitle(f"{column}: original values by split", fontsize=12)
        figure.savefig(args.output_dir / f"{column}.png", dpi=180)
        plt.close(figure)
        overview_ax.set_title(column, fontsize=11)

        source = {name: frame[column] for name, frame in splits.items()}
        summary.append({
            "column": column,
            "type": "numerical" if numerical else "categorical",
            "distance": "KS" if numerical else "TVD",
            "train_validation": distribution_distance(source["Fit train"], source["Validation"], numerical),
            "train_test": distribution_distance(source["Fit train"], source["Test"], numerical),
            "validation_test": distribution_distance(source["Validation"], source["Test"], numerical),
            "fit_unique": source["Fit train"].nunique(dropna=True),
            "validation_unique": source["Validation"].nunique(dropna=True),
            "test_unique": source["Test"].nunique(dropna=True),
            "validation_unseen_values": len(set(source["Validation"].dropna()) - set(source["Fit train"].dropna())) if not numerical else None,
            "test_unseen_values": len(set(source["Test"].dropna()) - set(source["Fit train"].dropna())) if not numerical else None,
        })
    overview.suptitle("HDD sequential: target distributions by split", fontsize=15)
    overview.savefig(args.output_dir / "overview.png", dpi=170)
    plt.close(overview)
    pd.DataFrame(summary).to_csv(args.output_dir / "distribution_shift.csv", index=False)
    metadata = {
        "train_csv": str(args.train_csv), "test_csv": str(args.test_csv),
        "seed": args.seed, "validation_fraction": args.validation_fraction,
        "split_method": method,
        "rows": {name: len(frame) for name, frame in splits.items()},
        "groups": {name: int(frame[args.group_column].nunique()) for name, frame in splits.items()},
        "numerical_columns": schema.numerical_columns,
        "categorical_columns": schema.categorical_columns,
        "constant_values": schema.constant_values,
        "excluded_columns": schema.excluded_columns,
    }
    (args.output_dir / "split_info.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"Saved plots and split summary to {args.output_dir}")


if __name__ == "__main__":
    main()
