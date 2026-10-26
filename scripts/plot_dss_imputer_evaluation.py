#!/usr/bin/env python3
"""Plot train or test LOCO reconstructions produced by eval_dss_imputer.py."""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

cache_dir = Path(tempfile.gettempdir()) / "dss_imputer_evaluation_plots"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", context="paper")

COLORS = {"ground_truth": "#2563EB", "model": "#DC2626"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot train or test LOCO estimates from eval_dss_imputer.py."
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--data-csv", type=Path, default=None)
    parser.add_argument("--test-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-label", default="DSS Imputer")
    parser.add_argument("--split-label", default=None)
    parser.add_argument("--plot-sample-size", type=int, default=20000)
    parser.add_argument("--max-discrete-values", type=int, default=50)
    parser.add_argument("--max-categorical-values", type=int, default=20)
    parser.add_argument("--iops-limits", nargs=2, type=float, default=None)
    parser.add_argument("--lat-limits", nargs=2, type=float, default=None)
    parser.add_argument("--iops-density-limits", nargs=2, type=float, default=None)
    parser.add_argument("--lat-density-limits", nargs=2, type=float, default=None)
    parser.add_argument(
        "--histogram-cols",
        nargs="*",
        default=[],
        help="Continuous columns to draw as histograms instead of KDE curves.",
    )
    parser.add_argument("--histogram-bins", type=int, default=50)
    parser.add_argument(
        "--density-scale",
        choices=["linear", "log10"],
        default="log10",
        help="Scale for continuous density plots; scatter plots remain linear.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_data_csv(args: argparse.Namespace, run_info: dict[str, Any]) -> Path:
    saved_truth = args.results_dir / "evaluation_truth.csv"
    if saved_truth.is_file():
        return saved_truth
    explicit = args.data_csv or args.test_csv
    if explicit is not None:
        return explicit
    stored = run_info.get("data_csv") or run_info.get("test_csv")
    if stored is None:
        raise KeyError("run_info.json contains neither data_csv nor test_csv.")
    path = Path(stored)
    if path.is_file():
        return path
    raise FileNotFoundError(
        f"CSV from run_info is unavailable: {path}. Pass --data-csv explicitly."
    )


def numeric(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def scaled_values(series: pd.Series, column: str) -> np.ndarray:
    values = numeric(series)
    if column == "iops":
        return values / 1e3
    if column == "lat":
        return values / 1e6
    return values


def axis_label(column: str) -> str:
    if column == "iops":
        return "IOPS ($10^3$)"
    if column == "lat":
        return "Latency ($\\mu$s)"
    return column


def target_unit(column: str) -> str:
    if column == "iops":
        return "IOPS"
    if column == "lat":
        return "microseconds"
    return "raw"


def target_divisor(column: str) -> float:
    if column == "lat":
        return 1e6
    return 1.0


def sampled(values: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if limit <= 0 or len(values) <= limit:
        return values
    indices = np.random.default_rng(seed).choice(len(values), size=limit, replace=False)
    return values[indices]


def density_transform(
    values: np.ndarray,
    column: str,
    density_scale: str = "log10",
) -> tuple[np.ndarray, str]:
    if density_scale not in {"linear", "log10"}:
        raise ValueError(f"Unsupported density scale: {density_scale}")
    if column == "lat" and density_scale == "log10":
        return (
            np.log10(1.0 + np.clip(values, 0.0, None)),
            "log10(1 + Latency [microseconds])",
        )
    return values, axis_label(column)


def density_limits(values: list[np.ndarray]) -> tuple[float, float]:
    combined = np.concatenate(values)
    low = float(np.quantile(combined, 0.001))
    high = float(np.quantile(combined, 0.999))
    if math.isclose(low, high):
        padding = max(abs(low) * 0.05, 1.0)
    else:
        padding = 0.04 * (high - low)
    return low - padding, high + padding


def is_discrete_integer(
    values: list[np.ndarray], max_discrete_values: int
) -> bool:
    combined = np.concatenate(values)
    if not len(combined):
        return False
    unique = np.unique(combined)
    return len(unique) <= max_discrete_values and bool(
        np.allclose(unique, np.rint(unique), rtol=0.0, atol=1e-9)
    )


def discrete_bin_edges(support: np.ndarray) -> np.ndarray:
    support = np.sort(np.unique(support.astype(float)))
    if len(support) == 1:
        return np.asarray([support[0] - 0.5, support[0] + 0.5])
    midpoints = (support[:-1] + support[1:]) / 2
    return np.concatenate(
        ([support[0] - (midpoints[0] - support[0])], midpoints, [support[-1] + (support[-1] - midpoints[-1])])
    )


def categorical_display_values(
    truth: pd.Series,
    prediction: pd.Series,
    max_values: int,
) -> tuple[dict[str, np.ndarray], list[str]]:
    reference = truth.dropna()
    numeric_categories = len(reference) > 0 and all(
        isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_))
        for value in reference
    )
    if numeric_categories:
        def normalize(series: pd.Series) -> pd.Series:
            values = series.astype(object)
            numbers = pd.to_numeric(values, errors="coerce")
            finite = numbers.notna() & np.isfinite(numbers)
            values.loc[finite] = numbers.loc[finite].map(
                lambda value: str(int(value)) if float(value).is_integer() else str(float(value))
            )
            return values
        truth, prediction = normalize(truth), normalize(prediction)
    actual = truth.fillna("__MISSING__").astype(str).reset_index(drop=True)
    predicted = prediction.fillna("__MISSING__").astype(str).reset_index(drop=True)
    counts = pd.concat([actual, predicted], ignore_index=True).value_counts()
    if len(counts) > max_values:
        retained = counts.index[: max_values - 1].tolist()
        support = retained + ["Other"]
        actual = actual.where(actual.isin(retained), "Other")
        predicted = predicted.where(predicted.isin(retained), "Other")
    else:
        support = counts.index.tolist()
    return {
        "ground_truth": actual.to_numpy(),
        "model": predicted.to_numpy(),
    }, support


def plot_distribution(
    truth: pd.Series,
    prediction: pd.Series,
    target: str,
    column_type: str,
    output_dir: Path,
    model_label: str,
    split_label: str,
    sample_size: int,
    max_discrete_values: int,
    seed: int,
    density_scale: str = "log10",
    display_limits: tuple[float, float] | None = None,
    histogram_columns: set[str] | None = None,
    histogram_bins: int = 50,
    max_categorical_values: int = 20,
    protocol_label: str = "LOCO",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = {"ground_truth": "Ground truth", "model": model_label}
    if column_type != "numerical":
        values_by_source, support = categorical_display_values(
            truth, prediction, max_categorical_values
        )

        def draw_categorical(
            ax: plt.Axes, sources: list[str], title: str
        ) -> None:
            positions = np.arange(len(support), dtype=float)
            width = 0.78 / len(sources)
            for source_index, source in enumerate(sources):
                probabilities = (
                    pd.Series(values_by_source[source])
                    .value_counts(normalize=True)
                    .reindex(support, fill_value=0.0)
                )
                offset = (source_index - (len(sources) - 1) / 2) * width
                ax.bar(
                    positions + offset,
                    probabilities.to_numpy(),
                    width=width,
                    alpha=0.68,
                    color=COLORS[source],
                    label=labels[source],
                )
            ax.set_xticks(positions)
            ax.set_xticklabels(support, rotation=40, ha="right")
            ax.set_xlabel(target)
            ax.set_ylabel("Probability")
            ax.set_title(title)
            ax.grid(True, axis="y", linestyle="--", alpha=0.5)
            if len(sources) > 1:
                ax.legend(frameon=True, loc="upper left", bbox_to_anchor=(1.01, 1.0))

        draw = draw_categorical
    else:
        raw = {
            "ground_truth": sampled(scaled_values(truth, target), sample_size, seed),
            "model": sampled(
                scaled_values(prediction, target), sample_size, seed + 1
            ),
        }
        discrete = is_discrete_integer(
            [numeric(truth), numeric(prediction)], max_discrete_values
        )
        continuous_histogram = target in (histogram_columns or set())
        transformed: dict[str, np.ndarray] = {}
        x_label = axis_label(target)
        if discrete:
            transformed = raw
            support = np.unique(np.concatenate(list(transformed.values())))
            bins = discrete_bin_edges(support)
            limits = (float(bins[0]), float(bins[-1]))
        else:
            for source, values in raw.items():
                transformed[source], x_label = density_transform(
                    values, target, density_scale
                )
            limits = density_limits(list(transformed.values()))
            if display_limits is not None:
                limit_values = np.asarray(display_limits, dtype=float)
                limits = tuple(
                    float(value)
                    for value in density_transform(
                        limit_values, target, density_scale
                    )[0]
                )
            if continuous_histogram:
                bins = np.linspace(limits[0], limits[1], histogram_bins + 1)

        def draw_numerical(ax: plt.Axes, sources: list[str], title: str) -> None:
            for source in sources:
                values = transformed[source]
                if discrete or continuous_histogram:
                    ax.hist(
                        values,
                        bins=bins,
                        density=True,
                        histtype="stepfilled",
                        alpha=0.30,
                        linewidth=1.2,
                        color=COLORS[source],
                        label=labels[source],
                    )
                elif len(np.unique(values)) > 1:
                    sns.kdeplot(
                        x=values,
                        fill=True,
                        alpha=0.26,
                        linewidth=1.3,
                        bw_adjust=0.75,
                        warn_singular=False,
                        color=COLORS[source],
                        label=labels[source],
                        ax=ax,
                    )
                else:
                    ax.axvline(
                        values[0],
                        color=COLORS[source],
                        linewidth=1.5,
                        label=labels[source],
                    )
            ax.set_xlim(*limits)
            ax.set_xlabel(x_label)
            ax.set_ylabel("Density")
            ax.set_title(title)
            ax.grid(True, linestyle="--", alpha=0.5)
            if len(sources) > 1:
                ax.legend(frameon=True, loc="upper left", bbox_to_anchor=(1.01, 1.0))

        draw = draw_numerical

    for source in ["ground_truth", "model"]:
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        draw(ax, [source], f"{protocol_label} {target} ({split_label}): {labels[source]}")
        fig.tight_layout()
        fig.savefig(output_dir / f"{source}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    draw(
        ax,
        ["ground_truth", "model"],
        f"{protocol_label} {target} ({split_label}): target distribution",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def scatter_limits(
    frames: list[pd.DataFrame],
) -> tuple[tuple[float, float], tuple[float, float]]:
    iops = np.concatenate([scaled_values(frame.iops, "iops") for frame in frames])
    latency = np.concatenate([scaled_values(frame.lat, "lat") for frame in frames])
    x_high = float(np.quantile(iops, 0.999))
    y_high = float(np.quantile(latency, 0.999))
    return (0.0, x_high * 1.05), (0.0, y_high * 1.12)


def draw_iops_latency(
    ax: plt.Axes,
    frame: pd.DataFrame,
    title: str,
    color: str,
    xlimits: tuple[float, float],
    ylimits: tuple[float, float],
    sample_size: int,
    seed: int,
    show_labels: bool,
) -> int:
    rows = len(frame)
    if sample_size > 0 and rows > sample_size:
        indices = np.sort(
            np.random.default_rng(seed).choice(rows, size=sample_size, replace=False)
        )
        frame = frame.iloc[indices]
    x = pd.to_numeric(frame.iops, errors="coerce").to_numpy(dtype=float) / 1e3
    y = pd.to_numeric(frame.lat, errors="coerce").to_numpy(dtype=float) / 1e6
    valid = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0)
    ax.scatter(
        x[valid],
        y[valid],
        s=0.1,
        c=color,
        marker="o",
        alpha=0.4,
        edgecolors="black",
        linewidth=0.2,
    )
    ax.set_xlim(*xlimits)
    ax.set_ylim(*ylimits)
    ax.set_title(title)
    if show_labels:
        ax.set_xlabel("IOPS ($10^3$)")
        ax.set_ylabel("Latency ($\\mu$s)")
    ax.grid(True, linestyle="--", alpha=0.6)
    return int((~valid).sum())


def plot_iops_latency(
    truth: pd.DataFrame,
    reconstruction: pd.DataFrame,
    scenario: str,
    output_dir: Path,
    model_label: str,
    sample_size: int,
    seed: int,
    xlimits: tuple[float, float] | None = None,
    ylimits: tuple[float, float] | None = None,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = {"ground_truth": truth, "model": reconstruction}
    automatic_xlimits, automatic_ylimits = scatter_limits(list(frames.values()))
    xlimits = xlimits or automatic_xlimits
    ylimits = ylimits or automatic_ylimits
    records: list[dict[str, Any]] = []
    labels = {"ground_truth": "Ground truth", "model": model_label}

    for source, frame in frames.items():
        fig, ax = plt.subplots(figsize=(3.5, 3))
        dropped = draw_iops_latency(
            ax,
            frame,
            f"{labels[source]}\n{scenario}",
            COLORS[source],
            xlimits,
            ylimits,
            sample_size,
            seed,
            show_labels=True,
        )
        fig.tight_layout()
        fig.savefig(output_dir / f"{source}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        records.append(
            {
                "scenario": scenario,
                "source": source,
                "xscale": "linear",
                "yscale": "linear",
                "xmin": xlimits[0],
                "xmax": xlimits[1],
                "ymin": ylimits[0],
                "ymax": ylimits[1],
                "nonfinite_or_negative_points_dropped": dropped,
            }
        )

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.1), sharex=True, sharey=True)
    for axis, source in zip(axes, ["ground_truth", "model"]):
        draw_iops_latency(
            axis,
            frames[source],
            labels[source],
            COLORS[source],
            xlimits,
            ylimits,
            sample_size,
            seed,
            show_labels=False,
        )
    fig.suptitle(scenario, fontsize=12)
    fig.supxlabel("IOPS ($10^3$)", fontsize=10)
    fig.supylabel("Latency ($\\mu$s)", fontsize=10)
    fig.tight_layout(rect=(0.04, 0.04, 1, 0.93))
    fig.savefig(output_dir / "comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return records


def plot_prediction_vs_truth(
    truth: pd.Series,
    prediction: pd.Series,
    target: str,
    output_path: Path,
    model_label: str,
    sample_size: int,
    seed: int,
) -> None:
    actual = scaled_values(truth, target)
    estimate = scaled_values(prediction, target)
    finite = np.isfinite(actual) & np.isfinite(estimate)
    actual = actual[finite]
    estimate = estimate[finite]
    if sample_size > 0 and len(actual) > sample_size:
        indices = np.random.default_rng(seed).choice(
            len(actual), size=sample_size, replace=False
        )
        actual = actual[indices]
        estimate = estimate[indices]
    high = float(np.quantile(np.concatenate([actual, estimate]), 0.999)) * 1.05

    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.scatter(
        actual,
        estimate,
        s=1.0,
        c=COLORS["model"],
        alpha=0.3,
        edgecolors="none",
    )
    if target == "lat":
        ax.set_xscale("symlog", linthresh=1.0, linscale=0.8)
        ax.set_yscale("symlog", linthresh=1.0, linscale=0.8)
        diagonal = np.concatenate(([0.0], np.geomspace(0.01, max(high, 0.01), 200)))
    else:
        diagonal = np.linspace(0.0, high, 200)
    ax.plot(diagonal, diagonal, color="#111827", linestyle="--", linewidth=1.0)
    ax.set_xlim(0.0, high)
    ax.set_ylim(0.0, high)
    ax.set_xlabel(f"Ground truth {axis_label(target)}")
    ax.set_ylabel(f"Predicted {axis_label(target)}")
    ax.set_title(f"{model_label}: LOCO {target}")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_row_comparison(
    truth: pd.Series,
    prediction: pd.Series,
    target: str,
    column_type: str,
    output_path: Path,
    sample_size: int,
    seed: int,
) -> None:
    available = truth.notna().to_numpy() & prediction.notna().to_numpy()
    positions = np.flatnonzero(available)
    if not len(positions):
        return
    limit = 250 if column_type == "numerical" else 30
    if sample_size > 0:
        limit = min(limit, sample_size)
    if len(positions) > limit:
        positions = np.sort(
            np.random.default_rng(seed).choice(positions, size=limit, replace=False)
        )

    if column_type == "numerical":
        divisor = 1e3 if target == "iops" else target_divisor(target)
        actual = pd.to_numeric(truth.iloc[positions]).to_numpy(dtype=float) / divisor
        predicted = (
            pd.to_numeric(prediction.iloc[positions]).to_numpy(dtype=float) / divisor
        )
        fig, ax = plt.subplots(figsize=(9.0, 3.4))
        ax.scatter(positions, actual, s=10, alpha=0.75, label="Ground truth")
        ax.scatter(positions, predicted, s=10, alpha=0.75, label="Prediction")
        ax.set_xlabel("Row index among observed targets")
        ax.set_ylabel(axis_label(target))
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
    else:
        actual = truth.iloc[positions].astype(str).to_numpy()
        predicted = prediction.iloc[positions].astype(str).to_numpy()
        rows = [
            [str(index), gt[:28], pred[:28]]
            for index, gt, pred in zip(positions, actual, predicted)
        ]
        fig, ax = plt.subplots(figsize=(8.0, max(2.0, 0.29 * len(rows) + 1.2)))
        ax.axis("off")
        table = ax.table(
            cellText=rows,
            colLabels=["Row", "Ground truth", "Prediction"],
            loc="center",
            cellLoc="left",
            colLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.35)
        for row_index, (gt, pred) in enumerate(zip(actual, predicted), start=1):
            table[(row_index, 2)].set_facecolor(
                "#DCFCE7" if gt == pred else "#FEE2E2"
            )
    ax.set_title(f"LOCO {target}: paired rows", pad=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_categorical_confusion(
    truth: pd.Series,
    prediction: pd.Series,
    target: str,
    output_path: Path,
    max_values: int,
) -> None:
    values, support = categorical_display_values(truth, prediction, max_values)
    if not support:
        return
    counts = pd.crosstab(
        pd.Series(values["ground_truth"], name="Ground truth"),
        pd.Series(values["model"], name="Prediction"),
    ).reindex(index=support, columns=support, fill_value=0)
    normalized = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    width = max(5.0, 0.45 * len(support) + 2)
    fig, ax = plt.subplots(figsize=(width, width))
    image = ax.imshow(normalized.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(support)), labels=support, rotation=45, ha="right")
    ax.set_yticks(range(len(support)), labels=support)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground truth")
    ax.set_title(f"LOCO {target}: row-normalized confusion")
    fig.colorbar(image, ax=ax, label="Fraction of ground-truth class")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_metrics_table(
    metrics: pd.DataFrame, path: Path, model_label: str, split_label: str
) -> None:
    if metrics.empty:
        return
    columns = ["column", "r2", "rmse_display", "mae_display", "nrmse", "unit"]
    display = metrics.loc[:, columns].copy()
    display.columns = ["Target", "R2", "RMSE", "MAE", "nRMSE", "Unit"]
    for column in ["R2", "RMSE", "MAE", "nRMSE"]:
        display[column] = display[column].map(lambda value: f"{value:.4g}")
    fig, ax = plt.subplots(figsize=(7.4, 2.2))
    ax.axis("off")
    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#CBD5E1")
        if row == 0:
            cell.set_facecolor("#334155")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F1F5F9")
    ax.set_title(
        f"{model_label}: {split_label} LOCO numerical metrics", fontsize=12, pad=12
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.max_discrete_values < 1:
        raise ValueError("max-discrete-values must be positive.")
    if args.max_categorical_values < 2:
        raise ValueError("max-categorical-values must be at least two.")
    if args.histogram_bins < 2:
        raise ValueError("histogram-bins must be at least 2.")
    run_info_path = args.results_dir / "run_info.json"
    run_info = json.loads(run_info_path.read_text())
    data_path = resolve_data_csv(args, run_info)
    split_label = args.split_label or run_info.get("split_label", "evaluation")
    truth = pd.read_csv(data_path, low_memory=False, na_values=["?"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = args.results_dir / "metrics_by_column.csv"
    metrics = pd.read_csv(metrics_path)
    augmented_rows: list[dict[str, Any]] = []
    scale_rows: list[dict[str, Any]] = []

    scenario_dirs = sorted(
        path
        for path in args.results_dir.glob("loco_*")
        if path.is_dir() and (path / "point_estimate.csv").is_file()
    )
    for scenario_index, scenario_dir in enumerate(scenario_dirs):
        target = scenario_dir.name.removeprefix("loco_")
        reconstruction = pd.read_csv(
            scenario_dir / "point_estimate.csv", low_memory=False
        )
        if len(reconstruction) != len(truth):
            raise ValueError(
                f"{scenario_dir} has {len(reconstruction)} rows; expected {len(truth)}"
            )
        if target not in reconstruction or target not in truth:
            raise ValueError(f"Target {target!r} is absent from the reconstruction")
        scenario_metrics = metrics[metrics["scenario"] == scenario_dir.name]
        if scenario_metrics.empty:
            raise ValueError(f"Metrics are missing for {scenario_dir.name}.")
        row = scenario_metrics.iloc[0].to_dict()
        column_type = str(row["type"])
        observed = truth[target].notna()
        target_truth = truth.loc[observed, target].reset_index(drop=True)
        target_prediction = reconstruction.loc[observed, target].reset_index(drop=True)

        scenario_output = args.output_dir / scenario_dir.name
        plot_distribution(
            target_truth,
            target_prediction,
            target,
            column_type,
            scenario_output / "distribution",
            args.model_label,
            split_label,
            args.plot_sample_size,
            args.max_discrete_values,
            args.seed + scenario_index * 100,
            args.density_scale,
            (
                tuple(args.iops_density_limits)
                if target == "iops" and args.iops_density_limits
                else tuple(args.lat_density_limits)
                if target == "lat" and args.lat_density_limits
                else None
            ),
            set(args.histogram_cols),
            args.histogram_bins,
            args.max_categorical_values,
        )
        plot_row_comparison(
            target_truth,
            target_prediction,
            target,
            column_type,
            scenario_output / "row_comparison.png",
            args.plot_sample_size,
            args.seed + scenario_index * 100,
        )
        if target in {"iops", "lat"} and {"iops", "lat"}.issubset(truth.columns):
            scale_rows.extend(
                plot_iops_latency(
                    truth[["iops", "lat"]],
                    reconstruction[["iops", "lat"]],
                    f"LOCO {target} ({split_label})",
                    scenario_output / "iops_latency",
                    args.model_label,
                    args.plot_sample_size,
                    args.seed + scenario_index * 100,
                    tuple(args.iops_limits) if args.iops_limits else None,
                    tuple(args.lat_limits) if args.lat_limits else None,
                )
            )
        if column_type == "numerical":
            plot_prediction_vs_truth(
                target_truth,
                target_prediction,
                target,
                scenario_output / "prediction_vs_truth.png",
                args.model_label,
                args.plot_sample_size,
                args.seed + scenario_index * 100,
            )
            actual = pd.to_numeric(target_truth, errors="coerce").to_numpy(
                dtype=float
            )
            variance = float(np.nanvar(actual))
            row["normalized_mse"] = row["mse"] / variance if variance > 0 else np.nan
            row["nrmse"] = math.sqrt(row["normalized_mse"])
            divisor = target_divisor(target)
            row["rmse_display"] = row["rmse"] / divisor
            row["mae_display"] = row["mae"] / divisor
            row["unit"] = target_unit(target)
            augmented_rows.append(row)
        else:
            plot_categorical_confusion(
                target_truth,
                target_prediction,
                target,
                scenario_output / "confusion_matrix.png",
                args.max_categorical_values,
            )

    augmented = pd.DataFrame(augmented_rows)
    augmented.to_csv(args.output_dir / "metrics_display_units.csv", index=False)
    pd.DataFrame(scale_rows).to_csv(args.output_dir / "plot_scales.csv", index=False)
    save_metrics_table(
        augmented,
        args.output_dir / "metrics_summary.png",
        args.model_label,
        split_label,
    )

    output_info = {
        "source_results": str(args.results_dir.resolve()),
        "data_csv": str(data_path.resolve()),
        "split_label": split_label,
        "output_dir": str(args.output_dir.resolve()),
        "model_label": args.model_label,
        "checkpoint": run_info.get("checkpoint"),
        "checkpoint_epoch": run_info.get("checkpoint_epoch"),
        "rows": len(truth),
        "samples_used_for_point_estimate": run_info.get("samples"),
        "scenarios": [path.name for path in scenario_dirs],
        "latency_display": "Raw latency divided by 1e6 and labeled microseconds.",
        "latency_scatter_scale": "Linear.",
        "density_scale": args.density_scale,
        "iops_density_limits": args.iops_density_limits,
        "lat_density_limits": args.lat_density_limits,
        "histogram_columns": args.histogram_cols,
        "histogram_bins": args.histogram_bins,
        "max_discrete_values": args.max_discrete_values,
        "max_categorical_values": args.max_categorical_values,
        "distribution_note": (
            "These are deterministic point-estimate distributions. Individual "
            "stochastic sample CSV files were not present in the source directory."
        ),
    }
    (args.output_dir / "run_info.json").write_text(json.dumps(output_info, indent=2))
    print(f"Saved plots to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
