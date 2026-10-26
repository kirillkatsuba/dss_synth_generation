#!/usr/bin/env python3
"""Build training and held-out imputation plots for the DSS diffusion model."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

_cache_dir = Path(tempfile.gettempdir()) / "dss_imputer_plot_cache"
_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_dir / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_dir / "xdg"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402


sns.set_theme(style="whitegrid", context="paper")

LOSS_PANELS = [
    ("Total loss", "train_loss", "validation_loss"),
    ("Numerical denoising MSE", "train_numerical_loss", "validation_numerical_loss"),
    (
        "Categorical cross-entropy",
        "train_categorical_loss",
        "validation_categorical_loss",
    ),
]
GENERATION_METRICS = [
    "overall_quality",
    "column_shapes",
    "column_pair_trends",
    "c2st",
    "alpha_precision",
    "beta_recall",
    "authenticity",
    "dcr",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot DSS imputer training history and held-out predictions."
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, default=None)
    parser.add_argument("--test-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--targets", nargs="+", default=["iops", "lat"])
    parser.add_argument("--sample-size", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def aligned_values(
    truth: pd.DataFrame,
    estimate: pd.DataFrame,
    column: str,
) -> tuple[np.ndarray, np.ndarray]:
    actual = pd.to_numeric(truth[column], errors="coerce").to_numpy(dtype=float)
    predicted = pd.to_numeric(estimate[column], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(actual) & np.isfinite(predicted)
    return actual[finite], predicted[finite]


def sample_pair(
    actual: np.ndarray,
    predicted: np.ndarray,
    sample_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if sample_size <= 0 or len(actual) <= sample_size:
        return actual, predicted
    indices = np.random.default_rng(seed).choice(
        len(actual), size=sample_size, replace=False
    )
    return actual[indices], predicted[indices]


def plot_training_losses(history: pd.DataFrame, output_dir: Path) -> None:
    panels = [panel for panel in LOSS_PANELS if set(panel[1:]) <= set(history)]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(4.3 * len(panels), 3.2))
    axes = np.atleast_1d(axes)
    best_epoch = None
    if "validation_loss" in history:
        best_epoch = int(history.loc[history["validation_loss"].idxmin(), "epoch"])
    for ax, (title, train_column, validation_column) in zip(axes, panels):
        ax.plot(history["epoch"], history[train_column], label="train", linewidth=1.7)
        ax.plot(
            history["epoch"],
            history[validation_column],
            label="validation",
            linewidth=1.7,
        )
        if best_epoch is not None:
            ax.axvline(
                best_epoch,
                color="black",
                linestyle="--",
                linewidth=0.9,
                alpha=0.7,
                label="best total val" if ax is axes[0] else None,
            )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
    save_figure(fig, output_dir / "training_losses.png")


def plot_mask_mixture(history: pd.DataFrame, output_dir: Path) -> None:
    prefix = "train_mask_"
    suffix = "_fraction"
    columns = [
        column
        for column in history
        if column.startswith(prefix) and column.endswith(suffix)
    ]
    if not columns:
        return
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    for column in columns:
        label = column[len(prefix) : -len(suffix)]
        ax.plot(history["epoch"], history[column], label=label, linewidth=1.4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Fraction of batches")
    ax.set_ylim(0, 1)
    ax.set_title("Sampled training mask mixture")
    ax.legend(ncol=min(3, len(columns)))
    save_figure(fig, output_dir / "training_mask_mixture.png")


def plot_history_metric(
    ax: plt.Axes,
    group: pd.DataFrame,
    metric: str,
    label: str,
    divisor: float = 1.0,
) -> None:
    valid = group.dropna(subset=[metric]).sort_values("epoch")
    if valid.empty:
        return
    x = pd.to_numeric(valid["epoch"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(valid[metric], errors="coerce").to_numpy(dtype=float) / divisor
    line = ax.plot(x, y, marker="o", markersize=3, label=label)[0]
    std_column = f"{metric}_std"
    if std_column in valid:
        std = (
            pd.to_numeric(valid[std_column], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
            / divisor
        )
        ax.fill_between(
            x,
            y - std,
            y + std,
            color=line.get_color(),
            alpha=0.15,
            linewidth=0,
        )


def plot_generative_metrics(history: pd.DataFrame, output_dir: Path) -> None:
    metrics = [metric for metric in GENERATION_METRICS if metric in history]
    if not metrics or "scenario" not in history:
        return
    columns = 2
    rows = int(np.ceil(len(metrics) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(9.2, 2.8 * rows))
    axes = np.asarray(axes).reshape(-1)
    for ax, metric in zip(axes, metrics):
        for scenario, group in history.groupby("scenario", sort=False):
            plot_history_metric(ax, group, metric, str(scenario))
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel("Epoch")
        ax.set_ylim(-0.05, 1.05)
    for ax in axes[len(metrics) :]:
        ax.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(3, len(labels)))
        fig.subplots_adjust(top=0.94)
    save_figure(fig, output_dir / "generative_metrics.png")


def target_scale(column: str) -> tuple[float, str]:
    if column == "iops":
        return 1e3, "IOPS (10^3)"
    if column == "lat":
        return 1e6, "Latency (microseconds)"
    return 1.0, column


def plot_generative_target_accuracy(
    history: pd.DataFrame,
    targets: list[str],
    output_dir: Path,
) -> None:
    available_targets = [
        target for target in targets if f"{target}_r2" in history.columns
    ]
    if not available_targets or "scenario" not in history:
        return
    fig, axes = plt.subplots(
        len(available_targets),
        3,
        figsize=(12.0, 3.0 * len(available_targets)),
        squeeze=False,
    )
    for row, target in enumerate(available_targets):
        scale, unit = target_scale(target)
        specs = [
            (f"{target}_r2", "R2", 1.0),
            (f"{target}_rmse", f"RMSE, {unit}", scale),
            (f"{target}_mae", f"MAE, {unit}", scale),
        ]
        for ax, (metric, title, divisor) in zip(axes[row], specs):
            if metric not in history:
                ax.set_visible(False)
                continue
            for scenario, group in history.groupby("scenario", sort=False):
                plot_history_metric(ax, group, metric, str(scenario), divisor=divisor)
            ax.set_title(f"{target}: {title}")
            ax.set_xlabel("Epoch")
            if metric.endswith("_r2"):
                ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(3, len(labels)))
        fig.subplots_adjust(top=0.91)
    save_figure(fig, output_dir / "generative_target_accuracy.png")


def plot_test_metrics(
    metrics: pd.DataFrame,
    targets: list[str],
    output_dir: Path,
) -> None:
    metrics = metrics[
        (metrics["type"] == "numerical") & metrics["column"].isin(targets)
    ].copy()
    if metrics.empty:
        return
    available_targets = [
        target for target in targets if target in set(metrics["column"])
    ]
    fig, axes = plt.subplots(
        len(available_targets),
        3,
        figsize=(12.0, 3.1 * len(available_targets)),
        squeeze=False,
    )
    for row, target in enumerate(available_targets):
        target_rows = metrics[metrics["column"] == target]
        scale, unit = target_scale(target)
        for ax, (metric, title, divisor) in zip(
            axes[row],
            [
                ("r2", "R2", 1.0),
                ("rmse", f"RMSE, {unit}", scale),
                ("mae", f"MAE, {unit}", scale),
            ],
        ):
            values = pd.to_numeric(target_rows[metric], errors="coerce") / divisor
            positions = np.arange(len(target_rows))
            ax.bar(positions, values, color="tab:blue", alpha=0.8)
            ax.set_xticks(positions)
            ax.set_xticklabels(target_rows["scenario"], rotation=30, ha="right")
            ax.set_title(f"{target}: {title}")
            if metric == "r2":
                ax.axhline(0, color="black", linewidth=0.8)
    save_figure(fig, output_dir / "test_metrics.png")


def plot_density(
    actual: np.ndarray,
    predicted: np.ndarray,
    target: str,
    scenario: str,
    output_path: Path,
) -> None:
    scale, unit = target_scale(target)
    combined_raw = np.concatenate([actual, predicted])
    combined_unique = np.unique(combined_raw)
    discrete = len(combined_unique) <= 50 and bool(
        np.allclose(combined_unique, np.rint(combined_unique), rtol=0.0, atol=1e-9)
    )
    actual = actual / scale
    predicted = predicted / scale
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    if discrete:
        support = np.unique(np.concatenate([actual, predicted]))
        if len(support) == 1:
            bins = np.asarray([support[0] - 0.5, support[0] + 0.5])
        else:
            midpoints = (support[:-1] + support[1:]) / 2
            bins = np.concatenate(
                (
                    [support[0] - (midpoints[0] - support[0])],
                    midpoints,
                    [support[-1] + (support[-1] - midpoints[-1])],
                )
            )
        ax.hist(actual, bins=bins, density=True, alpha=0.35, label="real")
        ax.hist(predicted, bins=bins, density=True, alpha=0.35, label="imputed")
    else:
        try:
            sns.kdeplot(
                actual,
                fill=True,
                cut=0,
                alpha=0.35,
                linewidth=1.3,
                label="real",
                ax=ax,
            )
            sns.kdeplot(
                predicted,
                fill=True,
                cut=0,
                alpha=0.35,
                linewidth=1.3,
                label="imputed",
                ax=ax,
            )
        except (TypeError, ValueError, np.linalg.LinAlgError):
            ax.hist(actual, bins=50, density=True, alpha=0.35, label="real")
            ax.hist(predicted, bins=50, density=True, alpha=0.35, label="imputed")
    ax.set_xlabel(unit)
    ax.set_ylabel("Density")
    ax.set_title(f"{scenario}: {target} distribution")
    ax.legend()
    save_figure(fig, output_path)


def plot_prediction_diagnostics(
    actual: np.ndarray,
    predicted: np.ndarray,
    target: str,
    scenario: str,
    output_path: Path,
    sample_size: int,
    seed: int,
) -> None:
    actual, predicted = sample_pair(actual, predicted, sample_size, seed)
    scale, unit = target_scale(target)
    actual = actual / scale
    predicted = predicted / scale
    residual = predicted - actual
    combined = np.concatenate([actual, predicted])
    lower, upper = np.quantile(combined, [0.005, 0.995])
    if not np.isfinite(lower) or not np.isfinite(upper) or lower == upper:
        lower, upper = float(np.min(combined)), float(np.max(combined) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5))
    axes[0].scatter(actual, predicted, s=2, alpha=0.25, linewidths=0)
    axes[0].plot([lower, upper], [lower, upper], color="black", linestyle="--")
    axes[0].set_xlim(lower, upper)
    axes[0].set_ylim(lower, upper)
    axes[0].set_xlabel(f"Real {unit}")
    axes[0].set_ylabel(f"Imputed {unit}")
    axes[0].set_title("Truth vs. reconstruction")

    axes[1].scatter(actual, residual, s=2, alpha=0.25, linewidths=0)
    axes[1].axhline(0, color="black", linestyle="--")
    axes[1].set_xlim(lower, upper)
    axes[1].set_xlabel(f"Real {unit}")
    axes[1].set_ylabel(f"Residual, {unit}")
    axes[1].set_title("Residuals")
    fig.suptitle(scenario)
    save_figure(fig, output_path)


def plot_iops_latency(
    frame: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    if not {"iops", "lat"} <= set(frame):
        return
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.scatter(
        pd.to_numeric(frame["iops"], errors="coerce").to_numpy() / 1e3,
        pd.to_numeric(frame["lat"], errors="coerce").to_numpy() / 1e6,
        s=0.1,
        c="blue",
        marker="o",
        alpha=0.4,
        edgecolors="black",
        linewidth=0.2,
    )
    ax.set_xlabel("IOPs ($10^3$)")
    ax.set_ylabel("Latency ($\\mu$s)")
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 14)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.6)
    save_figure(fig, output_path)


def plot_evaluation(
    evaluation_dir: Path,
    test_csv: Path,
    targets: list[str],
    output_dir: Path,
    sample_size: int,
    seed: int,
) -> None:
    truth = pd.read_csv(test_csv, low_memory=False, na_values=["?"])
    metrics_path = evaluation_dir / "metrics_by_column.csv"
    if metrics_path.exists():
        plot_test_metrics(pd.read_csv(metrics_path), targets, output_dir)

    plot_iops_latency(truth, output_dir / "real_iops_lat.png", "Real test data")
    for scenario_dir in sorted(evaluation_dir.glob("loco_*")):
        estimate_path = scenario_dir / "point_estimate.csv"
        if not estimate_path.exists():
            continue
        scenario = scenario_dir.name
        target = scenario.removeprefix("loco_")
        if target not in targets or target not in truth:
            continue
        estimate = pd.read_csv(estimate_path, low_memory=False, na_values=["?"])
        if len(estimate) != len(truth):
            raise ValueError(
                f"{estimate_path} has {len(estimate)} rows; expected {len(truth)}."
            )
        actual, predicted = aligned_values(truth, estimate, target)
        if not len(actual):
            continue
        plot_density(
            actual,
            predicted,
            target,
            scenario,
            output_dir / f"{scenario}_density.png",
        )
        plot_prediction_diagnostics(
            actual,
            predicted,
            target,
            scenario,
            output_dir / f"{scenario}_prediction_diagnostics.png",
            sample_size,
            seed,
        )
        plot_iops_latency(
            estimate,
            output_dir / f"{scenario}_iops_lat.png",
            f"Imputed test data: {scenario}",
        )


def plot_saved_validation(
    model_dir: Path,
    targets: list[str],
    output_dir: Path,
    sample_size: int,
    seed: int,
) -> None:
    real_path = model_dir / "generative_validation_real.csv"
    if not real_path.exists():
        return
    truth = pd.read_csv(real_path, low_memory=False, na_values=["?"])
    plot_iops_latency(
        truth,
        output_dir / "validation_real_iops_lat.png",
        "Real validation data",
    )
    pattern = "generative_validation_loco_*_latest.csv"
    for estimate_path in sorted(model_dir.glob(pattern)):
        scenario = estimate_path.stem
        scenario = scenario.removeprefix("generative_validation_")
        scenario = scenario.removesuffix("_latest")
        target = scenario.removeprefix("loco_")
        if target not in targets or target not in truth:
            continue
        estimate = pd.read_csv(estimate_path, low_memory=False, na_values=["?"])
        if len(estimate) != len(truth):
            raise ValueError(
                f"{estimate_path} has {len(estimate)} rows; expected {len(truth)}."
            )
        actual, predicted = aligned_values(truth, estimate, target)
        if not len(actual):
            continue
        prefix = f"validation_{scenario}"
        plot_density(
            actual,
            predicted,
            target,
            prefix,
            output_dir / f"{prefix}_density.png",
        )
        plot_prediction_diagnostics(
            actual,
            predicted,
            target,
            prefix,
            output_dir / f"{prefix}_prediction_diagnostics.png",
            sample_size,
            seed,
        )
        plot_iops_latency(
            estimate,
            output_dir / f"{prefix}_iops_lat.png",
            f"Validation reconstruction: {scenario}",
        )


def main() -> None:
    args = parse_args()
    if args.sample_size < 0:
        raise ValueError("sample-size cannot be negative.")
    output_dir = args.output_dir or args.model_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    history_path = args.model_dir / "history.csv"
    if history_path.exists():
        history = pd.read_csv(history_path)
        plot_training_losses(history, output_dir)
        plot_mask_mixture(history, output_dir)
    else:
        print(f"Training history not found: {history_path}", flush=True)

    generative_path = args.model_dir / "generative_history.csv"
    if generative_path.exists():
        generative = pd.read_csv(generative_path)
        plot_generative_metrics(generative, output_dir)
        plot_generative_target_accuracy(generative, args.targets, output_dir)
    else:
        print(f"Generative history not found: {generative_path}", flush=True)

    plot_saved_validation(
        args.model_dir,
        args.targets,
        output_dir,
        args.sample_size,
        args.seed,
    )

    if args.evaluation_dir is not None or args.test_csv is not None:
        if args.evaluation_dir is None or args.test_csv is None:
            raise ValueError("evaluation-dir and test-csv must be provided together.")
        plot_evaluation(
            args.evaluation_dir,
            args.test_csv,
            args.targets,
            output_dir,
            args.sample_size,
            args.seed,
        )

    print(f"Plots saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
