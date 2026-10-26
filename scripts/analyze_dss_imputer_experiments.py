#!/usr/bin/env python3
# ruff: noqa: E402
"""Analyze multi-seed DSS imputer runs and build comparison artifacts."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

cache_dir = Path(tempfile.gettempdir()) / "dss_imputer_experiment_analysis"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn.metrics import mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dss_imputer.generative_metrics import (
    alpha_precision_beta_recall,
    logistic_detection_quality,
)


sns.set_theme(style="whitegrid", context="paper")

LOCO_TARGETS = {"loco_iops": "iops", "loco_lat": "lat"}
GEN_METRICS = [
    "overall_quality",
    "column_shapes",
    "column_pair_trends",
    "c2st",
    "alpha_precision",
    "beta_recall",
]
COLORS = {
    "raw": "#7C3AED",
    "log": "#D97706",
    "old_lr": "#DC2626",
    "new_lr": "#2563EB",
}


@dataclass(frozen=True)
class RunData:
    path: Path
    name: str
    seed: int
    config: dict[str, Any]
    history: pd.DataFrame
    generative: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-run-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--raw-baseline-dir", type=Path, required=True)
    parser.add_argument("--log-baseline-dir", type=Path, required=True)
    parser.add_argument("--external-summary", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--outlier-quantile", type=float, default=0.005)
    return parser.parse_args()


def load_run(path: Path) -> RunData:
    config = json.loads((path / "run_config.json").read_text())
    history = pd.read_csv(path / "history.csv")
    generative = pd.read_csv(path / "generative_history.csv")
    return RunData(
        path=path,
        name=path.name,
        seed=int(config.get("seed", 42)),
        config=config,
        history=history,
        generative=generative,
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def aggregate_curve(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    available = frame.dropna(subset=[value])
    return available.groupby("epoch", as_index=False).agg(
        **{
            value: (value, "mean"),
            f"{value}_between_seed_std": (value, "std"),
            "count": (value, "count"),
        }
    )


def draw_band(
    ax: plt.Axes,
    frame: pd.DataFrame,
    value: str,
    label: str,
    color: str,
    linestyle: str = "-",
) -> None:
    if frame.empty:
        return
    x = frame["epoch"].to_numpy(dtype=float)
    y = frame[value].to_numpy(dtype=float)
    ax.plot(x, y, label=label, color=color, linestyle=linestyle, linewidth=1.7)
    std_column = f"{value}_between_seed_std"
    if std_column in frame:
        std = frame[std_column].fillna(0.0).to_numpy(dtype=float)
        ax.fill_between(x, y - std, y + std, color=color, alpha=0.16, linewidth=0)


def combined_history(runs: list[RunData], kind: str) -> pd.DataFrame:
    frames = []
    for run in runs:
        frame = run.history if kind == "history" else run.generative
        current = frame.copy()
        current["seed"] = run.seed
        current["run"] = run.name
        frames.append(current)
    return pd.concat(frames, ignore_index=True)


def plot_mean_training_curves(runs: list[RunData], output: Path) -> None:
    history = combined_history(runs, "history")
    common_max_epoch = min(int(run.history.epoch.max()) for run in runs)
    history = history[history.epoch <= common_max_epoch]
    panels = [
        ("loss", "Total loss"),
        ("numerical_loss", "Numerical denoising MSE"),
        ("categorical_loss", "Categorical cross-entropy"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.5))
    for ax, (metric, title) in zip(axes, panels):
        for prefix, color in [("train", "#2563EB"), ("validation", "#D97706")]:
            value = f"{prefix}_{metric}"
            draw_band(ax, aggregate_curve(history, value), value, prefix, color)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
    fig.suptitle("DSS log, lr=5e-5: mean across three seeds +/- 1 SD")
    save_figure(fig, output)


def plot_mean_generative_curves(runs: list[RunData], output: Path) -> None:
    generative = combined_history(runs, "generative")
    scenarios = ["mcar_30", "loco_iops", "loco_lat"]
    fig, axes = plt.subplots(3, 6, figsize=(19.0, 9.0), sharex=True)
    for row, scenario in enumerate(scenarios):
        scenario_frame = generative[generative.scenario == scenario]
        for column, metric in enumerate(GEN_METRICS):
            ax = axes[row, column]
            draw_band(
                ax,
                aggregate_curve(scenario_frame, metric),
                metric,
                scenario,
                "#2563EB",
            )
            ax.set_title(metric.replace("_", " ").title())
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel("Epoch")
            if column == 0:
                ax.set_ylabel(f"{scenario}\nScore")
    fig.suptitle(
        "Generative validation: mean of per-seed means +/- between-seed SD"
    )
    save_figure(fig, output)


def plot_mean_loco_accuracy(runs: list[RunData], output: Path) -> None:
    generative = combined_history(runs, "generative")
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.8), sharex=True)
    for row, (scenario, target) in enumerate(LOCO_TARGETS.items()):
        frame = generative[generative.scenario == scenario]
        panels = [
            (f"{target}_r2", "R2", 1.0),
            (f"{target}_normalized_mse", "NRMSE", 1.0),
            (f"{target}_rmse", "RMSE", 1e3 if target == "iops" else 1e6),
        ]
        for ax, (metric, title, divisor) in zip(axes[row], panels):
            aggregate = aggregate_curve(frame, metric)
            aggregate[metric] /= divisor
            aggregate[f"{metric}_between_seed_std"] /= divisor
            draw_band(ax, aggregate, metric, scenario, "#2563EB")
            if metric.endswith("_r2"):
                ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
            ax.set_title(f"{target}: {title}")
            ax.set_xlabel("Epoch")
            if title == "RMSE":
                ax.set_ylabel("IOPS (10^3)" if target == "iops" else "Latency (us)")
    fig.suptitle("LOCO accuracy: mean across three seeds +/- 1 SD")
    save_figure(fig, output)


def plot_lr_comparison(
    baseline: RunData, new_runs: list[RunData], output: Path
) -> None:
    new_history = combined_history(new_runs, "history")
    common_max_epoch = min(int(run.history.epoch.max()) for run in new_runs)
    new_history = new_history[new_history.epoch <= common_max_epoch]
    new_generative = combined_history(new_runs, "generative")
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2))

    for ax, value, title in [
        (axes[0, 0], "validation_loss", "Validation total loss"),
        (axes[0, 1], "validation_numerical_loss", "Validation numerical loss"),
    ]:
        ax.plot(
            baseline.history.epoch,
            baseline.history[value],
            color=COLORS["old_lr"],
            label="lr=2e-4, seed 42",
        )
        draw_band(
            ax,
            aggregate_curve(new_history, value),
            value,
            "lr=5e-5, 3 seeds",
            COLORS["new_lr"],
        )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()

    for ax, metric, title in [
        (axes[1, 0], "c2st", "LOCO C2ST quality"),
        (axes[1, 1], "beta_recall", "LOCO beta-recall"),
    ]:
        for scenario, linestyle in [("loco_iops", "-"), ("loco_lat", "--")]:
            old = baseline.generative[baseline.generative.scenario == scenario]
            ax.plot(
                old.epoch,
                old[metric],
                color=COLORS["old_lr"],
                linestyle=linestyle,
                label=f"2e-4 {scenario}",
            )
            new = new_generative[new_generative.scenario == scenario]
            draw_band(
                ax,
                aggregate_curve(new, metric),
                metric,
                f"5e-5 {scenario}",
                COLORS["new_lr"],
                linestyle=linestyle,
            )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7)
    fig.suptitle("Learning-rate comparison (validation only)")
    save_figure(fig, output)


def plot_raw_log_comparison(raw: RunData, log: RunData, output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.2))
    for ax, value, title in zip(
        axes[0],
        ["validation_loss", "validation_numerical_loss", "validation_categorical_loss"],
        ["Total validation loss", "Numerical validation loss", "Categorical validation loss"],
    ):
        ax.plot(raw.history.epoch, raw.history[value], color=COLORS["raw"], label="raw")
        ax.plot(log.history.epoch, log.history[value], color=COLORS["log"], label="log")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()

    for ax, metric, title in zip(
        axes[1],
        ["overall_quality", "c2st", "beta_recall"],
        ["Overall quality", "C2ST quality", "Beta-recall"],
    ):
        for run, transform, color in [
            (raw, "raw", COLORS["raw"]),
            (log, "log", COLORS["log"]),
        ]:
            for scenario, linestyle in [("loco_iops", "-"), ("loco_lat", "--")]:
                frame = run.generative[run.generative.scenario == scenario]
                ax.plot(
                    frame.epoch,
                    frame[metric],
                    color=color,
                    linestyle=linestyle,
                    label=f"{transform} {scenario}",
                )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7)
    fig.suptitle("Raw vs log transform, lr=2e-4, seed 42")
    save_figure(fig, output)


def target_metadata(target: str) -> dict[str, Any]:
    return {
        "METADATA_SPEC_VERSION": "SINGLE_TABLE_V1",
        "columns": {target: {"sdtype": "numerical"}},
    }


def target_metrics(
    real: pd.Series,
    generated: pd.Series,
    target: str,
    quantile: float,
    central_only: bool,
    c2st_seed: int,
) -> dict[str, float]:
    truth = pd.to_numeric(real, errors="coerce").to_numpy(dtype=float)
    prediction = pd.to_numeric(generated, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(truth) & np.isfinite(prediction)
    truth = truth[finite]
    prediction = prediction[finite]
    lower, upper = np.quantile(truth, [quantile, 1 - quantile])
    outside_rate = float(np.mean((prediction < lower) | (prediction > upper)))
    if central_only:
        selected = (truth >= lower) & (truth <= upper)
        truth = truth[selected]
        prediction = prediction[selected]
    mse = float(mean_squared_error(truth, prediction))
    variance = float(np.var(truth))
    real_frame = pd.DataFrame({target: truth})
    generated_frame = pd.DataFrame({target: prediction})
    alpha, beta, _ = alpha_precision_beta_recall(real_frame, generated_frame)
    return {
        "rows": float(len(truth)),
        "lower_threshold": float(lower),
        "upper_threshold": float(upper),
        "generated_outside_central_rate": outside_rate,
        "r2": float(r2_score(truth, prediction)),
        "rmse": math.sqrt(mse),
        "nrmse": math.sqrt(mse / variance) if variance > 0 else np.nan,
        "ks_similarity": 1.0 - float(ks_2samp(truth, prediction).statistic),
        "c2st_quality": logistic_detection_quality(
            real_frame,
            generated_frame,
            target_metadata(target),
            seed=c2st_seed,
        ),
        "alpha_precision": alpha,
        "beta_recall": beta,
    }


def latest_trial_paths(run: RunData, scenario: str) -> list[Path]:
    paths = sorted(
        run.path.glob(f"generative_validation_{scenario}_latest_trial_*.csv")
    )
    if paths:
        return paths
    latest = run.path / f"generative_validation_{scenario}_latest.csv"
    return [latest] if latest.is_file() else []


def evaluate_latest_trials(
    runs: list[RunData], quantile: float, model: str
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in runs:
        real = pd.read_csv(run.path / "generative_validation_real.csv", low_memory=False)
        for scenario, target in LOCO_TARGETS.items():
            for trial_index, path in enumerate(latest_trial_paths(run, scenario)):
                generated = pd.read_csv(path, low_memory=False)
                for central_only, scope in [(False, "all"), (True, "central")]:
                    rows.append(
                        {
                            "model": model,
                            "run": run.name,
                            "seed": run.seed,
                            "scenario": scenario,
                            "target": target,
                            "trial": trial_index,
                            "scope": scope,
                            "trim_quantile": quantile,
                            **target_metrics(
                                real[target],
                                generated[target],
                                target,
                                quantile,
                                central_only,
                                c2st_seed=(
                                    710_000
                                    + 100 * list(LOCO_TARGETS).index(scenario)
                                    + int(central_only)
                                ),
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def summarize_outliers(rows: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "rows",
        "generated_outside_central_rate",
        "r2",
        "rmse",
        "nrmse",
        "ks_similarity",
        "c2st_quality",
        "alpha_precision",
        "beta_recall",
    ]
    per_seed = (
        rows.groupby(["model", "target", "scope", "seed"], as_index=False)
        .agg(
            trials=("trial", "count"),
            **{
                metric: (metric, "mean")
                for metric in metrics
            },
        )
    )
    return (
        per_seed.groupby(["model", "target", "scope"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            trials=("trials", "sum"),
            **{
                f"{metric}_mean": (metric, "mean")
                for metric in metrics
            },
            **{
                f"{metric}_between_seed_std": (metric, "std")
                for metric in metrics
            },
        )
        .sort_values(["model", "target", "scope"])
    )


def outlier_effect_summary(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "r2_mean",
        "nrmse_mean",
        "ks_similarity_mean",
        "c2st_quality_mean",
        "alpha_precision_mean",
        "beta_recall_mean",
    ]
    wide = summary.pivot(index=["model", "target"], columns="scope", values=metrics)
    rows: list[dict[str, Any]] = []
    for model, target in wide.index:
        row: dict[str, Any] = {"model": model, "target": target}
        for metric in metrics:
            all_value = wide.loc[(model, target), (metric, "all")]
            central_value = wide.loc[(model, target), (metric, "central")]
            stem = metric.removesuffix("_mean")
            row[f"{stem}_all"] = all_value
            row[f"{stem}_central"] = central_value
            row[f"{stem}_delta"] = central_value - all_value
        rows.append(row)
    return pd.DataFrame(rows)


def plot_outlier_sensitivity(rows: pd.DataFrame, output: Path) -> None:
    panels = [
        ("c2st_quality", "Target C2ST quality"),
        ("r2", "R2"),
        ("nrmse", "NRMSE"),
        ("ks_similarity", "KS similarity"),
        ("alpha_precision", "Alpha-precision"),
        ("beta_recall", "Beta-recall"),
    ]
    per_seed = (
        rows.groupby(["model", "target", "scope", "seed"], as_index=False)[
            [metric for metric, _ in panels]
        ]
        .mean()
    )
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.0))
    for ax, (metric, title) in zip(axes.reshape(-1), panels):
        sns.barplot(
            data=per_seed,
            x="target",
            y=metric,
            hue="scope",
            errorbar="sd",
            palette={"all": "#DC2626", "central": "#2563EB"},
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("")
        if metric not in {"r2", "nrmse"}:
            ax.set_ylim(-0.05, 1.05)
        legend = ax.get_legend()
        if ax is axes[0, 0] and legend is not None:
            legend.set_title("")
        elif legend is not None:
            legend.remove()
    fig.suptitle("Outlier sensitivity at the latest validation checkpoint")
    save_figure(fig, output)


def training_run_summary(runs: list[RunData]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in runs:
        best = run.history.loc[run.history.validation_loss.idxmin()]
        row: dict[str, Any] = {
            "run": run.name,
            "seed": run.seed,
            "learning_rate": run.config.get("learning_rate"),
            "log_transform": bool(run.config.get("log_continuous_cols")),
            "epochs_completed": int(run.history.epoch.max()),
            "best_total_epoch": int(best.epoch),
            "best_validation_loss": float(best.validation_loss),
            "validation_numerical_at_best_total": float(
                best.validation_numerical_loss
            ),
            "validation_categorical_at_best_total": float(
                best.validation_categorical_loss
            ),
            "minimum_validation_numerical": float(
                run.history.validation_numerical_loss.min()
            ),
            "minimum_validation_categorical": float(
                run.history.validation_categorical_loss.min()
            ),
        }
        for scenario, target in LOCO_TARGETS.items():
            metric = f"{target}_r2"
            if metric not in run.generative:
                continue
            subset = run.generative.dropna(subset=[metric])
            subset = subset[subset.scenario == scenario]
            if subset.empty:
                continue
            best_target = subset.loc[subset[metric].idxmax()]
            row[f"best_observed_{target}_r2"] = float(best_target[metric])
            row[f"best_observed_{target}_epoch"] = int(best_target.epoch)
            row[f"best_observed_{target}_rmse"] = float(
                best_target[f"{target}_rmse"]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def c2st_variability(runs: list[RunData]) -> pd.DataFrame:
    generative = combined_history(runs, "generative")
    rows: list[dict[str, Any]] = []
    for (scenario, epoch), group in generative.groupby(["scenario", "epoch"]):
        rows.append(
            {
                "scenario": scenario,
                "epoch": int(epoch),
                "seed_count": int(group.seed.nunique()),
                "c2st_mean": float(group.c2st.mean()),
                "c2st_between_seed_std": float(group.c2st.std(ddof=1)),
                "c2st_within_sample_std_mean": float(group.c2st_std.mean()),
            }
        )
    return pd.DataFrame(rows)


def format_pm(mean: Any, std: Any) -> str:
    if pd.isna(mean):
        return ""
    if pd.isna(std):
        return f"{float(mean):.3f}"
    return f"{float(mean):.3f} +/- {float(std):.3f}"


def model_comparison_table(
    external_summary: Path | None,
    raw_latest: pd.DataFrame,
    log_latest: pd.DataFrame,
    new_latest: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if external_summary is not None and external_summary.is_file():
        external = pd.read_csv(external_summary)
        external = external[
            (external.dataset_key == "dss_hdd_sequential_iops")
            & external.target.isin(["iops", "lat"])
        ]
        for record in external.to_dict("records"):
            rows.append(
                {
                    "model": record["model"],
                    "target": record["target"],
                    "split": "held-out test",
                    "checkpoint": "external trained model",
                    "runs": 1,
                    "trials": record.get("trials"),
                    "r2_mean": record.get("r2"),
                    "r2_std": np.nan,
                    "nrmse_mean": record.get("nrmse"),
                    "nrmse_std": np.nan,
                    "c2st_mean": record.get("target_c2st_quality"),
                    "c2st_std": np.nan,
                    "alpha_mean": record.get("target_alpha_precision"),
                    "alpha_std": np.nan,
                    "beta_mean": record.get("target_beta_recall"),
                    "beta_std": np.nan,
                    "status": "available",
                }
            )
        present = set(zip(external.model_key, external.target))
        for target in ["iops", "lat"]:
            if ("tabsyn", target) not in present:
                rows.append(
                    {
                        "model": "TabSyn",
                        "target": target,
                        "split": "held-out test",
                        "checkpoint": "missing HDD results",
                        "runs": 0,
                        "trials": 0,
                        "status": "missing",
                    }
                )

    def add_validation(
        frame: pd.DataFrame,
        model: str,
        checkpoint: str,
        runs: int,
    ) -> None:
        all_rows = frame[frame.scope == "all"]
        for target, group in all_rows.groupby("target"):
            metrics = [
                "r2",
                "nrmse",
                "c2st_quality",
                "alpha_precision",
                "beta_recall",
            ]
            per_seed = group.groupby("seed", as_index=False)[metrics].mean()
            rows.append(
                {
                    "model": model,
                    "target": target,
                    "split": "validation",
                    "checkpoint": checkpoint,
                    "runs": runs,
                    "trials": len(group),
                    "r2_mean": per_seed.r2.mean(),
                    "r2_std": per_seed.r2.std(ddof=1),
                    "nrmse_mean": per_seed.nrmse.mean(),
                    "nrmse_std": per_seed.nrmse.std(ddof=1),
                    "c2st_mean": per_seed.c2st_quality.mean(),
                    "c2st_std": per_seed.c2st_quality.std(ddof=1),
                    "alpha_mean": per_seed.alpha_precision.mean(),
                    "alpha_std": per_seed.alpha_precision.std(ddof=1),
                    "beta_mean": per_seed.beta_recall.mean(),
                    "beta_std": per_seed.beta_recall.std(ddof=1),
                    "status": "validation only",
                }
            )

    add_validation(raw_latest, "DSS raw, lr=2e-4", "latest saved validation", 1)
    add_validation(log_latest, "DSS log, lr=2e-4", "latest saved validation", 1)
    add_validation(
        new_latest,
        "DSS log, lr=5e-5",
        "latest validation, mean over seeds",
        int(new_latest.seed.nunique()),
    )
    return pd.DataFrame(rows)


def write_markdown_table(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        path.write_text("No data.\n")
        return
    header = "| " + " | ".join(frame.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    path.write_text("\n".join([header, separator, *rows]) + "\n")


def save_model_table_artifacts(table: pd.DataFrame, output_dir: Path) -> None:
    table.to_csv(output_dir / "model_comparison.csv", index=False)
    display = table.copy()
    for metric in ["r2", "nrmse", "c2st", "alpha", "beta"]:
        display[metric.upper()] = [
            format_pm(mean, std)
            for mean, std in zip(
                display[f"{metric}_mean"], display[f"{metric}_std"]
            )
        ]
    display = display[
        [
            "model",
            "split",
            "checkpoint",
            "target",
            "runs",
            "trials",
            "R2",
            "NRMSE",
            "C2ST",
            "ALPHA",
            "BETA",
            "status",
        ]
    ].rename(
        columns={
            "model": "Model",
            "split": "Split",
            "checkpoint": "Checkpoint",
            "target": "Target",
            "runs": "Runs",
            "trials": "Trials",
            "status": "Status",
        }
    )
    write_markdown_table(display.fillna(""), output_dir / "model_comparison.md")

    plot_display = display[display.Model != "TabSyn"].drop(columns="Checkpoint").copy()
    without_std = plot_display.Model == "DSS log, lr=5e-5"
    for metric in ["R2", "NRMSE", "C2ST", "ALPHA", "BETA"]:
        plot_display.loc[without_std, metric] = table.loc[
            (table.model == "DSS log, lr=5e-5"), f"{metric.lower()}_mean"
        ].map(lambda value: f"{value:.3f}")

    fig, ax = plt.subplots(
        figsize=(14.5, max(3.0, 0.5 * len(plot_display) + 1.4))
    )
    ax.axis("off")
    rendered = ax.table(
        cellText=plot_display.fillna("").values,
        colLabels=plot_display.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(7.2)
    rendered.scale(1.0, 1.35)
    for index in range(len(plot_display.columns)):
        rendered[(0, index)].set_facecolor("#E2E8F0")
        rendered[(0, index)].set_text_props(weight="bold")
    ax.set_title(
        "Available model metrics; test and validation rows are not directly comparable",
        fontsize=11,
        pad=12,
    )
    save_figure(fig, output_dir / "model_comparison.png")


def latest_generative_quality_table(
    runs: list[RunData], target_trials: pd.DataFrame
) -> pd.DataFrame:
    full_rows = []
    for run in runs:
        latest_epoch = int(run.generative.epoch.max())
        current = run.generative[run.generative.epoch == latest_epoch].copy()
        current["seed"] = run.seed
        full_rows.append(current)
    full = pd.concat(full_rows, ignore_index=True)
    full_metrics = [
        "overall_quality",
        "column_shapes",
        "column_pair_trends",
        "c2st",
        "alpha_precision",
        "beta_recall",
        "dcr",
    ]
    rows: list[dict[str, Any]] = []
    for scenario, scenario_rows in full.groupby("scenario"):
        per_seed = scenario_rows.groupby("seed", as_index=False)[full_metrics].mean()
        row: dict[str, Any] = {
            "scenario": scenario,
            "seeds": int(per_seed.seed.nunique()),
            "samples_per_seed": int(scenario_rows.samples.min()),
        }
        for metric in full_metrics:
            row[f"full_{metric}_mean"] = per_seed[metric].mean()
            row[f"full_{metric}_std"] = per_seed[metric].std(ddof=1)

        target = LOCO_TARGETS.get(scenario)
        row["target"] = target
        if target is not None:
            selected = target_trials[
                (target_trials.scope == "all")
                & (target_trials.scenario == scenario)
            ]
            target_metrics = [
                "ks_similarity",
                "c2st_quality",
                "alpha_precision",
                "beta_recall",
            ]
            target_per_seed = selected.groupby("seed", as_index=False)[
                target_metrics
            ].mean()
            for metric in target_metrics:
                row[f"target_{metric}_mean"] = target_per_seed[metric].mean()
                row[f"target_{metric}_std"] = target_per_seed[metric].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("scenario")


def save_generative_quality_artifacts(
    table: pd.DataFrame, output_dir: Path
) -> None:
    table.to_csv(output_dir / "generative_quality_summary.csv", index=False)
    display = pd.DataFrame(
        {
            "Scenario": table.scenario,
            "Target": table.target.fillna("all masked columns"),
            "Full overall": [
                format_pm(mean, std)
                for mean, std in zip(
                    table.full_overall_quality_mean,
                    table.full_overall_quality_std,
                )
            ],
            "Full shapes": [
                format_pm(mean, std)
                for mean, std in zip(
                    table.full_column_shapes_mean,
                    table.full_column_shapes_std,
                )
            ],
            "Full pairs": [
                format_pm(mean, std)
                for mean, std in zip(
                    table.full_column_pair_trends_mean,
                    table.full_column_pair_trends_std,
                )
            ],
            "Full C2ST": [
                format_pm(mean, std)
                for mean, std in zip(table.full_c2st_mean, table.full_c2st_std)
            ],
            "Target KS": [
                format_pm(mean, std)
                for mean, std in zip(
                    table.target_ks_similarity_mean,
                    table.target_ks_similarity_std,
                )
            ],
            "Target C2ST": [
                format_pm(mean, std)
                for mean, std in zip(
                    table.target_c2st_quality_mean,
                    table.target_c2st_quality_std,
                )
            ],
            "Target alpha": [
                format_pm(mean, std)
                for mean, std in zip(
                    table.target_alpha_precision_mean,
                    table.target_alpha_precision_std,
                )
            ],
            "Target beta": [
                format_pm(mean, std)
                for mean, std in zip(
                    table.target_beta_recall_mean,
                    table.target_beta_recall_std,
                )
            ],
            "DCR": [
                format_pm(mean, std)
                for mean, std in zip(table.full_dcr_mean, table.full_dcr_std)
            ],
        }
    )
    write_markdown_table(
        display.fillna(""), output_dir / "generative_quality_summary.md"
    )

    fig, ax = plt.subplots(figsize=(16.5, 3.2))
    ax.axis("off")
    rendered = ax.table(
        cellText=display.fillna("").values,
        colLabels=display.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(7.4)
    rendered.scale(1.0, 1.5)
    for index in range(len(display.columns)):
        rendered[(0, index)].set_facecolor("#E2E8F0")
        rendered[(0, index)].set_text_props(weight="bold")
    ax.set_title(
        "Latest validation generative quality, mean +/- SD across three seeds",
        fontsize=11,
        pad=12,
    )
    save_figure(fig, output_dir / "generative_quality_summary.png")


def configuration_summary(
    raw: RunData, log: RunData, new_runs: list[RunData]
) -> pd.DataFrame:
    rows = []
    for label, runs in [
        ("raw_lr2e4", [raw]),
        ("log_lr2e4", [log]),
        ("log_lr5e5", new_runs),
    ]:
        per_run = training_run_summary(runs)
        row: dict[str, Any] = {"configuration": label, "runs": len(runs)}
        for metric in [
            "epochs_completed",
            "best_total_epoch",
            "best_validation_loss",
            "validation_numerical_at_best_total",
            "validation_categorical_at_best_total",
            "minimum_validation_numerical",
            "minimum_validation_categorical",
            "best_observed_iops_r2",
            "best_observed_iops_epoch",
            "best_observed_lat_r2",
            "best_observed_lat_epoch",
        ]:
            if metric in per_run:
                row[f"{metric}_mean"] = per_run[metric].mean()
                row[f"{metric}_std"] = per_run[metric].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def relative_change(new: float, old: float) -> float:
    return 100.0 * (new - old) / abs(old)


def seed_metric_summary(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "r2",
        "rmse",
        "nrmse",
        "c2st_quality",
        "alpha_precision",
        "beta_recall",
    ]
    selected = frame[frame.scope == "all"]
    per_seed = selected.groupby(["target", "seed"], as_index=False)[metrics].mean()
    return per_seed.groupby("target")[metrics].agg(["mean", "std"])


def write_text_reports(
    output_dir: Path,
    configurations: pd.DataFrame,
    variability: pd.DataFrame,
    new_latest: pd.DataFrame,
    outlier_effect: pd.DataFrame,
) -> None:
    indexed = configurations.set_index("configuration")
    raw = indexed.loc["raw_lr2e4"]
    old_log = indexed.loc["log_lr2e4"]
    new_log = indexed.loc["log_lr5e5"]
    targets = seed_metric_summary(new_latest)

    loss_log_delta = relative_change(
        old_log.best_validation_loss_mean,
        raw.best_validation_loss_mean,
    )
    loss_lr_delta = relative_change(
        new_log.best_validation_loss_mean,
        old_log.best_validation_loss_mean,
    )
    num_log_delta = relative_change(
        old_log.validation_numerical_at_best_total_mean,
        raw.validation_numerical_at_best_total_mean,
    )
    cat_log_delta = relative_change(
        old_log.validation_categorical_at_best_total_mean,
        raw.validation_categorical_at_best_total_mean,
    )

    iops = targets.loc["iops"]
    latency = targets.loc["lat"]
    new_effect = outlier_effect[outlier_effect.model == "DSS log, lr=5e-5"].set_index(
        "target"
    )

    variability_lines = []
    for scenario in ["mcar_30", "loco_iops", "loco_lat"]:
        group = variability[variability.scenario == scenario]
        variability_lines.append(
            f"- `{scenario}`: between-seed SD {group.c2st_between_seed_std.mean():.3f} "
            f"on average (maximum {group.c2st_between_seed_std.max():.3f}); "
            f"mean reported within-checkpoint trial/evaluator SD "
            f"{group.c2st_within_sample_std_mean.mean():.3f}."
        )

    report = f"""# HDD sequential: experiment analysis

## Three-seed result

For `log(iops + eps), log(lat + eps)` with `lr=5e-5`, the latest saved validation reconstructions give:

- IOPS LOCO: R2 {iops[('r2', 'mean')]:.3f} +/- {iops[('r2', 'std')]:.3f}, NRMSE {iops[('nrmse', 'mean')]:.3f} +/- {iops[('nrmse', 'std')]:.3f}, RMSE {iops[('rmse', 'mean')] / 1e3:.3f} +/- {iops[('rmse', 'std')] / 1e3:.3f} thousand IOPS.
- Latency LOCO: R2 {latency[('r2', 'mean')]:.3f} +/- {latency[('r2', 'std')]:.3f}, NRMSE {latency[('nrmse', 'mean')]:.3f} +/- {latency[('nrmse', 'std')]:.3f}, RMSE {latency[('rmse', 'mean')] / 1e6:.1f} +/- {latency[('rmse', 'std')] / 1e6:.1f} us.

The mean and SD above are calculated across three seed-level means. Each seed-level mean uses five stochastic imputations.

## Learning rate and transform

- Changing raw values to log values at `lr=2e-4` changed the best total validation loss from {raw.best_validation_loss_mean:.3f} to {old_log.best_validation_loss_mean:.3f} ({loss_log_delta:.1f}%). The numerical term changed by {num_log_delta:.1f}% and the categorical term by {cat_log_delta:+.1f}%.
- Lowering the learning rate from `2e-4` to `5e-5` changed the best total validation loss from {old_log.best_validation_loss_mean:.3f} to {new_log.best_validation_loss_mean:.3f} +/- {new_log.best_validation_loss_std:.3f} ({loss_lr_delta:.1f}% for the mean).
- The minimum total validation loss is reached around epoch {new_log.best_total_epoch_mean:.1f}, while the best observed IOPS R2 is reached around epoch {new_log.best_observed_iops_epoch_mean:.1f}. Model selection by total loss therefore does not select the strongest LOCO checkpoint.

## C2ST variability

{chr(10).join(variability_lines)}

Between-seed variation dominates the reported within-checkpoint variation. Historical C2ST trials used random classifier folds, so their standard deviation combines stochastic imputation and evaluator noise. Future evaluations now fix these folds. In these runs the training seed also controls both model initialization and the grouped train/validation split, so those two sources cannot yet be separated.

## Outliers

The central subset keeps rows whose real target lies between the 0.5% and 99.5% quantiles. Generated values never determine which rows are removed.

- IOPS: removing the tails changes R2 by {new_effect.loc['iops', 'r2_delta']:+.3f} and target C2ST quality by {new_effect.loc['iops', 'c2st_quality_delta']:+.3f}.
- Latency: removing the tails changes R2 by {new_effect.loc['lat', 'r2_delta']:+.3f} and target C2ST quality by {new_effect.loc['lat', 'c2st_quality_delta']:+.3f}.

The extreme tails do not explain the instability: trimming does not improve R2 and barely changes distributional scores.

## Comparison status

`model_comparison.md` contains every currently available HDD imputation result. TabDiff has held-out test results. The DSS runs currently have saved validation reconstructions only, and HDD TabSyn imputation outputs are absent. These rows must not be ranked as if they used the same split.

## Conclusion

The smaller learning rate gives a modest average improvement in best validation loss and a useful late-epoch IOPS reconstruction. The old single-seed loss lies inside the range of the three new runs, so this is not yet a statistically established learning-rate effect. It does not establish a general model improvement: Latency reconstruction is unstable and poor on average, checkpoint selection is misaligned with LOCO accuracy, and the new runs have not yet been evaluated on the common held-out test set.
"""
    (output_dir / "analysis_summary.md").write_text(report)

    supervisor = f"""Провел сравнение на HDD sequential для трех seed. При `lr=5e-5` лучший validation loss составил {new_log.best_validation_loss_mean:.3f} +/- {new_log.best_validation_loss_std:.3f} против {old_log.best_validation_loss_mean:.3f} при `lr=2e-4`, то есть среднее значение ниже примерно на {abs(loss_lr_delta):.1f}%. Однако старый результат попадает в диапазон новых запусков, а seed также меняет validation split, поэтому это пока не является строгим подтверждением эффекта learning rate. Для LOCO IOPS на последних сохраненных validation-реконструкциях получили R2 {iops[('r2', 'mean')]:.3f} +/- {iops[('r2', 'std')]:.3f}; для Latency результат нестабилен: R2 {latency[('r2', 'mean')]:.3f} +/- {latency[('r2', 'std')]:.3f}.

Разброс C2ST в основном связан с различиями между seed. Например, для LOCO IOPS средний between-seed SD равен {variability[variability.scenario == 'loco_iops'].c2st_between_seed_std.mean():.3f}, а reported within-checkpoint SD равен {variability[variability.scenario == 'loco_iops'].c2st_within_sample_std_mean.mean():.3f}. В старых логах второй показатель также включает случайность folds самого C2ST-классификатора; для следующих запусков folds зафиксированы. Удаление крайних 0.5% с каждой стороны не улучшило R2, поэтому выбросы не являются основным объяснением нестабильности.

Log-преобразование снизило лучший validation loss с {raw.best_validation_loss_mean:.3f} до {old_log.best_validation_loss_mean:.3f}; основной выигрыш пришелся на численный loss. При этом минимум общего validation loss возникает около 13-й эпохи, а лучшие LOCO-метрики заметно позже, поэтому текущий выбор `best.pt` по общему loss не оптимален для imputation.

Честный вывод: на поздних чекпоинтах для IOPS получен положительный результат во всех трех запусках, но общего улучшения модели пока не подтверждено из-за нестабильной Latency и отсутствия оценки трех новых запусков на едином test split. В сводной таблице эти validation-результаты отделены от test-метрик TabDiff; результатов TabSyn imputation для HDD в выгрузке нет.
"""
    (output_dir / "supervisor_message.md").write_text(supervisor)


def main() -> None:
    args = parse_args()
    if not 0 <= args.outlier_quantile < 0.5:
        raise ValueError("outlier-quantile must be in [0, 0.5).")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    new_runs = [load_run(path.resolve()) for path in args.new_run_dirs]
    raw = load_run(args.raw_baseline_dir.resolve())
    log = load_run(args.log_baseline_dir.resolve())

    plot_mean_training_curves(
        new_runs, args.output_dir / "mean_training_curves.png"
    )
    plot_mean_generative_curves(
        new_runs, args.output_dir / "mean_generative_curves.png"
    )
    plot_mean_loco_accuracy(
        new_runs, args.output_dir / "mean_loco_accuracy.png"
    )
    plot_lr_comparison(log, new_runs, args.output_dir / "lr_comparison.png")
    plot_raw_log_comparison(raw, log, args.output_dir / "raw_vs_log.png")

    per_run = pd.concat(
        [training_run_summary([raw, log]), training_run_summary(new_runs)],
        ignore_index=True,
    )
    per_run.to_csv(args.output_dir / "training_run_summary.csv", index=False)
    configuration_summary(raw, log, new_runs).to_csv(
        args.output_dir / "configuration_summary.csv", index=False
    )
    variability = c2st_variability(new_runs)
    variability.to_csv(args.output_dir / "c2st_variability.csv", index=False)

    new_latest = evaluate_latest_trials(
        new_runs, args.outlier_quantile, "DSS log, lr=5e-5"
    )
    raw_latest = evaluate_latest_trials(
        [raw], args.outlier_quantile, "DSS raw, lr=2e-4"
    )
    log_latest = evaluate_latest_trials(
        [log], args.outlier_quantile, "DSS log, lr=2e-4"
    )
    latest = pd.concat([raw_latest, log_latest, new_latest], ignore_index=True)
    latest.to_csv(args.output_dir / "latest_validation_trial_metrics.csv", index=False)
    generative_quality = latest_generative_quality_table(new_runs, new_latest)
    save_generative_quality_artifacts(generative_quality, args.output_dir)
    outlier_summary = summarize_outliers(latest)
    outlier_summary.to_csv(
        args.output_dir / "outlier_sensitivity_summary.csv", index=False
    )
    outlier_effect = outlier_effect_summary(outlier_summary)
    outlier_effect.to_csv(args.output_dir / "outlier_effect.csv", index=False)
    plot_outlier_sensitivity(
        new_latest, args.output_dir / "outlier_sensitivity.png"
    )

    model_table = model_comparison_table(
        args.external_summary,
        raw_latest,
        log_latest,
        new_latest,
    )
    save_model_table_artifacts(model_table, args.output_dir)
    configurations = configuration_summary(raw, log, new_runs)
    write_text_reports(
        args.output_dir,
        configurations,
        variability,
        new_latest,
        outlier_effect,
    )

    run_info = {
        "new_runs": [str(run.path) for run in new_runs],
        "raw_baseline": str(raw.path),
        "log_baseline": str(log.path),
        "external_summary": (
            str(args.external_summary.resolve())
            if args.external_summary is not None
            else None
        ),
        "outlier_definition": (
            "Rows outside target quantiles are removed based only on the real target; "
            "the paired generated row is then removed."
        ),
        "outlier_quantile": args.outlier_quantile,
        "comparison_warning": (
            "External models use held-out test data. DSS entries currently use saved "
            "validation reconstructions and must not be ranked against test entries."
        ),
        "seed_warning": (
            "The current runs use seed for both model initialization and grouped split, "
            "so between-seed variance includes split variance."
        ),
    }
    (args.output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))
    print(f"Analysis saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
