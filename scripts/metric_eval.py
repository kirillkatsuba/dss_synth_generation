#!/usr/bin/env python3
"""SDV/SDMetrics-based evaluation for DSS synthetic tabular data."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import warnings
from pathlib import Path
from typing import Iterable

_cache_dir = Path(tempfile.gettempdir()) / "dss_synth_metric_eval_cache"
_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_dir / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_dir / "xdg"))
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sdmetrics.reports.single_table import DiagnosticReport, QualityReport
from sdmetrics.single_table import (
    DCROverfittingProtection,
    LinearRegression,
    LogisticDetection,
)
from sdv.metadata import SingleTableMetadata
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler


sns.set_theme(style="whitegrid", context="paper")
warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores.*",
    category=UserWarning,
    module="joblib.externals.loky.backend.context",
)

TARGET_ALIASES = {
    "IOPS": "iops",
    "iops": "iops",
    "Latency": "lat",
    "latency": "lat",
    "lat": "lat",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute SDV/SDMetrics quality metrics and plots for synthetic DSS data."
    )
    parser.add_argument(
        "--real-train",
        type=Path,
        default=Path("dataset/pools/train_hdd_sequential.csv"),
        help="CSV with real training data.",
    )
    parser.add_argument(
        "--real-test",
        type=Path,
        default=Path("dataset/pools/test_hdd_sequential.csv"),
        help="CSV with real holdout data for DCR and downstream MLE.",
    )
    parser.add_argument(
        "--synthetic",
        type=Path,
        default=Path("synth_data/synth_train_hdd_sequential.csv"),
        help="CSV with synthetic data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/metric_eval"),
        help="Directory for CSV/JSON metrics and PNG plots.",
    )
    parser.add_argument(
        "--target-cols",
        nargs="+",
        default=["iops", "lat"],
        help="Regression targets. Aliases IOPS and Latency are accepted.",
    )
    parser.add_argument(
        "--drop-cols",
        nargs="*",
        default=[],
        help="Columns to exclude from all evaluations.",
    )
    parser.add_argument(
        "--model-drop-cols",
        nargs="*",
        default=["id"],
        help=(
            "Columns excluded from SDMetrics model-based metrics (C2ST and MLE). "
            "The default removes id-like columns because SDMetrics efficacy/detection "
            "does not ignore unseen categories."
        ),
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20000,
        help="Max rows per dataset for alpha/beta and plots. Use 0 for all rows.",
    )
    parser.add_argument(
        "--dcr-sample-size",
        type=int,
        default=1000,
        help="Max rows per dataset for SDMetrics DCR. Use 0 for all rows.",
    )
    parser.add_argument(
        "--synthetic-label",
        default="ctgan",
        help="Label used in synthetic IOPS/Latency plot filename, e.g. ctgan_iops_lat.png.",
    )
    parser.add_argument(
        "--alpha-beta-level",
        type=float,
        default=0.95,
        help="Support quantile used for scalar alpha-precision and beta-recall.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def resolve_targets(targets: Iterable[str], columns: Iterable[str]) -> list[str]:
    available = set(columns)
    resolved: list[str] = []
    for target in targets:
        candidate = TARGET_ALIASES.get(target, target)
        if candidate not in available:
            raise ValueError(f"Target '{target}' resolved to missing column '{candidate}'.")
        resolved.append(candidate)
    return list(dict.fromkeys(resolved))


def load_and_align(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    real_train = pd.read_csv(args.real_train)
    real_test = pd.read_csv(args.real_test)
    synthetic = pd.read_csv(args.synthetic)

    common_cols = list(real_train.columns.intersection(real_test.columns).intersection(synthetic.columns))
    common_cols = [col for col in common_cols if col not in set(args.drop_cols)]
    if not common_cols:
        raise ValueError("No common columns remain after applying --drop-cols.")

    return real_train[common_cols].copy(), real_test[common_cols].copy(), synthetic[common_cols].copy()


def sample_frame(df: pd.DataFrame, n: int, random_state: int) -> pd.DataFrame:
    if n <= 0 or len(df) <= n:
        return df.reset_index(drop=True)
    return df.sample(n=n, random_state=random_state).reset_index(drop=True)


def json_default(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if pd.isna(value):
        return None
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def dump_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=json_default))


def build_metadata(data: pd.DataFrame, output_dir: Path) -> dict:
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)
    metadata_dict = metadata.to_dict()
    dump_json(output_dir / "sdv_metadata.json", metadata_dict)
    return metadata_dict


def run_quality_reports(
    real_train: pd.DataFrame,
    synthetic: pd.DataFrame,
    metadata: dict,
    output_dir: Path,
) -> dict[str, float]:
    print("Computing SDMetrics QualityReport...", flush=True)
    quality = QualityReport()
    quality.generate(real_train, synthetic, metadata, verbose=False)
    quality.save(output_dir / "sdv_quality_report.pkl")
    quality.get_properties().to_csv(output_dir / "sdv_quality_properties.csv", index=False)
    quality.get_details("Column Shapes").to_csv(
        output_dir / "sdv_column_shapes.csv", index=False
    )
    quality.get_details("Column Pair Trends").to_csv(
        output_dir / "sdv_column_pair_trends.csv", index=False
    )

    print("Computing SDMetrics DiagnosticReport...", flush=True)
    diagnostic = DiagnosticReport()
    diagnostic.generate(real_train, synthetic, metadata, verbose=False)
    diagnostic.save(output_dir / "sdv_diagnostic_report.pkl")
    diagnostic.get_properties().to_csv(output_dir / "sdv_diagnostic_properties.csv", index=False)
    for prop in diagnostic.get_properties()["Property"]:
        filename = prop.lower().replace(" ", "_")
        diagnostic.get_details(prop).to_csv(
            output_dir / f"sdv_diagnostic_{filename}.csv", index=False
        )

    return {
        "sdv_quality_score": float(quality.get_score()),
        "sdv_diagnostic_score": float(diagnostic.get_score()),
    }


def model_columns(
    data: pd.DataFrame, target_cols: list[str], model_drop_cols: list[str]
) -> list[str]:
    blocked = set(model_drop_cols)
    cols = [col for col in data.columns if col not in blocked]
    missing_targets = [target for target in target_cols if target not in cols]
    if missing_targets:
        raise ValueError(f"Target columns cannot be in --model-drop-cols: {missing_targets}")
    return cols


def run_sdv_c2st(
    real_train: pd.DataFrame,
    synthetic: pd.DataFrame,
    metadata: dict,
    output_dir: Path,
) -> dict[str, float | str]:
    print("Computing SDMetrics C2ST (LogisticDetection)...", flush=True)
    try:
        score = float(LogisticDetection.compute(real_train, synthetic, metadata))
        distinguishability = 1.0 - score
        result: dict[str, float | str] = {
            "logistic_detection_score": score,
            "distinguishability": distinguishability,
            "effective_roc_auc": float((distinguishability + 1.0) / 2.0),
            "metric_backend": "sdmetrics.single_table.LogisticDetection",
        }
        return result
    except Exception as exc:
        result = {"error": f"{type(exc).__name__}: {exc}"}
        dump_json(output_dir / "sdv_c2st_logistic_error.json", result)
        return result


def scorer_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(r2_score(y_true, y_pred))


def scorer_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_squared_error(y_true, y_pred) ** 0.5)


def scorer_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def run_sdv_mle(
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    synthetic: pd.DataFrame,
    metadata: dict,
    target_cols: list[str],
    output_dir: Path,
) -> pd.DataFrame:
    print("Computing SDMetrics MLE (LinearRegression)...", flush=True)
    rows: list[dict[str, object]] = []
    scorers = [scorer_r2, scorer_rmse, scorer_mae]
    for train_name, train_data in [("real_train", real_train), ("synthetic", synthetic)]:
        for target in target_cols:
            try:
                r2, rmse, mae = LinearRegression.compute(
                    real_test,
                    train_data,
                    metadata=metadata,
                    target=target,
                    scorer=scorers,
                )
                rows.append(
                    {
                        "train_data": train_name,
                        "target": target,
                        "r2": float(r2),
                        "rmse": float(rmse),
                        "mae": float(mae),
                        "metric_backend": "sdmetrics.single_table.LinearRegression",
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "train_data": train_name,
                        "target": target,
                        "error": f"{type(exc).__name__}: {exc}",
                        "metric_backend": "sdmetrics.single_table.LinearRegression",
                    }
                )
    mle = pd.DataFrame(rows)
    mle.to_csv(output_dir / "sdv_mle_downstream_metrics.csv", index=False)
    return mle


def run_sdv_dcr(
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    synthetic: pd.DataFrame,
    metadata: dict,
    output_dir: Path,
    sample_size: int,
    random_state: int,
) -> dict[str, float | str]:
    print("Computing SDMetrics DCR Overfitting Protection...", flush=True)
    train_s = sample_frame(real_train, sample_size, random_state)
    test_s = sample_frame(real_test, sample_size, random_state)
    synth_s = sample_frame(synthetic, sample_size, random_state + 1)
    try:
        score = float(
            DCROverfittingProtection.compute(
                train_s,
                synth_s,
                test_s,
                metadata,
                num_rows_subsample=None,
                num_iterations=1,
            )
        )
        result: dict[str, float | str] = {
            "dcr_overfitting_protection_score": score,
            "sample_rows_real_train": len(train_s),
            "sample_rows_real_test": len(test_s),
            "sample_rows_synthetic": len(synth_s),
        }
    except Exception as exc:
        result = {"error": f"{type(exc).__name__}: {exc}"}
    dump_json(output_dir / "sdv_dcr_metrics.json", result)
    return result


def alpha_beta_embedding(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    real_parts: list[np.ndarray] = []
    synth_parts: list[np.ndarray] = []

    numeric_cols = [col for col in real.columns if pd.api.types.is_numeric_dtype(real[col])]
    categorical_cols = [col for col in real.columns if col not in numeric_cols]

    if numeric_cols:
        scaler = StandardScaler()
        real_num = real[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(real[numeric_cols].median())
        synth_num = synthetic[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(real[numeric_cols].median())
        real_parts.append(scaler.fit_transform(real_num))
        synth_parts.append(scaler.transform(synth_num))

    if categorical_cols:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        real_cat = real[categorical_cols].fillna("__MISSING__").astype(str)
        synth_cat = synthetic[categorical_cols].fillna("__MISSING__").astype(str)
        real_parts.append(encoder.fit_transform(real_cat))
        synth_parts.append(encoder.transform(synth_cat))

    return np.hstack(real_parts), np.hstack(synth_parts)


def run_alpha_beta(
    real_train: pd.DataFrame,
    synthetic: pd.DataFrame,
    output_dir: Path,
    sample_size: int,
    random_state: int,
    level: float,
) -> dict[str, float | str]:
    """TabSyn alpha/beta approximation. SDMetrics 0.28.0 has no equivalent class."""
    print("Computing alpha-precision and beta-recall approximation...", flush=True)
    real_s = sample_frame(real_train, sample_size, random_state)
    synth_s = sample_frame(synthetic, sample_size, random_state + 1)
    real_x, synth_x = alpha_beta_embedding(real_s, synth_s)

    real_nn = NearestNeighbors(n_neighbors=2, metric="euclidean", n_jobs=1).fit(real_x)
    real_self = real_nn.kneighbors(real_x, return_distance=True)[0][:, 1]
    synth_to_real = NearestNeighbors(n_neighbors=1, metric="euclidean", n_jobs=1).fit(real_x)
    real_to_synth = NearestNeighbors(n_neighbors=1, metric="euclidean", n_jobs=1).fit(synth_x)
    synth_nearest_real = synth_to_real.kneighbors(synth_x, return_distance=True)[0][:, 0]
    real_nearest_synth = real_to_synth.kneighbors(real_x, return_distance=True)[0][:, 0]

    levels = np.round(np.linspace(0.05, 0.95, 19), 2)
    curve_rows = []
    for q in levels:
        radius = float(np.quantile(real_self, q))
        curve_rows.append(
            {
                "level": float(q),
                "alpha_precision": float(np.mean(synth_nearest_real <= radius)),
                "beta_recall": float(np.mean(real_nearest_synth <= radius)),
                "support_radius": radius,
            }
        )
    curve = pd.DataFrame(curve_rows)
    curve.to_csv(output_dir / "alpha_precision_beta_recall_curve.csv", index=False)

    level = min(max(level, 0.01), 0.99)
    radius = float(np.quantile(real_self, level))
    result: dict[str, float | str] = {
        "level": level,
        "alpha_precision": float(np.mean(synth_nearest_real <= radius)),
        "beta_recall": float(np.mean(real_nearest_synth <= radius)),
        "support_radius": radius,
        "metric_backend": "local approximation; SDMetrics 0.28.0 has no TabSyn alpha/beta metric",
    }
    dump_json(output_dir / "alpha_precision_beta_recall_metrics.json", result)
    return result


def save_density_plots(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    columns: list[str],
    plots_dir: Path,
) -> None:
    dist_dir = plots_dir / "distributions"
    dist_dir.mkdir(parents=True, exist_ok=True)
    for col in columns:
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        if pd.api.types.is_numeric_dtype(real[col]):
            real_values = pd.to_numeric(real[col], errors="coerce").dropna()
            synth_values = pd.to_numeric(synthetic[col], errors="coerce").dropna()
            all_values = pd.concat([real_values, synth_values], ignore_index=True)
            discrete = all_values.nunique() <= 25
            if all_values.nunique() <= 1:
                ax.bar(["real", "synthetic"], [len(real_values), len(synth_values)], alpha=0.7)
                ax.set_ylabel("Count")
            elif discrete:
                sns.histplot(
                    real_values,
                    stat="density",
                    discrete=True,
                    fill=True,
                    alpha=0.35,
                    color="tab:blue",
                    label="real",
                    ax=ax,
                )
                sns.histplot(
                    synth_values,
                    stat="density",
                    discrete=True,
                    fill=True,
                    alpha=0.35,
                    color="tab:orange",
                    label="synthetic",
                    ax=ax,
                )
                ax.set_ylabel("Density")
            else:
                sns.kdeplot(real_values, fill=True, alpha=0.35, color="tab:blue", label="real", ax=ax)
                sns.kdeplot(
                    synth_values,
                    fill=True,
                    alpha=0.35,
                    color="tab:orange",
                    label="synthetic",
                    ax=ax,
                )
                ax.set_ylabel("Density")
        else:
            real_counts = real[col].fillna("__MISSING__").astype(str).value_counts(normalize=True)
            synth_counts = synthetic[col].fillna("__MISSING__").astype(str).value_counts(normalize=True)
            top = real_counts.add(synth_counts, fill_value=0).sort_values(ascending=False).head(20).index
            x = np.arange(len(top))
            ax.fill_between(x, real_counts.reindex(top, fill_value=0.0), step="mid", alpha=0.35, label="real")
            ax.fill_between(
                x,
                synth_counts.reindex(top, fill_value=0.0),
                step="mid",
                alpha=0.35,
                label="synthetic",
            )
            ax.set_xticks(x)
            ax.set_xticklabels(top, rotation=45, ha="right")
            ax.set_ylabel("Density")
        ax.set_title(f"Distribution: {col}")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels)
        fig.tight_layout()
        fig.savefig(dist_dir / f"{col}_density.png", dpi=300)
        plt.close(fig)


def save_iops_latency_plot(
    df: pd.DataFrame,
    output_path: Path,
    title: str = "Latency vs. IOPS",
) -> None:
    plt.figure(figsize=(3.5, 3))
    plt.scatter(
        df.iops.values / 1e3,
        df.lat.values / 1e6,
        s=0.1,
        c="blue",
        marker="o",
        alpha=0.4,
        edgecolors="black",
        linewidth=0.2,
    )
    plt.xlabel("IOPs ($10^3$)")
    plt.ylabel("Latency ($\\mu$s)")
    plt.ylim(0, 100)
    plt.xlim(0, 14)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_alpha_beta_plot(output_dir: Path, plots_dir: Path) -> None:
    curve = pd.read_csv(output_dir / "alpha_precision_beta_recall_curve.csv")
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    sns.lineplot(data=curve, x="level", y="alpha_precision", marker="o", label="alpha-precision", ax=ax)
    sns.lineplot(data=curve, x="level", y="beta_recall", marker="o", label="beta-recall", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("Alpha precision / beta recall")
    fig.tight_layout()
    fig.savefig(plots_dir / "alpha_precision_beta_recall.png", dpi=300)
    plt.close(fig)


def save_mle_plot(mle: pd.DataFrame, plots_dir: Path) -> None:
    if "error" in mle.columns and mle["error"].notna().all():
        return
    plot_df = mle.dropna(subset=["r2", "rmse", "mae"], how="all")
    if plot_df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.8))
    for ax, metric in zip(axes, ["r2", "rmse", "mae"]):
        sns.barplot(data=plot_df, x="target", y=metric, hue="train_data", ax=ax)
        ax.set_title(metric.upper())
    fig.tight_layout()
    fig.savefig(plots_dir / "sdv_mle_downstream_metrics.png", dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    real_train, real_test, synthetic = load_and_align(args)
    target_cols = resolve_targets(args.target_cols, real_train.columns)
    metadata = build_metadata(real_train, args.output_dir)
    model_cols = model_columns(real_train, target_cols, args.model_drop_cols)
    model_metrics_dir = args.output_dir / "model_metrics"
    model_metrics_dir.mkdir(exist_ok=True)
    model_metadata = build_metadata(real_train[model_cols], model_metrics_dir)

    print(f"Real train shape: {real_train.shape}", flush=True)
    print(f"Real test shape: {real_test.shape}", flush=True)
    print(f"Synthetic shape: {synthetic.shape}", flush=True)
    print(f"Targets: {target_cols}", flush=True)
    print(f"SDMetrics model columns: {model_cols}", flush=True)

    summary: dict[str, object] = {
        "real_train_rows": len(real_train),
        "real_test_rows": len(real_test),
        "synthetic_rows": len(synthetic),
        "target_cols": target_cols,
        "model_drop_cols": args.model_drop_cols,
    }
    summary.update(run_quality_reports(real_train, synthetic, metadata, args.output_dir))

    c2st = run_sdv_c2st(
        real_train[model_cols],
        synthetic[model_cols],
        model_metadata,
        args.output_dir,
    )
    dump_json(args.output_dir / "sdv_c2st_logistic_metrics.json", c2st)
    summary["c2st"] = c2st

    mle = run_sdv_mle(
        real_train[model_cols],
        real_test[model_cols],
        synthetic[model_cols],
        model_metadata,
        target_cols,
        args.output_dir,
    )
    summary["mle"] = mle.to_dict(orient="records")

    dcr = run_sdv_dcr(
        real_train,
        real_test,
        synthetic,
        metadata,
        args.output_dir,
        args.dcr_sample_size,
        args.random_state,
    )
    summary["dcr"] = dcr

    alpha_beta = run_alpha_beta(
        real_train,
        synthetic,
        args.output_dir,
        args.sample_size,
        args.random_state,
        args.alpha_beta_level,
    )
    summary["alpha_beta"] = alpha_beta

    print("Saving density and IOPS/Latency plots...", flush=True)
    plot_real = sample_frame(real_train, args.sample_size, args.random_state)
    plot_synth = sample_frame(synthetic, args.sample_size, args.random_state + 1)
    save_density_plots(plot_real, plot_synth, list(real_train.columns), plots_dir)
    save_iops_latency_plot(plot_real, plots_dir / "real_iops_lat.png")
    save_iops_latency_plot(plot_synth, plots_dir / f"{args.synthetic_label}_iops_lat.png")
    save_alpha_beta_plot(args.output_dir, plots_dir)
    save_mle_plot(mle, plots_dir)

    dump_json(args.output_dir / "metrics_summary.json", summary)
    print(f"Saved metrics and plots to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
