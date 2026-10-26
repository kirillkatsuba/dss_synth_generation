#!/usr/bin/env python3
"""Build deterministic and generative reports for LOCO imputation models."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

cache_dir = Path(tempfile.gettempdir()) / "dss_imputation_comparison_report"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp, pearsonr, spearmanr, wasserstein_distance
from sdmetrics.reports.single_table import QualityReport
from sdmetrics.single_table import DCROverfittingProtection
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

from dss_imputer.generative_metrics import (
    METRIC_PROTOCOL_VERSION,
    alpha_precision_beta_recall,
    logistic_detection_quality,
    prepare_metric_tables,
    quality_report_scores,
)


sns.set_theme(style="whitegrid", context="paper")

MODEL_LABELS = {
    "tabdiff": "TabDiff",
    "tabsyn": "TabSyn",
    "dss_raw": "DSS diffusion",
    "dss_log": "DSS diffusion + log",
}
MODEL_COLORS = {
    "ground_truth": "#2563EB",
    "tabdiff": "#DC2626",
    "tabsyn": "#059669",
    "dss_raw": "#7C3AED",
    "dss_log": "#D97706",
}
DSS_MODEL_COLORS = [
    "#7C3AED",
    "#D97706",
    "#0891B2",
    "#DB2777",
    "#65A30D",
    "#4F46E5",
    "#EA580C",
]
DATASET_LABELS = {
    "dss_hdd_sequential_iops": "HDD sequential",
    "dss_ssd_sequential_iops": "SSD sequential",
    "dss_ssd_random_iops": "SSD random",
    "dss_cache_iops": "Cache",
}
SCATTER_SCALES = {
    "dss_hdd_sequential_iops": ("linear", "linear"),
    "dss_ssd_sequential_iops": ("linear", "linear"),
    "dss_ssd_random_iops": ("linear", "linear"),
    "dss_cache_iops": ("linear", "linear"),
}
SCATTER_LIMITS = {
    "dss_hdd_sequential_iops": {
        "iops": (0.0, 14.0),
        "lat": (0.0, 100.0),
    },
}


@dataclass(frozen=True)
class DatasetSchema:
    columns: list[str]
    numerical: set[str]
    categorical: set[str]

    def metadata(self, columns: Iterable[str] | None = None) -> dict[str, Any]:
        selected = list(columns) if columns is not None else self.columns
        return {
            "METADATA_SPEC_VERSION": "SINGLE_TABLE_V1",
            "columns": {
                column: {
                    "sdtype": (
                        "categorical" if column in self.categorical else "numerical"
                    )
                }
                for column in selected
            },
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare LOCO imputations using point accuracy, "
            "distribution metrics, density plots, and IOPS/latency plots."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("external/tabdiff/data"))
    parser.add_argument("--tabdiff-root", type=Path, default=Path("reports/imputation"))
    parser.add_argument(
        "--tabsyn-root", type=Path, default=Path("reports/imputation_tabsyn")
    )
    parser.add_argument(
        "--dss-raw-root",
        type=Path,
        default=None,
        help="Evaluation root containing loco_<target> directories.",
    )
    parser.add_argument(
        "--dss-log-root",
        type=Path,
        default=None,
        help="Evaluation root containing loco_<target> directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/imputation_tabdiff_tabsyn_comparison"),
    )
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_LABELS))
    parser.add_argument(
        "--models",
        nargs="+",
        default=["tabdiff", "tabsyn"],
    )
    parser.add_argument(
        "--dss-model",
        action="append",
        nargs=3,
        metavar=("KEY", "LABEL", "ROOT"),
        default=[],
        help=(
            "Register an additional DSS evaluation root. KEY must start with "
            "'dss_'; ROOT must contain loco_<target>/trial_*.csv. Repeatable."
        ),
    )
    parser.add_argument(
        "--targets", nargs="+", default=["all"], help="LOCO columns, or all."
    )
    parser.add_argument(
        "--numerical-cols", nargs="+", default=None,
        help="Common numerical metric columns for every model; others are categorical. Defaults to dataset info.json.",
    )
    parser.add_argument("--metric-sample-size", type=int, default=5000)
    parser.add_argument("--dcr-sample-size", type=int, default=1000)
    parser.add_argument("--energy-sample-size", type=int, default=1000)
    parser.add_argument("--plot-sample-size", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail instead of recording missing model/target results.",
    )
    return parser.parse_args()


def stable_seed(base: int, *parts: str) -> int:
    payload = "\0".join(parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return base + int.from_bytes(digest[:4], "little") % 1_000_000


def load_schema(data_dir: Path) -> DatasetSchema:
    info = json.loads((data_dir / "info.json").read_text())
    columns = list(info["column_names"])
    numerical_indices = set(info.get("num_col_idx", [])) | set(
        info.get("target_col_idx", [])
    )
    categorical_indices = set(info.get("cat_col_idx", []))
    numerical = {columns[index] for index in numerical_indices}
    categorical = {columns[index] for index in categorical_indices}
    untyped = set(columns) - numerical - categorical
    numerical.update(untyped)
    return DatasetSchema(columns, numerical, categorical)


def finite_numeric(values: pd.Series) -> np.ndarray:
    result = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return result[np.isfinite(result)]


def paired_finite_numeric(
    real: pd.Series, predicted: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    real_values = pd.to_numeric(real, errors="coerce").to_numpy(dtype=float)
    predicted_values = pd.to_numeric(predicted, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(real_values) & np.isfinite(predicted_values)
    return real_values[finite], predicted_values[finite]


def sample_indices(rows: int, size: int, seed: int) -> np.ndarray:
    if size <= 0 or rows <= size:
        return np.arange(rows)
    return np.sort(np.random.default_rng(seed).choice(rows, size=size, replace=False))


def trial_paths(directory: Path) -> list[Path]:
    return sorted(directory.glob("trial_*.csv")) if directory.is_dir() else []


def model_result_directory(model: str, root: Path, dataset: str, target: str) -> Path:
    if model.startswith("dss_"):
        return root / f"loco_{target}"
    return root / dataset / "loco" / target


def register_dss_models(
    specifications: list[list[str]], roots: dict[str, Path]
) -> None:
    color_index = sum(key.startswith("dss_") for key in roots)
    for key, label, root_value in specifications:
        if not key.startswith("dss_"):
            raise ValueError(f"DSS model key must start with 'dss_': {key!r}")
        if key in roots:
            raise ValueError(f"Duplicate model key: {key!r}")
        if not label.strip():
            raise ValueError(f"Model label cannot be empty for {key!r}")
        roots[key] = Path(root_value).resolve()
        MODEL_LABELS[key] = label
        MODEL_COLORS[key] = DSS_MODEL_COLORS[color_index % len(DSS_MODEL_COLORS)]
        color_index += 1


def pooled_target(
    directory: Path,
    target: str,
    rows: int,
    numerical: bool,
    seed: int,
) -> tuple[pd.Series, int]:
    paths = trial_paths(directory)
    if not paths:
        raise FileNotFoundError(f"No trial CSV files in {directory}")
    choices = np.random.default_rng(seed).integers(0, len(paths), size=rows)
    output = np.empty(rows, dtype=object)
    for trial_index, path in enumerate(paths):
        values = pd.read_csv(path, usecols=[target], low_memory=False)[target]
        if len(values) != rows:
            raise ValueError(f"{path} has {len(values)} rows; expected {rows}")
        selected = choices == trial_index
        output[selected] = values.to_numpy()[selected]
    result = pd.Series(output, name=target)
    if numerical:
        result = pd.to_numeric(result, errors="coerce")
    return result, len(paths)


def point_target(
    directory: Path,
    target: str,
    rows: int,
    numerical: bool,
) -> pd.Series:
    path = directory / "point_estimate.csv"
    if path.is_file():
        result = pd.read_csv(path, usecols=[target], low_memory=False)[target]
        if len(result) != rows:
            raise ValueError(f"{path} has {len(result)} rows; expected {rows}")
        return pd.to_numeric(result, errors="coerce") if numerical else result

    paths = trial_paths(directory)
    if not paths:
        raise FileNotFoundError(f"No point estimate or trials in {directory}")
    values = pd.concat(
        [
            pd.read_csv(path, usecols=[target], low_memory=False)[target]
            for path in paths
        ],
        axis=1,
    )
    if len(values) != rows:
        raise ValueError(f"Trials in {directory} have an unexpected row count")
    if numerical:
        return values.apply(pd.to_numeric, errors="coerce").mean(axis=1)
    return values.mode(axis=1, dropna=False).iloc[:, 0]


def deterministic_metrics(
    real: pd.Series, predicted: pd.Series, numerical: bool
) -> dict[str, float | str | int]:
    if numerical:
        truth, estimate = paired_finite_numeric(real, predicted)
        if not len(truth):
            raise ValueError("No finite target/prediction pairs")
        mse = float(mean_squared_error(truth, estimate))
        rmse = math.sqrt(mse)
        mae = float(mean_absolute_error(truth, estimate))
        scale = float(np.std(truth))
        return {
            "task_type": "numerical",
            "evaluated_rows": len(truth),
            "r2": float(r2_score(truth, estimate)) if len(truth) > 1 else np.nan,
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "median_absolute_error": float(np.median(np.abs(truth - estimate))),
            "normalized_mse": mse / scale**2 if scale > 0 else np.nan,
            "nrmse": rmse / scale if scale > 0 else np.nan,
            "nmae": mae / scale if scale > 0 else np.nan,
            "accuracy": np.nan,
            "macro_f1": np.nan,
            "macro_recall": np.nan,
        }

    truth = real.fillna("__MISSING__").astype(str)
    estimate = predicted.fillna("__MISSING__").astype(str)
    return {
        "task_type": "categorical",
        "evaluated_rows": len(truth),
        "r2": np.nan,
        "mse": np.nan,
        "rmse": np.nan,
        "mae": np.nan,
        "median_absolute_error": np.nan,
        "normalized_mse": np.nan,
        "nrmse": np.nan,
        "nmae": np.nan,
        "accuracy": float(accuracy_score(truth, estimate)),
        "macro_f1": float(f1_score(truth, estimate, average="macro", zero_division=0)),
        "macro_recall": float(
            recall_score(truth, estimate, average="macro", zero_division=0)
        ),
    }


def paired_sample(
    real: pd.DataFrame, generated: pd.DataFrame, size: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    indices = sample_indices(len(real), size, seed)
    return (
        real.iloc[indices].reset_index(drop=True),
        generated.iloc[indices].reset_index(drop=True),
    )


def safe_alpha_beta(
    real: pd.DataFrame, generated: pd.DataFrame, schema: DatasetSchema
) -> tuple[float, float, float]:
    return alpha_precision_beta_recall(
        real, generated,
        numerical_columns=[column for column in real if column in schema.numerical],
        categorical_columns=[column for column in real if column in schema.categorical],
    )


def tv_complement(real: pd.Series, generated: pd.Series) -> float:
    real_probability = (
        real.fillna("__MISSING__").astype(str).value_counts(normalize=True)
    )
    generated_probability = (
        generated.fillna("__MISSING__").astype(str).value_counts(normalize=True)
    )
    support = real_probability.index.union(generated_probability.index)
    distance = (
        0.5
        * np.abs(
            real_probability.reindex(support, fill_value=0.0)
            - generated_probability.reindex(support, fill_value=0.0)
        ).sum()
    )
    return float(1.0 - distance)


def normalized_energy_distance(
    real: pd.DataFrame, generated: pd.DataFrame, size: int, seed: int
) -> float:
    real_s, generated_s = paired_sample(real, generated, size, seed)
    real_x = real_s.apply(pd.to_numeric, errors="coerce")
    generated_x = generated_s.apply(pd.to_numeric, errors="coerce")
    medians = real_x.median()
    real_array = real_x.fillna(medians).to_numpy(dtype=float)
    generated_array = generated_x.fillna(medians).to_numpy(dtype=float)
    scaler = StandardScaler().fit(real_array)
    real_array = scaler.transform(real_array)
    generated_array = scaler.transform(generated_array)
    cross = float(cdist(real_array, generated_array).mean())
    real_within = float(cdist(real_array, real_array).mean())
    generated_within = float(cdist(generated_array, generated_array).mean())
    return float(max(0.0, 2 * cross - real_within - generated_within))


def correlation_metrics(
    real: pd.DataFrame, generated: pd.DataFrame
) -> dict[str, float]:
    real_values = real[["iops", "lat"]].apply(pd.to_numeric, errors="coerce")
    generated_values = generated[["iops", "lat"]].apply(pd.to_numeric, errors="coerce")
    real_values = real_values.dropna()
    generated_values = generated_values.dropna()
    real_pearson = float(pearsonr(real_values.iops, real_values.lat).statistic)
    generated_pearson = float(
        pearsonr(generated_values.iops, generated_values.lat).statistic
    )
    real_spearman = float(spearmanr(real_values.iops, real_values.lat).statistic)
    generated_spearman = float(
        spearmanr(generated_values.iops, generated_values.lat).statistic
    )
    return {
        "real_iops_lat_pearson": real_pearson,
        "generated_iops_lat_pearson": generated_pearson,
        "iops_lat_pearson_absolute_error": abs(real_pearson - generated_pearson),
        "real_iops_lat_spearman": real_spearman,
        "generated_iops_lat_spearman": generated_spearman,
        "iops_lat_spearman_absolute_error": abs(real_spearman - generated_spearman),
    }


def generative_metrics(
    real: pd.DataFrame,
    generated: pd.DataFrame,
    train: pd.DataFrame,
    target: str,
    schema: DatasetSchema,
    metric_sample_size: int,
    dcr_sample_size: int,
    energy_sample_size: int,
    seed: int,
) -> dict[str, Any]:
    real, generated, train, metadata = prepare_metric_tables(
        real, generated, train,
        numerical_columns=[column for column in real if column in schema.numerical],
        categorical_columns=[column for column in real if column in schema.categorical],
    )
    real_s, generated_s = paired_sample(real, generated, metric_sample_size, seed)
    errors: list[str] = []
    result: dict[str, Any] = {"metric_protocol": METRIC_PROTOCOL_VERSION}

    try:
        report = QualityReport()
        report.real_correlation_threshold = 0.0
        report.real_association_threshold = 0.0
        report.generate(real_s, generated_s, metadata, verbose=False)
        scores, quality_errors = quality_report_scores(report, [target], real_s.columns)
        errors.extend(quality_errors)
        result.update({f"full_{name}": scores[name] for name in (
            "column_shapes", "column_pair_trends", "overall_quality"
        ) if name in scores})
        shapes = report.get_details("Column Shapes")
        target_shape = shapes[shapes["Column"] == target]
        if not target_shape.empty:
            result["target_shape_metric"] = str(target_shape.iloc[0]["Metric"])
            if "target_column_shapes" in scores:
                result["target_shape_score"] = scores["target_column_shapes"]
        pairs = report.get_details("Column Pair Trends")
        result["target_context_pair_count"] = int(scores["target_pair_count"])
        result["target_context_pair_valid_count"] = int(scores["target_pair_valid_count"])
        if "target_column_pair_trends" in scores:
            result["target_context_pair_score"] = scores["target_column_pair_trends"]
        if {"iops", "lat"}.issubset(real_s.columns):
            physical_pair = pairs[
                ((pairs["Column 1"] == "iops") & (pairs["Column 2"] == "lat"))
                | ((pairs["Column 1"] == "lat") & (pairs["Column 2"] == "iops"))
            ]
            if not physical_pair.empty:
                result["iops_lat_pair_score"] = float(
                    pd.to_numeric(physical_pair["Score"], errors="coerce").iloc[0]
                )
    except Exception as exc:
        errors.append(f"quality: {type(exc).__name__}: {exc}")

    target_real = real_s[[target]]
    target_generated = generated_s[[target]]
    try:
        alpha, beta, authenticity = safe_alpha_beta(target_real, target_generated, schema)
        result.update(
            {
                "target_alpha_precision": alpha,
                "target_beta_recall": beta,
                "target_authenticity": authenticity,
            }
        )
    except Exception as exc:
        errors.append(f"target_alpha_beta: {type(exc).__name__}: {exc}")

    try:
        alpha, beta, authenticity = safe_alpha_beta(real_s, generated_s, schema)
        result.update(
            {
                "full_alpha_precision": alpha,
                "full_beta_recall": beta,
                "full_authenticity": authenticity,
            }
        )
    except Exception as exc:
        errors.append(f"full_alpha_beta: {type(exc).__name__}: {exc}")

    try:
        result["target_c2st_quality"] = logistic_detection_quality(
            target_real, target_generated, schema.metadata([target]), seed
        )
    except Exception as exc:
        errors.append(f"target_c2st: {type(exc).__name__}: {exc}")
    try:
        result["full_c2st_quality"] = logistic_detection_quality(real_s, generated_s, metadata, seed)
    except Exception as exc:
        errors.append(f"full_c2st: {type(exc).__name__}: {exc}")

    if target in schema.numerical:
        real_values, generated_values = paired_finite_numeric(
            real_s[target], generated_s[target]
        )
        if len(real_values):
            ks_distance = float(ks_2samp(real_values, generated_values).statistic)
            wasserstein = float(wasserstein_distance(real_values, generated_values))
            robust_scale = float(
                np.quantile(real_values, 0.99) - np.quantile(real_values, 0.01)
            )
            if robust_scale <= 0:
                robust_scale = float(np.std(real_values))
            normalized = wasserstein / robust_scale if robust_scale > 0 else np.nan
            result.update(
                {
                    "ks_complement": 1.0 - ks_distance,
                    "tv_complement": np.nan,
                    "wasserstein_distance": wasserstein,
                    "normalized_wasserstein": normalized,
                    "wasserstein_similarity": (
                        1.0 / (1.0 + normalized) if np.isfinite(normalized) else np.nan
                    ),
                }
            )
    else:
        result.update(
            {
                "ks_complement": np.nan,
                "tv_complement": tv_complement(real_s[target], generated_s[target]),
                "wasserstein_distance": np.nan,
                "normalized_wasserstein": np.nan,
                "wasserstein_similarity": np.nan,
            }
        )

    if target in {"iops", "lat"} and {"iops", "lat"}.issubset(real_s.columns):
        pair_columns = ["iops", "lat"]
        real_pair = real_s[pair_columns]
        generated_pair = generated_s[pair_columns]
        try:
            result.update(correlation_metrics(real_pair, generated_pair))
            result["iops_lat_energy_distance"] = normalized_energy_distance(
                real_pair,
                generated_pair,
                energy_sample_size,
                seed + 17,
            )
            result["iops_lat_c2st_quality"] = logistic_detection_quality(
                real_pair, generated_pair, schema.metadata(pair_columns), seed
            )
            alpha, beta, authenticity = safe_alpha_beta(real_pair, generated_pair, schema)
            result.update(
                {
                    "iops_lat_alpha_precision": alpha,
                    "iops_lat_beta_recall": beta,
                    "iops_lat_authenticity": authenticity,
                }
            )
        except Exception as exc:
            errors.append(f"iops_lat: {type(exc).__name__}: {exc}")

    try:
        train_s = train.sample(
            n=min(dcr_sample_size, len(train)), random_state=seed
        ).reset_index(drop=True)
        dcr_real, dcr_generated = paired_sample(
            real, generated, dcr_sample_size, seed + 1
        )
        result["dcr_descriptive"] = float(
            DCROverfittingProtection.compute(
                train_s,
                dcr_generated,
                dcr_real,
                schema.metadata(train_s.columns),
                num_rows_subsample=None,
                num_iterations=1,
            )
        )
    except Exception as exc:
        errors.append(f"dcr: {type(exc).__name__}: {exc}")

    result["metric_errors"] = " | ".join(errors)
    return result


def display_unit(column: str) -> tuple[float, str]:
    if column == "iops":
        return 1e3, "IOPS ($10^3$)"
    if column == "lat":
        return 1e6, "Latency ($\\mu$s)"
    return 1.0, column


def source_label(source: str) -> str:
    return "Ground truth" if source == "ground_truth" else MODEL_LABELS[source]


def numerical_axis(
    series_by_source: dict[str, pd.Series], column: str
) -> tuple[str, tuple[float, float]]:
    divisor, _ = display_unit(column)
    arrays = [finite_numeric(series) / divisor for series in series_by_source.values()]
    combined = np.concatenate([values for values in arrays if len(values)])
    positive = combined[combined > 0]
    log_scale = False
    if len(positive) >= max(2, int(0.98 * len(combined))):
        low = float(np.quantile(positive, 0.005))
        high = float(np.quantile(positive, 0.995))
        log_scale = low > 0 and high / low >= 100
    values = positive if log_scale else combined
    low = float(np.quantile(values, 0.001))
    high = float(np.quantile(values, 0.999))
    if math.isclose(low, high):
        padding = max(abs(low) * 0.05, 1.0)
        return "linear", (low - padding, high + padding)
    if log_scale:
        return "log", (low / 1.15, high * 1.15)
    padding = 0.04 * (high - low)
    lower = low - padding
    if column in {"iops", "lat"} and combined.min() >= 0:
        lower = max(0.0, lower)
    return "linear", (lower, high + padding)


def plot_numerical_distribution(
    series_by_source: dict[str, pd.Series],
    column: str,
    output_dir: Path,
    dataset_label: str,
    sample_size: int,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    divisor, axis_label = display_unit(column)
    xscale, limits = numerical_axis(series_by_source, column)
    scaled: dict[str, np.ndarray] = {}
    for source, series in series_by_source.items():
        values = finite_numeric(series) / divisor
        if xscale == "log":
            values = values[values > 0]
        indices = sample_indices(len(values), sample_size, seed)
        scaled[source] = values[indices]

    combined_unique = np.unique(np.concatenate(list(scaled.values())))
    discrete = len(combined_unique) <= 50 and bool(
        np.allclose(combined_unique, np.rint(combined_unique), rtol=0.0, atol=1e-9)
    )

    def draw(ax: plt.Axes, sources: list[str], title: str) -> None:
        for source in sources:
            values = scaled[source]
            color = MODEL_COLORS[source]
            label = source_label(source)
            if discrete:
                sns.histplot(
                    values,
                    stat="density",
                    discrete=True,
                    fill=True,
                    element="step",
                    alpha=0.28,
                    color=color,
                    label=label,
                    ax=ax,
                )
            elif len(np.unique(values)) > 1:
                try:
                    sns.kdeplot(
                        x=values,
                        fill=True,
                        alpha=0.24,
                        linewidth=1.2,
                        color=color,
                        label=label,
                        warn_singular=False,
                        ax=ax,
                    )
                except (TypeError, ValueError, np.linalg.LinAlgError):
                    sns.histplot(
                        values,
                        stat="density",
                        bins=60,
                        fill=True,
                        element="step",
                        alpha=0.24,
                        color=color,
                        label=label,
                        ax=ax,
                    )
            else:
                ax.axvline(values[0], color=color, linewidth=1.5, label=label)
        ax.set_xscale(xscale)
        ax.set_xlim(*limits)
        ax.set_xlabel(axis_label)
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.45)
        if len(sources) > 1:
            ax.legend(frameon=True, loc="upper left", bbox_to_anchor=(1.01, 1.0))

    for source in series_by_source:
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        draw(ax, [source], f"{dataset_label}: {column}\n{source_label(source)}")
        fig.tight_layout()
        fig.savefig(output_dir / f"{source}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    draw(ax, list(series_by_source), f"{dataset_label}: {column}")
    fig.tight_layout()
    fig.savefig(output_dir / "comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_categorical_distribution(
    series_by_source: dict[str, pd.Series],
    column: str,
    output_dir: Path,
    dataset_label: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    probabilities = {
        source: series.fillna("__MISSING__").astype(str).value_counts(normalize=True)
        for source, series in series_by_source.items()
    }
    support = probabilities["ground_truth"].index.tolist()
    extras = sorted(
        set().union(*(set(values.index) for values in probabilities.values()))
        - set(support)
    )
    support.extend(extras)
    x = np.arange(len(support))

    def draw(ax: plt.Axes, sources: list[str], title: str) -> None:
        width = 0.78 / len(sources)
        for index, source in enumerate(sources):
            values = probabilities[source].reindex(support, fill_value=0.0)
            offset = (index - (len(sources) - 1) / 2) * width
            ax.bar(
                x + offset,
                values,
                width=width,
                color=MODEL_COLORS[source],
                alpha=0.72,
                label=source_label(source),
            )
        ax.set_xticks(x)
        ax.set_xticklabels(support, rotation=40, ha="right")
        ax.set_ylabel("Probability")
        ax.set_xlabel(column)
        ax.set_title(title)
        ax.grid(True, axis="y", linestyle="--", alpha=0.45)
        if len(sources) > 1:
            ax.legend(frameon=True, loc="upper left", bbox_to_anchor=(1.01, 1.0))

    for source in series_by_source:
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        draw(ax, [source], f"{dataset_label}: {column}\n{source_label(source)}")
        fig.tight_layout()
        fig.savefig(output_dir / f"{source}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    draw(ax, list(series_by_source), f"{dataset_label}: {column}")
    fig.tight_layout()
    fig.savefig(output_dir / "comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def robust_scatter_limits(
    frames: dict[str, pd.DataFrame], axis: str, scale: str
) -> tuple[float, float]:
    divisor = 1e3 if axis == "iops" else 1e6
    values = np.concatenate(
        [finite_numeric(frame[axis]) / divisor for frame in frames.values()]
    )
    if scale == "log":
        values = values[values > 0]
    low = float(np.quantile(values, 0.001))
    high = float(np.quantile(values, 0.999))
    if scale == "log":
        return low / 1.12, high * 1.12
    padding = 0.04 * (high - low)
    lower = low - padding
    if values.min() >= 0:
        lower = max(0.0, lower)
    return lower, high + padding


def save_unavailable_plot(path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 3))
    ax.axis("off")
    ax.text(
        0.5,
        0.58,
        title,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.43,
        "Result unavailable",
        ha="center",
        va="center",
        fontsize=9,
        color="#64748B",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_scatter(
    ax: plt.Axes,
    frame: pd.DataFrame,
    title: str,
    color: str,
    xscale: str,
    yscale: str,
    xlimits: tuple[float, float],
    ylimits: tuple[float, float],
    sample_size: int,
    seed: int,
) -> tuple[int, int]:
    indices = sample_indices(len(frame), sample_size, seed)
    x = pd.to_numeric(frame.iloc[indices].iops, errors="coerce").to_numpy() / 1e3
    y = pd.to_numeric(frame.iloc[indices].lat, errors="coerce").to_numpy() / 1e6
    valid = np.isfinite(x) & np.isfinite(y)
    if xscale == "log":
        valid &= x > 0
    if yscale == "log":
        valid &= y > 0
    dropped = int((~valid).sum())
    outside_view = (
        (x < xlimits[0])
        | (x > xlimits[1])
        | (y < ylimits[0])
        | (y > ylimits[1])
    )
    outside = int((valid & outside_view).sum())
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
    ax.set_xlabel("IOPS ($10^3$)")
    ax.set_ylabel("Latency ($\\mu$s)")
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim(*xlimits)
    ax.set_ylim(*ylimits)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.6)
    finite_count = int(valid.sum())
    if outside and finite_count:
        ax.text(
            0.98,
            0.02,
            f"{outside / finite_count:.1%} outside view",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            color="#475569",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": "#CBD5E1",
                "alpha": 0.85,
            },
        )
    return dropped, outside


def plot_iops_latency(
    frames: dict[str, pd.DataFrame],
    models: list[str],
    output_dir: Path,
    dataset: str,
    scenario: str,
    sample_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    xscale, yscale = SCATTER_SCALES.get(dataset, ("linear", "linear"))
    fixed_limits = SCATTER_LIMITS.get(dataset, {})
    reference = {"ground_truth": frames.get("ground_truth", next(iter(frames.values())))}
    xlimits = fixed_limits.get("iops")
    if xlimits is None:
        xlimits = robust_scatter_limits(reference, "iops", xscale)
    ylimits = fixed_limits.get("lat")
    if ylimits is None:
        ylimits = robust_scatter_limits(reference, "lat", yscale)
    records: list[dict[str, Any]] = []

    for source, frame in frames.items():
        fig, ax = plt.subplots(figsize=(3.5, 3))
        dropped, outside = draw_scatter(
            ax,
            frame,
            f"{source_label(source)}\n{scenario}",
            MODEL_COLORS[source],
            xscale,
            yscale,
            xlimits,
            ylimits,
            sample_size,
            seed,
        )
        fig.tight_layout()
        fig.savefig(output_dir / f"{source}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        records.append(
            {
                "dataset_key": dataset,
                "scenario": scenario,
                "source": source,
                "xscale": xscale,
                "yscale": yscale,
                "xmin": xlimits[0],
                "xmax": xlimits[1],
                "ymin": ylimits[0],
                "ymax": ylimits[1],
                "nonpositive_or_nonfinite_points_dropped": dropped,
                "points_outside_view": outside,
            }
        )

    for source in models:
        if source not in frames:
            save_unavailable_plot(
                output_dir / f"{source}.png", f"{source_label(source)}\n{scenario}"
            )

    ordered = ["ground_truth", *models]
    subplot_count = len(ordered)
    grid_columns = min(4, subplot_count)
    grid_rows = math.ceil(subplot_count / grid_columns)
    fig, axes = plt.subplots(
        grid_rows,
        grid_columns,
        figsize=(3.5 * grid_columns, 3.0 * grid_rows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    flat_axes = axes.ravel()
    for axis, source in zip(flat_axes, ordered):
        if source not in frames:
            axis.axis("off")
            axis.text(
                0.5,
                0.55,
                source_label(source),
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
            )
            axis.text(
                0.5,
                0.42,
                "Result unavailable",
                ha="center",
                va="center",
                fontsize=9,
                color="#64748B",
            )
            continue
        draw_scatter(
            axis,
            frames[source],
            source_label(source),
            MODEL_COLORS[source],
            xscale,
            yscale,
            xlimits,
            ylimits,
            sample_size,
            seed,
        )
        axis.set_xlabel("")
        axis.set_ylabel("")
    for axis in flat_axes[subplot_count:]:
        axis.axis("off")
    fig.suptitle(f"{DATASET_LABELS.get(dataset, dataset)}: {scenario}", fontsize=12)
    fig.supxlabel("IOPS ($10^3$)", fontsize=10)
    fig.supylabel("Latency ($\\mu$s)", fontsize=10)
    fig.tight_layout(rect=(0.035, 0.04, 1, 0.94))
    fig.savefig(output_dir / "comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return records


def save_heatmap(
    frame: pd.DataFrame,
    value: str,
    path: Path,
    title: str,
    lower_is_better: bool = False,
) -> None:
    available = frame.dropna(subset=[value]).copy()
    if available.empty:
        return
    available["row"] = (
        available["dataset"].astype(str) + " | " + available["target"].astype(str)
    )
    pivot = available.pivot_table(
        index="row", columns="model", values=value, aggfunc="mean"
    )
    figure_height = max(3.2, 0.32 * len(pivot) + 1.5)
    fig, ax = plt.subplots(figsize=(6.0, figure_height))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="mako_r" if lower_is_better else "mako",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={
            "label": "Lower is better" if lower_is_better else "Higher is better"
        },
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_generative_metric_bars(frame: pd.DataFrame, path: Path) -> None:
    metric_labels = {
        "target_shape_score": "Target\nshape",
        "target_context_pair_score": "Target-context\npairs",
        "target_c2st_quality": "C2ST\nquality",
        "target_alpha_precision": "Alpha\nprecision",
        "target_beta_recall": "Beta\nrecall",
        "ks_complement": "KS\nsimilarity",
        "wasserstein_similarity": "Wasserstein\nsimilarity",
        "iops_lat_pair_score": "IOPS-latency\npair",
        "dcr_descriptive": "DCR\n(descriptive)",
    }
    available = frame[
        (frame["status"] == "ok") & frame["target"].isin(["iops", "lat"])
    ].copy()
    present_metrics = [
        metric for metric in metric_labels if metric in available.columns
    ]
    if available.empty or not present_metrics:
        return

    long = available.melt(
        id_vars=["target", "model", "model_key"],
        value_vars=present_metrics,
        var_name="metric",
        value_name="score",
    ).dropna(subset=["score"])
    if long.empty:
        return
    long["metric"] = long["metric"].map(metric_labels)
    target_order = [target for target in ["iops", "lat"] if target in set(long.target)]
    model_order = [
        MODEL_LABELS[model] for model in MODEL_LABELS if model in set(long.model_key)
    ]
    palette = {
        MODEL_LABELS[model]: MODEL_COLORS[model]
        for model in MODEL_LABELS
        if model in set(long.model_key)
    }
    fig, axes = plt.subplots(
        len(target_order),
        1,
        figsize=(10.5, 3.25 * len(target_order)),
        squeeze=False,
        sharex=True,
    )
    for target, ax in zip(target_order, axes[:, 0]):
        subset = long[long.target == target]
        sns.barplot(
            data=subset,
            x="metric",
            y="score",
            hue="model",
            hue_order=model_order,
            palette=palette,
            errorbar=None,
            ax=ax,
        )
        ax.set_title(f"LOCO {target}: stochastic generative quality")
        ax.set_xlabel("")
        ax.set_ylabel("Score (higher is better)")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, axis="y", linestyle="--", alpha=0.45)
        legend = ax.get_legend()
        if target == target_order[0] and legend is not None:
            legend.set_title("")
            legend.set_bbox_to_anchor((1.01, 1.0))
            legend.set_loc("upper left")
        elif legend is not None:
            legend.remove()
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_model_quality_summary(
    deterministic: pd.DataFrame,
    generative: pd.DataFrame,
    output_dir: Path,
) -> None:
    keys = ["dataset_key", "dataset", "model_key", "model", "target"]
    deterministic_columns = [*keys, "r2", "nrmse"]
    generative_metrics = [
        "trials",
        "target_shape_score",
        "target_context_pair_score",
        "target_c2st_quality",
        "target_alpha_precision",
        "target_beta_recall",
        "iops_lat_pair_score",
        "dcr_descriptive",
    ]
    deterministic_ok = deterministic[
        (deterministic["status"] == "ok")
        & deterministic["target"].isin(["iops", "lat"])
    ].loc[:, deterministic_columns]
    available_generative_metrics = [
        column for column in generative_metrics if column in generative.columns
    ]
    generative_ok = generative[
        (generative["status"] == "ok") & generative["target"].isin(["iops", "lat"])
    ].loc[:, [*keys, *available_generative_metrics]]
    summary = deterministic_ok.merge(generative_ok, on=keys, how="outer")

    model_order = {model: index for index, model in enumerate(MODEL_LABELS)}
    if not summary.empty:
        summary["_model_order"] = summary["model_key"].map(model_order)
        summary = summary.sort_values(["dataset_key", "target", "_model_order"]).drop(
            columns="_model_order"
        )
    summary.to_csv(output_dir / "model_quality_summary.csv", index=False)

    display_columns = {
        "dataset": "Dataset",
        "target": "Target",
        "model": "Model",
        "trials": "Trials",
        "r2": "R2",
        "nrmse": "NRMSE",
        "target_shape_score": "Shape",
        "target_context_pair_score": "Context pair",
        "target_c2st_quality": "C2ST",
        "target_alpha_precision": "Alpha",
        "target_beta_recall": "Beta",
        "iops_lat_pair_score": "IOPS-lat pair",
        "dcr_descriptive": "DCR",
    }
    selected = [column for column in display_columns if column in summary]
    display = summary.loc[:, selected].rename(columns=display_columns).copy()
    numeric_columns = [
        column
        for column in display
        if column not in {"Dataset", "Target", "Model", "Trials"}
    ]
    for column in numeric_columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{float(value):.3f}"
        )
    if "Trials" in display:
        display["Trials"] = display["Trials"].map(
            lambda value: "" if pd.isna(value) else str(int(value))
        )

    markdown_path = output_dir / "model_quality_summary.md"
    if display.empty:
        markdown_path.write_text("No comparable model results are available.\n")
        return
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    markdown_path.write_text("\n".join([header, separator, *rows]) + "\n")

    figure_height = max(2.2, 0.42 * len(display) + 1.2)
    fig, ax = plt.subplots(figsize=(16.5, figure_height))
    ax.axis("off")
    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.35)
    for column_index in range(len(display.columns)):
        table[(0, column_index)].set_facecolor("#E2E8F0")
        table[(0, column_index)].set_text_props(weight="bold")
    ax.set_title(
        "LOCO model quality on held-out test data (higher is better except NRMSE)",
        fontsize=11,
        pad=12,
    )
    fig.tight_layout()
    fig.savefig(
        output_dir / "model_quality_summary.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def aggregate_metrics(frame: pd.DataFrame, columns: list[str], output: Path) -> None:
    available = frame[frame["status"] == "ok"].copy()
    if available.empty:
        pd.DataFrame().to_csv(output, index=False)
        return
    aggregate = (
        available.groupby(
            ["dataset_key", "dataset", "model_key", "model"], as_index=False
        )
        .agg(
            targets=("target", "count"),
            **{column: (column, "mean") for column in columns},
        )
        .sort_values(["dataset_key", "model_key"])
    )
    aggregate.to_csv(output, index=False)


def json_dump(path: Path, value: Any) -> None:
    def convert(item: Any) -> Any:
        if isinstance(item, (np.integer, np.floating)):
            return item.item()
        if pd.isna(item):
            return None
        raise TypeError(f"Cannot serialize {type(item).__name__}")

    path.write_text(json.dumps(value, indent=2, default=convert))


def main() -> None:
    args = parse_args()
    roots = {
        "tabdiff": args.tabdiff_root.resolve(),
        "tabsyn": args.tabsyn_root.resolve(),
    }
    if args.dss_raw_root is not None:
        roots["dss_raw"] = args.dss_raw_root.resolve()
    if args.dss_log_root is not None:
        roots["dss_log"] = args.dss_log_root.resolve()
    register_dss_models(args.dss_model, roots)
    missing_roots = [model for model in args.models if model not in roots]
    if missing_roots:
        options = ", ".join(
            f"--{model.replace('_', '-')}-root" for model in missing_roots
        )
        raise ValueError(f"Missing result roots for {missing_roots}; pass {options}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots_root = args.output_dir / "plots"
    plots_root.mkdir(exist_ok=True)

    deterministic_rows: list[dict[str, Any]] = []
    generative_rows: list[dict[str, Any]] = []
    availability_rows: list[dict[str, Any]] = []
    scale_rows: list[dict[str, Any]] = []

    for dataset in args.datasets:
        data_dir = args.data_root.resolve() / dataset
        train_path = data_dir / "train.csv"
        test_path = data_dir / "test.csv"
        if not train_path.is_file() or not test_path.is_file():
            message = f"Prepared data is missing for {dataset}: {data_dir}"
            if args.strict:
                raise FileNotFoundError(message)
            print(f"WARNING: {message}", flush=True)
            continue

        schema = load_schema(data_dir)
        if args.numerical_cols is not None:
            numerical = set(args.numerical_cols)
            unknown = numerical - set(schema.columns)
            if unknown:
                raise ValueError(f"Unknown metric columns in {dataset}: {sorted(unknown)}")
            schema = DatasetSchema(schema.columns, numerical, set(schema.columns) - numerical)
        train = pd.read_csv(train_path, low_memory=False).loc[:, schema.columns]
        test = pd.read_csv(test_path, low_memory=False).loc[:, schema.columns]
        expected_targets = [
            column
            for column in schema.columns
            if test[column].nunique(dropna=False) > 1
        ]
        if args.targets != ["all"]:
            requested = set(args.targets)
            expected_targets = [
                column for column in expected_targets if column in requested
            ]

        dataset_plot_root = plots_root / dataset
        for target_index, target in enumerate(expected_targets):
            source_targets: dict[str, pd.Series] = {"ground_truth": test[target]}
            scatter_frames: dict[str, pd.DataFrame] = {
                "ground_truth": test[["iops", "lat"]].copy()
            }
            for model in args.models:
                directory = model_result_directory(model, roots[model], dataset, target)
                paths = trial_paths(directory)
                available = bool(paths)
                availability_rows.append(
                    {
                        "dataset_key": dataset,
                        "dataset": DATASET_LABELS.get(dataset, dataset),
                        "model_key": model,
                        "model": MODEL_LABELS[model],
                        "target": target,
                        "available": available,
                        "trials": len(paths),
                        "point_estimate_available": (
                            directory / "point_estimate.csv"
                        ).is_file(),
                        "directory": str(directory),
                    }
                )
                base = {
                    "dataset_key": dataset,
                    "dataset": DATASET_LABELS.get(dataset, dataset),
                    "model_key": model,
                    "model": MODEL_LABELS[model],
                    "target": target,
                    "rows": len(test),
                    "trials": len(paths),
                }
                if not available:
                    message = f"Missing {model}/{dataset}/{target}"
                    print(f"WARNING: {message}", flush=True)
                    deterministic_rows.append(
                        {**base, "status": "missing", "error": message}
                    )
                    generative_rows.append(
                        {**base, "status": "missing", "error": message}
                    )
                    if args.strict:
                        raise FileNotFoundError(message)
                    continue

                print(
                    f"Evaluating {MODEL_LABELS[model]} / {dataset} / {target}",
                    flush=True,
                )
                seed = stable_seed(args.seed, dataset, target)
                try:
                    point = point_target(
                        directory,
                        target,
                        len(test),
                        target in schema.numerical,
                    )
                    deterministic_rows.append(
                        {
                            **base,
                            "status": "ok",
                            "error": "",
                            **deterministic_metrics(
                                test[target], point, target in schema.numerical
                            ),
                        }
                    )
                    pooled, trials = pooled_target(
                        directory,
                        target,
                        len(test),
                        target in schema.numerical,
                        seed,
                    )
                    source_targets[model] = pooled
                    reconstructed = test.copy()
                    reconstructed[target] = pooled.to_numpy()
                    if target in {"iops", "lat"}:
                        scatter_frames[model] = reconstructed[["iops", "lat"]]
                    metrics = generative_metrics(
                        test,
                        reconstructed,
                        train,
                        target,
                        schema,
                        args.metric_sample_size,
                        args.dcr_sample_size,
                        args.energy_sample_size,
                        seed,
                    )
                    generative_rows.append(
                        {
                            **base,
                            "trials": trials,
                            "status": "ok",
                            "error": "",
                            **metrics,
                        }
                    )
                except Exception as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    print(
                        f"WARNING: failed {model}/{dataset}/{target}: {message}",
                        flush=True,
                    )
                    deterministic_rows.append(
                        {**base, "status": "error", "error": message}
                    )
                    generative_rows.append(
                        {**base, "status": "error", "error": message}
                    )
                    if args.strict:
                        raise

            distribution_dir = dataset_plot_root / "distributions" / target
            if target in schema.numerical:
                plot_numerical_distribution(
                    source_targets,
                    target,
                    distribution_dir,
                    DATASET_LABELS.get(dataset, dataset),
                    args.plot_sample_size,
                    stable_seed(args.seed, dataset, target, "density"),
                )
            else:
                plot_categorical_distribution(
                    source_targets,
                    target,
                    distribution_dir,
                    DATASET_LABELS.get(dataset, dataset),
                )
            for model in args.models:
                if model not in source_targets:
                    save_unavailable_plot(
                        distribution_dir / f"{model}.png",
                        f"{MODEL_LABELS[model]}\n{target}",
                    )

            if target in {"iops", "lat"}:
                scale_rows.extend(
                    plot_iops_latency(
                        scatter_frames,
                        args.models,
                        dataset_plot_root / "iops_latency" / f"loco_{target}",
                        dataset,
                        f"LOCO {target}",
                        args.plot_sample_size,
                        stable_seed(args.seed, dataset, target, "scatter"),
                    )
                )

    deterministic = pd.DataFrame(deterministic_rows)
    generative = pd.DataFrame(generative_rows)
    availability = pd.DataFrame(availability_rows)
    deterministic.to_csv(
        args.output_dir / "deterministic_metrics_by_target.csv", index=False
    )
    generative.to_csv(args.output_dir / "generative_metrics_by_target.csv", index=False)
    availability.to_csv(args.output_dir / "availability.csv", index=False)
    pd.DataFrame(scale_rows).to_csv(args.output_dir / "plot_scales.csv", index=False)

    deterministic_columns = [
        "r2",
        "mse",
        "rmse",
        "mae",
        "normalized_mse",
        "nrmse",
        "nmae",
        "accuracy",
        "macro_f1",
        "macro_recall",
    ]
    aggregate_metrics(
        deterministic,
        deterministic_columns,
        args.output_dir / "deterministic_metrics_by_dataset.csv",
    )
    generative_columns = [
        "target_shape_score",
        "target_context_pair_score",
        "target_alpha_precision",
        "target_beta_recall",
        "target_c2st_quality",
        "ks_complement",
        "tv_complement",
        "wasserstein_similarity",
        "full_column_shapes",
        "full_column_pair_trends",
        "full_overall_quality",
        "full_alpha_precision",
        "full_beta_recall",
        "full_c2st_quality",
        "dcr_descriptive",
    ]
    aggregate_metrics(
        generative,
        generative_columns,
        args.output_dir / "generative_metrics_by_dataset.csv",
    )

    join_columns = [
        "dataset_key",
        "dataset",
        "model_key",
        "model",
        "target",
    ]
    important_deterministic = deterministic[
        deterministic.target.isin(["iops", "lat"]) & (deterministic.status == "ok")
    ]
    important_generative = generative[
        generative.target.isin(["iops", "lat"]) & (generative.status == "ok")
    ]
    iops_lat = important_deterministic.merge(
        important_generative,
        on=join_columns,
        how="outer",
        suffixes=("_deterministic", "_generative"),
    )
    iops_lat.to_csv(args.output_dir / "iops_lat_metrics.csv", index=False)
    save_model_quality_summary(deterministic, generative, args.output_dir)

    save_heatmap(
        important_deterministic,
        "r2",
        args.output_dir / "deterministic_r2_heatmap.png",
        "LOCO point estimate: R2",
    )
    save_heatmap(
        important_deterministic,
        "nrmse",
        args.output_dir / "deterministic_nrmse_heatmap.png",
        "LOCO point estimate: normalized RMSE",
        lower_is_better=True,
    )
    save_heatmap(
        generative[generative.status == "ok"],
        "target_shape_score",
        args.output_dir / "generative_target_shape_heatmap.png",
        "Stochastic LOCO sample: target distribution similarity",
    )
    save_heatmap(
        important_generative,
        "target_context_pair_score",
        args.output_dir / "generative_iops_lat_pair_heatmap.png",
        "Stochastic LOCO sample: target-context pair similarity",
    )
    save_generative_metric_bars(
        generative,
        args.output_dir / "generative_metrics_comparison.png",
    )

    run_info = {
        "metric_protocol": METRIC_PROTOCOL_VERSION,
        "numerical_cols_override": args.numerical_cols,
        "data_root": str(args.data_root.resolve()),
        "roots": {key: str(value) for key, value in roots.items()},
        "output_dir": str(args.output_dir.resolve()),
        "datasets": args.datasets,
        "models": args.models,
        "targets": args.targets,
        "metric_sample_size": args.metric_sample_size,
        "dcr_sample_size": args.dcr_sample_size,
        "energy_sample_size": args.energy_sample_size,
        "plot_sample_size": args.plot_sample_size,
        "seed": args.seed,
        "generative_sample": (
            "One stochastic target value per row, selected deterministically across "
            "the saved trial_*.csv files."
        ),
        "deterministic_estimate": (
            "point_estimate.csv (trial mean for numerical targets and mode for "
            "categorical targets)."
        ),
        "metric_direction": (
            "All quality/similarity, R2, accuracy, alpha, beta, C2ST quality, and DCR "
            "scores are higher-is-better. Error, normalized Wasserstein, correlation "
            "error, energy distance, MSE, RMSE, and MAE are lower-is-better."
        ),
        "c2st_note": (
            "SDMetrics LogisticDetection is reported as a quality score, not raw AUC; "
            "1 means the classifier cannot distinguish the samples."
        ),
        "full_table_warning": (
            "Full-table LOCO scores are inflated by copied observed columns. Use the "
            "target and target-context metrics as the primary comparison."
        ),
        "dcr_warning": (
            "DCR is descriptive only for conditional reconstruction because observed "
            "test cells are copied exactly."
        ),
        "latency_unit": "Raw latency divided by 1e6 and displayed as microseconds.",
        "scatter_scales": SCATTER_SCALES,
        "scatter_limits": SCATTER_LIMITS,
        "scatter_layout": "At most four model panels per row.",
    }
    json_dump(args.output_dir / "run_info.json", run_info)
    print(f"Saved comparison report to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
