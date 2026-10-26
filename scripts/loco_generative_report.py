#!/usr/bin/env python3
"""Compare distribution quality of LOCO reconstructions.

Diffusion methods are represented by one conditional draw per row, pooled
deterministically across their saved trials. XGBoost predictions are rebuilt
from the selected hyperparameters stored by ``xgboost_loco.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

cache_dir = Path(tempfile.gettempdir()) / "dss_loco_generative_report"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sdmetrics.reports.single_table import QualityReport
from sdmetrics.single_table import DCROverfittingProtection, LogisticDetection
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from xgboost import XGBClassifier, XGBRegressor


MODEL_LABELS = {
    "tabdiff": "TabDiff",
    "tabsyn": "TabSyn",
    "xgboost": "XGBoost",
}
DATASET_LABELS = {
    "dss_hdd_sequential_iops": "HDD sequential",
    "dss_ssd_sequential_iops": "SSD sequential",
    "dss_ssd_random_iops": "SSD random",
    "dss_cache_iops": "Cache",
}
METRIC_COLUMNS = [
    "column_shapes",
    "column_pair_trends",
    "overall_quality",
    "alpha_precision",
    "beta_recall",
    "c2st",
    "dcr",
]
METRIC_LABELS = {
    "column_shapes": "Shapes",
    "column_pair_trends": "Pair Trends",
    "overall_quality": "Overall",
    "alpha_precision": "Alpha Precision",
    "beta_recall": "Beta Recall",
    "c2st": "C2ST",
    "dcr": "DCR*",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build PNG tables of generative metrics for LOCO reconstruction."
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("external/tabdiff/data")
    )
    parser.add_argument("--tabdiff-root", type=Path, default=Path("imputation"))
    parser.add_argument(
        "--tabsyn-root", type=Path, default=Path("imputation_tabsyn")
    )
    parser.add_argument(
        "--xgboost-root", type=Path, default=Path("reports/xgboost_loco")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/loco_generative_comparison"),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_LABELS),
        help="Prepared dataset directory names.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_LABELS),
        default=list(MODEL_LABELS),
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["all"],
        help="LOCO targets to evaluate, or all.",
    )
    parser.add_argument(
        "--metric-sample-size",
        type=int,
        default=5000,
        help="Rows used by quality, alpha/beta, and C2ST; 0 means all rows.",
    )
    parser.add_argument("--dcr-sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def json_dump(path: Path, value: Any) -> None:
    def convert(item: Any) -> Any:
        if isinstance(item, (np.integer, np.floating)):
            return item.item()
        if pd.isna(item):
            return None
        raise TypeError(f"Cannot serialize {type(item).__name__}")

    path.write_text(json.dumps(value, indent=2, default=convert))


def dataset_targets(root: Path, dataset: str, model: str) -> set[str]:
    if model in {"tabdiff", "tabsyn"}:
        loco = root / dataset / "loco"
        return {
            path.name
            for path in loco.iterdir()
            if path.is_dir() and list(path.glob("trial_*.csv"))
        } if loco.is_dir() else set()
    directory = root / dataset
    return {
        path.stem
        for path in directory.glob("*.json")
        if path.name != "run_info.json"
    } if directory.is_dir() else set()


def pooled_diffusion_reconstruction(
    real: pd.DataFrame,
    target: str,
    directory: Path,
    seed: int,
) -> tuple[pd.DataFrame, int]:
    trial_paths = sorted(directory.glob("trial_*.csv"))
    if not trial_paths:
        raise FileNotFoundError(f"No trial CSV files in {directory}")
    trials = [pd.read_csv(path, low_memory=False) for path in trial_paths]
    columns = list(trials[0].columns)
    if target not in columns:
        raise ValueError(f"Target {target!r} is absent from {trial_paths[0]}")
    if any(len(frame) != len(real) or list(frame.columns) != columns for frame in trials):
        raise ValueError(f"Inconsistent trial tables in {directory}")

    reconstruction = real.loc[:, columns].copy()
    choices = np.random.default_rng(seed).integers(0, len(trials), size=len(real))
    values = np.empty(len(real), dtype=object)
    for trial_index, frame in enumerate(trials):
        selected = choices == trial_index
        values[selected] = frame.loc[selected, target].to_numpy()
    reconstruction[target] = values
    if pd.api.types.is_numeric_dtype(real[target]):
        reconstruction[target] = pd.to_numeric(reconstruction[target], errors="coerce")
    return reconstruction, len(trials)


def make_features(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
) -> tuple[Any, Any, Any]:
    feature_columns = [column for column in train.columns if column != target]
    categorical = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(train[column])
    ]
    numerical = [column for column in feature_columns if column not in categorical]
    transformer = ColumnTransformer(
        [
            ("num", "passthrough", numerical),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                categorical,
            ),
        ]
    )
    return (
        transformer.fit_transform(train[feature_columns]),
        transformer.transform(validation[feature_columns]),
        transformer.transform(test[feature_columns]),
    )


def xgboost_reconstruction(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    result_path: Path,
    seed: int,
) -> tuple[pd.DataFrame, int]:
    saved = json.loads(result_path.read_text())
    params = saved["best_params"]
    train_part, validation = train_test_split(
        train, test_size=1 / 9, random_state=seed
    )
    x_train, _, x_test = make_features(train_part, validation, test, target)
    reconstruction = test.copy()
    common = {
        **params,
        "tree_method": "hist",
        "device": "cpu",
        "random_state": seed,
        "n_jobs": -1,
    }

    if saved["task"] == "classification":
        encoder = LabelEncoder().fit(train[target].astype(str))
        y_train = encoder.transform(train_part[target].astype(str))
        class_count = len(encoder.classes_)
        model = XGBClassifier(
            **common,
            objective="binary:logistic" if class_count == 2 else "multi:softprob",
            num_class=class_count if class_count > 2 else None,
        )
        model.fit(x_train, y_train, verbose=False)
        reconstruction[target] = encoder.inverse_transform(model.predict(x_test))
    else:
        y_train = train_part[target].to_numpy(dtype=np.float64)
        if saved.get("authors_target_transform", False):
            y_train = np.log(np.clip(y_train, 1, 20000))
        model = XGBRegressor(
            **common,
            objective="reg:squarederror",
        )
        prediction = model.fit(x_train, y_train, verbose=False).predict(x_test)
        reconstruction[target] = prediction
    return reconstruction, 1


def paired_sample(
    real: pd.DataFrame,
    reconstructed: pd.DataFrame,
    size: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if size <= 0 or len(real) <= size:
        indices = np.arange(len(real))
    else:
        indices = np.random.default_rng(seed).choice(len(real), size=size, replace=False)
    return (
        real.iloc[indices].reset_index(drop=True),
        reconstructed.iloc[indices].reset_index(drop=True),
    )


def alpha_beta_embedding(
    real: pd.DataFrame, reconstructed: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    numerical = [
        column for column in real if pd.api.types.is_numeric_dtype(real[column])
    ]
    categorical = [column for column in real if column not in numerical]
    real_parts: list[np.ndarray] = []
    reconstructed_parts: list[np.ndarray] = []

    if numerical:
        real_num = real[numerical].apply(pd.to_numeric, errors="coerce")
        medians = real_num.median()
        reconstructed_num = reconstructed[numerical].apply(
            pd.to_numeric, errors="coerce"
        )
        real_parts.append(real_num.fillna(medians).to_numpy(dtype=float))
        reconstructed_parts.append(
            reconstructed_num.fillna(medians).to_numpy(dtype=float)
        )
    if categorical:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        real_cat = real[categorical].fillna("__MISSING__").astype(str)
        reconstructed_cat = (
            reconstructed[categorical].fillna("__MISSING__").astype(str)
        )
        real_parts.append(encoder.fit_transform(real_cat))
        reconstructed_parts.append(encoder.transform(reconstructed_cat))

    real_x = np.hstack(real_parts)
    reconstructed_x = np.hstack(reconstructed_parts)
    scaler = MinMaxScaler().fit(real_x)
    return scaler.transform(real_x), scaler.transform(reconstructed_x)


def alpha_precision_beta_recall(
    real: pd.DataFrame, reconstructed: pd.DataFrame
) -> tuple[float, float, float]:
    """SynthCity's naive AlphaPrecision implementation without its dependency."""
    real_x, reconstructed_x = alpha_beta_embedding(real, reconstructed)
    center = real_x.mean(axis=0)
    alphas = np.linspace(0, 1, 30)
    radii = np.quantile(np.linalg.norm(real_x - center, axis=1), alphas)
    synth_center = reconstructed_x.mean(axis=0)
    synth_to_center = np.linalg.norm(reconstructed_x - center, axis=1)

    real_nn = NearestNeighbors(n_neighbors=2, n_jobs=1).fit(real_x)
    real_to_real = real_nn.kneighbors(real_x, return_distance=True)[0][:, 1]
    synth_nn = NearestNeighbors(n_neighbors=1, n_jobs=1).fit(reconstructed_x)
    real_to_synth, nearest_indices = synth_nn.kneighbors(
        real_x, return_distance=True
    )
    real_to_synth = real_to_synth[:, 0]
    nearest_indices = nearest_indices[:, 0]
    closest_synth = reconstructed_x[nearest_indices]
    closest_synth_distance = np.linalg.norm(closest_synth - synth_center, axis=1)
    closest_synth_radii = np.quantile(closest_synth_distance, alphas)

    precision_curve = np.array(
        [np.mean(synth_to_center <= radius) for radius in radii]
    )
    coverage_curve = np.array(
        [
            np.mean(
                (real_to_synth <= real_to_real)
                & (closest_synth_distance <= closest_synth_radii[index])
            )
            for index in range(len(alphas))
        ]
    )
    alpha_precision = 1 - np.abs(alphas - precision_curve).sum() / alphas.sum()
    beta_recall = 1 - np.abs(alphas - coverage_curve).sum() / alphas.sum()
    authenticity = np.mean(real_to_real[nearest_indices] < real_to_synth)
    return float(alpha_precision), float(beta_recall), float(authenticity)


def metadata_for(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "METADATA_SPEC_VERSION": "SINGLE_TABLE_V1",
        "columns": {
            column: {
                "sdtype": (
                    "numerical"
                    if pd.api.types.is_numeric_dtype(frame[column])
                    else "categorical"
                )
            }
            for column in frame.columns
        },
    }


def quality_metrics(
    real: pd.DataFrame,
    reconstructed: pd.DataFrame,
    train: pd.DataFrame,
    sample_size: int,
    dcr_sample_size: int,
    seed: int,
) -> dict[str, float]:
    real_s, reconstructed_s = paired_sample(real, reconstructed, sample_size, seed)
    metadata = metadata_for(real_s)
    quality = QualityReport()
    quality.real_correlation_threshold = 0.0
    quality.real_association_threshold = 0.0
    quality.generate(real_s, reconstructed_s, metadata, verbose=False)
    properties = quality.get_properties().set_index("Property")["Score"]
    alpha_precision, beta_recall, authenticity = alpha_precision_beta_recall(
        real_s, reconstructed_s
    )

    c2st = float(LogisticDetection.compute(real_s, reconstructed_s, metadata))
    train_columns = train.loc[:, real.columns]
    train_s = train_columns.sample(
        n=min(dcr_sample_size, len(train_columns)), random_state=seed
    ).reset_index(drop=True)
    dcr_real, dcr_reconstructed = paired_sample(
        real, reconstructed, dcr_sample_size, seed + 1
    )
    dcr = float(
        DCROverfittingProtection.compute(
            train_s,
            dcr_reconstructed,
            dcr_real,
            metadata_for(train_s),
            num_rows_subsample=None,
            num_iterations=1,
        )
    )
    return {
        "column_shapes": float(properties["Column Shapes"]),
        "column_pair_trends": float(properties["Column Pair Trends"]),
        "overall_quality": float(quality.get_score()),
        "alpha_precision": alpha_precision,
        "beta_recall": beta_recall,
        "authenticity": authenticity,
        "c2st": c2st,
        "dcr": dcr,
    }


def formatted_table(frame: pd.DataFrame, include_dataset: bool) -> pd.DataFrame:
    columns = (["dataset"] if include_dataset else []) + ["model", "targets"]
    result = frame.loc[:, columns + METRIC_COLUMNS].copy()
    result = result.rename(
        columns={
            "dataset": "Dataset",
            "model": "Model",
            "targets": "N",
            **METRIC_LABELS,
        }
    )
    for column in METRIC_LABELS.values():
        result[column] = result[column].map(lambda value: f"{value:.3f}")
    return result


def save_table_image(frame: pd.DataFrame, path: Path, title: str) -> None:
    rows, columns = frame.shape
    figure_width = max(10.5, 1.35 * columns)
    figure_height = max(2.2, 0.42 * rows + 1.45)
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    ax.axis("off")
    table = ax.table(
        cellText=frame.values,
        colLabels=frame.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.45)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#CBD5E1")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#1F4E5F")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#EEF4F5")
        else:
            cell.set_facecolor("white")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=16)
    fig.text(
        0.01,
        0.015,
        "Higher is better. DCR* is descriptive only for conditional reconstruction.",
        fontsize=8,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_reconstruction(
    model: str,
    dataset: str,
    target: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    roots: dict[str, Path],
    seed: int,
) -> tuple[pd.DataFrame, int]:
    if model in {"tabdiff", "tabsyn"}:
        return pooled_diffusion_reconstruction(
            test,
            target,
            roots[model] / dataset / "loco" / target,
            seed,
        )
    return xgboost_reconstruction(
        train,
        test,
        target,
        roots[model] / dataset / f"{target}.json",
        seed,
    )


def main() -> None:
    args = parse_args()
    roots = {
        "tabdiff": args.tabdiff_root.resolve(),
        "tabsyn": args.tabsyn_root.resolve(),
        "xgboost": args.xgboost_root.resolve(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reconstructed_dir = args.output_dir / "xgboost_reconstructions"
    reconstructed_dir.mkdir(exist_ok=True)

    available: dict[tuple[str, str], set[str]] = {}
    for dataset in args.datasets:
        for model in args.models:
            available[(model, dataset)] = dataset_targets(
                roots[model], dataset, model
            )

    rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        data_dir = args.data_root / dataset
        train_path = data_dir / "train.csv"
        test_path = data_dir / "test.csv"
        if not train_path.is_file() or not test_path.is_file():
            print(f"Skipping {dataset}: prepared train/test CSV is missing", flush=True)
            continue
        train = pd.read_csv(train_path, low_memory=False)
        test = pd.read_csv(test_path, low_memory=False)
        for model in args.models:
            targets = available[(model, dataset)]
            if args.targets != ["all"]:
                targets &= set(args.targets)
            for target_index, target in enumerate(sorted(targets)):
                print(
                    f"Evaluating {MODEL_LABELS[model]} / {dataset} / {target}",
                    flush=True,
                )
                reconstruction, trials = build_reconstruction(
                    model,
                    dataset,
                    target,
                    train,
                    test,
                    roots,
                    args.seed + target_index,
                )
                if model == "xgboost":
                    output = reconstructed_dir / dataset
                    output.mkdir(exist_ok=True)
                    reconstruction.to_csv(output / f"{target}.csv", index=False)
                real = test.loc[:, reconstruction.columns]
                metrics = quality_metrics(
                    real,
                    reconstruction,
                    train,
                    args.metric_sample_size,
                    args.dcr_sample_size,
                    args.seed + target_index,
                )
                rows.append(
                    {
                        "dataset_key": dataset,
                        "dataset": DATASET_LABELS.get(dataset, dataset),
                        "model_key": model,
                        "model": MODEL_LABELS[model],
                        "target": target,
                        "rows": len(real),
                        "trials": trials,
                        **metrics,
                    }
                )

    by_target = pd.DataFrame(rows)
    if by_target.empty:
        raise RuntimeError("No matching LOCO results were found")
    by_target.to_csv(args.output_dir / "metrics_by_target.csv", index=False)

    aggregate_columns = METRIC_COLUMNS + ["authenticity"]
    by_dataset = (
        by_target.groupby(
            ["dataset_key", "dataset", "model_key", "model"], as_index=False
        )
        .agg(
            targets=("target", "count"),
            **{metric: (metric, "mean") for metric in aggregate_columns},
        )
        .sort_values(["dataset_key", "model_key"])
    )
    by_dataset.to_csv(args.output_dir / "metrics_by_dataset.csv", index=False)

    target_sets = [
        set(zip(group.dataset_key, group.target))
        for _, group in by_target.groupby("model_key")
    ]
    common_tasks = set.intersection(*target_sets) if target_sets else set()
    common = by_target[
        by_target.apply(lambda row: (row.dataset_key, row.target) in common_tasks, axis=1)
    ]
    overall = (
        common.groupby(["model_key", "model"], as_index=False)
        .agg(
            targets=("target", "count"),
            **{metric: (metric, "mean") for metric in aggregate_columns},
        )
        .sort_values("model_key")
    )
    overall.to_csv(args.output_dir / "metrics_common_overall.csv", index=False)

    dataset_image = formatted_table(by_dataset, include_dataset=True)
    save_table_image(
        dataset_image,
        args.output_dir / "generative_metrics_by_dataset.png",
        "LOCO reconstruction: distribution quality by dataset",
    )
    overall_image = formatted_table(overall, include_dataset=False)
    save_table_image(
        overall_image,
        args.output_dir / "generative_metrics_common_overall.png",
        f"LOCO reconstruction: mean over {len(common_tasks)} common tasks",
    )
    xgboost_image = formatted_table(
        by_dataset[by_dataset.model_key == "xgboost"], include_dataset=True
    )
    save_table_image(
        xgboost_image,
        args.output_dir / "xgboost_generative_metrics.png",
        "XGBoost LOCO reconstruction: distribution quality",
    )

    for dataset_key, group in by_target.groupby("dataset_key"):
        target_table = group.loc[
            :, ["target", "model"] + METRIC_COLUMNS
        ].rename(columns={"target": "Target", "model": "Model", **METRIC_LABELS})
        for column in METRIC_LABELS.values():
            target_table[column] = target_table[column].map(lambda value: f"{value:.3f}")
        save_table_image(
            target_table,
            args.output_dir / f"generative_metrics_{dataset_key}.png",
            f"LOCO distribution quality: {DATASET_LABELS.get(dataset_key, dataset_key)}",
        )

    run_info = {
        "data_root": str(args.data_root.resolve()),
        "roots": {key: str(value) for key, value in roots.items()},
        "metric_sample_size": args.metric_sample_size,
        "dcr_sample_size": args.dcr_sample_size,
        "diffusion_sampling": "one prediction per row pooled across saved trials",
        "alpha_beta_backend": "SynthCity AlphaPrecision naive implementation",
        "c2st_backend": "sdmetrics.single_table.LogisticDetection",
        "quality_backend": (
            "SDMetrics QualityReport with correlation and association thresholds set "
            "to zero so every column pair contributes"
        ),
        "common_tasks": len(common_tasks),
        "dcr_warning": (
            "Conditional reconstructions copy observed test cells exactly; DCR is "
            "reported descriptively and is not a standalone privacy assessment."
        ),
    }
    json_dump(args.output_dir / "run_info.json", run_info)
    print(f"Saved report to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
