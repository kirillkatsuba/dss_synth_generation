from __future__ import annotations

import hashlib
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from .schema import _value_key

METRIC_PROTOCOL_VERSION = "explicit_schema_masked_targets_c2st_union_v3"


def generative_metric_seed(base_seed: int, scenario: str) -> int:
    """Keep metric samples fixed when scenarios are reordered or filtered."""
    digest = hashlib.sha256(scenario.encode("utf-8")).digest()
    return (base_seed + int.from_bytes(digest[:4], "little")) % (2**32)


def aggregate_metric_trials(
    trials: list[dict[str, float]],
) -> dict[str, float]:
    """Aggregate repeated stochastic evaluations as mean and sample std."""
    if not trials:
        raise ValueError("At least one metric trial is required.")

    result: dict[str, float] = {}
    names = sorted(set().union(*(trial.keys() for trial in trials)))
    for name in names:
        values = np.asarray(
            [trial[name] for trial in trials if name in trial], dtype=float
        )
        values = values[np.isfinite(values)]
        if not len(values):
            continue
        result[name] = float(values.mean())
        result[f"{name}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        result[f"{name}_trials"] = float(len(values))
    return result


def _paired_sample(
    real: pd.DataFrame,
    generated: pd.DataFrame,
    size: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    indices = _sample_indices(len(real), size, seed)
    return (
        real.iloc[indices].reset_index(drop=True),
        generated.iloc[indices].reset_index(drop=True),
    )


def _sample_indices(rows: int, size: int, seed: int) -> np.ndarray:
    if size <= 0 or rows <= size:
        return np.arange(rows)
    return np.random.default_rng(seed).choice(rows, size=size, replace=False)


def _metadata_for(
    frame: pd.DataFrame,
    numerical_columns: Iterable[str] | None = None,
    categorical_columns: Iterable[str] | None = None,
) -> dict[str, Any]:
    columns = set(frame.columns)
    numerical = None if numerical_columns is None else set(numerical_columns)
    categorical = None if categorical_columns is None else set(categorical_columns)
    if numerical is None and categorical is None:
        numerical = {
            column
            for column in frame
            if pd.api.types.is_numeric_dtype(frame[column])
            and not pd.api.types.is_bool_dtype(frame[column])
        }
    if numerical is None:
        numerical = columns - categorical
    if categorical is None:
        categorical = columns - numerical
    if numerical & categorical or numerical | categorical != columns:
        raise ValueError("Metric schema must assign every column exactly one type.")
    return {
        "METADATA_SPEC_VERSION": "SINGLE_TABLE_V1",
        "columns": {
            column: {"sdtype": ("numerical" if column in numerical else "categorical")}
            for column in frame.columns
        },
    }


def prepare_metric_tables(
    real: pd.DataFrame,
    generated: pd.DataFrame,
    train_reference: pd.DataFrame | None = None,
    *,
    numerical_columns: Iterable[str] | None = None,
    categorical_columns: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, dict[str, Any]]:
    """Use one semantic schema and representation for every evaluated table.

    Explicit types are recommended; dtype inference remains for legacy callers.
    Numeric category values 1 and 1.0 share a token. Unknown categories retain
    their own tokens rather than being silently coerced to missing values.
    """
    if not real.columns.is_unique or not len(real.columns):
        raise ValueError("Metric columns must be nonempty and unique.")
    if list(real.columns) != list(generated.columns) or len(real) != len(generated):
        raise ValueError(
            "Real and generated tables must have matching columns and rows."
        )
    if train_reference is not None and list(real.columns) != list(
        train_reference.columns
    ):
        raise ValueError("Training reference columns must match evaluation columns.")
    if real.empty or (train_reference is not None and train_reference.empty):
        raise ValueError("Metric tables must contain rows.")
    metadata = _metadata_for(real, numerical_columns, categorical_columns)
    frames = [real.copy(), generated.copy()]
    if train_reference is not None:
        frames.append(train_reference.copy())
    for column, specification in metadata["columns"].items():
        if specification["sdtype"] == "numerical":
            for frame in frames:
                values = pd.to_numeric(frame[column], errors="raise")
                values = values.to_numpy(dtype=float, na_value=np.nan)
                if np.isinf(values).any():
                    raise ValueError(
                        f"Numerical metric column {column!r} contains infinity."
                    )
                frame[column] = values
        else:
            reference = pd.concat(
                (
                    [real[column], train_reference[column]]
                    if train_reference is not None
                    else [real[column]]
                ),
                ignore_index=True,
            ).dropna()
            numeric_categories = len(reference) > 0 and all(
                isinstance(value, (int, float, np.number))
                and not isinstance(value, (bool, np.bool_))
                for value in reference
            )
            for frame in frames:
                values = frame[column].astype(object)
                if numeric_categories:
                    numbers = pd.to_numeric(values, errors="coerce")
                    values = values.where(numbers.isna(), numbers.astype(object))
                frame[column] = values.map(_value_key).astype(object)
    return frames[0], frames[1], frames[2] if len(frames) == 3 else None, metadata


def quality_report_scores(
    report: Any,
    target_columns: Iterable[str] | None = None,
    expected_columns: Iterable[str] | None = None,
) -> tuple[dict[str, float], list[str]]:
    """Reject partial report aggregates instead of averaging away failed metrics."""
    metrics: dict[str, float] = {}
    errors: list[str] = []
    details: dict[str, pd.DataFrame] = {}
    valid: dict[str, np.ndarray] = {}
    columns = None if expected_columns is None else set(expected_columns)
    for property_name, name in (
        ("Column Shapes", "column_shapes"),
        ("Column Pair Trends", "column_pair_trends"),
    ):
        frame = report.get_details(property_name)
        details[name] = frame
        scores = pd.to_numeric(frame["Score"], errors="coerce").to_numpy(dtype=float)
        failures = (
            frame["Error"].notna() & frame["Error"].astype(str).str.len().gt(0)
            if "Error" in frame
            else pd.Series(False, index=frame.index)
        )
        valid[name] = np.isfinite(scores) & ~failures.to_numpy()
        for index in np.flatnonzero(~valid[name]):
            row = frame.iloc[index]
            label = (
                row["Column"]
                if name == "column_shapes"
                else f"{row['Column 1']} / {row['Column 2']}"
            )
            errors.append(
                f"quality/{property_name}/{label}: {row.get('Error', 'non-finite score')}"
            )
        expected = len(frame)
        if columns is not None:
            expected = (
                len(columns)
                if name == "column_shapes"
                else len(columns) * (len(columns) - 1) // 2
            )
        if len(frame) != expected:
            errors.append(
                f"quality/{property_name}: expected {expected} entries, got {len(frame)}"
            )
        if len(frame) == expected and len(frame) and valid[name].all():
            metrics[name] = float(scores.mean())
    no_pairs_expected = columns is not None and len(columns) == 1
    if "column_shapes" in metrics and (
        "column_pair_trends" in metrics or no_pairs_expected
    ):
        metrics["overall_quality"] = float(
            np.mean(
                [
                    metrics[name]
                    for name in ("column_shapes", "column_pair_trends")
                    if name in metrics
                ]
            )
        )
    if target_columns is not None:
        targets = set(target_columns)
        shapes = details["column_shapes"]
        pairs = details["column_pair_trends"]
        selections = {
            "column_shapes": shapes["Column"].isin(targets).to_numpy(),
            "column_pair_trends": (
                pairs["Column 1"].isin(targets) | pairs["Column 2"].isin(targets)
            ).to_numpy(),
        }
        for name, selected in selections.items():
            count = int(selected.sum())
            valid_count = int((selected & valid[name]).sum())
            expected = count
            if columns is not None:
                n, k = len(columns), len(targets)
                expected = (
                    k if name == "column_shapes" else k * (n - k) + k * (k - 1) // 2
                )
            kind = "shape" if name == "column_shapes" else "pair"
            metrics[f"target_{kind}_count"] = float(expected)
            metrics[f"target_{kind}_valid_count"] = float(valid_count)
            if count == expected and count > 0 and valid_count == count:
                metrics[f"target_{name}"] = float(
                    pd.to_numeric(details[name].loc[selected, "Score"]).mean()
                )
        if "target_column_shapes" in metrics and (
            "target_column_pair_trends" in metrics or metrics["target_pair_count"] == 0
        ):
            metrics["target_overall_quality"] = float(
                np.mean(
                    [
                        metrics[name]
                        for name in (
                            "target_column_shapes",
                            "target_column_pair_trends",
                        )
                        if name in metrics
                    ]
                )
            )
    return metrics, errors


def _alpha_beta_embedding(
    real: pd.DataFrame,
    generated: pd.DataFrame,
    numerical_columns: Iterable[str] | None = None,
    categorical_columns: Iterable[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    real, generated, _, metadata = prepare_metric_tables(
        real,
        generated,
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
    )
    numerical = [
        column
        for column, spec in metadata["columns"].items()
        if spec["sdtype"] == "numerical"
    ]
    categorical = [column for column in real if column not in numerical]
    real_parts: list[np.ndarray] = []
    generated_parts: list[np.ndarray] = []

    if numerical:
        real_num = real[numerical].apply(pd.to_numeric, errors="coerce")
        medians = real_num.median()
        generated_num = generated[numerical].apply(pd.to_numeric, errors="coerce")
        real_parts.append(real_num.fillna(medians).to_numpy(dtype=float))
        generated_parts.append(generated_num.fillna(medians).to_numpy(dtype=float))
    if categorical:
        categories = [
            np.sort(pd.concat([real[column], generated[column]]).unique())
            for column in categorical
        ]
        encoder = OneHotEncoder(
            categories=categories, handle_unknown="error", sparse_output=False
        )
        real_cat = real[categorical].fillna("__MISSING__").astype(str)
        generated_cat = generated[categorical].fillna("__MISSING__").astype(str)
        real_parts.append(encoder.fit_transform(real_cat))
        generated_parts.append(encoder.transform(generated_cat))

    real_x = np.hstack(real_parts)
    generated_x = np.hstack(generated_parts)
    scaler = MinMaxScaler().fit(real_x)
    return scaler.transform(real_x), scaler.transform(generated_x)


def alpha_precision_beta_recall(
    real: pd.DataFrame,
    generated: pd.DataFrame,
    *,
    numerical_columns: Iterable[str] | None = None,
    categorical_columns: Iterable[str] | None = None,
) -> tuple[float, float, float]:
    """SynthCity's naive AlphaPrecision implementation without its dependency."""
    if len(real) < 2 or len(generated) < 2:
        raise ValueError("Alpha/beta metrics require at least two rows.")

    real_x, generated_x = _alpha_beta_embedding(
        real, generated, numerical_columns, categorical_columns
    )
    center = real_x.mean(axis=0)
    alphas = np.linspace(0, 1, 30)
    radii = np.quantile(np.linalg.norm(real_x - center, axis=1), alphas)
    generated_center = generated_x.mean(axis=0)
    generated_to_center = np.linalg.norm(generated_x - center, axis=1)

    real_nn = NearestNeighbors(n_neighbors=2, n_jobs=1).fit(real_x)
    real_to_real = real_nn.kneighbors(real_x, return_distance=True)[0][:, 1]
    generated_nn = NearestNeighbors(n_neighbors=1, n_jobs=1).fit(generated_x)
    real_to_generated, nearest_indices = generated_nn.kneighbors(
        real_x, return_distance=True
    )
    real_to_generated = real_to_generated[:, 0]
    nearest_indices = nearest_indices[:, 0]
    closest_generated = generated_x[nearest_indices]
    closest_generated_distance = np.linalg.norm(
        closest_generated - generated_center, axis=1
    )
    closest_generated_radii = np.quantile(closest_generated_distance, alphas)

    precision_curve = np.array(
        [np.mean(generated_to_center <= radius) for radius in radii]
    )
    coverage_curve = np.array(
        [
            np.mean(
                (real_to_generated <= real_to_real)
                & (closest_generated_distance <= closest_generated_radii[index])
            )
            for index in range(len(alphas))
        ]
    )
    alpha_precision = 1 - np.abs(alphas - precision_curve).sum() / alphas.sum()
    beta_recall = 1 - np.abs(alphas - coverage_curve).sum() / alphas.sum()
    authenticity = np.mean(real_to_real[nearest_indices] < real_to_generated)
    return float(alpha_precision), float(beta_recall), float(authenticity)


def logistic_detection_quality(
    real: pd.DataFrame,
    generated: pd.DataFrame,
    metadata: dict[str, Any],
    seed: int,
) -> float:
    """Compute SDMetrics C2ST with a shared categorical support and fixed folds.

    SDMetrics fits its categorical transformer on real data only. Explicit
    one-hot encoding of the union preserves generated-only categories instead
    of crashing or silently mapping them to an all-zero unknown category.
    """
    from sdmetrics.single_table import LogisticDetection

    categorical = [
        name for name, spec in metadata["columns"].items()
        if spec["sdtype"] == "categorical"
    ]
    if categorical:
        numerical = [name for name in real if name not in categorical]
        r, g, _, _ = prepare_metric_tables(
            real, generated, numerical_columns=numerical,
            categorical_columns=categorical,
        )
        encoder = OneHotEncoder(handle_unknown="error", sparse_output=False)
        encoder.fit(pd.concat([r[categorical], g[categorical]], ignore_index=True))
        real_x = np.hstack([r[numerical].to_numpy(float), encoder.transform(r[categorical])])
        generated_x = np.hstack([g[numerical].to_numpy(float), encoder.transform(g[categorical])])
        names = [f"feature_{index}" for index in range(real_x.shape[1])]
        real, generated = pd.DataFrame(real_x, columns=names), pd.DataFrame(generated_x, columns=names)
        metadata = _metadata_for(real, names, [])

    random_state = np.random.get_state()
    try:
        np.random.seed(seed)
        return float(LogisticDetection.compute(real, generated, metadata))
    finally:
        np.random.set_state(random_state)


def evaluate_masked_numerical_accuracy(
    real: pd.DataFrame,
    generated: pd.DataFrame,
    target_mask: np.ndarray,
    model_columns: list[str],
    numerical_columns: list[str],
) -> dict[str, float]:
    """Measure row-wise accuracy only on artificially hidden numerical cells."""
    if target_mask.shape != (len(real), len(model_columns)):
        raise ValueError("Target mask shape does not match the modeled table.")

    metrics: dict[str, float] = {}
    for column in numerical_columns:
        column_index = model_columns.index(column)
        selected = target_mask[:, column_index]
        truth = pd.to_numeric(real.loc[selected, column], errors="coerce").to_numpy(
            dtype=float
        )
        prediction = pd.to_numeric(
            generated.loc[selected, column], errors="coerce"
        ).to_numpy(dtype=float)
        finite = np.isfinite(truth) & np.isfinite(prediction)
        truth = truth[finite]
        prediction = prediction[finite]
        if not len(truth):
            continue

        errors = prediction - truth
        mse = float(np.mean(errors**2))
        variance = float(np.var(truth))
        normalized_mse = mse / variance if variance > 0 else float("nan")
        metrics.update(
            {
                f"{column}_masked_cells": float(len(truth)),
                f"{column}_r2": 1.0 - normalized_mse,
                f"{column}_mse": mse,
                f"{column}_rmse": float(np.sqrt(mse)),
                f"{column}_mae": float(np.mean(np.abs(errors))),
                f"{column}_normalized_mse": normalized_mse,
            }
        )
    return metrics


def evaluate_generative_quality(
    real: pd.DataFrame,
    generated: pd.DataFrame,
    train_reference: pd.DataFrame,
    sample_size: int,
    dcr_sample_size: int,
    seed: int,
    c2st_seed: int | None = None,
    *,
    numerical_columns: Iterable[str] | None = None,
    categorical_columns: Iterable[str] | None = None,
    target_columns: Iterable[str] | None = None,
    target_mask: np.ndarray | None = None,
    compute_dcr: bool = True,
) -> tuple[dict[str, float], list[str]]:
    """Evaluate one stochastic conditional reconstruction against real validation."""
    try:
        from sdmetrics.reports.single_table import QualityReport
        from sdmetrics.single_table import DCROverfittingProtection
    except ImportError as exc:
        raise RuntimeError(
            "Generative validation requires SDMetrics. Install the project's SDV "
            "dependencies in the training environment."
        ) from exc

    real, generated, train_reference, metadata = prepare_metric_tables(
        real,
        generated,
        train_reference,
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
    )
    if dcr_sample_size <= 0:
        raise ValueError("DCR sample size must be positive.")
    targets = None if target_columns is None else list(dict.fromkeys(target_columns))
    if targets is not None and (not targets or set(targets) - set(real.columns)):
        raise ValueError("Target columns must be a nonempty subset of metric columns.")
    if target_mask is not None:
        target_mask = np.asarray(target_mask)
        if target_mask.shape != real.shape or target_mask.dtype != bool:
            raise ValueError(
                "Target mask must be boolean and match the metric table shape."
            )
        if targets is None:
            targets = [
                column
                for index, column in enumerate(real)
                if target_mask[:, index].any()
            ]

    real_s, generated_s = _paired_sample(real, generated, sample_size, seed)
    metrics: dict[str, float] = {}
    errors: list[str] = []

    try:
        quality = QualityReport()
        quality.real_correlation_threshold = 0.0
        quality.real_association_threshold = 0.0
        quality.generate(real_s, generated_s, metadata, verbose=False)
        scores, report_errors = quality_report_scores(
            quality, targets if target_mask is None else None, real.columns
        )
        metrics.update(scores)
        errors.extend(report_errors)
        if targets is not None and target_mask is not None:
            macro_scores: dict[str, list[float]] = {}
            for column in targets:
                selected = np.flatnonzero(target_mask[:, real.columns.get_loc(column)])
                selected = selected[_sample_indices(len(selected), sample_size, seed)]
                metrics[f"target_{column}_rows"] = float(len(selected))
                if len(selected) < 2:
                    errors.append(f"target/{column}: fewer than two hidden rows")
                    continue
                target_report = quality
                if not (
                    len(selected) == len(real_s)
                    and np.array_equal(
                        selected, _sample_indices(len(real), sample_size, seed)
                    )
                ):
                    target_report = QualityReport()
                    target_report.real_correlation_threshold = 0.0
                    target_report.real_association_threshold = 0.0
                    target_report.generate(
                        real.iloc[selected].reset_index(drop=True),
                        generated.iloc[selected].reset_index(drop=True),
                        metadata,
                        verbose=False,
                    )
                target_scores, target_errors = quality_report_scores(
                    target_report, [column], real.columns
                )
                errors.extend(f"target/{column}: {error}" for error in target_errors)
                for name, value in target_scores.items():
                    if name.startswith("target_"):
                        suffix = name.removeprefix("target_")
                        metrics[f"target_{column}_{suffix}"] = value
                        if suffix in {
                            "column_shapes",
                            "column_pair_trends",
                            "overall_quality",
                        }:
                            macro_scores.setdefault(suffix, []).append(value)
                target_real = real.iloc[selected][[column]].reset_index(drop=True)
                target_generated = generated.iloc[selected][[column]].reset_index(
                    drop=True
                )
                is_numerical = metadata["columns"][column]["sdtype"] == "numerical"
                try:
                    alpha, beta, authenticity = alpha_precision_beta_recall(
                        target_real,
                        target_generated,
                        numerical_columns=[column] if is_numerical else [],
                        categorical_columns=[] if is_numerical else [column],
                    )
                    for name, value in (
                        ("alpha_precision", alpha),
                        ("beta_recall", beta),
                        ("authenticity", authenticity),
                    ):
                        metrics[f"target_{column}_{name}"] = value
                        macro_scores.setdefault(name, []).append(value)
                except Exception as exc:
                    errors.append(
                        f"target/{column}/alpha_beta: {type(exc).__name__}: {exc}"
                    )
                try:
                    value = logistic_detection_quality(
                        target_real,
                        target_generated,
                        {
                            "METADATA_SPEC_VERSION": "SINGLE_TABLE_V1",
                            "columns": {column: metadata["columns"][column]},
                        },
                        seed if c2st_seed is None else c2st_seed,
                    )
                    metrics[f"target_{column}_c2st"] = value
                    macro_scores.setdefault("c2st", []).append(value)
                except Exception as exc:
                    errors.append(f"target/{column}/c2st: {type(exc).__name__}: {exc}")
            for name, values in macro_scores.items():
                metrics[f"target_{name}_valid_targets"] = float(len(values))
                if len(values) == len(targets):
                    metrics[f"target_{name}"] = float(np.mean(values))
    except Exception as exc:  # SDMetrics failures should not abort model training.
        errors.append(f"quality: {type(exc).__name__}: {exc}")

    try:
        alpha_precision, beta_recall, authenticity = alpha_precision_beta_recall(
            real_s,
            generated_s,
            numerical_columns=[
                column
                for column, spec in metadata["columns"].items()
                if spec["sdtype"] == "numerical"
            ],
            categorical_columns=[
                column
                for column, spec in metadata["columns"].items()
                if spec["sdtype"] == "categorical"
            ],
        )
        metrics.update(
            {
                "alpha_precision": alpha_precision,
                "beta_recall": beta_recall,
                "authenticity": authenticity,
            }
        )
    except Exception as exc:
        errors.append(f"alpha_beta: {type(exc).__name__}: {exc}")

    try:
        metrics["c2st"] = logistic_detection_quality(
            real_s,
            generated_s,
            metadata,
            seed=seed if c2st_seed is None else c2st_seed,
        )
    except Exception as exc:
        errors.append(f"c2st: {type(exc).__name__}: {exc}")

    if not compute_dcr:
        return metrics, errors

    try:
        train_s = train_reference.sample(
            n=min(dcr_sample_size, len(train_reference)), random_state=seed
        ).reset_index(drop=True)
        dcr_real, dcr_generated = _paired_sample(
            real, generated, dcr_sample_size, seed + 1
        )
        metrics["dcr"] = float(
            DCROverfittingProtection.compute(
                train_s,
                dcr_generated,
                dcr_real,
                metadata,
                num_rows_subsample=None,
                num_iterations=1,
            )
        )
    except Exception as exc:
        errors.append(f"dcr: {type(exc).__name__}: {exc}")

    return metrics, errors
