#!/usr/bin/env python3
"""Supervised XGBoost LOCO baseline using the hyperparameter grid from TabDiff."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier, XGBRegressor, __version__ as xgboost_version


GRID = tuple(
    {
        "n_estimators": n_estimators,
        "min_child_weight": min_child_weight,
        "max_depth": max_depth,
        "gamma": gamma,
    }
    for n_estimators, min_child_weight, max_depth, gamma in product(
        (10, 50, 100), (1, 10), (5, 10, 20), (0.0, 1.0)
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the supervised XGBoost reference used by TabDiff."
    )
    parser.add_argument("--dataname", required=True)
    parser.add_argument(
        "--tabdiff-repo", type=Path, default=Path("external/tabdiff")
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        default=["all"],
        help="Target columns, or 'all' for every non-constant column.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--authors-target-transform",
        action="store_true",
        help="Apply log(clip(y, 1, 20000)) used in TabDiff's regression evaluator.",
    )
    return parser.parse_args()


def xgb_device_kwargs(device: str) -> dict[str, object]:
    major = int(xgboost_version.split(".", 1)[0])
    if device == "cuda":
        if major >= 2:
            return {"tree_method": "hist", "device": "cuda"}
        return {"tree_method": "gpu_hist"}
    return {"tree_method": "hist", "device": "cpu"} if major >= 2 else {
        "tree_method": "hist"
    }


def make_features(
    train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, target: str
) -> tuple[object, object, object]:
    feature_cols = [column for column in train.columns if column != target]
    categorical = [
        column
        for column in feature_cols
        if not pd.api.types.is_numeric_dtype(train[column])
    ]
    numerical = [column for column in feature_cols if column not in categorical]
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
        transformer.fit_transform(train[feature_cols]),
        transformer.transform(validation[feature_cols]),
        transformer.transform(test[feature_cols]),
    )


def regression_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, prediction)),
        "rmse": float(mean_squared_error(y_true, prediction) ** 0.5),
        "mae": float(mean_absolute_error(y_true, prediction)),
    }


def classification_metrics(
    y_true: np.ndarray,
    prediction: np.ndarray,
    probabilities: np.ndarray,
    class_count: int,
) -> dict[str, float]:
    if class_count == 2:
        auc = roc_auc_score(y_true, probabilities[:, 1])
    else:
        auc = roc_auc_score(
            y_true,
            probabilities,
            labels=np.arange(class_count),
            multi_class="ovr",
            average="macro",
        )
    return {
        "auc": float(auc),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "macro_recall": float(
            recall_score(y_true, prediction, average="macro", zero_division=0)
        ),
    }


def evaluate_regression(
    x_train: object,
    y_train: np.ndarray,
    x_validation: object,
    y_validation: np.ndarray,
    x_test: object,
    y_test: np.ndarray,
    common_kwargs: dict[str, object],
) -> dict[str, object]:
    trials = []
    for params in GRID:
        model = XGBRegressor(
            **params,
            **common_kwargs,
            objective="reg:squarederror",
            n_jobs=-1,
        )
        model.fit(x_train, y_train, verbose=False)
        metrics = regression_metrics(y_validation, model.predict(x_validation))
        trials.append({"params": params, **metrics})

    best = min(trials, key=lambda row: row["rmse"])
    model = XGBRegressor(
        **best["params"],
        **common_kwargs,
        objective="reg:squarederror",
        n_jobs=-1,
    )
    model.fit(x_train, y_train, verbose=False)
    return {
        "selection_metric": "validation_rmse",
        "best_params": best["params"],
        "validation": {key: best[key] for key in ("r2", "rmse", "mae")},
        "test": regression_metrics(y_test, model.predict(x_test)),
    }


def evaluate_classification(
    x_train: object,
    y_train: np.ndarray,
    x_validation: object,
    y_validation: np.ndarray,
    x_test: object,
    y_test: np.ndarray,
    common_kwargs: dict[str, object],
) -> dict[str, object]:
    class_count = len(np.unique(y_train))
    objective = "binary:logistic" if class_count == 2 else "multi:softprob"
    trials = []
    for params in GRID:
        model = XGBClassifier(
            **params,
            **common_kwargs,
            objective=objective,
            num_class=class_count if class_count > 2 else None,
            n_jobs=-1,
        )
        model.fit(x_train, y_train, verbose=False)
        metrics = classification_metrics(
            y_validation,
            model.predict(x_validation),
            model.predict_proba(x_validation),
            class_count,
        )
        trials.append({"params": params, **metrics})

    best = max(trials, key=lambda row: row["auc"])
    model = XGBClassifier(
        **best["params"],
        **common_kwargs,
        objective=objective,
        num_class=class_count if class_count > 2 else None,
        n_jobs=-1,
    )
    model.fit(x_train, y_train, verbose=False)
    return {
        "selection_metric": "validation_auc",
        "best_params": best["params"],
        "validation": {
            key: best[key]
            for key in ("auc", "accuracy", "macro_f1", "macro_recall")
        },
        "test": classification_metrics(
            y_test,
            model.predict(x_test),
            model.predict_proba(x_test),
            class_count,
        ),
    }


def main() -> None:
    args = parse_args()
    data_dir = args.tabdiff_repo / "data" / args.dataname
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    columns = args.columns
    if columns == ["all"]:
        columns = [column for column in train.columns if train[column].nunique() > 1]
    unknown = sorted(set(columns) - set(train.columns))
    if unknown:
        raise ValueError(f"Unknown columns: {unknown}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    common_kwargs = {"random_state": args.seed, **xgb_device_kwargs(args.device)}
    summary = []

    for target in columns:
        train_part, validation = train_test_split(
            train, test_size=1 / 9, random_state=args.seed
        )
        x_train, x_validation, x_test = make_features(
            train_part, validation, test, target
        )
        categorical_target = not pd.api.types.is_numeric_dtype(train[target])
        if categorical_target:
            encoder = LabelEncoder().fit(train[target].astype(str))
            y_train = encoder.transform(train_part[target].astype(str))
            y_validation = encoder.transform(validation[target].astype(str))
            y_test = encoder.transform(test[target].astype(str))
            result = evaluate_classification(
                x_train,
                y_train,
                x_validation,
                y_validation,
                x_test,
                y_test,
                common_kwargs,
            )
            task = "classification"
        else:
            y_train = train_part[target].to_numpy(dtype=np.float64)
            y_validation = validation[target].to_numpy(dtype=np.float64)
            y_test = test[target].to_numpy(dtype=np.float64)
            if args.authors_target_transform:
                y_train = np.log(np.clip(y_train, 1, 20000))
                y_validation = np.log(np.clip(y_validation, 1, 20000))
                y_test = np.log(np.clip(y_test, 1, 20000))
            result = evaluate_regression(
                x_train,
                y_train,
                x_validation,
                y_validation,
                x_test,
                y_test,
                common_kwargs,
            )
            task = "regression"

        payload = {
            "dataname": args.dataname,
            "target": target,
            "task": task,
            "xgboost_version": xgboost_version,
            "device": args.device,
            "authors_target_transform": args.authors_target_transform,
            **result,
        }
        (args.output_dir / f"{target}.json").write_text(json.dumps(payload, indent=2))
        summary.append({"target": target, "task": task, **result["test"]})
        print(target, result["test"], flush=True)

    pd.DataFrame(summary).to_csv(args.output_dir / "summary.csv", index=False)


if __name__ == "__main__":
    main()
