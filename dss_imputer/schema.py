from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def _python_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


def _value_key(value: Any) -> str:
    value = _python_value(value)
    if isinstance(value, bool):
        payload = {"type": "bool", "value": value}
    elif isinstance(value, (int, float)) and math.isfinite(float(value)):
        numeric_value: int | float = float(value)
        if numeric_value.is_integer():
            numeric_value = int(numeric_value)
        payload = {"type": "number", "value": numeric_value}
    else:
        payload = {"type": type(value).__name__, "value": value}
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
    )


@dataclass
class EncodedTable:
    numerical: np.ndarray
    categorical: np.ndarray
    numerical_available: np.ndarray
    categorical_available: np.ndarray

    @property
    def available(self) -> np.ndarray:
        return np.concatenate(
            [self.numerical_available, self.categorical_available], axis=1
        )


class TablePreprocessor:
    """Train-fitted encoding that preserves an explicit tabular schema."""

    def __init__(
        self,
        continuous_columns: Iterable[str],
        excluded_columns: Iterable[str] = ("id",),
        log_continuous_columns: Iterable[str] = (),
        log_epsilon: float = 1e-6,
    ) -> None:
        if not math.isfinite(log_epsilon) or log_epsilon <= 0:
            raise ValueError("log_epsilon must be a finite positive number.")
        self.requested_continuous = list(dict.fromkeys(continuous_columns))
        self.excluded_columns = list(dict.fromkeys(excluded_columns))
        self.log_continuous_columns = list(dict.fromkeys(log_continuous_columns))
        self.log_epsilon = float(log_epsilon)
        self.original_columns: list[str] = []
        self.numerical_columns: list[str] = []
        self.categorical_columns: list[str] = []
        self.constant_values: dict[str, Any] = {}
        self.numerical_mean: dict[str, float] = {}
        self.numerical_std: dict[str, float] = {}
        self.numerical_min: dict[str, float] = {}
        self.numerical_max: dict[str, float] = {}
        self.numerical_integer: dict[str, bool] = {}
        self.categories: dict[str, list[Any]] = {}
        self._category_maps: dict[str, dict[str, int]] = {}
        self.fitted = False

    @property
    def model_columns(self) -> list[str]:
        return self.numerical_columns + self.categorical_columns

    @property
    def categorical_sizes(self) -> list[int]:
        return [len(self.categories[column]) for column in self.categorical_columns]

    def fit(self, frame: pd.DataFrame) -> "TablePreprocessor":
        self.original_columns = frame.columns.tolist()
        unknown_continuous = sorted(
            set(self.requested_continuous) - set(self.original_columns)
        )
        if unknown_continuous:
            raise ValueError(f"Unknown continuous columns: {unknown_continuous}")
        unknown_log = sorted(
            set(self.log_continuous_columns) - set(self.requested_continuous)
        )
        if unknown_log:
            raise ValueError(
                "Log-transformed columns must also be continuous columns: "
                f"{unknown_log}"
            )

        excluded = set(self.excluded_columns)
        candidates = [
            column for column in self.original_columns if column not in excluded
        ]
        self.constant_values = {}
        for column in candidates:
            values = frame[column].dropna()
            if values.nunique(dropna=True) <= 1:
                self.constant_values[column] = (
                    _python_value(values.iloc[0]) if len(values) else None
                )

        model_candidates = [
            column for column in candidates if column not in self.constant_values
        ]
        self.numerical_columns = [
            column for column in self.requested_continuous if column in model_candidates
        ]
        self.categorical_columns = [
            column
            for column in model_candidates
            if column not in self.numerical_columns
        ]
        if not self.model_columns:
            raise ValueError(
                "No non-constant model columns remain after preprocessing."
            )

        self.numerical_mean = {}
        self.numerical_std = {}
        self.numerical_min = {}
        self.numerical_max = {}
        self.numerical_integer = {}
        for column in self.numerical_columns:
            numeric = pd.to_numeric(frame[column], errors="coerce")
            observed = numeric.dropna()
            if observed.empty:
                raise ValueError(f"Numerical column {column!r} has no observed values.")
            transformed = self._forward_numerical(column, observed)
            mean = float(transformed.mean())
            std = float(transformed.std(ddof=0))
            self.numerical_mean[column] = mean
            self.numerical_std[column] = std if std > 1e-12 else 1.0
            self.numerical_min[column] = float(observed.min())
            self.numerical_max[column] = float(observed.max())
            self.numerical_integer[column] = bool(
                pd.api.types.is_integer_dtype(frame[column].dtype)
            )

        self.categories = {}
        self._category_maps = {}
        for column in self.categorical_columns:
            observed = frame[column].dropna()
            values = [_python_value(value) for value in pd.unique(observed)]
            if not values:
                raise ValueError(
                    f"Categorical column {column!r} has no observed values."
                )
            self.categories[column] = values
            self._category_maps[column] = {
                _value_key(value): index for index, value in enumerate(values)
            }

        self.fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> EncodedTable:
        self._check_fitted()
        missing_columns = sorted(set(self.model_columns) - set(frame.columns))
        if missing_columns:
            raise ValueError(f"Input table is missing columns: {missing_columns}")

        rows = len(frame)
        numerical = np.zeros((rows, len(self.numerical_columns)), dtype=np.float32)
        numerical_available = np.zeros_like(numerical, dtype=bool)
        for index, column in enumerate(self.numerical_columns):
            values = pd.to_numeric(frame[column], errors="coerce")
            available = values.notna().to_numpy()
            numerical_available[:, index] = available
            transformed = self._forward_numerical(column, values)
            normalized = (
                transformed.to_numpy(dtype=np.float64, na_value=np.nan)
                - self.numerical_mean[column]
            ) / self.numerical_std[column]
            numerical[:, index] = np.nan_to_num(normalized, nan=0.0).astype(np.float32)

        categorical = np.zeros((rows, len(self.categorical_columns)), dtype=np.int64)
        categorical_available = np.zeros_like(categorical, dtype=bool)
        for index, column in enumerate(self.categorical_columns):
            mapping = self._category_maps[column]
            values = frame[column].to_numpy(dtype=object)
            mask_index = len(self.categories[column])
            categorical[:, index] = mask_index
            for row in np.flatnonzero(~pd.isna(values)):
                key = _value_key(values[row])
                if key in mapping:
                    categorical[row, index] = mapping[key]
                    categorical_available[row, index] = True

        return EncodedTable(
            numerical=numerical,
            categorical=categorical,
            numerical_available=numerical_available,
            categorical_available=categorical_available,
        )

    def inverse_transform(
        self, numerical: np.ndarray, categorical: np.ndarray
    ) -> pd.DataFrame:
        self._check_fitted()
        if numerical.shape[1] != len(self.numerical_columns):
            raise ValueError("Unexpected number of numerical columns.")
        if categorical.shape[1] != len(self.categorical_columns):
            raise ValueError("Unexpected number of categorical columns.")

        result: dict[str, Any] = {}
        for index, column in enumerate(self.numerical_columns):
            transformed = (
                numerical[:, index] * self.numerical_std[column]
                + self.numerical_mean[column]
            )
            values = self._inverse_numerical(column, transformed)
            values = np.clip(
                values,
                self.numerical_min[column],
                self.numerical_max[column],
            )
            if self.numerical_integer[column]:
                values = np.rint(values)
            result[column] = values

        for index, column in enumerate(self.categorical_columns):
            choices = self.categories[column]
            indices = np.clip(categorical[:, index], 0, len(choices) - 1)
            result[column] = [choices[int(value)] for value in indices]

        return pd.DataFrame(result)

    def fill_constants(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        for column, value in self.constant_values.items():
            if column not in result:
                result[column] = value
            else:
                result[column] = result[column].where(result[column].notna(), value)
        return result

    def to_dict(self) -> dict[str, Any]:
        self._check_fitted()
        return {
            "requested_continuous": self.requested_continuous,
            "excluded_columns": self.excluded_columns,
            "log_continuous_columns": self.log_continuous_columns,
            "log_epsilon": self.log_epsilon,
            "original_columns": self.original_columns,
            "numerical_columns": self.numerical_columns,
            "categorical_columns": self.categorical_columns,
            "constant_values": self.constant_values,
            "numerical_mean": self.numerical_mean,
            "numerical_std": self.numerical_std,
            "numerical_min": self.numerical_min,
            "numerical_max": self.numerical_max,
            "numerical_integer": self.numerical_integer,
            "categories": self.categories,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TablePreprocessor":
        instance = cls(
            data["requested_continuous"],
            data.get("excluded_columns", []),
            data.get("log_continuous_columns", []),
            data.get("log_epsilon", 1e-6),
        )
        instance.original_columns = list(data["original_columns"])
        instance.numerical_columns = list(data["numerical_columns"])
        instance.categorical_columns = list(data["categorical_columns"])
        instance.constant_values = dict(data["constant_values"])
        instance.numerical_mean = {
            key: float(value) for key, value in data["numerical_mean"].items()
        }
        instance.numerical_std = {
            key: float(value) for key, value in data["numerical_std"].items()
        }
        instance.numerical_min = {
            key: float(value) for key, value in data.get("numerical_min", {}).items()
        }
        instance.numerical_max = {
            key: float(value) for key, value in data.get("numerical_max", {}).items()
        }
        for column in instance.numerical_columns:
            instance.numerical_min.setdefault(column, -float("inf"))
            instance.numerical_max.setdefault(column, float("inf"))
        instance.numerical_integer = {
            key: bool(value) for key, value in data["numerical_integer"].items()
        }
        instance.categories = {
            key: list(value) for key, value in data["categories"].items()
        }
        instance._category_maps = {
            column: {
                _value_key(value): index
                for index, value in enumerate(instance.categories[column])
            }
            for column in instance.categorical_columns
        }
        instance.fitted = True
        return instance

    def _forward_numerical(self, column: str, values: pd.Series) -> pd.Series:
        if column not in self.log_continuous_columns:
            return values
        observed = values.dropna()
        invalid = observed + self.log_epsilon <= 0
        if invalid.any():
            minimum = float(observed.min())
            raise ValueError(
                f"Column {column!r} contains values incompatible with "
                f"log(x + eps): min={minimum}, eps={self.log_epsilon}."
            )
        return np.log(values + self.log_epsilon)

    def _inverse_numerical(self, column: str, values: np.ndarray) -> np.ndarray:
        if column not in self.log_continuous_columns:
            return values
        lower = math.log(self.numerical_min[column] + self.log_epsilon)
        upper = math.log(self.numerical_max[column] + self.log_epsilon)
        return np.exp(np.clip(values, lower, upper)) - self.log_epsilon

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=True))

    @classmethod
    def load(cls, path: Path) -> "TablePreprocessor":
        return cls.from_dict(json.loads(path.read_text()))

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("TablePreprocessor must be fitted first.")
