from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def split_indices(
    frame: pd.DataFrame,
    validation_fraction: float,
    group_column: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    if group_column and group_column in frame.columns:
        groups = frame[group_column].fillna("__missing_group__").astype(str)
        if groups.nunique() >= 2:
            splitter = GroupShuffleSplit(
                n_splits=1, test_size=validation_fraction, random_state=seed
            )
            train_indices, validation_indices = next(
                splitter.split(frame, groups=groups)
            )
            return train_indices, validation_indices, f"group:{group_column}"

    indices = np.random.default_rng(seed).permutation(len(frame))
    validation_rows = max(1, round(len(frame) * validation_fraction))
    return indices[validation_rows:], indices[:validation_rows], "random_rows"
