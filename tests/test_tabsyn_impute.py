from __future__ import annotations

import pandas as pd
import pytest

from scripts.tabsyn_impute import select_model_columns


INFO = {
    "idx_name_mapping": {"0": "iops", "1": "lat", "2": "io_type"},
    "target_col_idx": [0],
    "num_col_idx": [1],
    "cat_col_idx": [2],
}


def test_select_model_columns_drops_unmodeled_prepared_column() -> None:
    frame = pd.DataFrame(
        {
            "iops": [100],
            "lat": [200],
            "io_type": ["read"],
            "3": ["unused-id"],
        }
    )

    selected = select_model_columns(frame, INFO)

    assert list(selected.columns) == ["iops", "lat", "io_type"]
    assert selected.iloc[0].to_dict() == {
        "iops": 100,
        "lat": 200,
        "io_type": "read",
    }


def test_select_model_columns_rejects_missing_model_column() -> None:
    frame = pd.DataFrame({"iops": [100], "io_type": ["read"]})

    with pytest.raises(ValueError, match="missing model columns: \\['lat'\\]"):
        select_model_columns(frame, INFO)
