import numpy as np
import pandas as pd

from timestransfer.data import fixed_train_test_split
from timestransfer.features import build_lag_feature_bundle


def _toy_df():
    return pd.DataFrame(
        {
            "unique_id": ["a"] * 12 + ["b"] * 12,
            "ds": list(range(1, 13)) * 2,
            "y": list(range(1, 13)) + list(range(101, 113)),
        }
    )


def test_lag_features_are_capped_and_have_expected_test_rows():
    split = fixed_train_test_split(_toy_df(), horizon=3)

    bundle = build_lag_feature_bundle(
        split.train,
        split.test,
        horizon=3,
        seasonality=3,
        lags=[1, 2, 5],
        train_row_cap=7,
        seed=123,
    )

    assert len(bundle.X_train) == 7
    assert len(bundle.y_train) == 7
    assert len(bundle.X_test) == 6
    assert bundle.test_index[["unique_id", "horizon"]].values.tolist() == [
        ["a", 1],
        ["a", 2],
        ["a", 3],
        ["b", 1],
        ["b", 2],
        ["b", 3],
    ]
    assert {"lag_1", "lag_2", "lag_5", "season_sin", "season_cos"}.issubset(bundle.X_train.columns)
    assert np.isfinite(bundle.X_test["lag_1"]).all()


def test_lag_feature_targets_do_not_use_holdout_values():
    split = fixed_train_test_split(_toy_df(), horizon=3)
    max_train_ds = split.train.groupby("unique_id")["ds"].max().to_dict()
    min_test_ds = split.test.groupby("unique_id")["ds"].min().to_dict()

    bundle = build_lag_feature_bundle(
        split.train,
        split.test,
        horizon=3,
        seasonality=3,
        lags=[1],
        train_row_cap=None,
        seed=123,
    )

    for _, row in bundle.train_index.iterrows():
        uid = row["unique_id"]
        assert row["target_ds"] <= max_train_ds[uid]
        assert row["target_ds"] < min_test_ds[uid]
