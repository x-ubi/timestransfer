import pandas as pd

import pytest

from timestransfer.data import fixed_train_test_split, rolling_origin_splits, select_series


def test_fixed_train_test_split_uses_last_horizon_per_series():
    df = pd.DataFrame(
        {
            "unique_id": ["a"] * 6 + ["b"] * 6,
            "ds": [1, 2, 3, 4, 5, 6] * 2,
            "y": list(range(6)) + list(range(10, 16)),
        }
    )

    split = fixed_train_test_split(df, horizon=2)

    assert len(split.train) == 8
    assert len(split.test) == 4
    for unique_id in ["a", "b"]:
        train_group = split.train[split.train["unique_id"] == unique_id]
        test_group = split.test[split.test["unique_id"] == unique_id]
        assert train_group["ds"].max() == 4
        assert test_group["ds"].tolist() == [5, 6]
        assert train_group["ds"].max() < test_group["ds"].min()


def test_offset_split_drops_observations_after_the_window():
    df = pd.DataFrame({"unique_id": ["a"] * 10, "ds": range(1, 11), "y": range(10)})

    split = fixed_train_test_split(df, horizon=2, offset=3)

    assert split.test["ds"].tolist() == [6, 7]
    assert split.train["ds"].tolist() == [1, 2, 3, 4, 5]


def test_rolling_origin_splits_are_non_overlapping_and_final_first():
    df = pd.DataFrame({"unique_id": ["a"] * 10, "ds": range(1, 11), "y": range(10)})

    splits = rolling_origin_splits(df, horizon=2, n_windows=3)

    assert [s.test["ds"].tolist() for s in splits] == [[9, 10], [7, 8], [5, 6]]
    assert all(s.train["ds"].max() < s.test["ds"].min() for s in splits)


def test_rolling_origin_splits_reject_too_short_series():
    df = pd.DataFrame({"unique_id": ["a"] * 5, "ds": range(1, 6), "y": range(5)})

    with pytest.raises(ValueError, match="offset"):
        rolling_origin_splits(df, horizon=2, n_windows=3)


def test_select_series_is_deterministic():
    df = pd.DataFrame(
        {
            "unique_id": ["H2", "H1", "H3", "H1"],
            "ds": [1, 1, 1, 2],
            "y": [2.0, 1.0, 3.0, 1.5],
        }
    )

    selected = select_series(df, series_limit=2)

    assert selected["unique_id"].drop_duplicates().tolist() == ["H1", "H2"]
