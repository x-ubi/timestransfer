import pandas as pd

from timestransfer.data import fixed_train_test_split, select_series


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
