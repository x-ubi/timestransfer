import pandas as pd
import pytest

from timestransfer.data import ensure_datetime_ds, load_dataset, load_ett_h1, load_long_horizon2


def test_load_ett_h1_returns_canonical_schema(monkeypatch, tmp_path):
    from datasetsforecast import long_horizon2

    raw = pd.DataFrame(
        {
            "unique_id": ["OT", "HUFL", "OT", "HUFL"],
            "ds": [
                "2016-07-01 01:00:00",
                "2016-07-01 00:00:00",
                "2016-07-01 00:00:00",
                "2016-07-01 01:00:00",
            ],
            "y": ["1.5", "2.0", "1.0", "2.5"],
        }
    )

    monkeypatch.setattr(
        long_horizon2.LongHorizon2,
        "load",
        staticmethod(lambda directory, group, normalize=False: raw),
    )

    df = load_ett_h1(tmp_path, expected_n_series=2)

    assert list(df.columns) == ["unique_id", "ds", "y"]
    assert df["unique_id"].drop_duplicates().tolist() == ["HUFL", "OT"]
    assert pd.api.types.is_datetime64_any_dtype(df["ds"])
    assert df["y"].dtype == float


def test_load_long_horizon2_library_group_uses_library_loader(monkeypatch, tmp_path):
    from datasetsforecast import long_horizon2

    raw = pd.DataFrame(
        {
            "unique_id": ["OT", "OT"],
            "ds": ["2016-07-01 00:00:00", "2016-07-01 01:00:00"],
            "y": ["1.0", "2.0"],
        }
    )
    seen = {}

    def fake_load(directory, group, normalize=False):
        seen["group"] = group
        return raw

    monkeypatch.setattr(long_horizon2.LongHorizon2, "load", staticmethod(fake_load))

    df = load_long_horizon2(tmp_path, group="ETTh2", expected_n_series=1)

    assert seen["group"] == "ETTh2"
    assert list(df.columns) == ["unique_id", "ds", "y"]


def test_load_long_horizon2_reads_unregistered_groups_from_csv(monkeypatch, tmp_path):
    from datasetsforecast import long_horizon2

    csv_dir = tmp_path / "longhorizon2" / "all_six_datasets" / "Exchange"
    csv_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": ["1990-01-01", "1990-01-02"],
            "0": [0.78, 0.79],
            "OT": [0.59, 0.60],
        }
    ).to_csv(csv_dir / "Y_df.csv", index=False)
    monkeypatch.setattr(
        long_horizon2.LongHorizon2, "download", staticmethod(lambda directory: None)
    )

    df = load_long_horizon2(tmp_path, group="Exchange", expected_n_series=2)

    assert list(df.columns) == ["unique_id", "ds", "y"]
    assert df["unique_id"].drop_duplicates().tolist() == ["0", "OT"]
    assert pd.api.types.is_datetime64_any_dtype(df["ds"])
    assert len(df) == 4


def test_load_long_horizon2_rejects_unknown_group(tmp_path):
    with pytest.raises(ValueError, match="Unknown LongHorizon2 group"):
        load_long_horizon2(tmp_path, group="Nope")


def test_load_dataset_dispatches_long_horizon2(monkeypatch, tmp_path):
    from datasetsforecast import long_horizon2

    raw = pd.DataFrame(
        {"unique_id": ["OT"], "ds": ["2016-07-01 00:00:00"], "y": ["1.0"]}
    )
    monkeypatch.setattr(
        long_horizon2.LongHorizon2,
        "load",
        staticmethod(lambda directory, group, normalize=False: raw),
    )

    df = load_dataset(
        data_dir=tmp_path,
        dataset_config={"name": "ett_h2_h48", "loader": "long_horizon2", "group": "ETTh2"},
    )

    assert list(df.columns) == ["unique_id", "ds", "y"]


def test_ensure_datetime_ds_maps_integer_ds_to_synthetic_hours():
    train = pd.DataFrame({"unique_id": ["a"] * 3, "ds": [1, 2, 3], "y": [1.0, 2.0, 3.0]})
    test = pd.DataFrame({"unique_id": ["a"] * 2, "ds": [4, 5], "y": [4.0, 5.0]})

    train_dt, test_dt = ensure_datetime_ds(train, test, time_freq="h")

    assert pd.api.types.is_datetime64_any_dtype(train_dt["ds"])
    assert train_dt["ds"].iloc[0] == pd.Timestamp("2000-01-01 00:00:00")
    assert train_dt["ds"].iloc[-1] == pd.Timestamp("2000-01-01 02:00:00")
    assert test_dt["ds"].iloc[0] == pd.Timestamp("2000-01-01 03:00:00")
    assert train_dt["ds"].max() < test_dt["ds"].min()
    # Originals are untouched.
    assert train["ds"].tolist() == [1, 2, 3]


def test_ensure_datetime_ds_passes_datetime_through():
    train = pd.DataFrame(
        {"unique_id": ["a"], "ds": pd.to_datetime(["2016-07-01"]), "y": [1.0]}
    )
    test = pd.DataFrame(
        {"unique_id": ["a"], "ds": pd.to_datetime(["2016-07-02"]), "y": [2.0]}
    )

    train_out, test_out = ensure_datetime_ds(train, test, time_freq="h")

    assert train_out is train
    assert test_out is test
