import pandas as pd

from timestransfer.data import load_ett_h1


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
