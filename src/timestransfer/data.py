from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class DatasetSplit:
    train: pd.DataFrame
    test: pd.DataFrame


def load_m4_hourly(data_dir: str | Path, expected_n_series: int | None = 414) -> pd.DataFrame:
    """Download/cache M4 Hourly and return canonical columns: unique_id, ds, y."""
    try:
        from datasetsforecast.m4 import M4
    except ImportError as exc:  # pragma: no cover - covered by integration environment.
        raise RuntimeError("datasetsforecast is required to load M4 Hourly.") from exc

    df, _, _ = M4.load(directory=str(data_dir), group="Hourly")
    df = df.loc[:, ["unique_id", "ds", "y"]].copy()
    df["unique_id"] = df["unique_id"].astype(str)
    df["ds"] = pd.to_numeric(df["ds"], errors="raise").astype(int)
    df["y"] = pd.to_numeric(df["y"], errors="raise").astype(float)
    df = df.sort_values(["unique_id", "ds"], kind="mergesort").reset_index(drop=True)

    n_series = df["unique_id"].nunique()
    if expected_n_series is not None and n_series != expected_n_series:
        raise ValueError(f"Expected {expected_n_series} M4 Hourly series, found {n_series}.")
    return df


# Groups registered in datasetsforecast.long_horizon2.LongHorizon2Info; the library's
# load() raises for anything else even though the downloaded archive contains more.
_LONG_HORIZON2_LIBRARY_GROUPS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "ECL", "TrafficL", "Weather")
# Present in the same archive but not registered in the library -> direct CSV read.
_LONG_HORIZON2_CSV_GROUPS = ("Exchange", "ILI")


def load_long_horizon2(
    data_dir: str | Path,
    group: str,
    expected_n_series: int | None = None,
) -> pd.DataFrame:
    """Download/cache a LongHorizon2 group and return canonical columns: unique_id, ds, y.

    Library-registered groups go through LongHorizon2.load (which mimics the Google
    n_time truncation for ETT). Exchange and ILI are melted directly from the extracted
    Y_df.csv with the same wrangling, minus the truncation (none is defined for them).
    """
    try:
        from datasetsforecast.long_horizon2 import LongHorizon2
    except ImportError as exc:  # pragma: no cover - covered by integration environment.
        raise RuntimeError("datasetsforecast is required to load LongHorizon2 datasets.") from exc

    if group in _LONG_HORIZON2_LIBRARY_GROUPS:
        df = LongHorizon2.load(directory=str(data_dir), group=group, normalize=False)
    elif group in _LONG_HORIZON2_CSV_GROUPS:
        LongHorizon2.download(str(data_dir))
        csv_path = Path(data_dir) / "longhorizon2" / "all_six_datasets" / group / "Y_df.csv"
        wide = pd.read_csv(csv_path)
        df = wide.set_index("date").melt(ignore_index=False).reset_index()
        df = df.rename(columns={"date": "ds", "variable": "unique_id", "value": "y"})
    else:
        raise ValueError(
            f"Unknown LongHorizon2 group {group!r}. Known groups: "
            f"{_LONG_HORIZON2_LIBRARY_GROUPS + _LONG_HORIZON2_CSV_GROUPS}"
        )

    df = df.loc[:, ["unique_id", "ds", "y"]].copy()
    df["unique_id"] = df["unique_id"].astype(str)
    df["ds"] = pd.to_datetime(df["ds"], errors="raise")
    df["y"] = pd.to_numeric(df["y"], errors="raise").astype(float)
    df = df.sort_values(["unique_id", "ds"], kind="mergesort").reset_index(drop=True)

    n_series = df["unique_id"].nunique()
    if expected_n_series is not None and n_series != expected_n_series:
        raise ValueError(f"Expected {expected_n_series} {group} series, found {n_series}.")
    return df


def load_ett_h1(data_dir: str | Path, expected_n_series: int | None = 7) -> pd.DataFrame:
    """Download/cache ETTh1 and return canonical columns: unique_id, ds, y."""
    return load_long_horizon2(data_dir, group="ETTh1", expected_n_series=expected_n_series)


def load_dataset(data_dir: str | Path, dataset_config: dict[str, Any]) -> pd.DataFrame:
    """Load a configured benchmark dataset into canonical long format."""
    loader = dataset_config.get("loader", dataset_config.get("name"))
    expected_n_series = dataset_config.get("expected_n_series")
    if loader == "m4_hourly":
        return load_m4_hourly(data_dir=data_dir, expected_n_series=expected_n_series)
    if loader == "ett_h1":
        return load_ett_h1(data_dir=data_dir, expected_n_series=expected_n_series)
    if loader == "long_horizon2":
        return load_long_horizon2(
            data_dir=data_dir,
            group=str(dataset_config["group"]),
            expected_n_series=expected_n_series,
        )
    raise ValueError(f"Unknown dataset loader: {loader!r}")


def ensure_datetime_ds(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    time_freq: str,
    origin: str = "2000-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map integer ds (e.g. M4) onto synthetic timestamps: origin + (ds - 1) * time_freq.

    Datetime ds passes through unchanged. M4 ds values are consecutive integers per
    series, so train and test stay contiguous on the synthetic time axis.
    """
    if pd.api.types.is_datetime64_any_dtype(train_df["ds"]):
        return train_df, test_df

    offset = pd.tseries.frequencies.to_offset(time_freq)
    if not isinstance(offset, pd.tseries.offsets.Tick):
        raise ValueError(
            f"time_freq {time_freq!r} is not a fixed-length frequency; "
            "cannot build synthetic timestamps for integer ds."
        )
    origin_ts = pd.Timestamp(origin)

    def _convert(df: pd.DataFrame) -> pd.DataFrame:
        converted = df.copy()
        steps = pd.to_numeric(converted["ds"], errors="raise").astype("int64") - 1
        converted["ds"] = origin_ts + pd.to_timedelta(steps * offset.nanos, unit="ns")
        return converted

    return _convert(train_df), _convert(test_df)


def select_series(df: pd.DataFrame, series_limit: int | None) -> pd.DataFrame:
    """Select the first N sorted series for a deterministic fast Phase 1 benchmark."""
    if series_limit is None:
        return df.copy()
    if series_limit <= 0:
        raise ValueError("series_limit must be positive or null.")

    selected_ids = sorted(df["unique_id"].unique())[:series_limit]
    selected = df[df["unique_id"].isin(selected_ids)].copy()
    return selected.sort_values(["unique_id", "ds"], kind="mergesort").reset_index(drop=True)


def fixed_train_test_split(df: pd.DataFrame, horizon: int) -> DatasetSplit:
    """Use the last horizon observations of every series as test."""
    if horizon <= 0:
        raise ValueError("horizon must be positive.")

    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for unique_id, group in df.sort_values(["unique_id", "ds"]).groupby("unique_id", sort=True):
        if len(group) <= horizon:
            raise ValueError(
                f"Series {unique_id!r} has length {len(group)}, which is not greater than "
                f"horizon {horizon}."
            )
        test_parts.append(group.tail(horizon))
        train_parts.append(group.iloc[:-horizon])

    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    return DatasetSplit(train=train, test=test)
