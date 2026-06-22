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


def load_ett_h1(data_dir: str | Path, expected_n_series: int | None = 7) -> pd.DataFrame:
    """Download/cache ETTh1 and return canonical columns: unique_id, ds, y."""
    try:
        from datasetsforecast.long_horizon2 import LongHorizon2
    except ImportError as exc:  # pragma: no cover - covered by integration environment.
        raise RuntimeError("datasetsforecast is required to load ETTh1.") from exc

    df = LongHorizon2.load(directory=str(data_dir), group="ETTh1", normalize=False)
    df = df.loc[:, ["unique_id", "ds", "y"]].copy()
    df["unique_id"] = df["unique_id"].astype(str)
    df["ds"] = pd.to_datetime(df["ds"], errors="raise")
    df["y"] = pd.to_numeric(df["y"], errors="raise").astype(float)
    df = df.sort_values(["unique_id", "ds"], kind="mergesort").reset_index(drop=True)

    n_series = df["unique_id"].nunique()
    if expected_n_series is not None and n_series != expected_n_series:
        raise ValueError(f"Expected {expected_n_series} ETTh1 series, found {n_series}.")
    return df


def load_dataset(data_dir: str | Path, dataset_config: dict[str, Any]) -> pd.DataFrame:
    """Load a configured benchmark dataset into canonical long format."""
    loader = dataset_config.get("loader", dataset_config.get("name"))
    expected_n_series = dataset_config.get("expected_n_series")
    if loader == "m4_hourly":
        return load_m4_hourly(data_dir=data_dir, expected_n_series=expected_n_series)
    if loader == "ett_h1":
        return load_ett_h1(data_dir=data_dir, expected_n_series=expected_n_series)
    raise ValueError(f"Unknown dataset loader: {loader!r}")


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
