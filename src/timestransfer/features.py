from __future__ import annotations

from dataclasses import dataclass
from math import cos, log, pi, sin

import numpy as np
import pandas as pd

EPSILON = 1e-8


@dataclass(frozen=True)
class FeatureBundle:
    X_train: pd.DataFrame
    y_train: pd.Series
    train_index: pd.DataFrame
    X_test: pd.DataFrame
    test_index: pd.DataFrame
    feature_columns: list[str]
    series_stats: pd.DataFrame


@dataclass(frozen=True)
class _SeriesArrays:
    unique_id: str
    values: np.ndarray
    ds: np.ndarray
    time_idx: np.ndarray
    scale: float
    nonnegative: bool


def build_lag_feature_bundle(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    horizon: int,
    seasonality: int,
    lags: list[int],
    train_row_cap: int | None,
    seed: int,
) -> FeatureBundle:
    """Build pooled supervised rows from train history and forecast-origin test rows."""
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if seasonality <= 0:
        raise ValueError("seasonality must be positive.")
    if not lags or any(lag <= 0 for lag in lags):
        raise ValueError("lags must contain positive integers.")

    lags = sorted(set(int(lag) for lag in lags))
    feature_columns = _feature_columns(lags)
    series = _series_arrays(train_df)

    train_rows: list[dict[str, float]] = []
    train_index_rows: list[dict[str, object]] = []
    targets: list[float] = []
    rng = np.random.default_rng(seed)
    counts = {item.unique_id: _candidate_count(len(item.values), horizon) for item in series}
    allocations = _allocate_rows(counts, train_row_cap)

    for item in series:
        count = counts[item.unique_id]
        allocation = allocations[item.unique_id]
        if count == 0 or allocation == 0:
            continue
        local_indices = np.arange(count)
        if allocation < count:
            local_indices = rng.choice(local_indices, size=allocation, replace=False)
        local_indices = np.sort(local_indices)
        cutoff_offsets = _cutoff_offsets(len(item.values), horizon)
        for local_index in local_indices:
            cutoff, h = _decode_local_index(int(local_index), cutoff_offsets)
            target_pos = cutoff + h
            train_rows.append(
                _make_feature_row(
                    values=item.values,
                    cutoff=cutoff,
                    target_time_idx=int(item.time_idx[target_pos]),
                    horizon_step=h,
                    max_horizon=horizon,
                    seasonality=seasonality,
                    lags=lags,
                    scale=item.scale,
                )
            )
            targets.append(float(item.values[target_pos] / item.scale))
            train_index_rows.append(
                {
                    "unique_id": item.unique_id,
                    "cutoff_ds": item.ds[cutoff],
                    "target_ds": item.ds[target_pos],
                    "cutoff_time_idx": int(item.time_idx[cutoff]),
                    "target_time_idx": int(item.time_idx[target_pos]),
                    "horizon": h,
                    "target_y": float(item.values[target_pos]),
                }
            )

    if not train_rows:
        raise ValueError("No supervised training rows were generated.")

    X_train = pd.DataFrame(train_rows, columns=feature_columns)
    y_train = pd.Series(targets, name="y", dtype=float)
    train_index = pd.DataFrame(train_index_rows)

    X_test_rows: list[dict[str, float]] = []
    test_index_rows: list[dict[str, object]] = []
    test_groups = {
        unique_id: group.sort_values("ds").reset_index(drop=True)
        for unique_id, group in test_df.groupby("unique_id", sort=True)
    }

    for item in series:
        test_group = test_groups.get(item.unique_id)
        if test_group is None:
            raise ValueError(f"Missing test rows for series {item.unique_id!r}.")
        if len(test_group) != horizon:
            raise ValueError(
                f"Expected {horizon} test rows for {item.unique_id!r}, found {len(test_group)}."
            )

        cutoff = len(item.values) - 1
        for h in range(1, horizon + 1):
            test_row = test_group.iloc[h - 1]
            X_test_rows.append(
                _make_feature_row(
                    values=item.values,
                    cutoff=cutoff,
                    target_time_idx=int(item.time_idx[-1] + h),
                    horizon_step=h,
                    max_horizon=horizon,
                    seasonality=seasonality,
                    lags=lags,
                    scale=item.scale,
                )
            )
            test_index_rows.append(
                {
                    "unique_id": item.unique_id,
                    "horizon": h,
                    "ds": test_row["ds"],
                    "y_true": float(test_row["y"]),
                    "scale": item.scale,
                    "train_nonnegative": item.nonnegative,
                }
            )

    X_test = pd.DataFrame(X_test_rows, columns=feature_columns)
    test_index = pd.DataFrame(test_index_rows)
    series_stats = pd.DataFrame(
        [
            {
                "unique_id": item.unique_id,
                "scale": item.scale,
                "train_nonnegative": item.nonnegative,
                "train_length": len(item.values),
            }
            for item in series
        ]
    )

    return FeatureBundle(
        X_train=X_train,
        y_train=y_train,
        train_index=train_index,
        X_test=X_test,
        test_index=test_index,
        feature_columns=feature_columns,
        series_stats=series_stats,
    )


def _series_arrays(train_df: pd.DataFrame) -> list[_SeriesArrays]:
    arrays: list[_SeriesArrays] = []
    for unique_id, group in train_df.sort_values(["unique_id", "ds"]).groupby("unique_id", sort=True):
        values = group["y"].to_numpy(dtype=float)
        ds = group["ds"].to_numpy()
        time_idx = np.arange(1, len(group) + 1, dtype=int)
        scale = float(np.mean(np.abs(values)) + EPSILON)
        arrays.append(
            _SeriesArrays(
                unique_id=str(unique_id),
                values=values,
                ds=ds,
                time_idx=time_idx,
                scale=scale,
                nonnegative=bool(np.nanmin(values) >= 0),
            )
        )
    return arrays


def _feature_columns(lags: list[int]) -> list[str]:
    return [
        "horizon",
        "horizon_frac",
        "season_sin",
        "season_cos",
        "log_scale",
        *[f"lag_{lag}" for lag in lags],
    ]


def _make_feature_row(
    *,
    values: np.ndarray,
    cutoff: int,
    target_time_idx: int,
    horizon_step: int,
    max_horizon: int,
    seasonality: int,
    lags: list[int],
    scale: float,
) -> dict[str, float]:
    phase = ((target_time_idx - 1) % seasonality) / seasonality
    row = {
        "horizon": float(horizon_step),
        "horizon_frac": float(horizon_step / max_horizon),
        "season_sin": float(sin(2.0 * pi * phase)),
        "season_cos": float(cos(2.0 * pi * phase)),
        "log_scale": float(log(scale)),
    }
    for lag in lags:
        pos = cutoff - lag + 1
        row[f"lag_{lag}"] = float(values[pos] / scale) if pos >= 0 else np.nan
    return row


def _candidate_count(n_values: int, horizon: int) -> int:
    if n_values <= 1:
        return 0
    return int(sum(min(horizon, n_values - 1 - cutoff) for cutoff in range(n_values - 1)))


def _cutoff_offsets(n_values: int, horizon: int) -> np.ndarray:
    per_cutoff = [min(horizon, n_values - 1 - cutoff) for cutoff in range(n_values - 1)]
    return np.cumsum(per_cutoff)


def _decode_local_index(local_index: int, cutoff_offsets: np.ndarray) -> tuple[int, int]:
    cutoff = int(np.searchsorted(cutoff_offsets, local_index, side="right"))
    previous = int(cutoff_offsets[cutoff - 1]) if cutoff > 0 else 0
    horizon_step = local_index - previous + 1
    return cutoff, int(horizon_step)


def _allocate_rows(counts: dict[str, int], train_row_cap: int | None) -> dict[str, int]:
    total = sum(counts.values())
    if train_row_cap is None or total <= train_row_cap:
        return counts.copy()
    if train_row_cap <= 0:
        raise ValueError("train_row_cap must be positive or null.")

    keys = [key for key, count in counts.items() if count > 0]
    raw = {key: counts[key] / total * train_row_cap for key in keys}
    allocations = {key: min(counts[key], int(np.floor(raw[key]))) for key in keys}

    if train_row_cap >= len(keys):
        for key in keys:
            if allocations[key] == 0:
                allocations[key] = 1

    while sum(allocations.values()) > train_row_cap:
        key = max(allocations, key=lambda item: allocations[item])
        allocations[key] -= 1

    remainders = sorted(keys, key=lambda key: raw[key] - np.floor(raw[key]), reverse=True)
    idx = 0
    while sum(allocations.values()) < train_row_cap and remainders:
        key = remainders[idx % len(remainders)]
        if allocations[key] < counts[key]:
            allocations[key] += 1
        idx += 1
        if idx > len(remainders) * 2 and all(allocations[key] >= counts[key] for key in remainders):
            break

    return {key: allocations.get(key, 0) for key in counts}
