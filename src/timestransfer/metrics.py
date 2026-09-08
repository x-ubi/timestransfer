from __future__ import annotations

import numpy as np
import pandas as pd

EPSILON = 1e-8


def seasonal_naive_scales(train_df: pd.DataFrame, seasonality: int) -> pd.DataFrame:
    """Per-series in-sample MAE of the seasonal-naive forecast on the TRAIN series.

    Falls back to the lag-1 naive when a series is not longer than the season;
    the scale is NaN when a series has fewer than 2 observations.
    Returns columns [unique_id, mase_scale].
    """
    if seasonality <= 0:
        raise ValueError("seasonality must be positive.")

    rows: list[dict[str, object]] = []
    grouped = train_df.sort_values(["unique_id", "ds"], kind="mergesort").groupby(
        "unique_id", sort=True
    )
    for unique_id, group in grouped:
        values = group["y"].to_numpy(dtype=float)
        lag = seasonality if len(values) > seasonality else 1
        if len(values) < 2:
            scale = float("nan")
        else:
            scale = float(np.mean(np.abs(values[lag:] - values[:-lag])))
        rows.append({"unique_id": str(unique_id), "mase_scale": scale})
    return pd.DataFrame(rows, columns=["unique_id", "mase_scale"])


def compute_metric_rows(
    forecasts: pd.DataFrame,
    mase_scales: pd.DataFrame | None = None,
) -> pd.DataFrame:
    scored = forecasts.copy()
    if mase_scales is not None:
        scored = scored.merge(
            mase_scales.loc[:, ["dataset", "unique_id", "mase_scale"]],
            on=["dataset", "unique_id"],
            how="left",
        )
    else:
        scored["mase_scale"] = np.nan

    rows: list[dict[str, object]] = []
    for (dataset, model), group in scored.groupby(["dataset", "model"], sort=True):
        forecast_horizon = int(group["horizon"].max())
        rows.append(
            _metric_row(
                dataset=dataset,
                model=model,
                scope="overall",
                horizon="",
                unique_id="",
                forecast_horizon=forecast_horizon,
                group=group,
            )
        )
        for horizon, horizon_group in group.groupby("horizon", sort=True):
            rows.append(
                _metric_row(
                    dataset=dataset,
                    model=model,
                    scope="horizon",
                    horizon=int(horizon),
                    unique_id="",
                    forecast_horizon=forecast_horizon,
                    group=horizon_group,
                )
            )
        for unique_id, series_group in group.groupby("unique_id", sort=True):
            rows.append(
                _metric_row(
                    dataset=dataset,
                    model=model,
                    scope="series",
                    horizon="",
                    unique_id=str(unique_id),
                    forecast_horizon=forecast_horizon,
                    group=series_group,
                )
            )
    return pd.DataFrame(rows)


def failure_metric_row(dataset: str, model: str, error: str, forecast_horizon: int | str = "") -> dict[str, object]:
    return {
        "dataset": dataset,
        "model": model,
        "scope": "model_status",
        "horizon": "",
        "unique_id": "",
        "forecast_horizon": forecast_horizon,
        "mae": np.nan,
        "rmse": np.nan,
        "smape": np.nan,
        "mase": np.nan,
        "n_obs": 0,
        "status": "failed",
        "error": error,
    }


def _metric_row(
    dataset: str,
    model: str,
    scope: str,
    horizon: int | str,
    unique_id: str,
    forecast_horizon: int,
    group: pd.DataFrame,
) -> dict[str, object]:
    y_true = group["y_true"].to_numpy(dtype=float)
    y_pred = group["y_pred"].to_numpy(dtype=float)
    error = y_pred - y_true
    return {
        "dataset": dataset,
        "model": model,
        "scope": scope,
        "horizon": horizon,
        "unique_id": unique_id,
        "forecast_horizon": forecast_horizon,
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "smape": float(np.mean(200.0 * np.abs(error) / (np.abs(y_true) + np.abs(y_pred) + EPSILON))),
        "mase": _mase(error=error, scales=group["mase_scale"].to_numpy(dtype=float)),
        "n_obs": int(len(group)),
        "status": "ok",
        "error": "",
    }


def _mase(*, error: np.ndarray, scales: np.ndarray) -> float:
    """Panel MASE: mean of |error| / seasonal-naive scale over rows with a valid scale."""
    valid = np.isfinite(scales) & (scales > 0)
    if not valid.any():
        return float("nan")
    return float(np.mean(np.abs(error[valid]) / scales[valid]))
