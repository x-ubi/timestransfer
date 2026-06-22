from __future__ import annotations

import numpy as np
import pandas as pd

EPSILON = 1e-8


def compute_metric_rows(forecasts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, model), group in forecasts.groupby(["dataset", "model"], sort=True):
        forecast_horizon = int(group["horizon"].max())
        rows.append(
            _metric_row(
                dataset=dataset,
                model=model,
                scope="overall",
                horizon="",
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
                    forecast_horizon=forecast_horizon,
                    group=horizon_group,
                )
            )
    return pd.DataFrame(rows)


def failure_metric_row(dataset: str, model: str, error: str, forecast_horizon: int | str = "") -> dict[str, object]:
    return {
        "dataset": dataset,
        "model": model,
        "scope": "model_status",
        "horizon": "",
        "forecast_horizon": forecast_horizon,
        "mae": np.nan,
        "rmse": np.nan,
        "smape": np.nan,
        "n_obs": 0,
        "status": "failed",
        "error": error,
    }


def _metric_row(
    dataset: str,
    model: str,
    scope: str,
    horizon: int | str,
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
        "forecast_horizon": forecast_horizon,
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "smape": float(np.mean(200.0 * np.abs(error) / (np.abs(y_true) + np.abs(y_pred) + EPSILON))),
        "n_obs": int(len(group)),
        "status": "ok",
        "error": "",
    }
