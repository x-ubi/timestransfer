from __future__ import annotations

from collections.abc import Iterable
import re

import numpy as np
import pandas as pd

EPSILON = 1e-8

_QUANTILE_COLUMN_PATTERN = re.compile(r"^q(\d+(?:\.\d+)?)$")


def quantile_column(level: float) -> str:
    """Forecast column name for a quantile level, e.g. 0.1 -> "q10", 0.025 -> "q2.5"."""
    return f"q{round(float(level) * 100, 6):g}"


def quantile_columns(columns: Iterable[str]) -> list[tuple[float, str]]:
    """(level, column) pairs of the quantile columns in ``columns``, sorted by level."""
    pairs = []
    for column in columns:
        match = _QUANTILE_COLUMN_PATTERN.match(str(column))
        if match:
            pairs.append((float(match.group(1)) / 100.0, str(column)))
    return sorted(pairs)


def quantile_loss(y_true: np.ndarray, y_quantiles: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Pinball loss per row and level; ``y_quantiles`` has shape (n_rows, n_levels)."""
    residual = np.asarray(y_true, dtype=float)[:, None] - np.asarray(y_quantiles, dtype=float)
    levels = np.asarray(levels, dtype=float)[None, :]
    return np.maximum(levels * residual, (levels - 1.0) * residual)


def weighted_quantile_loss(
    y_true: np.ndarray, y_quantiles: np.ndarray, levels: np.ndarray
) -> float:
    """Mean over levels of 2 * sum(pinball loss) / sum(|y|), pooled over all rows.

    The WQL of Chronos (Ansari et al., 2024) and GIFT-Eval (GluonTS
    ``mean_weighted_sum_quantile_loss``).
    """
    denominator = float(np.sum(np.abs(y_true)))
    if denominator <= 0 or not np.isfinite(y_quantiles).all():
        return float("nan")
    losses = quantile_loss(y_true, y_quantiles, levels)
    return float(np.mean(2.0 * losses.sum(axis=0) / denominator))


def scaled_quantile_loss(
    y_true: np.ndarray, y_quantiles: np.ndarray, levels: np.ndarray, scales: np.ndarray
) -> float:
    """Mean over rows with a valid scale of mean-over-levels 2 * pinball loss / MASE scale.

    The probabilistic analogue of panel MASE (fev's SQL): a point forecast repeated at
    levels symmetric around 0.5 scores exactly its MASE.
    """
    valid = np.isfinite(scales) & (scales > 0)
    if not valid.any() or not np.isfinite(y_quantiles[valid]).all():
        return float("nan")
    losses = quantile_loss(y_true[valid], y_quantiles[valid], levels)
    return float(np.mean(2.0 * losses.mean(axis=1) / scales[valid]))


def interval_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Share of observations inside the closed interval [lower, upper]."""
    if not (np.isfinite(lower).all() and np.isfinite(upper).all()):
        return float("nan")
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


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
    """Score forecasts at overall, per-window, per-horizon-step and per-series scope.

    Forecasts may carry a ``window`` column (rolling origins; absent means one window).
    Overall, horizon and series rows pool all windows; ``window`` rows score each one.
    MASE scales are matched per window, since each window has its own training history.
    Quantile columns (``q10``, ``q90``, ...; see ``quantile_column``) add ``wql``, ``sql``
    and, when q10 and q90 exist, ``coverage_80``; these are NaN for models without
    quantiles and absent when the frame has no quantile columns.
    """
    scored = forecasts.copy()
    if "window" not in scored.columns:
        scored["window"] = 0
    if mase_scales is not None:
        scales = mase_scales.copy()
        if "window" not in scales.columns:
            scales["window"] = 0
        scored = scored.merge(
            scales.loc[:, ["dataset", "window", "unique_id", "mase_scale"]],
            on=["dataset", "window", "unique_id"],
            how="left",
            validate="many_to_one",
        )
    else:
        scored["mase_scale"] = np.nan

    quantiles = quantile_columns(scored.columns)
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
                quantiles=quantiles,
            )
        )
        for window, window_group in group.groupby("window", sort=True):
            rows.append(
                _metric_row(
                    dataset=dataset,
                    model=model,
                    scope="window",
                    horizon="",
                    unique_id="",
                    forecast_horizon=forecast_horizon,
                    group=window_group,
                    window=int(window),
                    quantiles=quantiles,
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
                    quantiles=quantiles,
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
                    quantiles=quantiles,
                )
            )
    return pd.DataFrame(rows)


def failure_metric_row(
    dataset: str,
    model: str,
    error: str,
    forecast_horizon: int | str = "",
    status: str = "failed",
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "model": model,
        "scope": "model_status",
        "horizon": "",
        "unique_id": "",
        "window": "",
        "forecast_horizon": forecast_horizon,
        "mae": np.nan,
        "rmse": np.nan,
        "smape": np.nan,
        "mase": np.nan,
        "n_obs": 0,
        "status": status,
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
    window: int | str = "",
    quantiles: list[tuple[float, str]] | None = None,
) -> dict[str, object]:
    y_true = group["y_true"].to_numpy(dtype=float)
    y_pred = group["y_pred"].to_numpy(dtype=float)
    scales = group["mase_scale"].to_numpy(dtype=float)
    error = y_pred - y_true
    row: dict[str, object] = {
        "dataset": dataset,
        "model": model,
        "scope": scope,
        "horizon": horizon,
        "unique_id": unique_id,
        "window": window,
        "forecast_horizon": forecast_horizon,
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "smape": float(
            np.mean(200.0 * np.abs(error) / (np.abs(y_true) + np.abs(y_pred) + EPSILON))
        ),
        "mase": _mase(error=error, scales=scales),
    }
    if quantiles:
        row.update(_probabilistic_metrics(group, y_true=y_true, scales=scales, quantiles=quantiles))
    row.update({"n_obs": int(len(group)), "status": "ok", "error": ""})
    return row


def _probabilistic_metrics(
    group: pd.DataFrame,
    *,
    y_true: np.ndarray,
    scales: np.ndarray,
    quantiles: list[tuple[float, str]],
) -> dict[str, float]:
    levels = np.array([level for level, _ in quantiles])
    y_quantiles = group.loc[:, [column for _, column in quantiles]].to_numpy(dtype=float)
    metrics = {
        "wql": weighted_quantile_loss(y_true, y_quantiles, levels),
        "sql": scaled_quantile_loss(y_true, y_quantiles, levels, scales),
    }
    columns = dict((round(level, 6), column) for level, column in quantiles)
    if 0.1 in columns and 0.9 in columns:
        metrics["coverage_80"] = interval_coverage(
            y_true,
            group[columns[0.1]].to_numpy(dtype=float),
            group[columns[0.9]].to_numpy(dtype=float),
        )
    return metrics


def _mase(*, error: np.ndarray, scales: np.ndarray) -> float:
    """Panel MASE: mean of |error| / seasonal-naive scale over rows with a valid scale."""
    valid = np.isfinite(scales) & (scales > 0)
    if not valid.any():
        return float("nan")
    return float(np.mean(np.abs(error[valid]) / scales[valid]))
