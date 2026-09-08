from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "length",
    "mean",
    "std",
    "cv",
    "acf1",
    "acf_seasonal",
    "trend_strength",
    "seasonal_strength",
    "spectral_entropy",
]


def compute_series_features(
    train_df: pd.DataFrame,
    *,
    seasonality: int,
    max_points: int | None = 10000,
) -> pd.DataFrame:
    """Per-series characteristics computed on the (tail-capped) TRAIN series.

    length is the full train length; the remaining features use at most the last
    max_points observations to keep STL tractable on very long series.
    trend_strength / seasonal_strength follow the FPP3 STL-based definitions
    (max(0, 1 - Var(remainder) / Var(component + remainder))); spectral_entropy is
    the normalized Shannon entropy of the periodogram density (1 = white noise).
    Features that need more data than a series has are NaN.
    """
    if seasonality <= 0:
        raise ValueError("seasonality must be positive.")

    rows: list[dict[str, object]] = []
    grouped = train_df.sort_values(["unique_id", "ds"], kind="mergesort").groupby(
        "unique_id", sort=True
    )
    for unique_id, group in grouped:
        full_values = group["y"].to_numpy(dtype=float)
        values = full_values[-max_points:] if max_points is not None else full_values
        features = _series_features(values, seasonality=seasonality)
        features["length"] = int(len(full_values))
        rows.append({"unique_id": str(unique_id), **features})
    frame = pd.DataFrame(rows)
    return frame.loc[:, ["unique_id", *FEATURE_COLUMNS]]


def _series_features(values: np.ndarray, *, seasonality: int) -> dict[str, float]:
    mean = float(np.mean(values))
    std = float(np.std(values))
    cv = float(std / abs(mean)) if abs(mean) > 1e-12 else float("nan")

    features: dict[str, float] = {
        "mean": mean,
        "std": std,
        "cv": cv,
        "acf1": float("nan"),
        "acf_seasonal": float("nan"),
        "trend_strength": float("nan"),
        "seasonal_strength": float("nan"),
        "spectral_entropy": float("nan"),
    }
    n = len(values)
    if std <= 0 or n < 3:
        return features

    from statsmodels.tsa.stattools import acf

    max_lag = seasonality if n > seasonality + 1 else 1
    acf_values = acf(values, nlags=max_lag, fft=True)
    features["acf1"] = float(acf_values[1])
    if max_lag >= seasonality:
        features["acf_seasonal"] = float(acf_values[seasonality])

    if seasonality >= 2 and n >= 2 * seasonality + 1:
        from statsmodels.tsa.seasonal import STL

        decomposition = STL(values, period=seasonality, robust=True).fit()
        remainder = np.asarray(decomposition.resid, dtype=float)
        trend = np.asarray(decomposition.trend, dtype=float)
        seasonal = np.asarray(decomposition.seasonal, dtype=float)
        var_remainder = float(np.var(remainder))
        var_deseasonalized = float(np.var(trend + remainder))
        var_detrended = float(np.var(seasonal + remainder))
        if var_deseasonalized > 0:
            features["trend_strength"] = max(0.0, 1.0 - var_remainder / var_deseasonalized)
        if var_detrended > 0:
            features["seasonal_strength"] = max(0.0, 1.0 - var_remainder / var_detrended)

    if n >= 8:
        from scipy.signal import periodogram

        _, psd = periodogram(values)
        psd = psd[psd > 0]
        if len(psd) >= 2:
            density = psd / psd.sum()
            features["spectral_entropy"] = float(
                -(density * np.log(density)).sum() / np.log(len(density))
            )

    return features
