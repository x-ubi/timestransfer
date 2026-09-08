import numpy as np
import pandas as pd

from timestransfer.series_features import FEATURE_COLUMNS, compute_series_features


def _frame(unique_id: str, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {"unique_id": [unique_id] * len(values), "ds": range(len(values)), "y": values}
    )


def test_seasonal_trending_series_has_high_strengths_and_seasonal_acf():
    t = np.arange(480, dtype=float)
    values = 0.05 * t + 3.0 * np.sin(2.0 * np.pi * t / 24.0)

    features = compute_series_features(_frame("a", values), seasonality=24)

    row = features.iloc[0]
    assert list(features.columns) == ["unique_id", *FEATURE_COLUMNS]
    assert row["length"] == 480
    assert row["trend_strength"] > 0.8
    assert row["seasonal_strength"] > 0.8
    assert row["acf_seasonal"] > 0.5


def test_white_noise_has_low_structure_and_high_entropy():
    rng = np.random.default_rng(42)
    values = rng.normal(size=480)

    features = compute_series_features(_frame("a", values), seasonality=24)

    row = features.iloc[0]
    assert row["seasonal_strength"] < 0.5
    assert abs(row["acf1"]) < 0.2
    assert row["spectral_entropy"] > 0.8


def test_short_series_gets_nan_guards():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    features = compute_series_features(_frame("a", values), seasonality=24)

    row = features.iloc[0]
    assert np.isnan(row["acf_seasonal"])
    assert np.isnan(row["trend_strength"])
    assert np.isnan(row["seasonal_strength"])
    assert not np.isnan(row["acf1"])


def test_max_points_caps_feature_window_but_not_length():
    t = np.arange(1000, dtype=float)
    values = np.sin(2.0 * np.pi * t / 24.0)

    features = compute_series_features(_frame("a", values), seasonality=24, max_points=100)

    assert features.iloc[0]["length"] == 1000
