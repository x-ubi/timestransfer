import numpy as np
import pandas as pd

from timestransfer.metrics import compute_metric_rows, failure_metric_row, seasonal_naive_scales


def test_compute_metric_rows_overall_and_horizon():
    forecasts = pd.DataFrame(
        {
            "dataset": ["toy", "toy"],
            "model": ["linear_regression", "linear_regression"],
            "unique_id": ["a", "a"],
            "horizon": [1, 2],
            "ds": [10, 11],
            "y_true": [10.0, 20.0],
            "y_pred": [12.0, 18.0],
        }
    )

    rows = compute_metric_rows(forecasts)

    overall = rows[rows["scope"] == "overall"].iloc[0]
    assert overall["mae"] == 2.0
    assert overall["rmse"] == 2.0
    assert overall["n_obs"] == 2
    assert overall["forecast_horizon"] == 2
    assert np.isnan(overall["mase"])
    assert set(rows["scope"]) == {"overall", "horizon", "series"}


def test_compute_metric_rows_series_scope_and_mase():
    forecasts = pd.DataFrame(
        {
            "dataset": ["toy"] * 4,
            "model": ["linear_regression"] * 4,
            "unique_id": ["a", "a", "b", "b"],
            "horizon": [1, 2, 1, 2],
            "ds": [10, 11, 10, 11],
            "y_true": [10.0, 20.0, 5.0, 5.0],
            "y_pred": [12.0, 22.0, 6.0, 6.0],
        }
    )
    mase_scales = pd.DataFrame(
        {
            "dataset": ["toy", "toy"],
            "unique_id": ["a", "b"],
            "mase_scale": [1.0, 0.5],
        }
    )

    rows = compute_metric_rows(forecasts, mase_scales=mase_scales)

    series = rows[rows["scope"] == "series"].set_index("unique_id")
    assert len(series) == 2
    # Series a: |error| = 2 each, scale 1 -> MASE 2; series b: |error| = 1 each, scale 0.5 -> MASE 2.
    assert series.loc["a", "mase"] == 2.0
    assert series.loc["b", "mase"] == 2.0
    overall = rows[rows["scope"] == "overall"].iloc[0]
    assert overall["mase"] == 2.0
    assert overall["unique_id"] == ""


def test_compute_metric_rows_mase_ignores_invalid_scales():
    forecasts = pd.DataFrame(
        {
            "dataset": ["toy"] * 2,
            "model": ["m"] * 2,
            "unique_id": ["a", "b"],
            "horizon": [1, 1],
            "ds": [10, 10],
            "y_true": [10.0, 10.0],
            "y_pred": [11.0, 13.0],
        }
    )
    mase_scales = pd.DataFrame(
        {
            "dataset": ["toy", "toy"],
            "unique_id": ["a", "b"],
            "mase_scale": [0.0, 2.0],
        }
    )

    rows = compute_metric_rows(forecasts, mase_scales=mase_scales)

    overall = rows[rows["scope"] == "overall"].iloc[0]
    # Only series b has a valid (positive) scale: |13 - 10| / 2 = 1.5.
    assert overall["mase"] == 1.5
    series_a = rows[(rows["scope"] == "series") & (rows["unique_id"] == "a")].iloc[0]
    assert np.isnan(series_a["mase"])


def test_seasonal_naive_scales_seasonal_and_lag1_fallback():
    train = pd.DataFrame(
        {
            "unique_id": ["a"] * 6 + ["b"] * 3,
            "ds": list(range(6)) + list(range(3)),
            "y": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0] + [1.0, 3.0, 5.0],
        }
    )

    scales = seasonal_naive_scales(train, seasonality=2)

    scales = scales.set_index("unique_id")["mase_scale"]
    # Series a with season 2 repeats exactly -> scale 0.
    assert scales.loc["a"] == 0.0
    # Series b: values[2:] - values[:-2] = [5 - 1] -> scale 4.
    assert scales.loc["b"] == 4.0


def test_seasonal_naive_scales_short_series_falls_back_to_lag1():
    train = pd.DataFrame(
        {
            "unique_id": ["a"] * 3,
            "ds": range(3),
            "y": [1.0, 4.0, 9.0],
        }
    )

    scales = seasonal_naive_scales(train, seasonality=10)

    # Length 3 <= season 10 -> lag-1 naive: mean(|4-1|, |9-4|) = 4.
    assert scales.loc[0, "mase_scale"] == 4.0


def test_failure_metric_row_schema():
    row = failure_metric_row("m4_hourly", "tabpfn", "missing dependency")

    assert row["status"] == "failed"
    assert row["n_obs"] == 0
    assert row["model"] == "tabpfn"
    assert np.isnan(row["mase"])
    assert row["unique_id"] == ""
