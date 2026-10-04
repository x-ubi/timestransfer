import pytest
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
    assert set(rows["scope"]) == {"overall", "window", "horizon", "series"}


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


def test_compute_metric_rows_matches_mase_scale_per_window():
    forecasts = pd.DataFrame(
        {
            "dataset": ["d"] * 4,
            "model": ["m"] * 4,
            "unique_id": ["a"] * 4,
            "horizon": [1, 2, 1, 2],
            "window": [0, 0, 1, 1],
            "ds": [5, 6, 3, 4],
            "y_true": [1.0, 1.0, 1.0, 1.0],
            "y_pred": [2.0, 2.0, 3.0, 3.0],
        }
    )
    scales = pd.DataFrame(
        {"dataset": ["d", "d"], "window": [0, 1], "unique_id": ["a", "a"], "mase_scale": [1.0, 4.0]}
    )

    rows = compute_metric_rows(forecasts, mase_scales=scales)
    windows = rows[rows["scope"] == "window"].set_index("window")["mase"]
    overall = rows[rows["scope"] == "overall"].iloc[0]

    assert windows.loc[0] == pytest.approx(1.0)
    assert windows.loc[1] == pytest.approx(0.5)
    assert overall["mase"] == pytest.approx(0.75)
    assert overall["n_obs"] == 4


def test_failure_metric_row_records_skipped_status():
    row = failure_metric_row("ett_m1_h48", "auto_arima", "season too long", status="skipped")

    assert row["status"] == "skipped"
    assert row["scope"] == "model_status"


def test_failure_metric_row_schema():
    row = failure_metric_row("m4_hourly", "tabpfn_ts", "missing dependency")

    assert row["status"] == "failed"
    assert row["n_obs"] == 0
    assert row["model"] == "tabpfn_ts"
    assert np.isnan(row["mase"])
    assert row["unique_id"] == ""


def test_quantile_column_names_round_trip():
    from timestransfer.metrics import quantile_column, quantile_columns

    names = [quantile_column(level) for level in (0.1, 0.5, 0.9, 0.025)]
    assert names == ["q10", "q50", "q90", "q2.5"]
    assert quantile_columns(["y_pred", "q90", "q10", "q2.5", "quality"]) == [
        (0.025, "q2.5"),
        (0.1, "q10"),
        (0.9, "q90"),
    ]


def test_weighted_quantile_loss_hand_computed():
    from timestransfer.metrics import weighted_quantile_loss

    y = np.array([10.0, 20.0])
    # q0.1 forecasts [12, 18]: residuals [-2, 2] -> pinball [0.9*2, 0.1*2] = [1.8, 0.2].
    # q0.9 forecasts [8, 25]: residuals [2, -5] -> pinball [0.9*2, 0.1*5] = [1.8, 0.5].
    quantiles = np.array([[12.0, 8.0], [18.0, 25.0]])
    expected = np.mean([2 * 2.0 / 30.0, 2 * 2.3 / 30.0])
    assert weighted_quantile_loss(y, quantiles, np.array([0.1, 0.9])) == pytest.approx(expected)


def test_scaled_quantile_loss_of_point_forecast_equals_mase():
    from timestransfer.metrics import scaled_quantile_loss

    rng = np.random.default_rng(0)
    y = rng.normal(size=50)
    y_pred = rng.normal(size=50)
    scales = rng.uniform(0.5, 2.0, size=50)
    levels = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    degenerate = np.repeat(y_pred[:, None], len(levels), axis=1)

    sql = scaled_quantile_loss(y, degenerate, levels, scales)

    assert sql == pytest.approx(np.mean(np.abs(y - y_pred) / scales))


def test_compute_metric_rows_probabilistic_columns():
    forecasts = pd.DataFrame(
        {
            "dataset": ["toy"] * 4,
            "model": ["fm", "fm", "point_only", "point_only"],
            "unique_id": ["a", "a", "a", "a"],
            "horizon": [1, 2, 1, 2],
            "ds": [10, 11, 10, 11],
            "y_true": [10.0, 20.0, 10.0, 20.0],
            "y_pred": [10.0, 20.0, 10.0, 20.0],
            "q10": [9.0, 21.0, np.nan, np.nan],
            "q50": [10.0, 20.0, np.nan, np.nan],
            "q90": [11.0, 22.0, np.nan, np.nan],
        }
    )
    mase_scales = pd.DataFrame({"dataset": ["toy"], "unique_id": ["a"], "mase_scale": [2.0]})

    rows = compute_metric_rows(forecasts, mase_scales=mase_scales)

    overall = rows[rows["scope"] == "overall"].set_index("model")
    # Row 1 is inside [9, 11]; row 2 (20) is below q10 = 21.
    assert overall.loc["fm", "coverage_80"] == 0.5
    # Row 1 pinball: q10 0.1*1, q90 0.1*1; row 2: q10 0.9*1, q90 0.1*2. q50 is exact.
    pinball_sums = np.array([0.1 + 0.9, 0.0, 0.1 + 0.2])
    assert overall.loc["fm", "wql"] == pytest.approx(np.mean(2 * pinball_sums / 30.0))
    row_sql = [2 * (0.1 + 0.0 + 0.1) / 3 / 2.0, 2 * (0.9 + 0.0 + 0.2) / 3 / 2.0]
    assert overall.loc["fm", "sql"] == pytest.approx(np.mean(row_sql))
    assert np.isnan(overall.loc["point_only", "wql"])
    assert np.isnan(overall.loc["point_only", "coverage_80"])
    assert overall.loc["point_only", "mase"] == 0.0
    assert set(rows["scope"]) == {"overall", "window", "horizon", "series"}


def test_compute_metric_rows_without_quantiles_keeps_schema():
    forecasts = pd.DataFrame(
        {
            "dataset": ["toy"],
            "model": ["m"],
            "unique_id": ["a"],
            "horizon": [1],
            "ds": [1],
            "y_true": [1.0],
            "y_pred": [2.0],
        }
    )

    rows = compute_metric_rows(forecasts)

    assert list(rows.columns) == [
        "dataset",
        "model",
        "scope",
        "horizon",
        "unique_id",
        "window",
        "forecast_horizon",
        "mae",
        "rmse",
        "smape",
        "mase",
        "n_obs",
        "status",
        "error",
    ]
