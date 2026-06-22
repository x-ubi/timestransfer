import pandas as pd

from timestransfer.metrics import compute_metric_rows, failure_metric_row


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
    assert set(rows["scope"]) == {"overall", "horizon"}


def test_failure_metric_row_schema():
    row = failure_metric_row("m4_hourly", "tabpfn", "missing dependency")

    assert row["status"] == "failed"
    assert row["n_obs"] == 0
    assert row["model"] == "tabpfn"
