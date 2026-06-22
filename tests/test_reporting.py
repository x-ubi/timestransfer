import pandas as pd

from timestransfer.reporting import write_reporting_outputs


def test_write_reporting_outputs_creates_comparison_tables(tmp_path):
    metrics = pd.DataFrame(
        {
            "dataset": ["m4_hourly", "ett_h1_h48", "m4_hourly"],
            "model": ["linear_regression", "linear_regression", "linear_regression"],
            "scope": ["overall", "overall", "horizon"],
            "horizon": ["", "", 1],
            "forecast_horizon": [48, 48, 48],
            "mae": [1.0, 2.0, 1.5],
            "rmse": [1.0, 2.0, 1.5],
            "smape": [10.0, 20.0, 15.0],
            "n_obs": [10, 10, 2],
            "status": ["ok", "ok", "ok"],
            "error": ["", "", ""],
        }
    )

    paths = write_reporting_outputs(metrics, tmp_path)

    assert paths["overall_by_dataset"].exists()
    assert paths["horizon_metrics"].exists()
    assert paths["model_comparison_wide"].exists()

    wide = pd.read_csv(paths["model_comparison_wide"])
    assert {"dataset", "forecast_horizon", "smape_linear_regression"}.issubset(wide.columns)
