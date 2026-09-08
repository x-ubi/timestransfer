import pandas as pd

from timestransfer.reporting import write_reporting_outputs


def test_write_reporting_outputs_creates_comparison_tables(tmp_path):
    metrics = pd.DataFrame(
        {
            "dataset": ["m4_hourly", "ett_h1_h48", "m4_hourly", "m4_hourly"],
            "model": ["linear_regression"] * 4,
            "scope": ["overall", "overall", "horizon", "series"],
            "horizon": ["", "", 1, ""],
            "unique_id": ["", "", "", "H1"],
            "forecast_horizon": [48, 48, 48, 48],
            "mae": [1.0, 2.0, 1.5, 1.0],
            "rmse": [1.0, 2.0, 1.5, 1.0],
            "smape": [10.0, 20.0, 15.0, 10.0],
            "mase": [0.9, 1.1, 1.0, 0.9],
            "n_obs": [10, 10, 2, 48],
            "status": ["ok", "ok", "ok", "ok"],
            "error": ["", "", "", ""],
        }
    )

    paths = write_reporting_outputs(metrics, tmp_path)

    assert paths["overall_by_dataset"].exists()
    assert paths["horizon_metrics"].exists()
    assert paths["series_metrics"].exists()
    assert paths["model_comparison_wide"].exists()
    assert paths["overall_smape_plot"].exists()
    assert paths["horizon_smape_plot"].exists()
    assert paths["overall_rmse_plot"].exists()
    assert paths["horizon_rmse_plot"].exists()
    assert paths["overall_mase_plot"].exists()
    assert paths["horizon_mase_plot"].exists()

    series = pd.read_csv(paths["series_metrics"])
    assert series["unique_id"].tolist() == ["H1"]

    wide = pd.read_csv(paths["model_comparison_wide"])
    assert {
        "dataset",
        "forecast_horizon",
        "rmse_linear_regression",
        "smape_linear_regression",
        "mase_linear_regression",
    }.issubset(wide.columns)
