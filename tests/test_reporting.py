import pandas as pd

import numpy as np
import pytest

from timestransfer.reporting import (
    aggregate_relative_scores,
    relative_scores,
    window_spread,
    write_reporting_outputs,
)


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


def test_relative_scores_geometric_mean_over_available_tasks():
    overall = pd.DataFrame(
        {
            "dataset": ["t1", "t1", "t2", "t2", "t3"],
            "model": ["seasonal_naive", "m", "seasonal_naive", "m", "seasonal_naive"],
            "mase": [2.0, 1.0, 1.0, 4.0, 1.0],
            "mae": [2.0, 1.0, 1.0, 4.0, 1.0],
        }
    )

    relative = relative_scores(overall)
    aggregate = aggregate_relative_scores(relative).set_index("model")

    # m: ratios 0.5 and 4.0 -> geometric mean sqrt(2); t3 missing for m.
    assert aggregate.loc["m", "geomean_relative_mase"] == pytest.approx(np.sqrt(2.0))
    assert aggregate.loc["m", "n_tasks"] == 2
    assert aggregate.loc["seasonal_naive", "geomean_relative_mase"] == pytest.approx(1.0)
    assert aggregate.loc["seasonal_naive", "n_tasks"] == 3


def test_window_spread_summarises_window_rows():
    window_metrics = pd.DataFrame(
        {
            "dataset": ["t"] * 3,
            "model": ["m"] * 3,
            "window": [0, 1, 2],
            "mase": [1.0, 2.0, 3.0],
            "smape": [10.0, 20.0, 30.0],
        }
    )

    spread = window_spread(window_metrics).iloc[0]

    assert spread["n_windows"] == 3
    assert spread["mase_mean"] == pytest.approx(2.0)
    assert spread["mase_std"] == pytest.approx(1.0)
    assert spread["smape_max"] == pytest.approx(30.0)
