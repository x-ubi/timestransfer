import pandas as pd

from timestransfer.forecast_plots import select_plot_series, write_forecast_plots


def _toy_forecasts(errors_by_series: dict[str, float], model: str = "ref") -> pd.DataFrame:
    rows = []
    for unique_id, error in errors_by_series.items():
        for horizon in (1, 2):
            rows.append(
                {
                    "dataset": "toy",
                    "model": model,
                    "unique_id": unique_id,
                    "horizon": horizon,
                    "ds": horizon,
                    "y_true": 10.0,
                    "y_pred": 10.0 + error,
                }
            )
    return pd.DataFrame(rows)


def test_select_plot_series_picks_first_k_and_best_median_worst():
    forecasts = _toy_forecasts({"s1": 5.0, "s2": 1.0, "s3": 3.0, "s4": 0.1, "s5": 9.0})

    selected = select_plot_series(forecasts, reference_model="ref", first_k=1)

    # First sorted id, then best (s4), median (s3), worst (s5) by sMAPE.
    assert selected == ["s1", "s4", "s3", "s5"]


def test_select_plot_series_falls_back_when_reference_model_missing():
    forecasts = _toy_forecasts({"s1": 1.0, "s2": 2.0})

    selected = select_plot_series(forecasts, reference_model="not_there", first_k=1)

    assert "s1" in selected


def test_write_forecast_plots_creates_figures(tmp_path):
    train = pd.DataFrame(
        {
            "unique_id": ["s1"] * 10,
            "ds": range(1, 11),
            "y": [float(v) for v in range(1, 11)],
        }
    )
    forecasts = pd.concat(
        [
            _toy_forecasts({"s1": 1.0}, model="model_a"),
            _toy_forecasts({"s1": -1.0}, model="model_b"),
        ],
        ignore_index=True,
    )
    forecasts["ds"] = forecasts["horizon"] + 10

    paths = write_forecast_plots(
        train,
        forecasts,
        dataset_name="toy",
        output_dir=tmp_path,
        history_points=5,
        reference_model="model_a",
    )

    assert len(paths) == 1
    assert paths[0].name == "toy_s1.png"
    assert paths[0].exists()


def test_write_forecast_plots_sanitizes_unsafe_series_ids(tmp_path):
    train = pd.DataFrame(
        {
            "unique_id": ["H2OC (mmol/mol)"] * 6,
            "ds": range(1, 7),
            "y": [float(v) for v in range(1, 7)],
        }
    )
    forecasts = _toy_forecasts({"H2OC (mmol/mol)": 1.0}, model="m")
    forecasts["ds"] = forecasts["horizon"] + 6

    paths = write_forecast_plots(
        train,
        forecasts,
        dataset_name="toy",
        output_dir=tmp_path,
        history_points=5,
        reference_model="m",
    )

    assert len(paths) == 1
    assert "/" not in paths[0].name
    assert paths[0].name == "toy_H2OC_mmol_mol.png"
    assert paths[0].exists()


def test_write_forecast_plots_empty_frame_returns_nothing(tmp_path):
    train = pd.DataFrame({"unique_id": [], "ds": [], "y": []})
    forecasts = pd.DataFrame(
        columns=["dataset", "model", "unique_id", "horizon", "ds", "y_true", "y_pred"]
    )

    paths = write_forecast_plots(
        train, forecasts, dataset_name="toy", output_dir=tmp_path
    )

    assert paths == []
