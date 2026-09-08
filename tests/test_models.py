import pandas as pd

from timestransfer.data import fixed_train_test_split
from timestransfer.features import build_lag_feature_bundle
from timestransfer.models import (
    run_auto_arima,
    run_chronos_bolt,
    run_linear_regression,
    run_prophet,
    run_seasonal_naive,
    run_timesfm_2p5,
)

FORECAST_COLUMNS = [
    "dataset",
    "model",
    "unique_id",
    "horizon",
    "ds",
    "y_true",
    "y_pred",
]


def test_linear_regression_forecast_schema():
    df = pd.DataFrame(
        {
            "unique_id": ["a"] * 20 + ["b"] * 20,
            "ds": list(range(1, 21)) * 2,
            "y": list(range(1, 21)) + list(range(51, 71)),
        }
    )
    split = fixed_train_test_split(df, horizon=4)
    bundle = build_lag_feature_bundle(
        split.train,
        split.test,
        horizon=4,
        seasonality=4,
        lags=[1, 2, 4],
        train_row_cap=20,
        seed=42,
    )

    result = run_linear_regression(bundle, "toy")

    assert result.status == "ok"
    assert result.forecasts is not None
    assert list(result.forecasts.columns) == FORECAST_COLUMNS
    assert len(result.forecasts) == 8


def test_seasonal_naive_forecasts_repeat_last_season():
    df = pd.DataFrame(
        {
            "unique_id": ["a"] * 8 + ["b"] * 8,
            "ds": list(range(1, 9)) * 2,
            "y": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 9.0, 9.0] + [5.0] * 8,
        }
    )
    split = fixed_train_test_split(df, horizon=2)

    result = run_seasonal_naive(
        split.train,
        split.test,
        "toy",
        horizon=2,
        seasonality=2,
        freq=1,
        n_jobs=1,
    )

    assert result.status == "ok"
    assert result.forecasts is not None
    assert list(result.forecasts.columns) == FORECAST_COLUMNS
    assert len(result.forecasts) == 4
    predictions = result.forecasts.set_index(["unique_id", "horizon"])["y_pred"]
    # Train for series a ends with the season [1, 2]; series b is constant 5.
    assert predictions.loc[("a", 1)] == 1.0
    assert predictions.loc[("a", 2)] == 2.0
    assert predictions.loc[("b", 1)] == 5.0
    assert predictions.loc[("b", 2)] == 5.0
    assert (result.forecasts["model"] == "seasonal_naive").all()


def test_seasonal_naive_records_import_failure(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("statsforecast"):
            raise ModuleNotFoundError("No module named 'statsforecast'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    df = pd.DataFrame({"unique_id": ["a"] * 6, "ds": range(1, 7), "y": range(6)})

    result = run_seasonal_naive(
        df.iloc[:4],
        df.iloc[4:],
        "toy",
        horizon=2,
        seasonality=1,
        freq=1,
        n_jobs=1,
    )

    assert result.model == "seasonal_naive"
    assert result.status == "failed"
    assert "import failed" in result.details["message"]


def test_prophet_records_import_failure(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("prophet"):
            raise ModuleNotFoundError("No module named 'prophet'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    df = pd.DataFrame({"unique_id": ["a"] * 6, "ds": range(1, 7), "y": range(6)})

    result = run_prophet(
        df.iloc[:4],
        df.iloc[4:],
        "toy",
        horizon=2,
        time_freq="h",
        n_jobs=1,
    )

    assert result.model == "prophet"
    assert result.status == "failed"
    assert "import failed" in result.details["message"]


def test_chronos_bolt_records_import_failure(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("chronos"):
            raise ModuleNotFoundError("No module named 'chronos'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    df = pd.DataFrame({"unique_id": ["a"] * 6, "ds": range(1, 7), "y": range(6)})

    result = run_chronos_bolt(
        df.iloc[:4],
        df.iloc[4:],
        "toy",
        horizon=2,
    )

    assert result.model == "chronos_bolt"
    assert result.status == "failed"
    assert "import failed" in result.details["message"]


def test_timesfm_variant_name_flows_into_result(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "timesfm":
            raise ModuleNotFoundError("No module named 'timesfm'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    df = pd.DataFrame({"unique_id": ["a"] * 6, "ds": range(1, 7), "y": range(6)})

    result = run_timesfm_2p5(
        df.iloc[:4],
        df.iloc[4:],
        "toy",
        model_name="timesfm_2p5_ctx512",
        model_id="google/timesfm-2.5-200m-pytorch",
        max_context=512,
        max_horizon=2,
        horizon=2,
    )

    assert result.model == "timesfm_2p5_ctx512"
    assert result.status == "failed"


def test_auto_arima_records_import_failure(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("statsforecast"):
            raise ModuleNotFoundError("No module named 'statsforecast'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    df = pd.DataFrame({"unique_id": ["a"] * 6, "ds": range(1, 7), "y": range(6)})

    result = run_auto_arima(
        df.iloc[:4],
        df.iloc[4:],
        "toy",
        horizon=2,
        seasonality=1,
        freq=1,
        n_jobs=1,
        model_kwargs={"approximation": True, "nmodels": 2},
    )

    assert result.model == "auto_arima"
    assert result.status == "failed"
    assert "import failed" in result.details["message"]
