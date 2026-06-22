import pandas as pd

from timestransfer.data import fixed_train_test_split
from timestransfer.features import build_lag_feature_bundle
from timestransfer.models import run_auto_arima, run_linear_regression


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
    assert list(result.forecasts.columns) == [
        "dataset",
        "model",
        "unique_id",
        "horizon",
        "ds",
        "y_true",
        "y_pred",
    ]
    assert len(result.forecasts) == 8


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
