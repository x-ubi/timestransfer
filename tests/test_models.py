import pandas as pd
import pytest

from timestransfer.data import fixed_train_test_split
from timestransfer.features import build_lag_feature_bundle
from timestransfer.models import (
    run_auto_arima,
    run_chronos2,
    run_linear_regression,
    run_prophet,
    run_seasonal_naive,
    run_tabpfn_ts,
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


def test_chronos2_records_import_failure(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("chronos"):
            raise ModuleNotFoundError("No module named 'chronos'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    df = pd.DataFrame({"unique_id": ["a"] * 6, "ds": range(1, 7), "y": range(6)})

    result = run_chronos2(
        df.iloc[:4],
        df.iloc[4:],
        "toy",
        horizon=2,
    )

    assert result.model == "chronos2"
    assert result.status == "failed"
    assert "import failed" in result.details["message"]


def test_tabpfn_ts_records_import_failure(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("tabpfn_time_series"):
            raise ModuleNotFoundError("No module named 'tabpfn_time_series'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    df = pd.DataFrame({"unique_id": ["a"] * 6, "ds": range(1, 7), "y": range(6)})

    result = run_tabpfn_ts(df.iloc[:4], df.iloc[4:], "toy", horizon=2, time_freq="h")

    assert result.model == "tabpfn_ts"
    assert result.status == "failed"
    assert "import failed" in result.details["message"]


def test_tabpfn_ts_maps_integer_ds_and_clips(monkeypatch, tmp_path):
    import tabpfn_time_series

    checkpoint = tmp_path / "fake.ckpt"
    checkpoint.write_bytes(b"weights")
    seen = {}

    class FakeAdapter:
        model_config = {"model_path": checkpoint}

        def predict(self):
            pass

    class FakePipeline:
        def __init__(self, **kwargs):
            seen["init"] = kwargs
            self.predictor = type("P", (), {"inference_routine": FakeAdapter().predict})()
            self.feature_transformer = type("F", (), {"feature_generators": []})()

        def predict_df(self, context_df, future_df=None, quantiles=None):
            seen["context"] = context_df
            seen["future"] = future_df
            out = future_df.copy()
            # Series "a" gets negative predictions; "b" (negative history) must stay unclipped.
            out["target"] = -1.0
            return out.set_index(["item_id", "timestamp"])

    monkeypatch.setattr(tabpfn_time_series, "TabPFNTSPipeline", FakePipeline)
    df = pd.DataFrame(
        {
            "unique_id": ["a"] * 6 + ["b"] * 6,
            "ds": list(range(1, 7)) * 2,
            "y": [0.0, 1, 2, 3, 4, 5, -1, 1, 2, 3, 4, 5],
        }
    )
    split = fixed_train_test_split(df, horizon=2)

    result = run_tabpfn_ts(
        split.train, split.test, "toy", horizon=2, time_freq="h", max_context_length=8
    )

    assert result.status == "ok", result.details
    assert list(result.forecasts.columns) == FORECAST_COLUMNS
    assert result.forecasts["ds"].tolist() == [5, 6, 5, 6]
    assert result.forecasts["y_pred"].tolist() == [0.0, 0.0, -1.0, -1.0]
    assert seen["init"]["max_context_length"] == 8
    assert str(seen["future"]["timestamp"].iloc[0]) == "2000-01-01 04:00:00"
    assert set(seen["context"].columns) == {"item_id", "timestamp", "target"}
    assert result.details["checkpoint"] == "fake.ckpt"
    assert len(result.details["checkpoint_sha256"]) == 64


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


DECILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DECILE_COLUMNS = ["q10", "q20", "q30", "q40", "q50", "q60", "q70", "q80", "q90"]


def _two_series_split():
    # Series "a" has a nonnegative history (clipped); "b" does not.
    df = pd.DataFrame(
        {
            "unique_id": ["a"] * 6 + ["b"] * 6,
            "ds": list(range(1, 7)) * 2,
            "y": [0.0, 1, 2, 3, 4, 5, -1, 1, 2, 3, 4, 5],
        }
    )
    return fixed_train_test_split(df, horizon=2)


def _patch_fake_timesfm(monkeypatch):
    """Fake TimesFM whose quantile head returns crossed, partly negative deciles."""
    import numpy as np
    import timesfm

    from timestransfer import models

    seen = {}

    class FakeModel:
        def compile(self, config):
            seen["config"] = config

        def forecast(self, horizon, inputs):
            n = len(inputs)
            quantiles = np.zeros((n, horizon, 10))
            # Channels 1..9 = q0.1..q0.9 with offsets -4..4; channel 5 is the point.
            quantiles[..., 1:] = np.arange(-4.0, 5.0)
            quantiles[..., 0] = 99.0  # mean channel, must be ignored
            quantiles[..., [2, 3]] = quantiles[..., [3, 2]]  # cross q0.2 and q0.3
            return quantiles[..., 5].copy(), quantiles

    class FakeTimesFM:
        @staticmethod
        def from_pretrained(model_id, revision=None):
            return FakeModel()

    monkeypatch.setattr(timesfm, "TimesFM_2p5_200M_torch", FakeTimesFM)
    monkeypatch.setattr(models, "_TIMESFM_CHECKPOINT_CACHE", {})
    return seen


def _run_fake_timesfm(quantile_levels=None):
    split = _two_series_split()
    return run_timesfm_2p5(
        split.train,
        split.test,
        "toy",
        model_id="fake",
        max_context=8,
        max_horizon=2,
        horizon=2,
        forecast_config_overrides={"per_core_batch_size": 64},
        quantile_levels=quantile_levels,
    )


def test_timesfm_without_quantiles_keeps_point_schema(monkeypatch):
    seen = _patch_fake_timesfm(monkeypatch)

    result = _run_fake_timesfm()

    assert result.status == "ok", result.details
    assert list(result.forecasts.columns) == FORECAST_COLUMNS
    assert "quantile_levels" not in result.details
    assert seen["config"].per_core_batch_size == 64


def test_timesfm_quantiles_are_monotone_clipped_and_keep_point(monkeypatch):
    _patch_fake_timesfm(monkeypatch)

    point_only = _run_fake_timesfm()
    result = _run_fake_timesfm(quantile_levels=DECILES)

    assert result.status == "ok", result.details
    assert list(result.forecasts.columns) == FORECAST_COLUMNS + DECILE_COLUMNS
    assert result.forecasts["y_pred"].tolist() == point_only.forecasts["y_pred"].tolist()
    assert result.details["quantile_levels"] == DECILES
    values = result.forecasts.set_index("unique_id")[DECILE_COLUMNS]
    # Series b (negative history): sorted, unclipped offsets -4..4.
    assert values.loc["b"].iloc[0].tolist() == [-4.0, -3, -2, -1, 0, 1, 2, 3, 4]
    # Series a (nonnegative history): clipped at zero like the median.
    assert values.loc["a"].iloc[0].tolist() == [0.0, 0, 0, 0, 0, 1, 2, 3, 4]


def test_timesfm_rejects_levels_outside_quantile_head(monkeypatch):
    _patch_fake_timesfm(monkeypatch)

    result = _run_fake_timesfm(quantile_levels=[0.05, 0.5])

    assert result.status == "failed"
    assert "unsupported level" in result.details["error"]


def _patch_fake_chronos(monkeypatch):
    """Fake Chronos-2 returning level-dependent, crossed quantiles."""
    import chronos
    import torch

    from timestransfer import models

    seen = {}

    class Chronos2Pipeline:
        model_context_length = 8192

        def predict_quantiles(self, contexts, prediction_length, quantile_levels, **kwargs):
            seen["quantile_levels"] = list(quantile_levels)
            levels = torch.tensor(quantile_levels, dtype=torch.float32)
            values = (levels - 0.5) * 10.0
            if len(quantile_levels) > 2:
                values[[0, 1]] = values[[1, 0]]  # cross the two lowest levels
            out = [values.expand(1, prediction_length, -1).clone() for _ in contexts]
            return out, None

    monkeypatch.setattr(
        chronos.BaseChronosPipeline,
        "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: Chronos2Pipeline()),
    )
    monkeypatch.setattr(models, "_CHRONOS_PIPELINE_CACHE", {})
    return seen


def test_chronos2_point_only_requests_median(monkeypatch):
    seen = _patch_fake_chronos(monkeypatch)
    split = _two_series_split()

    result = run_chronos2(split.train, split.test, "toy", horizon=2, device_map="cpu")

    assert result.status == "ok", result.details
    assert seen["quantile_levels"] == [0.5]
    assert list(result.forecasts.columns) == FORECAST_COLUMNS
    assert result.forecasts["y_pred"].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_chronos2_quantiles_add_median_and_sort(monkeypatch):
    seen = _patch_fake_chronos(monkeypatch)
    split = _two_series_split()

    result = run_chronos2(
        split.train, split.test, "toy", horizon=2, device_map="cpu", quantile_levels=[0.9, 0.1]
    )

    assert result.status == "ok", result.details
    assert seen["quantile_levels"] == [0.1, 0.5, 0.9]
    assert list(result.forecasts.columns) == FORECAST_COLUMNS + ["q10", "q90"]
    by_series = result.forecasts.set_index("unique_id")
    # Requested [0.1, 0.5, 0.9] -> raw [-4, 0, 4] with the lowest two crossed -> [0, -4, 4];
    # the median is read by level, then the selected quantiles are sorted.
    assert by_series.loc["b", "y_pred"].tolist() == [-4.0, -4.0]
    assert by_series.loc["b", ["q10", "q90"]].iloc[0].tolist() == pytest.approx([0.0, 4.0])
    assert by_series.loc["a", ["q10", "q90"]].iloc[0].tolist() == pytest.approx([0.0, 4.0])
    assert by_series.loc["a", "y_pred"].tolist() == [0.0, 0.0]


def test_tabpfn_ts_quantile_columns(monkeypatch, tmp_path):
    import tabpfn_time_series

    checkpoint = tmp_path / "fake.ckpt"
    checkpoint.write_bytes(b"weights")
    seen = {}

    class FakeAdapter:
        model_config = {"model_path": checkpoint}

        def predict(self):
            pass

    class FakePipeline:
        def __init__(self, **kwargs):
            self.predictor = type("P", (), {"inference_routine": FakeAdapter().predict})()
            self.feature_transformer = type("F", (), {"feature_generators": []})()

        def predict_df(self, context_df, future_df=None, quantiles=None):
            seen["quantiles"] = quantiles
            out = future_df.copy()
            out["target"] = 1.0
            for q in quantiles:
                out[q] = (q - 0.5) * 10.0
            out[0.9], out[0.1] = out[0.1], out[0.9]  # crossed
            return out.set_index(["item_id", "timestamp"])

    monkeypatch.setattr(tabpfn_time_series, "TabPFNTSPipeline", FakePipeline)
    split = _two_series_split()

    result = run_tabpfn_ts(
        split.train, split.test, "toy", horizon=2, time_freq="h", quantile_levels=[0.1, 0.9]
    )

    assert result.status == "ok", result.details
    assert seen["quantiles"] == [0.1, 0.5, 0.9]
    assert list(result.forecasts.columns) == FORECAST_COLUMNS + ["q10", "q90"]
    by_series = result.forecasts.set_index("unique_id")
    assert by_series.loc["b", ["q10", "q90"]].iloc[0].tolist() == [-4.0, 4.0]
    assert by_series.loc["a", ["q10", "q90"]].iloc[0].tolist() == [0.0, 4.0]
    assert result.forecasts["y_pred"].tolist() == [1.0] * 4
    assert result.details["quantile_levels"] == [0.1, 0.9]


def test_quantile_level_validation():
    from timestransfer.models import _validate_quantile_levels

    assert _validate_quantile_levels(None) == []
    assert _validate_quantile_levels([0.9, 0.1, 0.1]) == [0.1, 0.9]
    with pytest.raises(ValueError):
        _validate_quantile_levels([0.0, 0.5])
    with pytest.raises(ValueError):
        _validate_quantile_levels([0.5, 1.0])
