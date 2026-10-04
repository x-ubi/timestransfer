import pytest

from timestransfer import runner as runner_module
from timestransfer.models import ModelRunResult
from timestransfer.runner import MODEL_RUNNERS, _dispatch_model


def test_registry_contains_expected_runners():
    expected = {
        "linear_regression",
        "tabpfn_ts",
        "timesfm_2p5",
        "auto_arima",
        "seasonal_naive",
        "auto_ets",
        "auto_theta",
        "chronos2",
    }
    assert expected <= set(MODEL_RUNNERS)


def test_dispatch_honors_runner_field_and_passes_entry_name(monkeypatch):
    seen = {}

    def fake_entry(model_name, model_cfg, ctx):
        seen["model_name"] = model_name
        seen["max_context"] = model_cfg.get("max_context")
        return ModelRunResult(model=model_name, status="ok", forecasts=None, details={})

    monkeypatch.setitem(runner_module.MODEL_RUNNERS, "fake_runner", fake_entry)

    result = _dispatch_model(
        "timesfm_2p5_ctx4096", {"runner": "fake_runner", "max_context": 4096}, ctx=None
    )

    assert result.model == "timesfm_2p5_ctx4096"
    assert seen == {"model_name": "timesfm_2p5_ctx4096", "max_context": 4096}


def test_dispatch_defaults_runner_to_entry_name(monkeypatch):
    def fake_entry(model_name, model_cfg, ctx):
        return ModelRunResult(model=model_name, status="ok", forecasts=None, details={})

    monkeypatch.setitem(runner_module.MODEL_RUNNERS, "some_model", fake_entry)

    result = _dispatch_model("some_model", {"enabled": True}, ctx=None)

    assert result.model == "some_model"


def test_dispatch_unknown_runner_raises():
    with pytest.raises(ValueError, match="Unknown model runner"):
        _dispatch_model("mystery", {}, ctx=None)


def test_statsforecast_entry_skips_season_length_above_limit():
    ctx = type("Ctx", (), {"seasonality": 96})()

    result = MODEL_RUNNERS["auto_arima"]("auto_arima", {"max_season_length": 52}, ctx)

    assert result.status == "skipped"
    assert "96" in result.details["message"]


def test_foundation_model_entries_pass_quantiles_and_batch_size(monkeypatch):
    seen = {}

    def fake_runner(name):
        def _run(*args, **kwargs):
            seen[name] = kwargs
            return ModelRunResult(
                model=kwargs["model_name"], status="ok", forecasts=None, details={}
            )

        return _run

    for name in ("run_timesfm_2p5", "run_chronos2", "run_tabpfn_ts"):
        monkeypatch.setattr(runner_module, name, fake_runner(name))
    split = type("Split", (), {"train": None, "test": None})()
    ctx = type(
        "Ctx",
        (),
        {"split": split, "dataset_name": "toy", "horizon": 2, "freq": "h", "dataset_config": {}},
    )()
    cfg = {"quantile_levels": [0.1, 0.9], "per_core_batch_size": 64, "normalize_inputs": True}

    MODEL_RUNNERS["timesfm_2p5"]("timesfm_2p5", cfg, ctx)
    MODEL_RUNNERS["chronos2"]("chronos2", {"quantile_levels": [0.1, 0.9]}, ctx)
    MODEL_RUNNERS["tabpfn_ts"]("tabpfn_ts", {}, ctx)

    assert seen["run_timesfm_2p5"]["quantile_levels"] == [0.1, 0.9]
    assert seen["run_timesfm_2p5"]["forecast_config_overrides"] == {
        "normalize_inputs": True,
        "per_core_batch_size": 64,
    }
    assert seen["run_chronos2"]["quantile_levels"] == [0.1, 0.9]
    assert seen["run_tabpfn_ts"]["quantile_levels"] is None
