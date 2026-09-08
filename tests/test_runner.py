import pytest

from timestransfer import runner as runner_module
from timestransfer.models import ModelRunResult
from timestransfer.runner import MODEL_RUNNERS, _dispatch_model


def test_registry_contains_expected_runners():
    expected = {
        "linear_regression",
        "tabpfn",
        "timesfm_2p5",
        "auto_arima",
        "seasonal_naive",
        "auto_ets",
        "auto_theta",
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
