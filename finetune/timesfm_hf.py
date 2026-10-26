"""HF Transformers TimesFM 2.5 helpers that mirror the benchmark's `timesfm_2p5` runner.

The benchmark (timesfm package, all ForecastConfig flags on) returns the median channel as
the point forecast, repairs quantile crossing outward from the median, and clips forecasts
at zero when a series' whole training history is nonnegative. `forecast` reproduces that
with `TimesFm2_5ModelForPrediction`, so zero-shot and fine-tuned models are scored through
the same inference path.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

MODEL_ID = "google/timesfm-2.5-200m-transformers"
REVISION = "5a9806b9b291fad9233b5249d88263f1846304d3"
MEDIAN = 5  # output channel 0 is the mean head, channels 1..9 are q0.1..q0.9


def load_model(device: str = "cuda", dtype: torch.dtype = torch.float32):
    from transformers import TimesFm2_5ModelForPrediction

    model = TimesFm2_5ModelForPrediction.from_pretrained(MODEL_ID, revision=REVISION)
    return model.to(device, dtype=dtype).eval()


def fix_quantile_crossing(full: np.ndarray) -> np.ndarray:
    """timesfm's fix_quantile_crossing on (..., horizon, 10) output channels."""
    full = full.copy()
    for i in (4, 3, 2, 1):
        full[..., i] = np.minimum(full[..., i], full[..., i + 1])
    for i in (6, 7, 8, 9):
        full[..., i] = np.maximum(full[..., i], full[..., i - 1])
    return full


def forecast(
    model,
    histories: Sequence[np.ndarray],
    horizon: int,
    context: int,
    *,
    batch_size: int = 64,
) -> np.ndarray:
    """Point (median) and decile forecasts as an array (n_series, horizon, 10).

    Channel 5 is the point forecast. Each history is the full training series of one
    window; the last ``context`` points are fed to the model.
    """
    device = next(model.parameters()).device
    outputs = []
    for start in range(0, len(histories), batch_size):
        batch = histories[start : start + batch_size]
        contexts = [
            torch.from_numpy(np.asarray(h[-context:], dtype=np.float32)).to(device) for h in batch
        ]
        with torch.no_grad():
            out = model(past_values=contexts, forecast_context_len=context, truncate_negative=False)
        outputs.append(out.full_predictions[:, :horizon, :].double().cpu().numpy())
    full = fix_quantile_crossing(np.concatenate(outputs, axis=0))
    nonnegative = np.array([np.nanmin(h) >= 0 for h in histories])
    full[nonnegative] = np.maximum(full[nonnegative], 0.0)
    return full
