import numpy as np
import pytest
import torch

from timesfm_hf import fix_quantile_crossing
from train_lora import pinball_loss, sample_batch, window_roles


def test_window_roles_hold_out_the_three_oldest_windows():
    assert window_roles(20) == (list(range(17)), [17, 18, 19])
    assert window_roles(10) == (list(range(7)), [7, 8, 9])
    with pytest.raises(ValueError):
        window_roles(3)


def test_samples_stay_inside_their_history_and_respect_lengths():
    rng = np.random.default_rng(0)
    histories = [np.arange(200, dtype=np.float32), np.arange(1000, 1100, dtype=np.float32)]
    contexts, targets = sample_batch(histories, 500, context=64, horizon=8, min_context=16, rng=rng)
    assert targets.shape == (500, 8)
    for context, target in zip(contexts, targets, strict=True):
        assert 16 <= len(context) <= 64
        # Consecutive values of one history: target continues the context, nothing beyond.
        assert target[0] == context[-1] + 1
        assert np.all(np.diff(target) == 1)
        source = histories[0] if context[0] < 1000 else histories[1]
        assert target[-1] <= source[-1]


def test_sample_batch_skips_too_short_histories():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        sample_batch([np.zeros(10, dtype=np.float32)], 4, 64, 8, 16, rng)


def test_pinball_loss_pairs_channels_with_their_levels():
    torch.manual_seed(0)
    context = [torch.randn(50)]
    target = torch.zeros(1, 4)
    # Exact quantiles of a degenerate target (all zeros) give zero loss.
    exact = torch.zeros(1, 4, 10)
    assert float(pinball_loss(exact, target, context)) == pytest.approx(0.0)
    # Raising only channel 1 (q0.1) above the target costs (1 - 0.1) per unit;
    # raising only channel 9 (q0.9) costs (1 - 0.9) per unit.
    low, high = exact.clone(), exact.clone()
    low[..., 1] = 1.0
    high[..., 9] = 1.0
    ratio = float(pinball_loss(low, target, context)) / float(pinball_loss(high, target, context))
    assert ratio == pytest.approx(0.9 / 0.1)
    # The mean head (channel 0) is ignored.
    mean_only = exact.clone()
    mean_only[..., 0] = 5.0
    assert float(pinball_loss(mean_only, target, context)) == pytest.approx(0.0)


def test_fix_quantile_crossing_matches_timesfm_rule():
    full = np.array([[[9.0, 3.0, 1.0, 2.0, 5.0, 4.0, 3.5, 6.0, 5.0, 7.0]]])
    fixed = fix_quantile_crossing(full)[0, 0]
    assert fixed[0] == 9.0  # mean head untouched
    assert np.all(np.diff(fixed[1:]) >= 0)
    assert list(fixed[1:6]) == [1.0, 1.0, 2.0, 4.0, 4.0]
    assert list(fixed[5:]) == [4.0, 4.0, 6.0, 6.0, 7.0]
