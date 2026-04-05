import sys
import unittest
from pathlib import Path

import torch

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_processing import PatchedInput
from timesfm import Config, TimesFM


class TimesFMModelTests(unittest.TestCase):
    def test_model_forward_returns_patch_forecasts(self):
        torch.manual_seed(0)
        config = Config(
            patch_length=2,
            num_layers=2,
            num_heads=2,
            hidden_size=8,
            intermediate_size=16,
            forecast_length=3,
            num_outputs=2,
            use_rotary_position_embeddings=False,
            attention_qk_norm="none",
            sdp_backend="math",
        )
        model = TimesFM(config)
        model.eval()

        inputs = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        mask = torch.tensor([[False, False, False, False]])
        patched = PatchedInput(inputs, mask, patch_length=config.patch_length, debug_mode=False)

        output = model(patched)

        self.assertEqual(tuple(output.shape), (1, 2, 3, 2))
        self.assertTrue(torch.isfinite(output).all())

    def test_model_rejects_hidden_size_not_divisible_by_num_heads(self):
        with self.assertRaises(ValueError):
            TimesFM(Config(patch_length=2, hidden_size=10, num_heads=3))

    def test_model_rejects_mismatched_input_patch_length(self):
        config = Config(
            patch_length=4,
            num_layers=1,
            num_heads=2,
            hidden_size=8,
            intermediate_size=16,
            use_rotary_position_embeddings=False,
            attention_qk_norm="none",
            sdp_backend="math",
        )
        model = TimesFM(config)
        patched = PatchedInput(
            torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            torch.tensor([[False, False, False, False]]),
            patch_length=2,
            debug_mode=False,
        )

        with self.assertRaises(ValueError):
            model(patched)


if __name__ == "__main__":
    unittest.main()
