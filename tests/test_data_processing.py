import math
import sys
import unittest
from pathlib import Path

import torch

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_processing import DEBUG_PADDING_VALUE, DataStats, PatchedInput, PatchedOutput
from timesfm import Config


class PatchedInputTests(unittest.TestCase):
    def test_patched_input_tracks_stats_and_masks_padding(self):
        inputs = torch.tensor([[DEBUG_PADDING_VALUE, 1.0, 2.0, 3.0, 4.0, 5.0]])
        mask = torch.tensor([[True, False, False, False, False, False]])

        patched = PatchedInput(inputs, mask, patch_length=3, debug_mode=False)

        expected_std = math.sqrt(2.0 / 3.0)
        self.assertEqual(tuple(patched.data.shape), (1, 2, 3))
        self.assertTrue(torch.equal(patched.mask, torch.tensor([[[True, False, False], [False, False, False]]])))
        self.assertTrue(torch.equal(patched.patch_valid, torch.tensor([[True, True]])))
        self.assertAlmostEqual(float(patched.stats.mean[0]), 4.0, places=6)
        self.assertAlmostEqual(float(patched.stats.std[0]), expected_std, places=6)
        self.assertEqual(float(patched.data[0, 0, 0]), 0.0)

    def test_debug_mode_preserves_padding_sentinel(self):
        inputs = torch.tensor([[DEBUG_PADDING_VALUE, 1.0, 2.0, 3.0]])
        mask = torch.tensor([[True, False, False, False]])

        patched = PatchedInput(inputs, mask, patch_length=2, debug_mode=True)

        self.assertLess(abs(float(patched.data[0, 0, 0]) - DEBUG_PADDING_VALUE), 32.0)

    def test_shift_sequence_by_valid_patches_rotates_to_first_valid_patch(self):
        inputs = torch.tensor([[0.0, 0.0, 10.0, 11.0]])
        mask = torch.tensor([[True, True, False, False]])
        patched = PatchedInput(inputs, mask, patch_length=2, debug_mode=False)
        sequence = torch.tensor([[[1.0], [2.0]]])

        shifted = patched.shift_sequence_by_valid_patches(sequence)

        torch.testing.assert_close(shifted, torch.tensor([[[2.0], [1.0]]]))


class PatchedOutputTests(unittest.TestCase):
    def test_postprocess_reshapes_and_denormalizes(self):
        config = Config(patch_length=2, forecast_length=2, num_outputs=1)
        stats = DataStats(mean=torch.tensor([10.0]), std=torch.tensor([2.0]))
        patched_output = PatchedOutput(
            output=torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
            config=config,
            stats=stats,
        )

        patched_output.postprocess()

        self.assertEqual(tuple(patched_output.output.shape), (1, 2, 2, 1))
        expected = torch.tensor([[[[12.0], [14.0]], [[16.0], [18.0]]]])
        torch.testing.assert_close(patched_output.output, expected)


if __name__ == "__main__":
    unittest.main()
