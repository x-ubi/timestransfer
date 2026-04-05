import sys
import unittest
from pathlib import Path

import torch

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from timesfm_blocks import (
    Attention,
    FeedForward,
    RotaryPositionalEmbedding,
    TransformerLayer,
    build_rope_positions,
)


class RopeHelperTests(unittest.TestCase):
    def test_build_rope_positions_renumbers_after_left_padding(self):
        padding_mask = torch.tensor([[True, True, False, False, False]])

        positions = build_rope_positions(padding_mask, sequence_length=5, device=padding_mask.device)

        torch.testing.assert_close(positions, torch.tensor([[0.0, 0.0, 0.0, 1.0, 2.0]]))

    def test_rotary_positional_embedding_preserves_shape(self):
        rope = RotaryPositionalEmbedding(head_dim=4)
        x = torch.randn(2, 5, 3, 4)

        output = rope(x)

        self.assertEqual(tuple(output.shape), tuple(x.shape))


class FeedForwardTests(unittest.TestCase):
    def test_feedforward_zeroes_fully_padded_patches(self):
        torch.manual_seed(0)
        ff = FeedForward(
            input_size=4,
            hidden_size=8,
            norm="none",
            activation="silu",
            zero_init_output=False,
        )
        x = torch.randn(1, 3, 4)
        padding_mask = torch.tensor([[False, True, False]])

        output = ff(x, padding_mask=padding_mask)

        self.assertEqual(tuple(output.shape), (1, 3, 4))
        torch.testing.assert_close(output[:, 1], torch.zeros_like(output[:, 1]))
        self.assertGreater(float(output[:, 0].abs().sum()), 0.0)


class AttentionTests(unittest.TestCase):
    def test_attention_returns_expected_shape_and_masks_padded_patches(self):
        torch.manual_seed(0)
        attention = Attention(
            d_model=8,
            n_heads=2,
            qk_norm="none",
            use_rope=False,
            use_per_dim_scale=False,
            sdp_backend="math",
        )
        attention.eval()
        x = torch.randn(1, 3, 8)
        padding_mask = torch.tensor([[[True, True], [False, False], [False, False]]])

        output = attention(x, padding_mask=padding_mask)

        self.assertEqual(tuple(output.shape), (1, 3, 8))
        torch.testing.assert_close(output[:, 0], torch.zeros_like(output[:, 0]))
        self.assertTrue(torch.isfinite(output).all())

    def test_attention_rejects_invalid_head_configuration(self):
        with self.assertRaises(ValueError):
            Attention(d_model=10, n_heads=3)


class TransformerLayerTests(unittest.TestCase):
    def test_transformer_layer_preserves_shape(self):
        torch.manual_seed(0)
        layer = TransformerLayer(
            input_size=8,
            num_heads=2,
            hidden_size=16,
            d_head=4,
            use_rotary_position_embeddings=False,
            sdp_backend="math",
        )
        layer.eval()
        x = torch.randn(2, 4, 8)
        padding_mask = torch.tensor(
            [
                [[False, False], [False, False], [True, True], [True, True]],
                [[False, False], [False, False], [False, False], [False, False]],
            ]
        )

        output = layer(x, padding_mask=padding_mask)

        self.assertEqual(tuple(output.shape), (2, 4, 8))
        self.assertTrue(torch.isfinite(output).all())

    def test_transformer_layer_rejects_mismatched_head_dim(self):
        with self.assertRaises(ValueError):
            TransformerLayer(
                input_size=8,
                num_heads=2,
                hidden_size=16,
                d_head=3,
                use_rotary_position_embeddings=False,
            )


if __name__ == "__main__":
    unittest.main()
