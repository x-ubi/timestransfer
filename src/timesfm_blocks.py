import math
import numpy as np
import torch
from torch import nn

from einmix import EinMix


class ResidualConnection(nn.Module):
    def __init__(self, layer) -> None:
        super().__init__()
        self.layer = layer

    def forward(self, x):
        return x + self.layer(x)


# possible TODO: Experiment with other activation functions
class ResidualBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
    ) -> None:
        super().__init__()

        self.hidden_layer = nn.Sequential(nn.Linear(input_size, hidden_size), nn.SiLU())
        self.output_layer = nn.Linear(hidden_size, output_size)

        self.residual_layer = ResidualConnection(nn.Linear(input_size, output_size))

    def forward(self, x):
        hidden = self.hidden_layer(x)
        output = self.output_layer(hidden)

        return self.residual_layer(output)


# @TODO: attention block
class Attention(nn.Module):
    def __init__(
        self,
        # hidden_size: int,
        dim_model: int,
        num_heads: int,
        dim_head: int,
        use_qk_norm: bool,
        sequence_length: int,
        d_k: int,
    ) -> None:
        super().__init__()

        assert dim_model % num_heads == 0, "num_q_heads must be divisible by num_kv_heads"

        def generate_einmix(
            signature: str = "... seq_len dim_model ->... n_heads seq_len d_k",
            weight_shape: str = "dim_model n_heads d_k",
        ) -> EinMix:
            return EinMix(
                signature=signature,
                weight_shape=weight_shape,
                dim_model=dim_model,
                n_heads=num_heads,
                d_k=d_k,
            )

        self.q = generate_einmix()
        self.k = generate_einmix()
        self.v = generate_einmix()

        self.output = generate_einmix(
            signature=".. n_heads seq_len d_k -> ... seq_len dim_model",
            weight_shape="n_heads d_k dim_model",
        )

        self.use_qk_norm = use_qk_norm

        if use_qk_norm:
            # TODO: look into why its like this, maybe change this later
            scale = 0.75 * sequence_length
            self.scale = nn.Parameter(torch.tensor(np.log2(scale**2 - scale) / num_heads))

    def forward(self, x: torch.Tensor, attn_mask: torch.BoolTensor) -> torch.Tensor:
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        if self.use_qk_norm:
            q = q / torch.linalg.norm(q, dim=-1, keepdim=True)
            k = k / torch.linalg.norm(k, dim=-1, keepdim=True)

        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            if self.use_qk_norm:
                q = self.scale * q
                k = self.scale * k
            sdp_scale = 1.0 if self.use_qk_norm else None
            sdp = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=True, attn_mask=~attn_mask, scale=sdp_scale
            )
        output = self.output(sdp)


# @TODO: look into RMSNorm
class MaskedFeedForward(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
    ) -> None:
        super().__init__()

        self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)

        self.input_layer = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU())
        nn.init.normal_(self.input_layer[0].weight, mean=0.0, std=(2 / input_size) ** 0.5)
        self.input_layer[0].bias.data.zero_()
        self.output_layer = nn.Linear(hidden_size, input_size)
        self.output_layer.weight.data.zero_()
        self.output_layer.bias.data.zero_()

    # @TODO: Possibly missing paddings
    def forward(self, x: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = self.input_layer(x)
        x = self.output_layer(x)
        x *= (~mask)[:, :, None]
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = float(d_model)

    def forward(self, sequence_length: int) -> torch.Tensor:
        num_timescales = self.d_model // 2
        position = torch.arange(sequence_length, dtype=torch.float32).unsqueeze(0).unsqueeze(2)
        log_timescale_increment = math.log(10000.0) / max(1, num_timescales - 1)

        inversed_timescales = torch.exp(torch.arange(num_timescales, dtype=torch.float32) * -log_timescale_increment)
        scaled_time = position * inversed_timescales.unsqueeze(0).unsqueeze(0)

        positional_encoding = torch.zeros(position.size(0), position.size(1), self.d_model, dtype=torch.float32)
        positional_encoding[:, :, 0::2] = torch.sin(scaled_time)
        positional_encoding[:, :, 1::2] = torch.cos(scaled_time)

        return positional_encoding


def build_rope_positions(
    padding_mask: torch.Tensor | None,
    sequence_length: int,
    device: torch.device,
    shift: int | torch.Tensor = 0,
) -> torch.Tensor:
    """
    Returns positions with shape [batch, sequence_length].

    Assumes `padding_mask` is True for padded tokens and False for valid tokens.
    If there is left padding, valid tokens are renumbered so the first real token
    starts at position 0, which is closer to TimesFM behavior.
    """
    if padding_mask is None:
        positions = torch.arange(sequence_length, device=device, dtype=torch.float32)
        return positions.unsqueeze(0)

    valid = (~padding_mask).to(torch.int64)
    positions = torch.cumsum(valid, dim=1) - 1
    positions = positions.clamp_min(0).to(torch.float32)

    if isinstance(shift, int):
        positions = positions + float(shift)
    else:
        positions = positions + shift[:, None].to(device=device, dtype=torch.float32)

    return positions


class RotaryPositionalEmbedding(nn.Module):
    """
    RoPE for tensors shaped [batch, sequence_length, num_heads, head_dim].
    """

    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE requires an even head_dim, got {head_dim}.")

        self.head_dim = head_dim
        self.base = float(base)

        inverse_frequencies = torch.exp(
            torch.arange(0, head_dim, 2, dtype=torch.float32) * (-math.log(self.base) / head_dim)
        )
        self.register_buffer("inverse_frequencies", inverse_frequencies, persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor | None = None,
        shift: int | torch.Tensor = 0,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"Expected x with shape [batch, sequence_length, num_heads, head_dim], got {tuple(x.shape)}."
            )
        if x.shape[-1] != self.head_dim:
            raise ValueError(f"Expected head_dim={self.head_dim}, got x.shape[-1]={x.shape[-1]}.")

        batch_size, sequence_length, _, _ = x.shape

        if positions is None:
            positions = build_rope_positions(
                padding_mask=None,
                sequence_length=sequence_length,
                device=x.device,
                shift=shift,
            )

        if positions.ndim == 1:
            positions = positions.unsqueeze(0)
        if positions.shape[0] == 1 and batch_size > 1:
            positions = positions.expand(batch_size, -1)

        angles = positions.to(device=x.device, dtype=torch.float32)[..., None]
        angles = angles * self.inverse_frequencies[None, None, :]

        cos = torch.repeat_interleave(torch.cos(angles), repeats=2, dim=-1)
        sin = torch.repeat_interleave(torch.sin(angles), repeats=2, dim=-1)

        cos = cos.unsqueeze(2).to(dtype=x.dtype)
        sin = sin.unsqueeze(2).to(dtype=x.dtype)

        return x * cos + self._rotate_half(x) * sin


class TransformerLayer(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_heads: int,
        hidden_size: int,
        d_head: int,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.attention_with_residual = ResidualConnection(Attention(input_size, num_heads=num_heads))
        self.feed_forward_with_residual = ResidualConnection(MaskedFeedForward(input_size, hidden_size))
        self.rms_norm = nn.RMSNorm(input_size, eps=rms_norm_eps)

    def forward(self, x) -> torch.Tensor:
        x = self.rms_norm(x)
        x = self.attention_with_residual(x)
        x = self.feed_forward_with_residual(x)
        return x
