import math
import torch
import torch.nn.functional as F
from torch import nn

from einmix import EinMix


class ResidualConnection(nn.Module):
    def __init__(self, layer) -> None:
        super().__init__()
        self.layer = layer

    def forward(self, x, *args, **kwargs):
        return x + self.layer(x, *args, **kwargs)


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
        self.skip = nn.Identity() if input_size == output_size else nn.Linear(input_size, output_size)

    def forward(self, x):
        residual = self.skip(x)
        hidden = self.hidden_layer(x)
        output = self.output_layer(hidden)

        return residual + output


def _available_sdpa_backends() -> dict[str, torch.nn.attention.SDPBackend]:
    backend_names = {
        "math": "MATH",
        "flash": "FLASH_ATTENTION",
        "efficient": "EFFICIENT_ATTENTION",
    }
    return {
        key: getattr(torch.nn.attention.SDPBackend, value)
        for key, value in backend_names.items()
        if hasattr(torch.nn.attention.SDPBackend, value)
    }


class PerDimScale(nn.Module):
    def __init__(self, head_dim: int) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.per_dim_scale = nn.Parameter(torch.zeros(head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = 1.442695041 / math.sqrt(self.head_dim) * F.softplus(self.per_dim_scale)
        return x * scale


class Attention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        qk_norm: str = "rms",
        use_rope: bool = True,
        use_per_dim_scale: bool = True,
        sdp_backend: str | None = None,
        attention_probs_dropout_rate: float = 0.0,
        out_dropout_rate: float = 0.0,
        norm_eps: float = 1e-6,
    ) -> None:
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}.")

        super().__init__()

        available_backends = _available_sdpa_backends()
        if sdp_backend not in (None, "auto", *available_backends.keys()):
            raise ValueError(
                f"Invalid sdp_backend '{sdp_backend}'."
                f" Expected one of {['auto', *available_backends.keys()]}."
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.qk_norm = qk_norm.lower()
        self.use_rope = use_rope
        self.use_per_dim_scale = use_per_dim_scale
        self.attention_probs_dropout_rate = attention_probs_dropout_rate
        self.out_dropout = nn.Dropout(out_dropout_rate)
        self.sdp_backends = (
            list(available_backends.values())
            if sdp_backend in (None, "auto")
            else [available_backends[sdp_backend]]
        )

        def make_projection(
            signature: str = "... seq_len d_model -> ... seq_len n_heads d_k",
            weight_shape: str = "d_model n_heads d_k",
        ) -> EinMix:
            return EinMix(
                signature=signature,
                weight_shape=weight_shape,
                d_model=d_model,
                n_heads=n_heads,
                d_k=self.d_k,
            )

        self.q = make_projection()
        self.k = make_projection()
        self.v = make_projection()
        self.output = make_projection(
            signature="... seq_len n_heads d_k -> ... seq_len d_model",
            weight_shape="n_heads d_k d_model",
        )

        if self.qk_norm == "rms":
            self.q_norm = nn.RMSNorm(self.d_k, eps=norm_eps)
            self.k_norm = nn.RMSNorm(self.d_k, eps=norm_eps)
        elif self.qk_norm in ("none", "l2"):
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
        else:
            raise ValueError(f"Unsupported qk_norm '{qk_norm}'.")

        if self.use_per_dim_scale:
            self.query_scale = PerDimScale(self.d_k)
        else:
            self.query_scale = nn.Identity()

        if self.use_rope:
            self.rope = RotaryPositionalEmbedding(self.d_k)

    def _qkv_and_transform(
        self,
        x: torch.Tensor,
        patch_padding_mask: torch.BoolTensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        if self.use_rope:
            positions = build_rope_positions(patch_padding_mask, x.shape[1], x.device)
            q = self.rope(q, positions=positions)
            k = self.rope(k, positions=positions)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.qk_norm == "l2":
            q = F.normalize(q, p=2, dim=-1)
            k = F.normalize(k, p=2, dim=-1)

        q = self.query_scale(q)
        return q, k, v

    def _build_attention_mask(
        self,
        patch_padding_mask: torch.BoolTensor | None,
        sequence_length: int,
        device: torch.device,
    ) -> torch.BoolTensor:
        causal = torch.tril(torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=device))
        attn_mask = causal[None, None, :, :]

        if patch_padding_mask is None:
            return attn_mask

        valid = ~patch_padding_mask
        valid_queries = valid[:, None, :, None]
        valid_keys = valid[:, None, None, :]
        return attn_mask & valid_queries & valid_keys

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        patch_padding_mask = _coerce_patch_padding_mask(padding_mask, x)
        q, k, v = self._qkv_and_transform(x, patch_padding_mask)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_mask = self._build_attention_mask(patch_padding_mask, x.shape[1], x.device)

        with torch.nn.attention.sdpa_kernel(backends=self.sdp_backends):
            sdp = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                is_causal=False,
                dropout_p=self.attention_probs_dropout_rate if self.training else 0.0,
                scale=1.0,
            )

        sdp = sdp.transpose(1, 2)
        output = self.out_dropout(self.output(sdp))

        if patch_padding_mask is not None:
            output = output.masked_fill(patch_padding_mask[:, :, None], 0.0)

        return output

def _make_activation(name: str) -> nn.Module:
    activation_name = name.lower()

    if activation_name == "relu":
        return nn.ReLU()
    if activation_name == "gelu":
        return nn.GELU()
    if activation_name == "silu":
        return nn.SiLU()

    raise ValueError(f"Unsupported activation '{name}'.")


def _make_norm(name: str, hidden_size: int, eps: float) -> nn.Module:
    norm_name = name.lower()

    if norm_name == "none":
        return nn.Identity()
    if norm_name == "layernorm":
        return nn.LayerNorm(hidden_size, eps=eps)
    if norm_name == "rmsnorm":
        return nn.RMSNorm(hidden_size, eps=eps)

    raise ValueError(f"Unsupported norm '{name}'.")


def _coerce_patch_padding_mask(
    padding_mask: torch.Tensor | None,
    x: torch.Tensor,
) -> torch.BoolTensor | None:
    if padding_mask is None:
        return None

    if padding_mask.shape[:2] != x.shape[:2]:
        raise ValueError(
            "padding_mask must start with [batch_size, num_patches]."
            f" Got {tuple(padding_mask.shape)} for x with shape {tuple(x.shape)}."
        )

    patch_padding_mask = padding_mask.to(dtype=torch.bool, device=x.device)
    while patch_padding_mask.ndim > 2:
        patch_padding_mask = patch_padding_mask.all(dim=-1)

    return patch_padding_mask


class FeedForward(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        norm: str = "rmsnorm",
        activation: str = "silu",
        norm_eps: float = 1e-6,
        zero_init_output: bool = True,
    ) -> None:
        super().__init__()

        self.norm = _make_norm(norm, input_size, eps=norm_eps)
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.activation = _make_activation(activation)
        self.output_layer = nn.Linear(hidden_size, input_size)

        nn.init.normal_(self.input_layer.weight, mean=0.0, std=(2 / input_size) ** 0.5)
        nn.init.zeros_(self.input_layer.bias)

        if zero_init_output:
            nn.init.zeros_(self.output_layer.weight)
            nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        patch_padding_mask = _coerce_patch_padding_mask(padding_mask, x)

        if patch_padding_mask is not None:
            x = x.masked_fill(patch_padding_mask[:, :, None], 0.0)

        x = self.norm(x)
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.output_layer(x)

        if patch_padding_mask is not None:
            x = x.masked_fill(patch_padding_mask[:, :, None], 0.0)

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
        attention_norm: str = "rmsnorm",
        attention_qk_norm: str = "rmsnorm",
        use_rotary_position_embeddings: bool = True,
        use_per_dim_scale: bool = True,
        sdp_backend: str | None = None,
        attention_probs_dropout_rate: float = 0.0,
        attention_out_dropout_rate: float = 0.0,
        ff_norm: str = "rmsnorm",
        ff_activation: str = "silu",
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()

        expected_d_head = input_size // num_heads
        if d_head != expected_d_head:
            raise ValueError(
                f"d_head={d_head} must equal input_size // num_heads={expected_d_head}."
            )

        qk_norm = attention_qk_norm.lower()
        if qk_norm == "rmsnorm":
            qk_norm = "rms"

        self.pre_attention_norm = _make_norm(attention_norm, input_size, eps=rms_norm_eps)
        self.post_attention_norm = _make_norm(attention_norm, input_size, eps=rms_norm_eps)
        self.attention = Attention(
            d_model=input_size,
            n_heads=num_heads,
            qk_norm=qk_norm,
            use_rope=use_rotary_position_embeddings,
            use_per_dim_scale=use_per_dim_scale,
            sdp_backend=sdp_backend,
            attention_probs_dropout_rate=attention_probs_dropout_rate,
            out_dropout_rate=attention_out_dropout_rate,
            norm_eps=rms_norm_eps,
        )

        self.pre_feed_forward_norm = _make_norm(ff_norm, input_size, eps=rms_norm_eps)
        self.post_feed_forward_norm = _make_norm(ff_norm, input_size, eps=rms_norm_eps)
        self.feed_forward = FeedForward(
            input_size,
            hidden_size,
            norm="none",
            activation=ff_activation,
            norm_eps=rms_norm_eps,
        )

    def forward(self, x, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_output = self.attention(
            self.pre_attention_norm(x),
            padding_mask=padding_mask,
        )
        x = x + self.post_attention_norm(attn_output)

        ff_output = self.feed_forward(
            self.pre_feed_forward_norm(x),
            padding_mask=padding_mask,
        )
        x = x + self.post_feed_forward_norm(ff_output)
        return x
