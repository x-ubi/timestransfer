from dataclasses import dataclass

import torch
from torch import nn

from data_processing import PatchedInput, PatchedOutput
from timesfm_blocks import TransformerLayer


@dataclass(frozen=True)
class Config:
    patch_length: int = 32
    num_layers: int = 20
    num_heads: int = 16
    hidden_size: int = 1280
    intermediate_size: int = 1280
    num_outputs: int = 1
    forecast_length: int = 128
    attention_norm: str = "rmsnorm"
    attention_qk_norm: str = "rmsnorm"
    ff_norm: str = "rmsnorm"
    ff_activation: str = "silu"
    use_rotary_position_embeddings: bool = True
    use_per_dim_scale: bool = True
    sdp_backend: str | None = None
    attention_probs_dropout_rate: float = 0.0
    attention_out_dropout_rate: float = 0.0
    rms_norm_eps: float = 1e-6


class TimesFM(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

        tokenizer_input_size = 2 * config.patch_length
        head_dim = config.hidden_size // config.num_heads
        if config.hidden_size % config.num_heads != 0:
            raise ValueError(
                f"hidden_size={config.hidden_size} must be divisible by num_heads={config.num_heads}."
            )

        self.input_projection = nn.Linear(tokenizer_input_size, config.hidden_size)
        self.transformer_layers = nn.ModuleList(
            [
                TransformerLayer(
                    input_size=config.hidden_size,
                    num_heads=config.num_heads,
                    hidden_size=config.intermediate_size,
                    d_head=head_dim,
                    attention_norm=config.attention_norm,
                    attention_qk_norm=config.attention_qk_norm,
                    use_rotary_position_embeddings=config.use_rotary_position_embeddings,
                    use_per_dim_scale=config.use_per_dim_scale,
                    sdp_backend=config.sdp_backend,
                    attention_probs_dropout_rate=config.attention_probs_dropout_rate,
                    attention_out_dropout_rate=config.attention_out_dropout_rate,
                    ff_norm=config.ff_norm,
                    ff_activation=config.ff_activation,
                    rms_norm_eps=config.rms_norm_eps,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.output_projection = nn.Linear(
            config.hidden_size,
            config.num_outputs * config.forecast_length,
        )

    def forward(self, input: PatchedInput) -> torch.Tensor:
        if input.data.shape[-1] != self.config.patch_length:
            raise ValueError(
                f"Patched input patch_length={input.data.shape[-1]} does not match"
                f" model patch_length={self.config.patch_length}."
            )

        patched_values = input.data.masked_fill(input.mask, 0.0)
        patch_padding_mask = input.mask
        tokenizer_inputs = torch.cat([patched_values, patch_padding_mask.to(patched_values.dtype)], dim=-1)

        hidden_states = self.input_projection(tokenizer_inputs)

        for layer in self.transformer_layers:
            hidden_states = layer(hidden_states, padding_mask=patch_padding_mask)

        model_output = PatchedOutput(
            output=self.output_projection(hidden_states),
            config=self.config,
            stats=input.stats,
        )
        model_output.postprocess()
        return model_output.output
