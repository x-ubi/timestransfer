from dataclasses import dataclass

import torch.nn as nn
import torch

from timesfm_blocks import (
    ResidualBlock,
    Attention,
    FeedForward,
    PositionalEmbedding,
    ResidualConnection,
    TransformerLayer,
)

from data_processing import PatchedInput, PatchedOutput


@dataclass(frozen=True)
class Config:
    num_layers: int = 20

    num_heads: int = 16

    num_kv_heads: int = 16

    hidden_size: int = 1280

    intermediate_size: int = 1280

    # futureproofing if i decide to add quantiles
    num_outputs: int = 1

    forecast_length: int = 128


class TimesFM(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self.config = config

        self.num_outputs = config.num_outputs

        self.input_mlp = ResidualBlock(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            output_size=config.hidden_size,
        )
        self.frequency_embedding = nn.Embedding(num_embeddings=3, embedding_dim=config.hidden_size)
        # self.transformer =

        self.transformer_layers = nn.Sequential(
            *[
                TransformerLayer(
                    # TODO: fill it in with params once i figure that mess out
                )
                for _ in range(config.num_layers)
            ]
        )
        self.positional_embedding = PositionalEmbedding()  # TODO: finish this

        self.output_mlp = ResidualBlock(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            output_size=config.num_outputs * config.forecast_length,
        )

    def forward(self, input: PatchedInput) -> torch.Tensor:
        x = input.data
        positional_mask = input.mask
        stats = input.stats

        input_data = torch.cat([x, positional_mask], dim=-1)
        model_input: torch.Tensor = self.input_mlp(input_data)

        batch_size = model_input.shape[0]
        sequence_length = model_input.shape[1]
        positional_embedding: torch.Tensor = self.positional_embedding(sequence_length).to(model_input.device)
        positional_embedding = input.shift_sequence_by_valid_patches(
            positional_embedding.expand(batch_size, sequence_length, -1)
        )
        model_input += positional_embedding
        model_input += self.frequency_embedding(model_input)

        model_output = self.transformer_layers(model_input)

        model_output = PatchedOutput(
            self.output_mlp(model_output),
            self.config,
            stats,
        )
        model_output.postprocess()

        return model_output.output
