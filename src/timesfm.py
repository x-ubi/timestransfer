from dataclasses import dataclass

import torch.nn as nn

from timesfm_blocks import ResidualBlock, Attention, FeedForward, ResidualConnection


@dataclass(frozen=True)
class Config:
    num_layers: int = 20

    num_heads: int = 16

    num_kv_heads: int = 16

    hidden_size: int = 1280

    intermediate_size: int = 1280


class TimesFM(nn.Module):

    def __init__(self, config: Config) -> None:
        super().__init__()

        self.config = config
        self.input_layer = ResidualBlock
