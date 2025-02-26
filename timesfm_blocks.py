import torch
from torch import nn

from einmix import EinMix


# @TODO: Experiment with other activation functions
class ResidualBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
    ) -> None:
        super().__init__()


# @TODO: attention block
class Attention(nn.Module):
    def __init__(
        self,
        model_dim,
        num_heads,
    ) -> None:
        super().__init__()

        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"
        self.head_dim = model_dim // num_heads


# @TODO: look into layer initialization
class FeedForward(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
    ) -> None:
        super().__init__()

        self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)

        self.input_layer = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU())
        self.output_layer = nn.Linear(hidden_size, input_size)

    # @TODO: Possibly missing paddings
    def forward(self, x) -> torch.Tensor:
        x = self.layer_norm(x)
        x = self.input_layer(x)
        x = self.output_layer(x)
        return x
