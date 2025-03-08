import torch
from torch import nn

from einmix import EinMix


class ResidualConnection(nn.Module):
    def __init__(self, layer) -> None:
        super().__init__()
        self.layer = layer

    def forward(self, x):
        return x + self.layer(x)


# @TODO: Experiment with other activation functions
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
        input_size,
        num_heads,
    ) -> None:
        super().__init__()

        assert input_size % num_heads == 0, "model_dim must be divisible by num_heads"
        self.head_dim = input_size // num_heads


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
        nn.init.normal_(self.input_layer[0].weight, mean=0.0, std=(2 / input_size) ** 0.5)
        self.input_layer[0].bias.data.zero_()
        self.output_layer = nn.Linear(hidden_size, input_size)
        self.output_layer.weight.data.zero_()
        self.output_layer.bias.data.zero_()

    # @TODO: Possibly missing paddings
    def forward(self, x) -> torch.Tensor:
        x = self.layer_norm(x)
        x = self.input_layer(x)
        x = self.output_layer(x)
        return x


class TransformerLayer(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_heads: int,
        hidden_size: int,
    ) -> None:
        super().__init__()

        self.attention_with_residual = ResidualConnection(Attention(input_size, num_heads=num_heads))
        self.feed_forward_with_residual = ResidualConnection(FeedForward(input_size, hidden_size))

    def forward(self, x) -> torch.Tensor:
        x = self.attention_with_residual(x)
        x = self.feed_forward_with_residual(x)
        return x
