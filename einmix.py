import torch
from torch import nn


from einops.layers.torch import EinMix as EinOpsEinMix


class EinMix(nn.Module):
    """EinMix is a wrapper around einops.einsum that allows for the use of ellipses in the signature."""

    def __init__(self, signature, weight_shape=None, bias_shape=None, **kwargs) -> None:
        super().__init__()
        self.change_anything = False
        if "..." in signature:
            self.change_anything = True
            self.og_signature = signature
            signature = signature.replace("...", "squeezed")
        self.layer = EinOpsEinMix(signature, weight_shape=weight_shape, bias_shape=bias_shape, **kwargs)
        if self.layer.bias is not None:
            self.layer.bias.data *= 0.0
        std = (2 / (5 * max(self.layer.weight.shape))) ** 0.5
        nn.init.normal_(self.layer.weight, mean=0.0, std=std)

    def forward(self, x) -> torch.Tensor:
        if not self.change_anything:
            return self.layer(x)
        beginning, end = self.og_signature.split("->")
        beginning = beginning.split()
        end = end.split()
        assert beginning[0] == end[0] == "..."
        contracted_dims = len(x.shape) - len(beginning) + 1 + (1 if "(" in "".join(beginning) else 0)
        ellipsis_shape = list(x.shape[:contracted_dims])
        newx = torch.reshape(x, [-1] + list(x.shape[contracted_dims:]))
        output = self.layer(newx)
        new_output = torch.reshape(output, ellipsis_shape + list(output.shape[1:]))
        return new_output
