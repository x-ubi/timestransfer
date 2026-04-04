from dataclasses import dataclass
from typing import TYPE_CHECKING
import einops
import torch

if TYPE_CHECKING:
    from timesfm import Config


DEBUG_PADDING_VALUE = 1123581321.0
TOLERANCE = 1e-5


@dataclass(frozen=True)
class DataStats:
    mean: torch.Tensor
    std: torch.Tensor


class PatchedInput:
    data: torch.Tensor  # shape: (batch_size, num_patches, patch_length)
    mask: torch.BoolTensor
    patch_valid: torch.BoolTensor
    stats: DataStats

    # possible TODO: check how much (if any) performance is lost from using rearrange instead of view
    def __init__(
        self, input_time_series: torch.Tensor, mask: torch.Tensor, patch_length: int, debug_mode: bool
    ) -> None:
        """
        input_time_series: shape (batch_size, seq_len) or (batch_size, seq_len, 1)
        """
        self.debug_mode = debug_mode

        if input_time_series.ndim == 3:
            if input_time_series.shape[-1] != 1:
                raise ValueError(
                    "PatchedInput only supports univariate inputs with shape"
                    f" [batch, seq_len] or [batch, seq_len, 1], got {tuple(input_time_series.shape)}."
                )
            input_time_series = input_time_series.squeeze(-1)
        elif input_time_series.ndim != 2:
            raise ValueError(
                "PatchedInput expects input_time_series with shape"
                f" [batch, seq_len] or [batch, seq_len, 1], got {tuple(input_time_series.shape)}."
            )

        if mask.ndim == 3:
            if mask.shape[-1] != 1:
                raise ValueError(
                    "PatchedInput only supports univariate masks with shape"
                    f" [batch, seq_len] or [batch, seq_len, 1], got {tuple(mask.shape)}."
                )
            mask = mask.squeeze(-1)
        elif mask.ndim != 2:
            raise ValueError(
                "PatchedInput expects mask with shape"
                f" [batch, seq_len] or [batch, seq_len, 1], got {tuple(mask.shape)}."
            )

        if input_time_series.shape != mask.shape:
            raise ValueError(
                "input_time_series and mask must have the same shape."
                f" Got {tuple(input_time_series.shape)} and {tuple(mask.shape)}."
            )
        if input_time_series.shape[1] % patch_length != 0:
            raise ValueError(
                f"Sequence length must be divisible by patch_length={patch_length},"
                f" got seq_len={input_time_series.shape[1]}."
            )

        self.data = einops.rearrange(
            input_time_series,
            "b (n p) -> b n p",
            p=patch_length,
        )
        self.mask = einops.rearrange(mask.to(torch.bool), "b (n p) -> b n p", p=patch_length)

        debug_padding_mask = (self.data - DEBUG_PADDING_VALUE).abs() < TOLERANCE
        self.mask = self.mask.masked_fill(debug_padding_mask, True)
        self.data = self.data.masked_fill(self.mask, 0.0)

        self.patch_valid = (~self.mask).any(dim=-1)

        self.stats = self._first_valid_patch_mean_std()

        self.data = self._normalize_input()
        if not self.debug_mode:
            self.data = self.data.masked_fill(self.mask, 0.0)

    def _normalize_input(
        self,
    ) -> torch.Tensor:
        normalized_input = (self.data - self.stats.mean[:, None, None]) / self.stats.std[:, None, None]

        if self.debug_mode:
            normalized_input = normalized_input.masked_fill(self.mask, DEBUG_PADDING_VALUE)

        return normalized_input

    def _first_valid_patch_mean_std(self) -> DataStats:
        """
        A patch is valid if it has at least three non-masked values.
        """

        non_masked_count = (~self.mask).sum(dim=-1)
        is_patch_valid = non_masked_count >= 3
        chosen_patches = torch.argmax(is_patch_valid.to(torch.int32), dim=-1)

        no_valid_patches = ~is_patch_valid.any(dim=1)
        chosen_patches[no_valid_patches] = 0

        batch_indices = torch.arange(self.data.shape[0], device=self.data.device)

        chosen_input = self.data[batch_indices, chosen_patches, :]
        chosen_mask = self.mask[batch_indices, chosen_patches, :]

        chosen_input = chosen_input.masked_fill(chosen_mask, float("nan"))

        mean = chosen_input.nanmean(dim=-1)
        variance = torch.nanmean((chosen_input - mean.unsqueeze(-1)) ** 2, dim=-1)
        std = variance.sqrt()

        mean = torch.where(no_valid_patches, torch.zeros_like(mean, device=mean.device), mean)
        std = torch.where(no_valid_patches, torch.ones_like(std, device=std.device), std)
        mean = torch.where(mean.isnan(), torch.zeros_like(mean, device=mean.device), mean)
        std = torch.where(std.isnan(), torch.zeros_like(std, device=std.device), std)
        std = torch.where(std < TOLERANCE, torch.ones_like(std, device=std.device), std)

        return DataStats(mean=mean, std=std)

    def shift_sequence_by_valid_patches(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim < 3:
            raise ValueError(f"sequence must have shape [batch_size, num_patches, ...], got {tuple(sequence.shape)}.")
        if sequence.shape[:2] != self.data.shape[:2]:
            raise ValueError(
                "Shape mismatch between sequence and patched input."
                f" Got {tuple(sequence.shape[:2])} and {tuple(self.data.shape[:2])}."
            )

        batch_size, num_patches = sequence.shape[:2]
        trailing_shape = sequence.shape[2:]

        first_valid_patches = self.patch_valid.to(torch.int32).argmax(dim=1)
        first_valid_patches[~self.patch_valid.any(dim=1)] = 0

        idx_ranges = torch.arange(num_patches, device=sequence.device).unsqueeze(0).expand(batch_size, num_patches)
        shifted_idx = (idx_ranges - first_valid_patches.unsqueeze(1)) % num_patches
        gather_index = shifted_idx.view(batch_size, num_patches, *([1] * len(trailing_shape))).expand(
            batch_size, num_patches, *trailing_shape
        )
        return sequence.gather(1, gather_index)


@dataclass
class PatchedOutput:
    output: torch.Tensor
    config: "Config"
    stats: DataStats

    def postprocess(self):
        self.output = einops.rearrange(
            self.output, "b n (h q) -> b n h q", h=self.config.forecast_length, q=self.config.num_outputs
        )
        self._reverse_normalization()

    def _reverse_normalization(self) -> None:
        self.output = self.output * self.stats.std[:, None, None, None] + self.stats.mean[:, None, None, None]
