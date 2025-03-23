from dataclasses import dataclass
from einops import rearrange
import torch


DEBUG_PADDING_VALUE = 1123581321.0
TOLERANCE = 1e-5


@dataclass(frozen=True)
class DataStats:
    mean: torch.Tensor
    std: torch.Tensor


class PatchedInput:
    data: torch.Tensor  # shape: (batch_size, )
    mask: torch.BoolTensor
    stats: DataStats

    # possible TODO: check how much (if any) performance is lost from using rearrange instead of view
    def __init__(
        self, input_time_series: torch.Tensor, mask: torch.Tensor, patch_length: int, debug_mode: bool
    ) -> None:
        """
        input_time_series: shape (batch_size, seq_len, num_features)
        """
        self.debug_mode = debug_mode

        self.data = rearrange(
            input_time_series,
            "batch_size (num_patches patch_length) -> batch_size num_patches patch_length",
            p=patch_length,
        )
        self.mask = rearrange(
            mask, "batch_size (num_patches patch_length) -> batch_size num_patches patch_length", p=patch_length
        )

        self.data = self.data.masked_fill_(mask, 0.0)
        self.mask = self.mask.masked_fill_((self.data - DEBUG_PADDING_VALUE).abs() < TOLERANCE, True)

        self.stats = self._first_valid_patch_mean_std()

        self.data = self._normalize_input()
        self.data = self.data.masked_fill(self.mask, 0.0)

    def _normalize_input(
        self,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]] | None:
        normalized_input = (self.data - self.mean[:, None, None]) / self.std[:, None, None]

        if self.debug_mode:
            normalized_input = normalized_input.masked_fill(
                (self.data - DEBUG_PADDING_VALUE).abs() < TOLERANCE, DEBUG_PADDING_VALUE
            )

        return normalized_input

    def _first_valid_patch_mean_std(self) -> DataStats:
        """
        A patch is valid if it has at least three non-masked values.
        """

        non_masked_count = (1 - self.mask).sum(dim=-1)
        is_patch_valid = non_masked_count >= 3
        chosen_patches = torch.argmax(is_patch_valid.to(torch.int32), dim=-1)

        no_valid_patches = ~is_patch_valid.any(dim=1)
        chosen_patches[no_valid_patches] = -1

        batch_indices = torch.arange(self.data.shape[0], device=self.data.device)

        chosen_input = self.data[batch_indices, chosen_patches, :]
        chosen_mask = self.mask[batch_indices, chosen_patches, :]

        chosen_input = chosen_input.masked_fill(chosen_mask, float("nan"))

        mean = chosen_input.nanmean(dim=-1)
        variance = torch.nanmean((chosen_input - mean.unsqueeze(-1)) ** 2, dim=-1)
        std = variance.sqrt()

        mean = torch.where(mean.isnan(), torch.zeros_like(mean), mean)
        std = torch.where(std.isnan(), torch.zeros_like(std), std)
        std = torch.where(std < TOLERANCE, torch.ones_like(std), std)

        return DataStats(mean=mean, std=std)


class PatchedOutput:
    stats: DataStats

    def reverse_normalization(self, output: torch.Tensor) -> torch.Tensor:
        return output * self.stats.std[:, None, None, None] + self.stats.mean[:, None, None, None]
