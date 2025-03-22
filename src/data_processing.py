from dataclasses import dataclass
from einops import rearrange
import torch


DEBUG_PADDING_VALUE = 1123581321.0
TOLERANCE = 1e-5


class PatchedInput:
    patched_input: torch.Tensor  # shape: (batch_size, )
    patched_mask: torch.BoolTensor

    # possible TODO: check how much (if any) performance is lost from using rearrange instead of view
    def __init__(
        self, input_time_series: torch.Tensor, mask: torch.Tensor, patch_length: int, debug_mode: bool
    ) -> None:
        """
        input_time_series: shape (batch_size, seq_len, num_features)
        """
        self.debug_mode = debug_mode

        self.patched_input = rearrange(
            input_time_series,
            "batch_size (num_patches patch_length) -> batch_size num_patches patch_length",
            p=patch_length,
        )
        self.patched_mask = rearrange(
            mask, "batch_size (num_patches patch_length) -> batch_size num_patches patch_length", p=patch_length
        )

        self.patched_input = self.patched_input.masked_fill_(mask, 0.0)
        self.patched_mask = self.patched_mask.masked_fill_(
            (self.patched_input - DEBUG_PADDING_VALUE).abs() < TOLERANCE, True
        )

        self.mean, self.std = self._first_valid_patch_mean_std()

        self.patched_input = self._normalize_input()
        self.patched_input = self.patched_input.masked_fill(self.patched_mask, 0.0)

    def _normalize_input(
        self,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]] | None:
        normalized_input = (self.patched_input - self.mean[:, None, None]) / self.std[:, None, None]

        if self.debug_mode:
            normalized_input = normalized_input.masked_fill(
                (self.patched_input - DEBUG_PADDING_VALUE).abs() < TOLERANCE, DEBUG_PADDING_VALUE
            )

        return normalized_input

    def _first_valid_patch_mean_std(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        A patch is valid if it has at least three non-masked values.
        """

        non_masked_count = (1 - self.patched_mask).sum(dim=-1)
        is_patch_valid = non_masked_count >= 3
        chosen_patches = torch.argmax(is_patch_valid.to(torch.int32), dim=-1)

        no_valid_patches = ~is_patch_valid.any(dim=1)
        chosen_patches[no_valid_patches] = -1

        batch_indices = torch.arange(self.patched_input.shape[0], device=self.patched_input.device)

        chosen_input = self.patched_input[batch_indices, chosen_patches, :]
        chosen_mask = self.patched_mask[batch_indices, chosen_patches, :]

        chosen_input = chosen_input.masked_fill(chosen_mask, float("nan"))

        mean = chosen_input.nanmean(dim=-1)
        variance = torch.nanmean((chosen_input - mean.unsqueeze(-1)) ** 2, dim=-1)
        std = variance.sqrt()

        mean = torch.where(mean.isnan(), torch.zeros_like(mean), mean)
        std = torch.where(std.isnan(), torch.zeros_like(std), std)
        std = torch.where(std < TOLERANCE, torch.ones_like(std), std)

        return mean, std
