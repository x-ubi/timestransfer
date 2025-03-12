from dataclasses import dataclass
from einops import rearrange
import torch

@dataclass
class PatchedTimeSeries:
    patched_time_series: torch.Tensor # shape: (batch_size, )
    mask: torch.Tensor

    # possible @TODO: check how much performance is lost from using rearrange instead of view
    def __init__(self, input_time_series: torch.Tensor, mask: torch.Tensor, patch_length: int) -> None:
        '''
        input_time_series: shape (batch_size, seq_len, num_features)
        '''

        batch_size = input_time_series.shape[0]
        patched_input = rearrange(input_time_series, 'batch_size (num_patches patch_length) -> batch_size num_patches patch_length', p=patch_length)
        patched_mask = rearrange(mask, 'batch_size (num_patches patch_length) -> batch_size num_patches patch_length', p=patch_length)

        patched_input = patched_input.masked_fill_




    def mean_std(self) -> tuple[torch.Tensor, torch.Tensor]:

