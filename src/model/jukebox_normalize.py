from pathlib import Path
from typing import Tuple, Optional

from einops import rearrange
import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


class RunningStats:

    def __init__(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0

    def standard_deviation(self):
        return torch.sqrt(self.variance())

        

class JukeboxNormalizer(nn.Module):

    def __init__(self, path: Optional[str] = None) -> None:
        super().__init__()
        self.register_buffer("mean", None)
        self.register_buffer("std", None)
        
        if path is not None:
            print(f"Loading jukebox normalizer from {path}")
            self.load_settings(path)

    def load_settings(self, path: str) -> None:
        self.mean, self.std = torch.load(path)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise ValueError("Mean and standard deviation must be set before normalizing the input tensor")

        assert x.dim() == 3, "Input tensor must be of shape (Batch, Temporal length, Channel/Feature)"
        mean = self.mean.unsqueeze(0).unsqueeze(0)
        std = self.std.unsqueeze(0).unsqueeze(0)

        return (x - mean) / std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise ValueError("Mean and standard deviation must be set before normalizing the input tensor")

        assert x.dim() == 3, "Input tensor must be of shape (Batch, Temporal length, Channel/Feature)"
        mean = self.mean.unsqueeze(0).unsqueeze(0)
        std = self.std.unsqueeze(0).unsqueeze(0)

        return x * std + mean


    def compute_stats(self, x: torch.Tensor):
        """ Compute the mean and standard deviation of the input tensor per channel.
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Temporal length, Channel/Feature)
        Returns:
            Tuple (torch.Tensor, torch.Tensor): Mean and standard deviation, each of shape (Channel/Feature, )
        """
        assert x.dim() == 3, "Input tensor must be of shape (Batch, Temporal length, Channel/Feature)"

        mean = torch.mean(x, dim=(0, 1))
        std = torch.std(x, dim=(0, 1))
        return mean, std

    def compute_stats_iter(self, dataloader: DataLoader, apply_fn=None, total=None) -> Tuple[torch.Tensor, torch.Tensor]:
        running_stats = RunningStats()
        for batch in tqdm(dataloader, total=total):
            if apply_fn is not None:
                batch = apply_fn(batch)
            batch = rearrange(batch, "b t c -> (b t) c")
            for s in batch:
                running_stats.push(s)
        
        return running_stats.mean(), running_stats.standard_deviation()

    def save_stats(self, filename: str, stats: Tuple[torch.Tensor, torch.Tensor]) -> None:
        torch.save(stats, filename)
    