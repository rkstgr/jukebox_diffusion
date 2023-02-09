from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

class StatsRecorder:
    def __init__(self, data) -> None:
        assert data.dim() == 3, "Input tensor must be of shape (Batch, Length, Channels)"
        self.n_observations = data.shape[0] * data.shape[1]
        self.mean = torch.mean(data, dim=(0, 1))
        self.std = torch.std(data, dim=(0, 1))
    
    def update(self, data) -> None:
        assert data.dim() == 3, "Input tensor must be of shape (Batch, Length, Channels)"

        n_observations = data.shape[0] * data.shape[1]
        new_mean = torch.mean(data, dim=(0, 1))
        new_std = torch.std(data, dim=(0, 1))
        
        m = self.n_observations * 1.0
        n = n_observations * 1.0

        mean_prev = self.mean

        # incremental mean
        self.mean = (m * mean_prev + n * new_mean) / (m + n)
        
        # incremental std
        self.std = torch.sqrt((m * (self.std ** 2) + n * (new_std ** 2) + (m * n * (mean_prev - new_mean) ** 2) / (m + n)) / (m + n))

        self.n_observations += n_observations
        

class JukeboxNormalizer(nn.Module):

    def __init__(self, mean=None, std=None) -> None:
        super().__init__()
        self.mean = mean
        self.std = std

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
        for i, batch in tqdm(enumerate(dataloader), total=total):
            if i == 0:
                if apply_fn is not None:
                    batch = apply_fn(batch)
                stats = StatsRecorder(batch)
            else:
                stats.update(batch)
        
        return stats.mean, stats.std

    def save_stats(self, filename: str, stats: Tuple[torch.Tensor, torch.Tensor]) -> None:
        torch.save(stats, filename)
    