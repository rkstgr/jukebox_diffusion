import abc

import torch


class DiffusionTimestepSampler(abc.ABC):
    def __init__(self, max_timestep: int = 1000):
        self.max_timestep = max_timestep

    @abc.abstractmethod
    def sample_timesteps(self, input_size: torch.Size = None) -> torch.Tensor:
        """
        Args:
            input_size [B, S]: Shape of the input tensor, used to determine the batch size and sequence length

        Returns:
            Long Tenor [B, S]: Timesteps for each element in the input tensor
        """
        raise NotImplementedError
