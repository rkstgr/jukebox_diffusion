import torch
from einops import repeat

from src.diffusion.timestep_sampler.diffusion_timestep_sampler import DiffusionTimestepSampler


class PartialSampler(DiffusionTimestepSampler):
    """
    Partial sampler:

    Constant over a fixed portion of the sequence (e.g second half/quarter) and 0 everywhere else
    """

    def __init__(self, max_timestep: int = 1000, start_fraction: float = 0.5, end_fraction: float = 1):
        """
        Args:
            max_timestep: Maximum timestep to sample
            start_fraction: Fraction of the sequence to start the constant portion
            end_fraction: Fraction of the sequence to end the constant portion
        """
        super().__init__(max_timestep=max_timestep)
        self.start_fraction = start_fraction
        self.end_fraction = end_fraction

    def sample_timesteps(self, input_size: torch.Size = None) -> torch.LongTensor:
        """
        Args:
            input_size [B, S, D]: Shape of the input tensor, used to determine the batch size and sequence length

        Returns:
            Long Tenor [B]: Only one timestep per batch
        """
        tx = torch.zeros(input_size[0], input_size[1]).long()
        start = int(self.start_fraction * input_size[1])
        end = int(self.end_fraction * input_size[1])
        t = torch.randint(0, self.max_timestep, (input_size[0],)).long()
        t = repeat(t, "b -> b s", s=end - start)
        tx[:, start:end] = t
        return tx
