import torch

from src.diffusion.timestep_sampler.diffusion_timestep_sampler import DiffusionTimestepSampler


class TimeConstantSampler(DiffusionTimestepSampler):
    """
    Time constant sampler:

    Sample a single timestep for each batch element, uniformly distributed between 0 and max_timestep
    Equivalent to standard unconditional diffusion

    """

    def __init__(self, max_timestep: int = 1000):
        super().__init__(max_timestep=max_timestep)

    def sample_timesteps(self, input_size: torch.Size = None) -> torch.Tensor:
        """
        Args:
            input_size [B, S, D]: Shape of the input tensor, used to determine the batch size and sequence length

        Returns:
            Long Tenor [B]: Only one timestep per batch
        """
        return torch.randint(0, self.max_timestep, (input_size[0],)).long()
