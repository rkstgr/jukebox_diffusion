import pytest
import torch

from src.diffusion.timestep_sampler.constant_sampler import TimeConstantSampler
from src.diffusion.timestep_sampler.partial_sampler import PartialSampler


@pytest.fixture
def x():
    return torch.rand(256, 2048, 64)


def test_constant_sampler(x):
    sampler = TimeConstantSampler(max_timestep=1000)
    t = sampler.sample_timesteps(x.shape)
    assert t.shape == (x.shape[0],)
    assert torch.all(t < 1000)
    assert torch.all(t >= 0)


def test_partial_sampler(x):
    sampler = PartialSampler(max_timestep=1000, start_fraction=0.5, end_fraction=1)
    t = sampler.sample_timesteps(x.shape)
    assert t.shape == (x.shape[0], x.shape[1])
    assert torch.all(t[:, :1024] == 0)
    assert torch.all(t[:, 1024:] < 1000)
    assert torch.all(t[:, 1024:] >= 0)
