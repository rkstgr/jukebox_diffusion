import pytest
import torch

from src.module.diffusion_attn_unet_1d import DiffusionAttnUnet1D


@pytest.fixture
def model():
    return DiffusionAttnUnet1D(
        io_channels=64,
        n_attn_layers=2,
        channel_sizes=[128] * 4 + [256] * 3
    )


def test_forward(model):
    x = torch.rand(1, 2048, 64)
    t = torch.randint(0, 1000, (1,)).long()
    assert model(x, t).shape == x.shape


def test_forward_timestep_scalar(model):
    x = torch.rand(4, 2048, 64)
    t = torch.arange(0, 1000).long()[100]
    assert model(x, t).shape == x.shape


def test_forward_timestep_2d(model):
    B, S, D = 2, 2048, 64
    x = torch.rand(B, S, D)
    t = torch.randint(0, 1000, (B, S)).long()
    assert model(x, t).shape == x.shape
