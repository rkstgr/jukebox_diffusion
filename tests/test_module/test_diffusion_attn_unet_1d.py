import torch

from src.module.diffusion_attn_unet_1d import DiffusionAttnUnet1D


def test_diffusion_attn_unet_1d():
    model = DiffusionAttnUnet1D(
        io_channels=64,
        n_attn_layers=2,
        channel_sizes=[128] * 4 + [256] * 3
    )
    x = torch.rand(1, 2048, 64)
    t = torch.randint(0, 100, (1,)).long()
    assert model(x, t).shape == x.shape
