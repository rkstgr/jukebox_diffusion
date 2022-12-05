import pytest
import torch

from src.model.jukebox_diffusion import UnconditionalJukeboxDiffusion
from src.module.diffusion_attn_unet_1d import DiffusionAttnUnet1D


@pytest.fixture
def model():
    yield UnconditionalJukeboxDiffusion(
        model=DiffusionAttnUnet1D(
            io_channels=64,
            channel_sizes=[128, 128, 128, 128, 256],
            n_attn_layers=2
        ),
    )


def test_jukebox_diffusion_forward(model):
    x = torch.randn(2, 2048, 64)
    loss = model(x)
    assert loss > 0


def test_jukebox_diffusion_unconditional_generation(model):
    model.eval()
    x = model.generate_unconditionally(batch_size=1, seq_len=2048, num_inference_steps=20)
    assert x.shape == (1, 2048, 64)
