import pytest
import torch
import torch.nn as nn
from diffusers import DDIMScheduler

from src.diffusion.pipeline import InpaintingPipeline
from src.diffusion.pipeline import UnconditionalPipeline


# Assume DDIMScheduler

class TestModel(nn.Module):
    def __init__(self, scheduler):
        super().__init__()
        self.scheduler = scheduler
        self.device = torch.device("cpu")
        self.output_dim = 2

    def forward(self, noisy_samples, t):
        original_samples = torch.ones_like(noisy_samples) * torch.tensor(0.5)

        sqrt_alpha_prod = (self.scheduler.alphas_cumprod[t] ** 0.5).unsqueeze(-1)
        sqrt_one_minus_alpha_prod = ((1 - self.scheduler.alphas_cumprod[t]) ** 0.5).unsqueeze(-1)

        noise = (noisy_samples - sqrt_alpha_prod * original_samples) / sqrt_one_minus_alpha_prod
        return noise


@pytest.fixture
def scheduler():
    return DDIMScheduler()


@pytest.fixture
def model(scheduler):
    return TestModel(scheduler)


def test_test_model(model, scheduler):
    x = torch.ones(4, 8) * torch.tensor(0.5)
    noise = torch.randn_like(x)
    t = torch.randint(0, len(scheduler.timesteps), (x.shape[0],))
    xt = scheduler.add_noise(x, noise, timesteps=t)

    assert torch.allclose(model(xt, t), noise)


def test_inpainting_pipeline(model, scheduler):
    pipeline = InpaintingPipeline(model, scheduler)
    prompt = torch.cat([torch.ones(1, 2, 2) * torch.tensor(0.5), torch.randn(1, 2, 2)], dim=1)
    target = torch.ones_like(prompt) * torch.tensor(0.5)

    x = pipeline(
        x=prompt,
        mask=torch.cat([torch.ones(1, 2), torch.zeros(1, 2)], dim=1),
        generator=torch.Generator().manual_seed(0),
        num_inference_steps=5,
    )

    assert torch.allclose(x, target)


def test_unconditional_pipeline(model, scheduler):
    pipeline = UnconditionalPipeline(model, scheduler)
    x = pipeline(
        generator=torch.Generator().manual_seed(0),
        batch_size=1,
        seq_len=4,
        num_inference_steps=5,
    )

    assert torch.allclose(x, torch.ones(1, 4, 2) * torch.tensor(0.5))
