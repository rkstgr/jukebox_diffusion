from src.model.jukebox_normalize import JukeboxNormalizer
import torch
from einops import rearrange

import os
import pytest

def test_normalizer():
    normalizer = JukeboxNormalizer()

    x = torch.randn(512, 16, 128, 4).to(torch.float64)

    iter_x = iter(x)

    mean, std = normalizer.compute_stats_iter(iter_x)

    assert torch.allclose(mean, torch.mean(x, dim=(0, 1, 2))), "Mean is not correct"
    assert torch.allclose(std, torch.std(x, dim=(0, 1, 2)), rtol=1e-3), "Standard deviation is not correct"


def test_normalizer_save():
    normalizer = JukeboxNormalizer()

    x = torch.randn(512, 16, 128, 4).to(torch.float64)

    iter_x = iter(x)

    mean, std = normalizer.compute_stats_iter(iter_x)

    normalizer.save_stats("tests/test_model/test_jukebox_normalize.pt", (mean, std))
    normalizer.load_settings("tests/test_model/test_jukebox_normalize.pt")

    assert torch.allclose(normalizer.mean, mean), "Mean is not correct"
    assert torch.allclose(normalizer.std, std, rtol=1e-3), "Standard deviation is not correct"

    os.remove("tests/test_model/test_jukebox_normalize.pt")


def test_normalizer_normalize_denormalize():
    normalizer = JukeboxNormalizer()

    x = torch.randn(512, 1024, 4).to(torch.float64)

    mean, std = normalizer.compute_stats(x)

    normalizer.mean = mean
    normalizer.std = std

    x_norm = normalizer.normalize(x)

    x_norm_mean = torch.mean(x_norm, dim=(0, 1))
    x_norm_std = torch.std(x_norm, dim=(0, 1))

    assert torch.allclose(x_norm_mean, torch.zeros_like(x_norm_mean)), "Mean is not correct"
    assert torch.allclose(x_norm_std, torch.ones_like(x_norm_std), rtol=1e-3), "Standard deviation is not correct"

    x_denorm = normalizer.denormalize(x_norm)

    assert torch.allclose(x, x_denorm), "Denormalization is not correct"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_normalize_cuda():
    normalizer = JukeboxNormalizer("config/normalizations/maestro_all_lvl_1.pt").to("cuda")

    x = torch.randn(512, 1024, 64).to("cuda")

    x_norm = normalizer.normalize(x)
    x_denorm = normalizer.denormalize(x_norm)

    assert torch.allclose(x, x_denorm), "Normalization is not invertible"