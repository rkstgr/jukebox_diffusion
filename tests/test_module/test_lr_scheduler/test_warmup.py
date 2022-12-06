import torch

from src.module.lr_scheduler.warmup import WarmupScheduler


def test_warmup_scheduler():
    final_lr = 1e-3

    optim = torch.optim.Adam([torch.nn.Parameter(torch.randn(1))], final_lr)
    scheduler = WarmupScheduler(optim, warmup_steps=1000)
    optim.step()  # just to suppress torch warning
    # warmup
    for i in range(1000):
        scheduler.step()
        assert round(scheduler.get_lr()[0], 8) == round(1e-3 * (i + 1) / 1000, 8)
    # stay constant
    for i in range(1000):
        scheduler.step()
        assert round(scheduler.get_lr()[0], 8) == final_lr
