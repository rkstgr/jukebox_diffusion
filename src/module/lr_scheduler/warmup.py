import torch


class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1, min_lr=1e-9):
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        min_lr = self.min_lr
        return [
            min_lr + (base_lr-min_lr) * min(1.0, self.last_epoch / self.warmup_steps) for base_lr in self.base_lrs
        ]
