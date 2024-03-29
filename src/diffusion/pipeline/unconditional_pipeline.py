from typing import Optional

import torch

from src.diffusion.pipeline import SequencePipeline


class UnconditionalPipeline(SequencePipeline):

    def __init__(self, unet, scheduler, *args, **kwargs):
        super().__init__()
        self.unet = unet
        self.scheduler = scheduler

    @property
    def config(self):
        return {"unet": self.unet, "scheduler": self.scheduler}

    @torch.no_grad()
    def __call__(
            self,
            seq_len: int,
            batch_size: int = 1,
            generator: Optional[torch.Generator] = None,
            num_inference_steps: int = 50,
            clip: bool = True,
            **kwargs,
    ):
        r"""
        Args:
            seq_len: The length of the sequence to generate.
            batch_size (`int`, *optional*, defaults to 1):
                The number of sequences to generate.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality sequence at the
                expense of slower inference.

        Returns:
            `torch.Tensor`: The generated sequence.
        """

        def report_stats(sample, desc=""):
            mean, std, smin, smax = torch.mean(sample).item(), torch.std(sample).item(), torch.min(sample).item(), torch.max(sample).item()
            print(f"\n[{desc}] | mean: {mean:.3f}, std: {std:.3f}, min: {smin:.3f}, max: {smax:.3f}")

        # Sample gaussian noise to begin loop
        seq = torch.randn(
            (batch_size, seq_len, self.unet.output_dim),
            generator=generator,
        )
        seq = seq.to(self.device)

        report_stats(seq, "Initial")

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            model_output = self.unet(seq, t)

            # 2. predict previous mean of seq x_t-1 and add variance depending on eta
            # do x_t -> x_t-1
            seq = self.scheduler.step(model_output, t, seq, return_dict=True).prev_sample
            seq = torch.clip(seq, -5, 5)
            report_stats(seq, f"Step {t}")

        return seq
