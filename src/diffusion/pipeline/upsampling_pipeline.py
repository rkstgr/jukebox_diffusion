from typing import Optional

import torch

from src.diffusion.pipeline import SequencePipeline


class UpsamplingPipeline(SequencePipeline):

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
            cond: torch.Tensor,
            generator: Optional[torch.Generator] = None,
            num_inference_steps: int = 50,
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
        batch_size = cond.shape[0]
        seq_len = cond.shape[1]

        # Sample gaussian noise to begin loop
        seq = torch.randn(
            (batch_size, seq_len, self.unet.output_dim),
            generator=generator,
        )
        seq = seq.to(self.device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            model_output = self.unet(seq, t, cond)

            # 2. predict previous mean of seq x_t-1 and add variance depending on eta
            # do x_t -> x_t-1
            seq = self.scheduler.step(model_output, t, seq, return_dict=True).prev_sample

        return seq
