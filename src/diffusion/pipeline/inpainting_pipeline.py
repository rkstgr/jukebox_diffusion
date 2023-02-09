from typing import Optional

import torch
from diffusers import DDIMScheduler
from einops import repeat
from tqdm import tqdm


class InpaintingPipeline:

    def __init__(self, unet, scheduler, *args, **kwargs):
        """
        Note: Works currently only for noise based diffusion models.

        :param unet:
        :param scheduler:
        :param args:
        :param kwargs:
        """
        super().__init__()
        self.unet = unet
        self.scheduler: DDIMScheduler = scheduler
        self.device = self.unet.device
    #
    # @property
    # def config(self):
    #     return {"unet": self.unet, "scheduler": self.scheduler}

    @torch.no_grad()
    def __call__(
            self,
            x: torch.Tensor,
            mask: torch.Tensor,
            generator: Optional[torch.Generator] = None,
            num_inference_steps: int = 50,
            **kwargs,
    ):
        r"""
        Args:
            x (B, S, D): Input tensor.
            mask (B, S): The mask to use for inpainting. 1 for keeping the context, 0 for generating.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality sequence at the
                expense of slower inference.
            return_complete_seq (`bool`, *optional*, defaults to False):
                If True, the complete sequence is returned. Otherwise, only the predicted sequence is returned.

        Returns:
            `torch.Tensor`: The generated sequence.
        """
        assert x.dim() == 3, "Input tensor must be of shape (B, S, D)."
        assert x.shape[:2] == mask.shape[:2], "x and mask must have the same batch and sequence dimensions."

        batch_size = x.shape[0]

        seq = torch.randn(
            x.shape,
            generator=generator,
        ).to(self.device)

        mask = repeat(mask, "b s -> b s d", d=x.shape[-1]).to(self.device)

        seq = x * mask + seq * (1 - mask)
        prediction = seq[torch.where(mask == 0)].view(batch_size, -1, x.shape[-1])

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in tqdm(self.scheduler.timesteps):
            # Build the timesteps
            timesteps = (
                    t * (1 - mask[:, :, 0]) * torch.ones(batch_size, x.shape[1], device=self.device)    
            ).long()
            sequence_output = self.unet(seq, timesteps)

            # Set the noise to zero for the context
            prediction_output = sequence_output[torch.where(mask == 0)].view(batch_size, -1, x.shape[-1])
            prediction = self.scheduler.step(prediction_output, t, prediction).prev_sample

            seq[torch.where(mask == 0)] = prediction.flatten()

        return seq
