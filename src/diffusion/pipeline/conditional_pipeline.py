import os
from typing import Optional, Tuple, Union, List, Dict

import torch
from src.dataset.acapella_dataset import AcapellaLanguage, AcapellaGender

from src.diffusion.pipeline import SequencePipeline


class AcapellaPipeline(SequencePipeline):
    def __init__(self, unet, scheduler, gender_embedding, language_embedding, singer_embedding, *args, **kwargs):
        super().__init__()
        self.unet = unet
        self.scheduler = scheduler
        self.singer_embedding = singer_embedding
        self.language_embedding = language_embedding
        self.gender_embedding = gender_embedding
        self.language_tokenizer = AcapellaLanguage
        self.gender_tokenizer = AcapellaGender

    @property
    def config(self):
        return {"unet": self.unet, "scheduler": self.scheduler}

    @torch.no_grad()
    def __call__(
            self,
            conditioning: Union[Dict[str, str], List[Dict[str, str]]],
            seq_len: int,
            guidance_scale: float = 1.0,
            generator: Optional[torch.Generator] = None,
            num_inference_steps: int = 50,
            clip: bool = True,
            **kwargs,
    ):
        r"""
        Args:
            context: The context to condition on. Dict with keys (gender, language, singer) and values
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
        if not isinstance(conditioning, list):
            conditioning = [conditioning]

        if any(cond.get("singer", "") != "" for cond in conditioning):
            raise NotImplementedError("Singer conditioning not implemented yet.")

        gender_id = torch.tensor([self.gender_tokenizer.get_id(cond.get(
            "gender", "")) for cond in conditioning]).long().to(self.device)
        language_id = torch.tensor([self.language_tokenizer.get_id(cond.get(
            "language", "")) for cond in conditioning]).long().to(self.device)
        singer_id = torch.zeros_like(language_id).to(self.device)

        singer_embedding = self.singer_embedding(singer_id)
        language_embedding = self.language_embedding(language_id)
        gender_embedding = self.gender_embedding(gender_id)

        # concatenate embeddings
        context = torch.cat([gender_embedding, language_embedding, singer_embedding
                             ], dim=1).unsqueeze(1)

        batch_size = len(conditioning)

        def report_stats(sample, desc=""):
            mean, std, smin, smax = torch.mean(sample).item(), torch.std(
                sample).item(), torch.min(sample).item(), torch.max(sample).item()
            print(
                f"\n[{desc}] | mean: {mean:.3f}, std: {std:.3f}, min: {smin:.3f}, max: {smax:.3f}")

        # Sample gaussian noise to begin loop
        seq = torch.randn(
            (batch_size, seq_len, self.unet.output_dim),
            generator=generator,
        )
        seq = seq.to(self.device)
        report_stats(seq, "Initial")

        unknown_context = torch.zeros_like(context) # this is only valid as long as the unknown ids are embedded with zeros

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            if guidance_scale == 1.0:
                # guidance of 1 means no guidance/ normal conditional diffusion
                seq_input = seq
                context_input = context
            else:
                # classifier free guidance needs two forward passes
                # We concat unconditional and conditional diffusion into a single batch
                seq_input = torch.cat([seq, seq], dim=0)
                context_input = torch.cat([unknown_context, context], dim=0)

            model_output = self.unet(seq_input, t, context_input)
            if guidance_scale != 1.0:
                model_output_uncond, model_output_cond = model_output.chunk(2)
                model_output = model_output_uncond + guidance_scale * \
                    (model_output_cond - model_output_uncond)

            seq = self.scheduler.step(
                model_output, t, seq, return_dict=True).prev_sample
            if clip:
                seq = torch.clip(seq, -5, 5)
            report_stats(seq, f"Step {t}")

        return seq
