import os
from pathlib import Path
from typing import Optional
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

import wandb
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchaudio
from diffusers import SchedulerMixin, PNDMScheduler
from einops import rearrange
from pytorch_lightning.loggers import WandbLogger
from transformers import JukeboxVQVAEConfig, JukeboxVQVAE
from src.diffusion.pipeline.conditional_pipeline import ConditionalPipeline
from src.diffusion.pipeline.unconditional_pipeline import UnconditionalPipeline

from src.diffusion.pipeline.upsampling_pipeline import UpsamplingPipeline
from src.diffusion.timestep_sampler.constant_sampler import TimeConstantSampler
from src.diffusion.timestep_sampler.diffusion_timestep_sampler import DiffusionTimestepSampler
from src.model.jukebox_normalize import JukeboxNormalizer
from src.model.jukebox_vqvae import JukeboxVQVAEModel
from src.module.lr_scheduler.warmup import WarmupScheduler


class JukeboxDiffusionUpsampler(pl.LightningModule):
    SAMPLE_RATE = 44100

    def __init__(
            self,
            model: torch.nn.Module,
            source_lvl: int = 2,
            target_lvl: int = 1,
            lr: float = 1e-4,
            lr_warmup_steps: int = 1000,
            lr_cycle_steps: int = 100_000,
            num_inference_steps: int = 50,
            inference_batch_size: int = 1,
            noise_scheduler: Optional[SchedulerMixin] = None,
            timestep_sampler: Optional[DiffusionTimestepSampler] = None,
            source_normalizer_path: Optional[Path] = None,
            target_normalizer_path: Optional[Path] = None,
            source_dropout: float = 0.0,
            source_aug_noise_std: float = 0.0,
            guidance_scales: Optional[list] = None,
            prompt_batch_idx: int = 0,
            log_train_audio: bool = False,
            skip_audio_logging: bool = False,
            clip_embeddings: bool = True,
            *args,
            **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "noise_scheduler", "timestep_sampler", "*args", "**kwargs"])
        self.model = model

        if noise_scheduler is None:
            self.noise_scheduler = PNDMScheduler(
                beta_start=1e-4,
                beta_end=1e-2,
                beta_schedule="linear",
                num_train_timesteps=1000
            )
        else:
            self.noise_scheduler = noise_scheduler

        if timestep_sampler is None:
            self.timestep_sampler = TimeConstantSampler(max_timestep=self.noise_scheduler.num_train_timesteps)
        else:
            self.timestep_sampler = timestep_sampler

        self.vqvae = JukeboxVQVAEModel(device=self.device)
        for param in self.vqvae.parameters():
            param.requires_grad = False

        if source_normalizer_path:
            self.register_module("source_normalizer", JukeboxNormalizer(source_normalizer_path))
        else:
            self.source_normalizer = None
        if target_normalizer_path:
            self.register_module("target_normalizer", JukeboxNormalizer(target_normalizer_path))
        else:
            self.target_normalizer = None
        
        self.guidance_scales = [1.0] if guidance_scales is None else guidance_scales
        
        self.lr_scheduler = None

    def forward(self, x, cond):
        """Computes the loss

        Args:
            x (B, T, C): input sequence
        """
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(x, dtype=x.dtype, device=x.device).float()

        # Sample a random timestep for each image
        timesteps = self.sample_timesteps(x.shape).to(x.device)

        # Add noise to the latents according to the noise magnitude at each timestep
        noisy_x = self.noise_scheduler.add_noise(x, noise, timesteps)
        model_out = self.model(noisy_x, timesteps, cond)

        # compute between noise & noise_pred only where timesteps > 0
        loss = F.mse_loss(x, model_out)

        return loss

    def training_step(self, batch, batch_idx):
        source = self.encode(batch, self.hparams.source_lvl)
        target = self.encode(batch, self.hparams.target_lvl)

        # dropout and add noise
        if (self.hparams.source_dropout > 0) and torch.rand(1) < self.hparams.source_dropout:
            source = torch.zeros_like(source)
        elif self.hparams.source_noise_std > 0:
                source = source + torch.randn_like(source) * self.hparams.source_noise_std

        loss = self(target, source)
        self.log_dict({
            "train/loss": loss,
            "train/lr": self.lr_schedulers().get_lr()[0],
        }, sync_dist=True, prog_bar=True)

        if self.logger and self.current_epoch == 0 and batch_idx == 0:
            if os.environ.get("SLURM_JOB_ID"):
                    self.logger.experiment.config["SLURM_JOB_ID"] = os.environ.get("SLURM_JOB_ID")

        if self.logger and batch_idx == 0 and self.current_epoch % 100 == 0 and self.hparams.log_train_audio:
                source_audio = self.decode(source[:self.hparams.inference_batch_size], self.hparams.source_lvl)
                target_audio = self.decode(target[:self.hparams.inference_batch_size], self.hparams.target_lvl)
                
                self.log_audio(source_audio, f"train/lvl{self.hparams.source_lvl}", f"epoch_{self.current_epoch}")
                self.log_audio(target_audio, f"train/lvl{self.hparams.target_lvl}", f"epoch_{self.current_epoch}")

                del source_audio, target_audio

        return loss

    def validation_step(self, batch, batch_idx):
        source = self.encode(batch, self.hparams.source_lvl)
        target = self.encode(batch, self.hparams.target_lvl)

        loss = self(target, source)
        self.log("val/loss", loss, sync_dist=True)

        if self.logger and batch_idx == self.hparams.prompt_batch_idx:
            return dict(
                source=source,
                target=target,
            )

    def validation_epoch_end(self, outputs) -> None:
        if self.logger:
            seed = torch.randint(0, 1000000, (1,)).item()
            data = {
                    "epoch": self.current_epoch,
                    "seed": seed,
                    "source": self.decode(outputs[0]["source"], self.hparams.source_lvl),
                    "target": self.decode(outputs[0]["target"], self.hparams.target_lvl),
                }

            for guidance_scale in self.guidance_scales:

                embeddings = self.generate_upsample(
                    source=outputs[0]["source"],
                    target_seq_len=outputs[0]["target"].shape[1],
                    guidance_scale=guidance_scale,
                    seed=seed,
                    num_inference_steps=10 if self.current_epoch == 0 else self.hparams.num_inference_steps
                )

                data[f"target_guidance={guidance_scale}"] = self.decode(embeddings, self.hparams.target_lvl)

            self.log_audio_table("val/audio", data)
        return super().validation_epoch_end(outputs)

    def sample_timesteps(self, x_shape: torch.Size):
        return self.timestep_sampler.sample_timesteps(x_shape)

    def log_audio_table(self, key, audio_dict):
        table_data = [] # list of lists
        # get the first item in audio_dict that is a tensor and return batch size
        batch_size = None
        for k, v in audio_dict.items():
            if isinstance(v, torch.Tensor):
                batch_size = v.shape[0]
                break
        if batch_size is None:
            raise ValueError("No tensor found in audio_dict. Provide at least one tensor to log audio table.")

        def to_audio(a):
            return wandb.Audio(a.cpu(), sample_rate=self.SAMPLE_RATE)
        
        keys = list(audio_dict.keys())
        table_data = [] # rows, where each row is a list of values
        for i in range(batch_size):
            row = []
            for k in keys:
                v = audio_dict[k]
                if isinstance(v, torch.Tensor):
                    # we assume it's audio
                    row.append(to_audio(v[i]))
                else:
                    row.append(v)
            table_data.append(row)
                

        table = wandb.Table(columns=list(keys), data=table_data)
        self.logger.experiment.log({key: table})

    def log_audio(self, audio, key, caption=None):
        if self.hparams.skip_audio_logging:
            return
        if isinstance(self.logger, WandbLogger):
            import wandb
            self.logger.experiment.log({
                f"audio/{key}": [wandb.Audio(a, sample_rate=self.SAMPLE_RATE, caption=caption) for a in audio.cpu()],
            })
        # tensorboard logging
        else:
            audio = torch.clamp(audio, -1, 1).cpu()
            for i, a in enumerate(audio):
                path = os.path.join(self.logger.save_dir, "audio", key, f"{caption}_batch{i}.wav")
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                torchaudio.save(filepath=path, src=a, sample_rate=self.SAMPLE_RATE)

    @torch.no_grad()
    def encode(self, audio: torch.Tensor, lvl: int):
        normalizer = self.source_normalizer if lvl == self.hparams.source_lvl else self.target_normalizer

        embeddings = self.vqvae.encode(audio, lvl=lvl)
        if normalizer:
            embeddings = normalizer.normalize(embeddings)
            if self.hparams.clip_embeddings:
                embeddings = embeddings.clamp(-5, 5)
        return embeddings

    @torch.no_grad()
    def decode(self, embeddings, lvl):
        normalizer = self.source_normalizer if lvl == self.hparams.source_lvl else self.target_normalizer

        if normalizer:
            embeddings = normalizer.denormalize(embeddings)
        audio = self.vqvae.decode(embeddings, lvl=lvl)
        return audio

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams.lr
        )

        lr_scheduler = CosineAnnealingWarmupRestarts(
            optim, 
            first_cycle_steps=self.hparams.lr_cycle_steps, 
            cycle_mult=1.0, 
            max_lr=self.hparams.lr, 
            min_lr=1e-8, 
            warmup_steps=self.hparams.lr_warmup_steps, 
            gamma=0.5
        )

        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }

    def generate_upsample(self, source: torch.Tensor, target_seq_len: int, num_inference_steps=None, seed=None, guidance_scale=1.0):
        if num_inference_steps is None:
            num_inference_steps = self.hparams.num_inference_steps
        generator = torch.Generator().manual_seed(seed) if seed is not None else None

        pipeline = ConditionalPipeline(
            unet=self.model,
            scheduler=self.noise_scheduler,
        ).to(self.device)

        embeddings = pipeline(
            conditioning=source,
            guidance_scale=guidance_scale,
            seq_len=target_seq_len,
            generator=generator,
            num_inference_steps=num_inference_steps,
            clip=self.hparams.clip_embeddings,
        )
        assert embeddings.shape[1] == target_seq_len, f"Generated embeddings have wrong length. Expected {target_seq_len}, got {embeddings.shape[1]}"

        return embeddings
