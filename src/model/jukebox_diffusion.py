import os
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from diffusers import DPMSolverMultistepScheduler, SchedulerMixin
from einops import rearrange
from transformers import JukeboxVQVAEConfig, JukeboxVQVAE

from src.diffusion.pipeline import SequencePipeline
from src.module.lr_scheduler.warmup import WarmupScheduler


class UnconditionalJukeboxDiffusion(pl.LightningModule):
    SAMPLE_RATE = 44100

    def __init__(
            self,
            model: torch.nn.Module,
            jukebox_embedding_lvl: int = 0,
            lr: float = 1e-4,
            lr_warmup_steps: int = 1000,
            weight_decay: float = 1e-2,
            beta_start: float = 0.02,
            beta_end: float = 1e-4,
            beta_schedule: str = "linear",
            num_train_timesteps: int = 1000,
            num_inference_steps: int = 50,
            inference_batch_size: int = 1,
            inference_seq_len: int = 2048,
            noise_scheduler: Optional[SchedulerMixin] = None,
            jukebox_latent_scaling_factor: float = 10.0,
            *args,
            **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "noise_scheduler"])
        self.model = model

        if noise_scheduler is None:
            self.noise_scheduler = DPMSolverMultistepScheduler(
                beta_start=beta_start,
                beta_end=beta_end,
                beta_schedule=beta_schedule,
                num_train_timesteps=num_train_timesteps,
                solver_order=3,
                prediction_type="epsilon",
            )
        else:
            self.noise_scheduler = noise_scheduler

        self.jukebox_vqvae = None
        self.lr_scheduler = None

    def prepare_data(self) -> None:
        self.jukebox_vqvae = self.load_jukebox_vqvae(os.environ["JUKEBOX_VQVAE_PATH"])

    def forward(self, x):
        """Computes the loss

        Args:
            x (B, T, C): input sequence
        """
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(x, dtype=x.dtype, device=x.device).float()
        bsz = x.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bsz,), device=x.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_x = self.noise_scheduler.add_noise(x, noise, timesteps)
        noise_pred = self.model(noisy_x, timesteps)
        loss = F.mse_loss(noise_pred, noise, reduction="mean")
        return loss

    def training_step(self, batch, batch_idx):
        loss = self(batch)
        self.log_dict({
            "train/loss": loss,
            "train/lr": self.lr_scheduler.get_last_lr()[0],
        })

        if self.logger and batch_idx == 0 and self.current_epoch % 40 == 0:
            audio = self.decode(batch[:self.hparams.inference_batch_size])
            self.log_audio(audio, "train", f"epoch_{self.current_epoch}_batch_{batch_idx}")

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self(batch)
        self.log("val/loss", loss)
        return loss

    def validation_epoch_end(self, outputs) -> None:
        if self.logger:
            seed = torch.randint(0, 1000000, (1,)).item()

            embeddings = self.generate_unconditionally(
                batch_size=self.hparams.inference_batch_size,
                seq_len=self.hparams.inference_seq_len,
                seed=seed,
            )

            audio = self.decode(embeddings)
            self.log_audio(audio, "generated", f"epoch_{self.current_epoch}_seed_{seed}")

        return super().validation_epoch_end(outputs)

    def log_audio(self, audio, key, caption=None):
        self.logger.experiment.log({
            f"audio/{key}": [wandb.Audio(a, sample_rate=self.SAMPLE_RATE, caption=caption) for a in audio.cpu()],
        })

    def load_jukebox_vqvae(self, vae_path):
        print("Loading Jukebox VAE")
        config = JukeboxVQVAEConfig.from_pretrained("openai/jukebox-1b-lyrics")
        vae = JukeboxVQVAE(config)
        vae.load_state_dict(torch.load(vae_path, map_location=self.device))
        vae.eval().to(self.device)
        return vae

    @torch.no_grad()
    def decode(self, embeddings):
        embeddings = rearrange(embeddings, "b t c -> b c t")
        # Use only lowest level
        decoder = self.jukebox_vqvae.decoders[self.hparams.jukebox_embedding_lvl]
        de_quantised_state = decoder([embeddings * self.hparams.jukebox_latent_scaling_factor], all_levels=False)
        de_quantised_state = de_quantised_state.permute(0, 2, 1)
        return de_quantised_state

    @torch.no_grad()
    def quantize_and_decode(self, embeddings):
        """Quantize the embeddings with the VQ-VAE first and then decode them"""
        embeddings = rearrange(embeddings, "b t c -> b c t")
        music_tokens = self.jukebox_vqvae.bottleneck.level_blocks[self.hparams.jukebox_embedding_lvl].encode(embeddings)
        latent_states = self.jukebox_vqvae.bottleneck.level_blocks[self.hparams.jukebox_embedding_lvl].decode(
            music_tokens)
        latent_states = rearrange(latent_states, "b c t -> b t c")
        return self.decode(latent_states)

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        # don't return, handled manually in optimizer step
        self.lr_scheduler = WarmupScheduler(optim, warmup_steps=self.hparams.lr_warmup_steps)
        return optim

    def optimizer_step(
            self,
            epoch: int,
            batch_idx: int,
            optimizer,
            optimizer_idx: int = 0,
            optimizer_closure=None,
            on_tpu: bool = False,
            using_native_amp: bool = False,
            using_lbfgs: bool = False,
    ) -> None:
        optimizer.step(closure=optimizer_closure)
        self.lr_scheduler.step()

    def generate_unconditionally(self, batch_size=1, seq_len=2048, num_inference_steps=None, seed=None):
        if num_inference_steps is None:
            num_inference_steps = self.hparams.num_inference_steps
        generator = torch.Generator().manual_seed(seed) if seed is not None else None

        pipeline = SequencePipeline(
            unet=self.model,
            scheduler=self.noise_scheduler,
        ).to(self.device)

        jukebox_latents = pipeline(
            generator=generator,
            batch_size=batch_size,
            seq_len=seq_len,
            num_inference_steps=num_inference_steps,
        )
        return jukebox_latents
