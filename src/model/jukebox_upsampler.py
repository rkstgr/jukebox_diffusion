import os
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchaudio
from diffusers import SchedulerMixin, PNDMScheduler
from einops import rearrange
from pytorch_lightning.loggers import WandbLogger
from transformers import JukeboxVQVAEConfig, JukeboxVQVAE

from src.diffusion.pipeline.upsampling_pipeline import UpsamplingPipeline
from src.diffusion.timestep_sampler.constant_sampler import TimeConstantSampler
from src.diffusion.timestep_sampler.diffusion_timestep_sampler import DiffusionTimestepSampler
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
            weight_decay: float = 1e-2,
            num_inference_steps: int = 50,
            inference_batch_size: int = 1,
            noise_scheduler: Optional[SchedulerMixin] = None,
            timestep_sampler: Optional[DiffusionTimestepSampler] = None,
            prompt_batch_idx: int = 0,
            log_train_audio: bool = False,
            skip_audio_logging: bool = False,
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

        self.jukebox_vqvae = self.load_jukebox_vqvae(os.environ["JUKEBOX_VQVAE_PATH"])
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
        noise_pred = self.model(noisy_x, timesteps, cond)

        # compute between noise & noise_pred only where timesteps > 0
        loss = F.mse_loss(
            noise_pred[torch.where(timesteps > 0)],
            noise[torch.where(timesteps > 0)]
        )

        return loss

    def training_step(self, batch, batch_idx):
        source = self.encode(batch, self.hparams.source_lvl)
        target = self.encode(batch, self.hparams.target_lvl)

        loss = self(target, source)
        self.log_dict({
            "train/loss": loss,
            "train/lr": self.lr_scheduler.get_last_lr()[0],
        }, sync_dist=True)

        if self.logger and batch_idx == 0 and self.current_epoch % 100 == 0 and self.hparams.log_train_audio:
                source_audio = self.decode(source[:self.hparams.inference_batch_size], self.hparams.source_lvl)
                target_audio = self.decode(target[:self.hparams.inference_batch_size], self.hparams.target_lvl)
                
                self.log_audio(source_audio, f"train/lvl{self.hparams.source_lvl}", f"epoch_{self.current_epoch}_source")
                self.log_audio(target_audio, f"train/lvl{self.hparams.target_lvl}", f"epoch_{self.current_epoch}_target")
                del source_audio, target_audio

        return loss

    def validation_step(self, batch, batch_idx):
        source = self.encode(batch, self.hparams.source_lvl)
        target = self.encode(batch, self.hparams.target_lvl)

        loss = self(target, source)
        self.log("val/loss", loss, sync_dist=True)

        if self.logger and batch_idx == self.hparams.prompt_batch_idx:
            source_audio = self.decode(source[:self.hparams.inference_batch_size], self.hparams.source_lvl)
            target_audio = self.decode(target[:self.hparams.inference_batch_size], self.hparams.target_lvl)

            self.log_audio(batch[:self.hparams.inference_batch_size], "val/original", f"epoch_{self.current_epoch}")
            self.log_audio(source_audio, f"val/lvl{self.hparams.source_lvl}", f"epoch_{self.current_epoch}_source")
            self.log_audio(target_audio, f"val/lvl{self.hparams.target_lvl}", f"epoch_{self.current_epoch}_target")

            return dict(
                source=source,
                target=target,
            )

    def validation_epoch_end(self, outputs) -> None:
        if self.logger:
            seed = torch.randint(0, 1000000, (1,)).item()

            embeddings = self.generate_upsample(
                source=outputs[0]["source"],
                target_shape=outputs[0]["target"].shape,
                seed=seed,
            )
            assert embeddings.shape == outputs[0]["target"].shape, f"Generated embeddings shape mismatch: {embeddings.shape} != {outputs[0]['target'].shape}"

            audio = self.decode(embeddings, self.hparams.target_lvl)
            self.log_audio(audio, "val/upsampled", f"epoch_{self.current_epoch}_seed_{seed}")

        return super().validation_epoch_end(outputs)

    def sample_timesteps(self, x_shape: torch.Size):
        return self.timestep_sampler.sample_timesteps(x_shape)

    def log_audio(self, audio, key, caption=None):
        if self.hparams.skip_audio_logging:
            return
        if isinstance(self.logger, WandbLogger):
            import wandb
            self.logger.experiment.log({
                f"audio/{key}": [wandb.Audio(a, sample_rate=self.SAMPLE_RATE, caption=caption) for a in audio.cpu()],
            })
        else:
            audio = torch.clamp(audio, -1, 1).cpu()
            for i, a in enumerate(audio):
                path = os.path.join(self.logger.save_dir, "audio", key, f"{caption}_batch{i}.wav")
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                torchaudio.save(filepath=path, src=a, sample_rate=self.SAMPLE_RATE)

    def load_jukebox_vqvae(self, vae_path):
        print("Loading Jukebox VAE")
        config = JukeboxVQVAEConfig.from_pretrained("openai/jukebox-1b-lyrics")
        vae = JukeboxVQVAE(config)
        vae.load_state_dict(torch.load(vae_path, map_location=self.device))
        vae.eval().to(self.device)
        return vae

    @staticmethod
    def preprocess(batch: torch.Tensor) -> torch.Tensor:
        return batch / torch.tensor(8)

    @staticmethod
    def postprocess(batch: torch.Tensor) -> torch.Tensor:
        return batch * torch.tensor(8)

    @torch.no_grad()
    def encode(self, audio: torch.Tensor, lvl: int):
        audio = rearrange(audio, "b t c -> b c t")
        encoder = self.jukebox_vqvae.encoders[lvl]
        embeddings = encoder(audio)[-1]
        embeddings = rearrange(embeddings, "b c t -> b t c")
        embeddings = self.preprocess(embeddings)
        return embeddings

    @torch.no_grad()
    def decode(self, embeddings, lvl):
        embeddings = self.postprocess(embeddings)
        embeddings = rearrange(embeddings, "b t c -> b c t")
        # Use only lowest level
        decoder = self.jukebox_vqvae.decoders[lvl]
        with torch.no_grad():
            de_quantised_state = decoder([embeddings], all_levels=False)
        de_quantised_state = de_quantised_state.permute(0, 2, 1)
        return de_quantised_state

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

    def generate_upsample(self, source: torch.Tensor, target_shape: torch.Size, num_inference_steps=None, seed=None):
        if num_inference_steps is None:
            num_inference_steps = self.hparams.num_inference_steps
        generator = torch.Generator().manual_seed(seed) if seed is not None else None

        pipeline = UpsamplingPipeline(
            unet=self.model,
            scheduler=self.noise_scheduler,
        ).to(self.device)

        jukebox_latents = pipeline(
            source=source,
            target_shape=target_shape,
            generator=generator,
            num_inference_steps=num_inference_steps,
        )
        return jukebox_latents
