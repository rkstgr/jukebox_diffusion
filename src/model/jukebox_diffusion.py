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

from src.model.jukebox_vqvae import JukeboxVQVAEModel
from src.model.jukebox_normalize import JukeboxNormalizer
from src.diffusion.pipeline.inpainting_pipeline import InpaintingPipeline
from src.diffusion.pipeline.unconditional_pipeline import UnconditionalPipeline
from src.diffusion.timestep_sampler.constant_sampler import TimeConstantSampler
from src.diffusion.timestep_sampler.diffusion_timestep_sampler import DiffusionTimestepSampler
from src.module.lr_scheduler.warmup import WarmupScheduler


class JukeboxDiffusion(pl.LightningModule):
    SAMPLE_RATE = 44100

    def __init__(
            self,
            model: torch.nn.Module,
            target_lvl: int = 2,
            lr: float = 1e-4,
            lr_warmup_steps: int = 1000,
            num_inference_steps: int = 50,
            inference_batch_size: int = 1,
            noise_scheduler: Optional[SchedulerMixin] = None,
            timestep_sampler: Optional[DiffusionTimestepSampler] = None,
            normalizer_path: Optional[Path] = None,
            generate_unconditional: bool = True,
            generate_continuation: bool = False,
            prompt_batch_idx: int = 0,
            log_train_audio: bool = False,
            skip_audio_logging: bool = False,
            weight_decay: float = 0.01,
            load_vqvae: bool = True,
            *args,
            **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "noise_scheduler", "timestep_sampler", "normalizer", "vqvae", "*args", "**kwargs"])
        self.model = model

        if noise_scheduler is None:
            print("Using default noise scheduler")
            self.noise_scheduler = PNDMScheduler(
                beta_start=1e-4,
                beta_end=1e-2,
                beta_schedule="linear",
                num_train_timesteps=1000
            )
        else:
            print("Using custom noise scheduler")
            self.noise_scheduler = noise_scheduler
            print(f"Number of training timesteps: {self.noise_scheduler.num_train_timesteps}")

        if timestep_sampler is None:
            self.timestep_sampler = TimeConstantSampler(max_timestep=self.noise_scheduler.num_train_timesteps)
            print(f"Using constant timestep sampler with max timestep {self.noise_scheduler.num_train_timesteps}")
        else:
            self.timestep_sampler = timestep_sampler

        if normalizer_path is not None:
            self.register_module("normalizer", JukeboxNormalizer(normalizer_path))
        else:
            self.normalizer = None

        if load_vqvae:
            self.vqvae = JukeboxVQVAEModel(device=self.device)
            # freeze vqvae
            for param in self.vqvae.parameters():
                param.requires_grad = False

        self.lr_scheduler = None

    def forward(self, x):
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
        model_output = self.model(noisy_x, timesteps)

        # compute between noise & noise_pred only where timesteps > 0
        loss = F.mse_loss(
            model_output,
            noise,
        )

        return loss

    def training_step(self, batch, batch_idx):
        target = self.encode(batch, debug=batch_idx == 0)
        loss = self(target)
        self.log_dict({
            "train/loss": loss,
            "train/lr": self.lr_scheduler.get_last_lr()[0],
        }, sync_dist=True)

        if self.logger and batch_idx == 0 and self.current_epoch % 100 == 0 and self.hparams.log_train_audio:
            with torch.no_grad():
                audio = self.decode(target[:self.hparams.inference_batch_size])
                self.log_audio(audio, "train", f"epoch_{self.current_epoch}_batch_{batch_idx}")
                del audio

        return loss

    def validation_step(self, batch, batch_idx):
        x = self.encode(batch, debug=batch_idx == 0)
        loss = self(x)
        self.log("val/loss", loss, sync_dist=True)
        if self.logger and batch_idx == self.hparams.prompt_batch_idx:
            audio = self.decode(x[:self.hparams.inference_batch_size])
            self.log_audio(audio, "val", f"epoch_{self.current_epoch}")
            return x

    def validation_epoch_end(self, outputs) -> None:
        if self.logger and self.hparams.generate_unconditional:
            seed = torch.randint(0, 1000000, (1,)).item()

            embeddings = self.generate_unconditionally(
                batch_size=self.hparams.inference_batch_size,
                seq_len=outputs[0].shape[1],
                seed=seed,
            )
            audio = self.decode(embeddings, debug=True)
            self.log_audio(audio, "unconditional", f"epoch_{self.current_epoch}_seed_{seed}")

        if self.logger and len(outputs) > 0 and self.hparams.generate_continuation:
            seed = torch.randint(0, 1000000, (1,)).item()
            sample = outputs[0]
            seq_len = sample.shape[1]
            prompt = sample[:self.hparams.inference_batch_size, :seq_len // 2]
            embeddings = self.generate_continuation(
                prompt=prompt,
                seed=seed,
                num_inference_steps=self.hparams.num_inference_steps,
            )
            audio = self.decode(embeddings)
            self.log_audio(audio, "val/continuation", f"epoch_{self.current_epoch}_seed_{seed}")
            # log original prompt
            audio = self.decode(prompt)
            self.log_audio(audio, "val/prompt", f"epoch_{self.current_epoch}_seed_{seed}")

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

    @torch.no_grad()
    def encode(self, audio: torch.Tensor, lvl=None, debug=False):
        if lvl is None:
            lvl = self.hparams.target_lvl

        embeddings = self.vqvae.encode(audio, lvl=lvl)
        if debug:
            print("Embeddings shape (before norm)", embeddings.shape)
            print("  Mean:", embeddings.mean().item())
            print("  Std :", embeddings.std().item())
            print("  Min :", embeddings.min().item())
            print("  Max :", embeddings.max().item())
        if self.normalizer is not None:
            embeddings = self.normalizer.normalize(embeddings)
            if debug:
                print("Mean (after norm):", embeddings.mean())
                print("Std  (after norm):", embeddings.std())
                print("Min  (after norm):", embeddings.min())
                print("Max  (after norm):", embeddings.max())
            embeddings = embeddings.clamp(-5, 5)

        return embeddings

    @torch.no_grad()
    def decode(self, embeddings, lvl=None, debug=False):
        if lvl is None:
            lvl = self.hparams.target_lvl

        if debug:
            print("** Decode shape:", embeddings.shape)
            print("Mean (before denorm):", embeddings.mean())
            print("Std  (before denorm):", embeddings.std())
            print("Min  (before denorm):", embeddings.min())
            print("Max  (before denorm):", embeddings.max())
            
        if self.normalizer is not None:
            embeddings = self.normalizer.denormalize(embeddings)
            if debug:
                print("Mean (after denorm):", embeddings.mean())
                print("Std  (after denorm):", embeddings.std())
                print("Min  (after denorm):", embeddings.min())
                print("Max  (after denorm):", embeddings.max())

        return self.vqvae.decode(embeddings, lvl=lvl)

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

    def generate_continuation(self, prompt: torch.Tensor, seed=None, num_inference_steps=50):
        generator = torch.Generator().manual_seed(seed) if seed is not None else None

        pipeline = InpaintingPipeline(
            unet=self.model,
            scheduler=self.noise_scheduler,
        )
        # double the sequence length to allow for the prompt
        context = torch.cat([prompt, torch.zeros_like(prompt)], dim=1)

        # mask the second half of prompt
        mask = torch.ones(context.shape[:2])
        start = prompt.shape[1]
        mask[:, start:] = 0

        return pipeline(
            x=context,
            mask=mask,
            seed=seed,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )

    def generate_unconditionally(self, batch_size=1, seq_len=2048, num_inference_steps=None, seed=None):
        if num_inference_steps is None:
            num_inference_steps = self.hparams.num_inference_steps
        generator = torch.Generator().manual_seed(seed) if seed is not None else None

        pipeline = UnconditionalPipeline(
            unet=self.model,
            scheduler=self.noise_scheduler,
        ).to(self.device)

        jukebox_latents = pipeline(
            generator=generator,
            batch_size=batch_size,
            seq_len=seq_len,
            num_inference_steps=num_inference_steps
        )
        return jukebox_latents
