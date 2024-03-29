import os
from pathlib import Path
from typing import Optional, Dict
import itertools
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchaudio
from diffusers import SchedulerMixin, PNDMScheduler
from einops import rearrange
from pytorch_lightning.loggers import WandbLogger
import wandb
from src.diffusion.pipeline.acapella_pipeline import AcapellaPipeline

from src.model.jukebox_vqvae import JukeboxVQVAEModel
from src.model.jukebox_normalize import JukeboxNormalizer
from src.diffusion.pipeline.inpainting_pipeline import InpaintingPipeline
from src.diffusion.pipeline.unconditional_pipeline import UnconditionalPipeline
from src.diffusion.timestep_sampler.constant_sampler import TimeConstantSampler
from src.diffusion.timestep_sampler.diffusion_timestep_sampler import DiffusionTimestepSampler
from src.module.lr_scheduler.warmup import WarmupScheduler


class AcapellaDiffusion(pl.LightningModule):
    SAMPLE_RATE = 44100

    def __init__(
            self,
            model: torch.nn.Module,
            target_lvl: int = 2,
            lr: float = 1e-4,
            lr_warmup_steps: int = 1000,
            loss_fn: str = "mse",
            cond_dropout: float = 0.15, # leave out conditional information with this probability
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
            conditioning = None,
            *args,
            **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=[
                                  "model", "noise_scheduler", "timestep_sampler", "normalizer", "vqvae", "*args", "**kwargs"])
        self.model = model

        self.singer_embedding = torch.nn.Embedding(262, 10, padding_idx=0)
        self.language_embedding = torch.nn.Embedding(17, 4, padding_idx=0)
        self.gender_embedding = torch.nn.Embedding(3, 2, padding_idx=0)

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
            print(
                f"Number of training timesteps: {self.noise_scheduler.num_train_timesteps}")

        if timestep_sampler is None:
            self.timestep_sampler = TimeConstantSampler(
                max_timestep=self.noise_scheduler.num_train_timesteps)
            print(
                f"Using constant timestep sampler with max timestep {self.noise_scheduler.num_train_timesteps}")
        else:
            self.timestep_sampler = timestep_sampler

        if normalizer_path is not None:
            self.register_module(
                "normalizer", JukeboxNormalizer(normalizer_path))
        else:
            self.normalizer = None

        if load_vqvae:
            self.vqvae = JukeboxVQVAEModel(device=self.device)
            # freeze vqvae
            for param in self.vqvae.parameters():
                param.requires_grad = False

        self.lr_scheduler = None

        # conditioining is Dict with gender, language, singer and a list of values
        # produce all combinations of conditioning
        self.conditioning = []
        for v in itertools.product(
            conditioning.get("gender", [""]),
            conditioning.get("language", [""]),
            conditioning.get("singer", [""]),
        ):
            self.conditioning.append(
                {
                    "gender": v[0],
                    "language": v[1],
                    "singer": v[2],
                }
            )


    def forward(self, x, cond):
        """Computes the loss

        Args:
            x (B, T, C): input sequence
            cond (dict): conditioning information
        """
        # prepare conditioning
        singer_id = cond["singer_id"].long()
        language_id = cond["language_id"].long()
        gender_id = cond["gender_id"].long()

        singer_embedding = self.singer_embedding(singer_id)
        language_embedding = self.language_embedding(language_id)
        gender_embedding = self.gender_embedding(gender_id)

        cond_emb = torch.cat([gender_embedding, language_embedding, singer_embedding
                             ], dim=1).unsqueeze(1)

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(x, dtype=x.dtype, device=x.device).float()

        # Sample a random timestep for each image
        timesteps = self.sample_timesteps(x.shape).to(x.device)

        # Add noise to the latents according to the noise magnitude at each timestep
        noisy_x = self.noise_scheduler.add_noise(x, noise, timesteps)
        model_output = self.model(noisy_x, timesteps, cond_emb)

        prediction_type = self.noise_scheduler.prediction_type
        loss_fn = F.mse_loss if self.hparams.loss_fn == "mse" else F.l1_loss
        if prediction_type == "sample":
            loss = loss_fn(
                x,
                model_output,
            )
        elif prediction_type == "epsilon":
            loss = loss_fn(
                noise,
                model_output,
            )
        elif prediction_type == "v-prediction":
            raise NotImplementedError

        return loss

    def training_step(self, batch, batch_idx):
        x = self.encode(batch["audio"], debug=batch_idx == 1)

        # dropout conditioning
        cond = {"singer_id": batch["singer_id"],
                "language_id": batch["language_id"], "gender_id": batch["gender_id"]}

        if self.hparams.cond_dropout > 0:
            rand = torch.rand(1).item()
            if rand < self.hparams.cond_dropout:
                cond = {"singer_id": torch.zeros_like(batch["singer_id"]),
                        "language_id": torch.zeros_like(batch["language_id"]),
                        "gender_id": torch.zeros_like(batch["gender_id"])}
        
        loss = self(x, cond)
        self.log_dict({
            "train/loss": loss,
            "train/lr": self.lr_schedulers().get_lr()[0],
        }, sync_dist=True)

        if self.logger and self.current_epoch == 0 and batch_idx == 0:
            if os.environ.get("SLURM_JOB_ID"):
                if self.logger.experiment.config.get("SLURM_JOB_ID") is None:
                    self.logger.experiment.config.update({"SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID")})
                else:
                    # append
                    new_job_id = self.logger.experiment.config.get("SLURM_JOB_ID") + "," + os.environ.get("SLURM_JOB_ID")
                    self.logger.experiment.config.update({"SLURM_JOB_ID": new_job_id}, allow_val_change=True)


        if self.logger and batch_idx == 0 and self.current_epoch % 100 == 0 and self.hparams.log_train_audio:
            with torch.no_grad():
                audio = self.decode(x[:self.hparams.inference_batch_size])
                self.log_audio(
                    audio, "train", f"epoch_{self.current_epoch}_batch_{batch_idx}")
                del audio

        return loss

    def validation_step(self, batch, batch_idx):
        x = self.encode(batch["audio"], debug=batch_idx == 0)
        cond = {"singer_id": batch["singer_id"],
                "language_id": batch["language_id"], "gender_id": batch["gender_id"]}
        loss = self(x, cond)
        self.log("val/loss", loss, sync_dist=True)
        if self.logger and batch_idx == self.hparams.prompt_batch_idx:
            audio = self.decode(x[:self.hparams.inference_batch_size])
            self.log_audio(audio, "val", f"epoch_{self.current_epoch}")
            return x

    def validation_epoch_end(self, outputs) -> None:
        if self.logger and self.hparams.generate_unconditional:
            seed = torch.randint(0, 1000000, (1,)).item()
            audio = []
            guidance_scales = self.hparams.guidance_scales
            for guidance in guidance_scales:
                embeddings = self.generate_conditionally(
                    conditioning=self.conditioning,
                    guidance_scale=guidance,
                    seq_len=outputs[0].shape[1],
                    seed=seed,
                )
                audio.append(self.decode(embeddings, debug=True))
            self.log_table(audio, guidance_scales, self.conditioning, "val/generated")

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
            self.log_audio(audio, "val/continuation",
                           f"epoch_{self.current_epoch}_seed_{seed}")
            # log original prompt
            audio = self.decode(prompt)
            self.log_audio(audio, "val/prompt",
                           f"epoch_{self.current_epoch}_seed_{seed}")

        return super().validation_epoch_end(outputs)

    def sample_timesteps(self, x_shape: torch.Size):
        return self.timestep_sampler.sample_timesteps(x_shape)

    def log_table(self, audios, guidance_scales, conditioning, key):
        def to_audio(a):
            return wandb.Audio(a.float().cpu(), sample_rate=self.SAMPLE_RATE)
        

        table_data = []
        for i in range(len(conditioning)):
            row = [conditioning[i]["gender"], conditioning[i]["language"], conditioning[i]["singer"]]
            for a, g in zip(audios, guidance_scales):
                row.append(to_audio(a[i]))
            table_data.append(row)

        table = wandb.Table(columns=["gender", "language", "singer", *[f"guidance={g}" for g in guidance_scales]], data=table_data)
        self.logger.experiment.log({
            key: table
        })

    def log_audio(self, audio, key, caption=None):
        if self.hparams.skip_audio_logging:
            return
        if isinstance(self.logger, WandbLogger):
            import wandb
            self.logger.experiment.log({
                f"audio/{key}": [wandb.Audio(a, sample_rate=self.SAMPLE_RATE, caption=caption) for a in audio.float().cpu()],
            })
        else:
            audio = torch.clamp(audio, -1, 1).float().cpu()
            for i, a in enumerate(audio):
                path = os.path.join(self.logger.save_dir,
                                    "audio", key, f"{caption}_batch{i}.wav")
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                torchaudio.save(filepath=path, src=a,
                                sample_rate=self.SAMPLE_RATE)

    @torch.no_grad()
    def encode(self, audio: dict, lvl=None, debug=False):
        if lvl is None:
            lvl = self.hparams.target_lvl

        embeddings = self.vqvae.encode(audio, lvl=lvl)
        if debug:
            print(f"Embeddings: {embeddings.shape} | Mean: {embeddings.mean().item():.4f} | Std: {embeddings.std().item():.4f} | Min: {embeddings.min().item():.4f} | Max: {embeddings.max().item():.4f}")
        if self.normalizer is not None:
            embeddings = self.normalizer.normalize(embeddings)
            if debug:
                print(f"(normalized) | Mean: {embeddings.mean().item():.4f} | Std: {embeddings.std().item():.4f} | Min: {embeddings.min().item():.4f} | Max: {embeddings.max().item():.4f}")
            embeddings.clamp_(-5, 5)
        return embeddings

    @torch.no_grad()
    def decode(self, embeddings, lvl=None, debug=False):
        if lvl is None:
            lvl = self.hparams.target_lvl

        if debug:
            print(f"Decode: {embeddings.shape} | Mean: {embeddings.mean().item():.4f} | Std: {embeddings.std().item():.4f} | Min: {embeddings.min().item():.4f} | Max: {embeddings.max().item():.4f}")

        if self.normalizer is not None:
            embeddings = self.normalizer.denormalize(embeddings)
            if debug:
                print(f"(denormalized) | Mean: {embeddings.mean().item():.4f} | Std: {embeddings.std().item():.4f} | Min: {embeddings.min().item():.4f} | Max: {embeddings.max().item():.4f}")

        return self.vqvae.decode(embeddings, lvl=lvl)

    def configure_optimizers(self):
        # model parameters and embeddings

        optim = torch.optim.Adam(
            list(self.parameters()) + 
            list(self.singer_embedding.parameters()) +
            list(self.language_embedding.parameters()) +
            list(self.gender_embedding.parameters()),
            lr=self.hparams.lr,
        )

        lr_scheduler = CosineAnnealingWarmupRestarts(
            optim, 
            first_cycle_steps=self.hparams.lr_cycle_steps, 
            cycle_mult=1.0, 
            max_lr=self.hparams.lr, 
            min_lr=self.hparams.lr/10, 
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

    def generate_conditionally(self, conditioning, guidance_scale=1.0, seq_len=2048, num_inference_steps=None, seed=None):
        if num_inference_steps is None:
            num_inference_steps = self.hparams.num_inference_steps
        generator = torch.Generator().manual_seed(seed) if seed is not None else None

        pipeline = AcapellaPipeline(
            unet=self.model,
            scheduler=self.noise_scheduler,
            singer_embedding=self.singer_embedding,
            language_embedding=self.language_embedding,
            gender_embedding=self.gender_embedding,
        ).to(self.device)

        jukebox_latents = pipeline(
            conditioning=conditioning, # len of conditioning dictates batch size
            guidance_scale=guidance_scale,
            generator=generator,
            seq_len=seq_len,
            num_inference_steps=num_inference_steps
        )
        return jukebox_latents
