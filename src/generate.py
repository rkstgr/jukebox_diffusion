""" Details
CPU oder GPU

which checkpoint and model for unconditional, upsample1, upsample2

batch size

length (may be dominated by models)

total length: autoregressive generation

generation strategy

"""
import torch

from src.model.jukebox_diffusion import JukeboxDiffusion
from src.model.jukebox_upsampler import JukeboxDiffusionUpsampler
from src.model.jukebox_vqvae import JukeboxVQVAEModel
from src.module.diffusion_attn_unet_1d import DiffusionAttnUnet1D
from diffusers import DPMSolverMultistepScheduler
from src.diffusion.timestep_sampler import TimeConstantSampler
import torchaudio

# ... -> lvl2
unconditional_config = {
    "model_config": {
        "target_lvl": 2,
        "model": DiffusionAttnUnet1D(
            io_channels=64,
            n_attn_layers=6,
            channel_sizes=[128, 128, 128, 128,  256, 256, 256, 256, 512, 512]
        ),
        "lr": 1e-4,
        "lr_warmup_steps": 2000,
        "inference_batch_size": 8,
        "noise_scheduler": DPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            beta_start=1e-4,
            beta_end=0.02,
            beta_schedule="linear",
            solver_order=3,
            prediction_type="epsilon"),
        "timestep_sampler": TimeConstantSampler(max_timestep=1000),
        "num_inference_steps": 100,
        "generate_unconditional": True,
        "generate_continuation": False,
        "skip_audio_logging": False,
    },
    "ckeckpoint_path": "2023-02-01_11-19-37/checkpoints/last.ckpt",
}

# lvl2 -> lvl1
upsampler_lvl2_config = {
    "model_config": {
        "source_lvl": 2,
        "target_lvl": 1,
        "model": DiffusionAttnUnet1D(
            io_channels=64,
            cond_channels=64,
            n_attn_layers=6,
            channel_sizes=[128, 128, 128, 256, 256, 256, 256, 512, 512]
        ),
        "lr": 1e-4,
        "lr_warmup_steps": 2000,
        "num_train_timesteps": 1000,
        "inference_batch_size": 16,
        "noise_scheduler": DPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            beta_start=1e-4,
            beta_end=0.02,
            beta_schedule="linear",
            solver_order=3,
            prediction_type="epsilon"
        ),
        "timestep_sampler": TimeConstantSampler(max_timestep=1000),
        "num_inference_steps": 80,
        "skip_audio_logging": False,
    },
    "ckeckpoint_path": "2023-01-10_10-23-16/checkpoints/last.ckpt",
}

# lvl1 -> lvl0
upsampler_lvl1_config = {
    "model_config": {
        "source_lvl": 1,
        "target_lvl": 0,
        "model": DiffusionAttnUnet1D(
            io_channels=64,
            cond_channels=64,
            n_attn_layers=6,
            channel_sizes=[128, 128, 128, 256, 256, 256, 256, 512, 512]
        ),
        "lr": 1e-4,
        "lr_warmup_steps": 2000,
        "num_train_timesteps": 1000,
        "inference_batch_size": 16,
        "noise_scheduler": DPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            beta_start=1e-4,
            beta_end=0.02,
            beta_schedule="linear",
            solver_order=3,
            prediction_type="epsilon"
        ),
        "timestep_sampler": TimeConstantSampler(max_timestep=1000),
        "num_inference_steps": 80,
        "skip_audio_logging": False,
    },
    "ckeckpoint_path": "2023-01-11_11-48-05/checkpoints/last.ckpt",
}

def load_unconditional(config, device="cpu"):
    ckpt = torch.load(config["ckeckpoint_path"], map_location=device)
    state_dict = {k: v for k, v in ckpt['state_dict'].items() if not k.startswith('jukebox_vqvae')}
    model = JukeboxDiffusion(model=config["model"], load_vqvae=True)
    model.load_state_dict(state_dict=state_dict)
    model = model.to(device).eval()
    return model

def load_upsampler(config, device="cpu"):
    ckpt = torch.load(config["ckeckpoint_path"], map_location=device)
    state_dict = {k: v for k, v in ckpt['state_dict'].items() if not k.startswith('jukebox_vqvae')}
    model = JukeboxDiffusionUpsampler(model=config["model"], load_vqvae=False)
    model.load_state_dict(state_dict=state_dict)
    model = model.to(device).eval()
    return model

def generate_unconditional(batch_size=1, num_inference_steps=50, seed=None):
    unc = load_unconditional(unconditional_config)
    ups1 = load_upsampler(upsampler1_config)
    ups2 = load_upsampler(upsampler2_config)

    embeddings_lvl2 = unc.generate_unconditional(batch_size=batch_size, seq_len=2048, num_inference_steps=num_inference_steps, seed=seed)
    embeddings_lvl1 = ups1.generate_upsample(embeddings_lvl2, target_seq_len=2048*4, num_inference_steps=num_inference_steps, seed=seed)
    embeddings_lvl0 = ups2.generate_upsample(embeddings_lvl1, target_seq_len=2048*16, num_inference_steps=num_inference_steps, seed=seed)

    audio_lvl2 = unc.decode(embeddings_lvl2, lvl=2)
    audio_lvl1 = unc.decode(embeddings_lvl1, lvl=1)
    audio_lvl0 = unc.decode(embeddings_lvl0, lvl=0)

    return [embeddings_lvl0, embeddings_lvl1, embeddings_lvl2], [audio_lvl0, audio_lvl1, audio_lvl2]

def upsample(audio_file, from_lvl, to_lvl):
    assert 2 >= from_lvl > to_lvl >= 0

    vqvae = JukeboxVQVAEModel()

    from_lvls = range(from_lvl, to_lvl, -1)

    full_audio = torchaudio.load(audio_file)[0].squeeze(-1)
    embeddings = vqvae.encode(full_audio, lvl=from_lvl)

    for lvl in from_lvls:
        upsampler = load_upsampler(upsampler_lvl1_config if lvl == 1 else upsampler_lvl2_config)
        embeddings = upsampler.generate_upsample(embeddings, target_seq_len=embeddings.shape[1]*4, num_inference_steps=50)

    audio = vqvae.decode(embeddings, lvl=to_lvl)
    return audio



if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()
    embeddings, audio = generate_unconditional(batch_size=args.batch_size, num_inference_steps=args.inference_steps, seed=args.seed)

    os.makedirs("generated_audio", exist_ok=True)
    for lvl in range(3):
        for batch in audio[lvl]:
            torchaudio.save(f"generated_audio/audio_lvl{lvl}_batch{batch}.wav", src=audio[lvl][batch].cpu(), sample_rate=44100)
