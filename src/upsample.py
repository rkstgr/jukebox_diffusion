import torch

from src.model.jukebox_diffusion import JukeboxDiffusion
from src.model.jukebox_upsampler import JukeboxDiffusionUpsampler
from src.model.jukebox_vqvae import JukeboxVQVAEModel
from src.module.diffusion_attn_unet_1d import DiffusionAttnUnet1D
from diffusers import DPMSolverMultistepScheduler
from src.diffusion.timestep_sampler import TimeConstantSampler
import torchaudio

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

def load_upsampler(config, device="cpu"):
    ckpt = torch.load("logs/train/runs/"+config["ckeckpoint_path"], map_location=device)
    # state_dict = {k: v for k, v in ckpt['state_dict'].items() if not k.startswith('jukebox_vqvae')}
    model = JukeboxDiffusionUpsampler(model=config["model_config"]["model"], load_vqvae=False)
    model.load_state_dict(state_dict=ckpt["state_dict"], strict=False)
    model = model.to(device).eval()
    return model

def upsample(audio_file, from_lvl, to_lvl, device="cpu"):
    assert 2 >= from_lvl > to_lvl >= 0

    upsamplers = {
    }
    from_lvls = range(from_lvl, to_lvl, -1)
    if 2 in from_lvls:
        upsamplers[2] = load_upsampler(upsampler_lvl2_config).to(device)
    if 1 in from_lvls:
        upsamplers[1] = load_upsampler(upsampler_lvl1_config).to(device)

    vqvae = JukeboxVQVAEModel(device=device)
    full_audio, sr = torchaudio.load(audio_file)
    full_audio = full_audio.unsqueeze(0).transpose(1, 2).to(device)

    embeddings = vqvae.encode(full_audio, lvl=from_lvl)
    print("Embeddings", embeddings.shape)

    to_lvl_embeddings = []
    parts = torch.split(embeddings, 1024, dim=1)
    for n_part, part in enumerate(parts):
        for lvl in from_lvls:
            print(f"Part {n_part+1}/{len(parts)}: Upsampling from lvl {lvl} to lvl {to_lvl}")
            part = upsamplers[lvl].generate_upsample(part, target_seq_len=part.shape[1]*4, num_inference_steps=50)
        to_lvl_embeddings.append(part)

    embeddings = torch.cat(to_lvl_embeddings, dim=1)
    audio = vqvae.decode(embeddings, lvl=to_lvl).squeeze(-1)

    return audio



if __name__ == "__main__":
    import argparse
    import os
    from tqdm import tqdm
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", type=str)
    parser.add_argument("--from_lvl", type=int, default=2)
    parser.add_argument("--to_lvl", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("Upsampling audio from lvl", args.from_lvl, "to lvl", args.to_lvl, "on", args.device, "device")

    path = Path(args.audio_path)
    if not path.exists():
        raise ValueError(f"File {path} does not exist")
    
    if path.is_file():
        audio = upsample(args.audio_path, args.from_lvl, args.to_lvl, args.device)
        print("audio", audio.shape)
        torchaudio.save(os.path.splitext(args.audio_path)[0] + f"_lvl{args.to_lvl}.wav", audio.cpu(), 44100)
    
    else:
        for file in tqdm(path.glob("*.wav")):
            audio = upsample(file, args.from_lvl, args.to_lvl, args.device)
            torchaudio.save(os.path.splitext(file)[0] + f"_lvl{args.to_lvl}.wav", audio.cpu(), 44100)


