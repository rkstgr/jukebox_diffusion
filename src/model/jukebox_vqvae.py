import os

import torch
import torch.nn as nn
from einops import rearrange
from transformers import JukeboxVQVAEConfig
from transformers import JukeboxVQVAE

class JukeboxVQVAEModel:
    def __init__(self, vae_path=None, device="cpu") -> None:
        super().__init__()
        if vae_path is None:
            vae_path = os.environ["JUKEBOX_VQVAE_PATH"]
            
        print("Loading Jukebox VAE from", vae_path)
        self.jukebox_vqvae = self.get_vqvae(vae_path, device).eval().to(device)

    def get_vqvae(self, vae_path, device="cpu"):
        config = JukeboxVQVAEConfig.from_pretrained("openai/jukebox-1b-lyrics")
        vae = JukeboxVQVAE(config)
        vae.load_state_dict(torch.load(vae_path, map_location=device))
        return vae
    
    @torch.no_grad()
    def encode(self, audio, lvl):
        audio = rearrange(audio, "b t c -> b c t")
        encoder = self.jukebox_vqvae.encoders[lvl].to(audio.device)
        embeddings = encoder(audio)[-1]
        embeddings = rearrange(embeddings, "b c t -> b t c")
        return embeddings

    @torch.no_grad()
    def decode(self, embeddings, lvl):
        embeddings = rearrange(embeddings, "b t c -> b c t")
        decoder = self.jukebox_vqvae.decoders[lvl].to(embeddings.device)
        audio = decoder([embeddings], all_levels=False)
        audio = audio.permute(0, 2, 1)
        return audio
        
    @torch.no_grad()
    def quantize_and_decode(self, embeddings):
        """Quantize the embeddings with the VQ-VAE first and then decode them"""
        embeddings = rearrange(embeddings, "b t c -> b c t")
        music_tokens = self.jukebox_vqvae.bottleneck.level_blocks[self.hparams.target_lvl].encode(embeddings)
        latent_states = self.jukebox_vqvae.bottleneck.level_blocks[self.hparams.target_lvl].decode(
            music_tokens)
        latent_states = rearrange(latent_states, "b c t -> b t c")
        return self.decode(latent_states)


if __name__ == "__main__":
    vqvae = JukeboxVQVAEModel()