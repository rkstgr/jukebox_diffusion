import os

import torch
import torch.nn as nn
from einops import rearrange
from transformers import JukeboxVQVAEConfig
from transformers import JukeboxVQVAE

class JukeboxTokenizer(nn.Module):
    sample_rate = 44100
    
    def __init__(self, lvl: int, path=None, device="cpu") -> None:
        super().__init__()
        assert lvl in [0, 1, 2], "Token level must be 0, 1 or 2"
        self.lvl = lvl

        if path is None:
            path = os.environ["JUKEBOX_VQVAE_PATH"]
            
        print("Loading Jukebox VAE from", path)
        self.jukebox_vqvae = self._get_vqvae(path, device).eval().to(device)

    def _get_vqvae(self, vae_path, device="cpu"):
        config = JukeboxVQVAEConfig.from_pretrained("openai/jukebox-1b-lyrics")
        vae = JukeboxVQVAE(config)
        vae.load_state_dict(torch.load(vae_path, map_location=device))
        return vae
    
    @torch.no_grad()
    def encode(self, audio):
        audio = rearrange(audio, "b t c -> b c t")
        encoder = self.jukebox_vqvae.encoders[self.lvl].to(audio.device)
        embeddings = encoder(audio)[-1]
        music_tokens = self.jukebox_vqvae.bottleneck.level_blocks[self.lvl].encode(embeddings) # [B, L]
        return music_tokens

    @torch.no_grad()
    def decode(self, music_tokens):
        embeddings = self.jukebox_vqvae.bottleneck.level_blocks[self.lvl].decode(music_tokens)
        decoder = self.jukebox_vqvae.decoders[self.lvl].to(embeddings.device)
        audio = decoder([embeddings], all_levels=False)
        audio = audio.permute(0, 2, 1)
        return audio