from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass

import numpy as np
import torch as th
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from discrete_diffusions.reparam_absorbing_diffusion import \
    ReparamAbsorbingDiffusion

from src.model.jukebox_vqvae import JukeboxVQVAEModel
from src.module.integer_encoding import IntegerFourierEmbedding


class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.
    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """

    @abstractmethod
    def weights(self):
        """
        Get a numpy array of weights, one per diffusion step.
        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size, device):
        """
        Importance-sample timesteps for a batch.
        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = th.from_numpy(indices_np).long().to(device)
        weights_np = 1 / p[indices_np]
        weights = th.from_numpy(weights_np).float().to(device)
        return indices, weights

class UniformSampler(ScheduleSampler):
    def __init__(self, diffusion_steps):
        self.diffusion_steps = diffusion_steps
        self._weights = np.ones([diffusion_steps])

    def weights(self):
        return self._weights

@dataclass
class RDMArgs:
    # diffusion
    diffusion_type: str = "reparam_absorbing"
    q_sample_mode: str = "coupled"

    num_diffusion_timesteps: int = 50
    mask_idx: int = 2048
    reweighting_type: str = "linear"
    not_diffusing_special_sym: bool = True
    label_smoothing: float = 0.1

    # architucture
    encoder_layers: int = 6
    encoder_heads: int = 8
    encoder_dim: int = 256
    encoder_ff_dim: int = 1024
    encoder_dropout: float = 0.1

    token_dict_size: int = 2048
    max_token_seq_len: int = 8192


RDMDecoderOut = namedtuple(
    "RDMDecoderOut",
    ["output_tokens", "output_scores", "auxiliary_output", "attn", "step", "max_step", "history"],
)


class RDMEncoder(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.token_embedding = nn.Embedding(args.token_dict_size + 1, args.encoder_dim, padding_idx=args.mask_idx)
        self.positional_embedding = IntegerFourierEmbedding(args.encoder_dim, min_index=0, max_index=args.max_token_seq_len)

        self.backbone = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=args.encoder_dim,
                nhead=args.encoder_heads,
                dim_feedforward=args.encoder_ff_dim,
                dropout=args.encoder_dropout,
                activation="gelu"
            ),
            num_layers=args.encoder_layers,
            norm=nn.LayerNorm(args.encoder_dim)
        )

        self.linear = nn.Linear(args.encoder_dim, args.token_dict_size + 1) # music tokens + absorbing state token
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x_t):
        """
        Args:
            x_t (torch.Tensor): (batch_size, token_seq_len, 2049). Noised/corrupted music tokens one-hot encoded with masked_idx token.
        Returns:
            torch.Tensor: (batch_size, token_seq_len, 2049). Softmaxed logits over token ids.
        """
        x_t_emb = self.token_embedding(x_t)
        x_t_emb = x_t_emb + self.positional_embedding(x_t_emb)
        x_t_emb = self.backbone(x_t_emb)
        x_t_emb = self.linear(x_t_emb)
        x_t = self.softmax(x_t_emb)
        return x_t


class RDM(pl.LightningModule):
    def __init__(self, args) -> None:
        super().__init__()

        if not isinstance(args, RDMArgs):
            args = RDMArgs(**args)

        self.args: RDMArgs = args
        self.unk = args.mask_idx

        self.pad_id, self.bos_id, self.eos_id = args.token_dict_size+2, args.token_dict_size+2, args.token_dict_size+2

        self.diffusion = ReparamAbsorbingDiffusion(
            args.num_diffusion_timesteps,
            args.mask_idx, # absorbing state token idx
            self.args.reweighting_type,
            self.args.not_diffusing_special_sym,
            self.pad_id, self.bos_id, self.eos_id
        )
        self.time_sampler = UniformSampler(args.num_diffusion_timesteps)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=args.encoder_dim,
                nhead=args.encoder_heads,
                dim_feedforward=args.encoder_ff_dim,
                dropout=args.encoder_dropout,
                activation="gelu"
            ),
            num_layers=args.encoder_layers,
            norm=nn.LayerNorm(args.encoder_dim)
        )

        self.vqvae = JukeboxVQVAEModel(vae_path=args.vae_path, device=self.device)
        
    def forward(self, x):
        pass

    def tokenize_audio(self, batch):
        assert batch.ndim == 3 and batch.shape[-1] == 1, "batch must be (batch_size, audio_seq_len, 1)"
        x = self.vqvae.encode_quantized(batch)
        assert x.ndim == 2, "x must be (batch_size, token_seq_len)"
        assert x.max() < self.args.token_dict_size, "x must be in range [0, token_dict_size)"
        return x

    def _prepare_batch(self, batch):
        """
        Args:
            batch (torch.Tensor): (batch_size, audio_seq_len, 1). The batch of audio data in raw float waveform format.
        Returns:
            Dict[str, torch.Tensor]:
                - "x_0": (batch_size, token_seq_len, ). Original music tokens one-hot encoded one of {0, 1, ..., 2047}
                - "x_t": (batch_size, token_seq_len, 2048). Noised/corrupted music tokens one-hot encoded.
                -   "t": (batch_size, ). The diffusion time step for each batch.
        """
        x_0 = self.tokenize_audio(batch)

        non_special_sym_mask = x_0.ne(self.args.mask_idx+1)

        if self.args.q_sample_mode == "default":
            # we use 1 sample for the default sampling trick.
            num_q_samples = 1            
        elif self.args.q_sample_mode in ["coupled", "multi-sample", "multi-step"]:
            # we use 2 samples by default for these advanced sampling tricks,
            # but feel free to specify as you like.
            num_q_samples = 2
            x_0 = x_0.repeat(num_q_samples, 1)

        batch_size = x_0.shape[0]
        device = x_0.device
        
        if self.args.q_sample_mode == "coupled":
            t1, weight_t = self.time_sampler.sample(batch_size, device)
            t2, _ = self.time_sampler.sample(batch_size, device)
            x_t, x_0_ignore, mask, t = self.diffusion.q_sample_coupled(x_0=x_0, t1=t1, t2=t2, non_special_sym_mask=non_special_sym_mask) 
            weight_t = weight_t.repeat(num_q_samples)

        elif self.args.q_sample_mode == "multi-sample":
            rets = []
            t, weight_t = self.time_sampler.sample(batch_size, device)
            for _ in range(num_q_samples):
                x_t, x_0_ignore, mask = self.diffusion.q_sample(x_0=x_0, t=t, non_special_sym_mask=non_special_sym_mask)
                rets.append((t, weight_t, x_t, x_0_ignore, mask))
            t, weight_t, x_t, x_0_ignore, mask = map(lambda x: th.cat(x, dim=0), zip(*rets))

        elif self.args.q_sample_mode == "multi-step":
            rets = []
            for _ in range(num_q_samples):
                t, weight_t = self.time_sampler.sample(batch_size, device)
                x_t, x_0_ignore, mask = self.diffusion.q_sample(x_0=x_0, t=t, non_special_sym_mask=non_special_sym_mask)
                rets.append((t, weight_t, x_t, x_0_ignore, mask))
            t, weight_t, x_t, x_0_ignore, mask = map(lambda x: th.cat(x, dim=0), zip(*rets))

        elif self.args.q_sample_mode == "default":
            t, weight_t = self.time_sampler.sample(batch_size, device)
            x_t, x_0_ignore, mask = self.diffusion.q_sample(x_0=x_0, t=t, non_special_sym_mask=non_special_sym_mask)

        diffusion_dict = {
                "x_t" : x_t,
                "x_0_ignore" : x_0_ignore,
                "masks" : mask,
                "t": t,
                "weight_t": weight_t
            }

        encoder_out = self.encoder(
            x_t=diffusion_dict["x_t"],
            t=diffusion_dict["t"],
        ) # a tuple ([B, N, C], None) or ([B, N, C], [B, N])

        diffusion_dict["decoder_outputs"] = encoder_out
        diffusion_dict["x_0"] = x_0

        return diffusion_dict

    def forward_decoder(self, decoder_out, **kwargs):
        if self.diffusion is None:
            raise NotImplementedError("No diffusion decoding function is provided.")

        def denoising_fn(x_t, t):
            return self.encoder(
                x_t=x_t,
                t=t,
            )

        new_decoder_out = self.diffusion.sample_step(
            decoder_out,
            denoising_fn,
            **kwargs
        )
        return new_decoder_out

    def training_step(self, batch, batch_idx):
        diffusion_dict = self._prepare_batch(batch)

        diffusion_losses, logging_outputs = self.diffusion.compute_loss(
            inputs=diffusion_dict, 
            label_smoothing=self.args.label_smoothing,
        )

        loss_dict = {
            "train/loss": diffusion_losses["diffusion_loss"],
            "train/nll_loss": diffusion_losses.get("diffusion_nll_loss", None),
        }

        self.log_dict(loss_dict)
        
        return loss_dict

    def validation_step(self, batch, batch_idx):
        diffusion_dict = self._prepare_batch(batch)

        diffusion_losses, logging_outputs = self.diffusion.compute_loss(
            inputs=diffusion_dict, 
            label_smoothing=self.args.label_smoothing,
        )

        loss_dict = {
            "validation/loss": diffusion_losses["diffusion_loss"],
            "validation/nll_loss": diffusion_losses.get("diffusion_nll_loss", None),
        }

        self.log_dict(loss_dict)
    
    def validation_epoch_end(self, outputs: None) -> None:
        music_tokens = self.generate(batch_size=16, token_seq_len=4096, device=self.device)

        self.log_music_tokens(music_tokens, "validation/generated")

        return super().validation_epoch_end(outputs)

    def initialize_tokens(self, batch_size, token_seq_len):
        if self.args.diffusion_type in ['absorbing', 'reparam-absorbing']:
            # for masking diffusion types,
            # we start with a whole [M=Masked Index] sequence.
            initial_output_tokens = th.zeros(
                batch_size, token_seq_len
            ).fill_(self.unk)
        else:
            raise NotImplementedError

        assert initial_output_tokens.shape == (batch_size, token_seq_len)

        return initial_output_tokens

    def generate(self, batch_size, token_seq_len, device, max_iter=100, retain_history=False):
        # 1 initialize xt ~ q_noise
        initial_output_tokens = self.initialize_tokens(batch_size, token_seq_len).to(device)
        initial_output_scores = initial_output_tokens.new_zeros(*initial_output_tokens.size()).type_as(initial_output_tokens)
        initial_output_masks = initial_output_tokens.ne(self.bos_id)

        prev_decoder_out = RDMDecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            auxiliary_output={
                "output_masks": initial_output_masks,
            },
            attn=None,
            step=0,
            max_step=0,
            history=None,
        )

        prev_output_tokens = prev_decoder_out.output_tokens.clone()

        if retain_history:
            prev_decoder_out = prev_decoder_out._replace(history=[prev_output_tokens])

        for i in range(max_iter):
            prev_decoder_out = prev_decoder_out._replace(step=i, max_step=max_iter)

            decoder_out = self.forward_decoder(prev_decoder_out)

            prev_decoder_out = decoder_out._replace(
                output_tokens=decoder_out.output_tokens,
                output_scores=decoder_out.output_scores,
                auxiliary_output={k : decoder_out.auxiliary_output[k] for k in decoder_out.auxiliary_output},
                attn=decoder_out.attn
                if (decoder_out.attn is not None and decoder_out.attn.size(0) > 0)
                else None,
                history=[h for h in decoder_out.history]
                if decoder_out.history is not None
                else None,
            )

            prev_output_tokens = prev_decoder_out.output_tokens.clone()
        
        return prev_decoder_out.output_tokens


    def log_music_tokens(self, music_tokens: th.Tensor, description: str):
        # music_tokens: [B, L]
        music = self.tokenizer.decode(music_tokens)
        self.log_audio(music, description)

    def log_audio(self, audio, description, caption=""):
        # audio: [B, L]
        if isinstance(self.logger, WandbLogger):
            import wandb
            self.logger.experiment.log({
                f"audio/{description}": [wandb.Audio(a, sample_rate=self.SAMPLE_RATE, caption=caption) for a in audio.cpu()],
        })

