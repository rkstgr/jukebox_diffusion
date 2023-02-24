import torch as th
import torch.nn as nn
import pytorch_lightning as pl
from discrete_diffusions.reparam_absorbing_diffusion import ReparamAbsorbingDiffusion
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np

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
    num_diffusion_timesteps: int = 1000
    mask_idx: int = 0
    reweighting_type: str = "none"
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

        if isinstance(args, dict):
            args = RDMArgs(**args)

        pad_id, bos_id, eos_id = args.token_dict_size+2, args.token_dict_size+2, args.token_dict_size+2

        self.diffusion = ReparamAbsorbingDiffusion(
            args.num_diffusion_timesteps,
            args.mask_idx, # absorbing state token idx
            self.args.reweighting_type,
            self.args.not_diffusing_special_sym,
            pad_id, bos_id, eos_id
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
        
    def forward(self, x):
        pass

    def _prepare_batch(self, batch):
        """
        Args:
            batch (torch.Tensor): (batch_size, audio_seq_len, 1). The batch of audio data in raw float waveform format.
        Returns:
            Dict[str, torch.Tensor]:
                - "x_0": (batch_size, token_seq_len, 2048). Original music tokens one-hot encoded.
                - "x_t": (batch_size, token_seq_len, 2048). Noised/corrupted music tokens one-hot encoded.
                -   "t": (batch_size, ). The diffusion time step for each batch.
        """
        x_0 = self.vqvae.encode_quantized(batch)
        
        non_special_sym_mask = x_0.ne(self.vqvae.pad_id)

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

    def training_step(self, batch, batch_idx):
        diffusion_dict = self._prepare_batch(batch)

        diffusion_losses, logging_outputs = self.diffusion.compute_loss(
            inputs=diffusion_dict, 
            label_smoothing=self.args.label_smoothing,
        )

        loss_dict = {
            "loss": diffusion_losses["diffusion_loss"],
            "nll_loss": diffusion_losses.get("diffusion_nll_loss", None),
        }
        
        return loss_dict