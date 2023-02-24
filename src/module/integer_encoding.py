import math

import torch
import torch.nn as nn


class IntegerFourierEmbedding(nn.Module):
    def __init__(self, emb_dim: int, min_index: int = 0, max_index: int = 1000):
        """
        Args:
            min_index: Smallest index (incl.)
            max_index:
        """
        super().__init__()
        self.min_index = min_index
        self.max_index = max_index
        max_len = max_index - min_index

        position = torch.arange(min_index, max_index + 1).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2) * (-math.log(float(max_len)) / emb_dim))
        pe = torch.zeros(max_len + 1, emb_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe: [Sequence, 1, d_model]
        self.register_buffer("pe", pe)

    def forward(self, idx):
        """
        Args:
            idx: Long Tensor, shape [...]

        Returns:
            Tensor, shape [..., emb_dim]
        """
        idx_flatten = idx.flatten()
        assert torch.all(
            idx_flatten >= self.min_index
        ), f"One of the given indices is smaller than min_index, set min_index to at least ({torch.min(idx_flatten)})"
        assert torch.all(
            idx_flatten < self.max_index
        ), f"One of the given indices is greater than max_index, set max_index to at least ({torch.max(idx_flatten) + 1})"

        pe = self.pe[idx_flatten + self.min_index]
        return pe.view(*idx.shape, self.pe.shape[1])

    def apply_1d(self, x, dim=-1):
        """
        Args:
            x: Tensor [...], Input for which positional encoding should be computed
            dim: int. Dimension of x with size D for which positional encoding should be computed
        Returns:
            Tensor [D, emb_dim]
        """
        t = torch.arange(x.shape[dim], device=x.device)
        return self(t)
