from __future__ import annotations

import math

from neuroforge.nn.activations import GELU
from neuroforge.nn.layers import Dropout, LayerNorm, Linear
from neuroforge.nn.module import Module, Sequential
from neuroforge.tensor import Tensor


class MultiHeadAttention(Module):
    """Scaled dot-product self-attention."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)
        self.dropout = Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        batch, steps, _ = x.shape
        q = self._split(self.q_proj(x), batch, steps)
        k = self._split(self.k_proj(x), batch, steps)
        v = self._split(self.v_proj(x), batch, steps)
        scores = (q @ k.swapaxes(-1, -2)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        attn = _softmax(scores, axis=-1)
        attn = self.dropout(attn)
        context = attn @ v
        context = context.transpose((0, 2, 1, 3)).reshape(batch, steps, self.embed_dim)
        return self.out_proj(context)

    def _split(self, x: Tensor, batch: int, steps: int) -> Tensor:
        return x.reshape(batch, steps, self.num_heads, self.head_dim).transpose((0, 2, 1, 3))


class TransformerEncoderBlock(Module):
    """Pre-norm Transformer encoder block."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.norm1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.drop1 = Dropout(dropout)
        self.norm2 = LayerNorm(embed_dim)
        self.ffn = Sequential(
            Linear(embed_dim, hidden),
            GELU(),
            Linear(hidden, embed_dim),
            Dropout(dropout),
        )

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = x + self.drop1(self.attn(self.norm1(x), mask=mask))
        return x + self.ffn(self.norm2(x))


def _softmax(x: Tensor, axis: int = -1) -> Tensor:
    shifted = x - x.data.max(axis=axis, keepdims=True)
    exps = shifted.exp()
    return exps / exps.sum(axis=axis, keepdims=True)
