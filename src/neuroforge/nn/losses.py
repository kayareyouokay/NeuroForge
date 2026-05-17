from __future__ import annotations

import numpy as np

from neuroforge.nn.module import Module
from neuroforge.tensor import Tensor


def mse_loss(prediction: Tensor, target: Tensor) -> Tensor:
    return ((prediction - target) ** 2).mean()


def cross_entropy(logits: Tensor, target: np.ndarray | Tensor) -> Tensor:
    """Mean sparse categorical cross-entropy over the last dimension."""

    labels = (
        target.data.astype(np.int64)
        if isinstance(target, Tensor)
        else np.asarray(target, dtype=np.int64)
    )
    shifted = logits - logits.data.max(axis=-1, keepdims=True)
    exp = shifted.exp()
    probs = exp / exp.sum(axis=-1, keepdims=True)
    flat_probs = probs.reshape(-1, probs.shape[-1])
    flat_labels = labels.reshape(-1)
    picked = flat_probs[np.arange(flat_labels.size), flat_labels]
    return -(picked.log()).mean()


def binary_cross_entropy(prediction: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
    clipped = prediction * (1.0 - 2.0 * eps) + eps
    return -(target * clipped.log() + (1.0 - target) * (1.0 - clipped).log()).mean()


class MSELoss(Module):
    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        return mse_loss(prediction, target)


class CrossEntropyLoss(Module):
    def forward(self, logits: Tensor, target: np.ndarray | Tensor) -> Tensor:
        return cross_entropy(logits, target)
