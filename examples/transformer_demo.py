"""Transformer encoder classification demo on synthetic sequence labels."""

from __future__ import annotations

import numpy as np

from neuroforge import Tensor
from neuroforge.nn import Embedding, Linear, Module, TransformerEncoderBlock, cross_entropy
from neuroforge.optim import Adam


class SequenceClassifier(Module):
    def __init__(self, vocab: int, width: int = 32) -> None:
        super().__init__()
        self.embed = Embedding(vocab, width)
        self.block = TransformerEncoderBlock(width, num_heads=4, dropout=0.0)
        self.head = Linear(width, 2)

    def forward(self, idx: np.ndarray) -> Tensor:
        x = self.block(self.embed(idx))
        pooled = x.mean(axis=1)
        return self.head(pooled)


def main() -> None:
    rng = np.random.default_rng(123)
    x = rng.integers(0, 32, size=(64, 12), dtype=np.int64)
    y = (x[:, :6].sum(axis=1) > x[:, 6:].sum(axis=1)).astype(np.int64)
    model = SequenceClassifier(vocab=32)
    opt = Adam(model.parameters(), lr=2e-3)
    for epoch in range(8):
        logits = model(x)
        loss = cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"epoch={epoch} loss={loss.item():.3f}")


if __name__ == "__main__":
    main()
