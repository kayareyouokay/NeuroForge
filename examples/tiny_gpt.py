"""Tiny GPT-style next-token model built from NeuroForge primitives."""

from __future__ import annotations

import numpy as np

from neuroforge import Tensor
from neuroforge.nn import Embedding, Linear, Module, TransformerEncoderBlock, cross_entropy
from neuroforge.optim import AdamW


class TinyGPT(Module):
    def __init__(self, vocab_size: int, block_size: int, width: int = 64, heads: int = 4) -> None:
        super().__init__()
        self.block_size = block_size
        self.token = Embedding(vocab_size, width)
        self.pos = Embedding(block_size, width)
        self.blocks = [TransformerEncoderBlock(width, heads, dropout=0.0) for _ in range(2)]
        self.head = Linear(width, vocab_size)

    def forward(self, idx: np.ndarray) -> Tensor:
        batch, steps = idx.shape
        pos = np.arange(steps, dtype=np.int64)
        x = self.token(idx) + self.pos(pos).reshape(1, steps, -1)
        mask = np.triu(np.ones((steps, steps), dtype=np.float32), k=1) * -1e9
        attn_mask = Tensor(mask.reshape(1, 1, steps, steps))
        for block in self.blocks:
            x = block(x, mask=attn_mask)
        return self.head(x)


def main() -> None:
    text = "to be or not to be, that is the question\n" * 32
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=np.int64)
    model = TinyGPT(len(chars), block_size=24)
    opt = AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    for step in range(10):
        starts = np.random.randint(0, len(data) - 25, size=4)
        x = np.stack([data[s : s + 24] for s in starts])
        y = np.stack([data[s + 1 : s + 25] for s in starts])
        logits = model(x)
        loss = cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"step={step} loss={loss.item():.3f}")


if __name__ == "__main__":
    main()
