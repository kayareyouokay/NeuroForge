"""Character-level language model using an LSTM."""

from __future__ import annotations

import numpy as np

from neuroforge import Tensor
from neuroforge.nn import LSTM, Embedding, Linear, Module, cross_entropy
from neuroforge.optim import AdamW

TEXT = "the quick brown fox jumps over the lazy dog. neural nets learn tiny rhythms. "


class CharLSTM(Module):
    def __init__(self, vocab_size: int, width: int = 48) -> None:
        super().__init__()
        self.embed = Embedding(vocab_size, width)
        self.lstm = LSTM(width, width)
        self.head = Linear(width, vocab_size)

    def forward(self, idx: np.ndarray) -> Tensor:
        x = self.embed(idx)
        out, _ = self.lstm(x)
        return self.head(out)


def main() -> None:
    chars = sorted(set(TEXT))
    stoi = {c: i for i, c in enumerate(chars)}
    encoded = np.array([stoi[c] for c in TEXT * 20], dtype=np.int64)
    model = CharLSTM(len(chars))
    opt = AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    block = 16
    for step in range(25):
        starts = np.random.randint(0, len(encoded) - block - 1, size=8)
        x = np.stack([encoded[s : s + block] for s in starts])
        y = np.stack([encoded[s + 1 : s + block + 1] for s in starts])
        logits = model(x)
        loss = cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 5 == 0:
            print(f"step={step} loss={loss.item():.3f}")


if __name__ == "__main__":
    main()
