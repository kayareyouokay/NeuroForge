"""CIFAR-shaped convolution smoke training on synthetic data."""

from __future__ import annotations

import numpy as np

from neuroforge import Tensor
from neuroforge.nn import Conv2D, Linear, ReLU, Sequential, cross_entropy
from neuroforge.optim import Adam


class TinyCNN(Sequential):
    def __init__(self) -> None:
        super().__init__(Conv2D(3, 8, 3, padding=1), ReLU(), Conv2D(8, 8, 3, padding=1), ReLU())
        self.head = Linear(8 * 32 * 32, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = super().forward(x)
        return self.head(x.reshape(x.shape[0], -1))


def main() -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(size=(16, 3, 32, 32)).astype(np.float32)
    y = rng.integers(0, 10, size=(16,), dtype=np.int64)
    model = TinyCNN()
    opt = Adam(model.parameters(), lr=1e-3)
    for step in range(5):
        logits = model(Tensor(x))
        loss = cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"step={step} loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
