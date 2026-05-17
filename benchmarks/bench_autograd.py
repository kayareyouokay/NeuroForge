from __future__ import annotations

import argparse
import time

import numpy as np

from neuroforge import Tensor
from neuroforge.nn import Linear, ReLU, Sequential, mse_loss
from neuroforge.optim import AdamW


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    model = Sequential(Linear(args.width, args.width), ReLU(), Linear(args.width, args.width))
    opt = AdamW(model.parameters(), lr=1e-3)
    x = Tensor(np.random.randn(args.batch, args.width).astype(np.float32))
    y = Tensor(np.random.randn(args.batch, args.width).astype(np.float32))

    start = time.perf_counter()
    for _ in range(args.steps):
        loss = mse_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    elapsed = time.perf_counter() - start
    print(f"{args.steps / elapsed:.2f} steps/s ({elapsed:.3f}s total)")


if __name__ == "__main__":
    main()
