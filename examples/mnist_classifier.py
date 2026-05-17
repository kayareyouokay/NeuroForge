"""Train a small MNIST MLP from raw IDX files.

Set MNIST_DIR to a directory containing the four gzip IDX files.  If the files
are absent, the script downloads them from the public mirror.
"""

from __future__ import annotations

import gzip
import os
import struct
import urllib.request
from pathlib import Path

import numpy as np

from neuroforge.nn import Linear, ReLU, Sequential, cross_entropy
from neuroforge.optim import AdamW
from neuroforge.training import Trainer

URL = "https://storage.googleapis.com/cvdf-datasets/mnist"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
}


def download_mnist(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for filename in FILES.values():
        target = root / filename
        if not target.exists():
            urllib.request.urlretrieve(f"{URL}/{filename}", target)


def read_images(path: Path, limit: int = 4096) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        _, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        data = np.frombuffer(handle.read(rows * cols * min(count, limit)), dtype=np.uint8)
    return (data.reshape(-1, rows * cols).astype(np.float32) / 255.0)[:limit]


def read_labels(path: Path, limit: int = 4096) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        _, _ = struct.unpack(">II", handle.read(8))
        return np.frombuffer(handle.read(limit), dtype=np.uint8).astype(np.int64)


def batches(x: np.ndarray, y: np.ndarray, batch_size: int = 64):
    order = np.random.permutation(len(x))
    for start in range(0, len(x), batch_size):
        idx = order[start : start + batch_size]
        yield x[idx], y[idx]


def main() -> None:
    root = Path(os.environ.get("MNIST_DIR", "data/mnist"))
    download_mnist(root)
    x = read_images(root / FILES["train_images"])
    y = read_labels(root / FILES["train_labels"])

    model = Sequential(Linear(784, 128), ReLU(), Linear(128, 10))
    trainer = Trainer(model, AdamW(model.parameters(), lr=2e-3), loss_fn=cross_entropy)
    for epoch in range(3):
        result = trainer.fit_epoch(batches(x, y))
        print(f"epoch={epoch + 1} loss={result.loss:.4f} accuracy={result.accuracy:.3f}")


if __name__ == "__main__":
    main()
