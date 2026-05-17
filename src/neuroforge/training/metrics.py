from __future__ import annotations

import numpy as np

from neuroforge.tensor import Tensor


class MeanMetric:
    def __init__(self) -> None:
        self.reset()

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    def compute(self) -> float:
        return self.total / max(1, self.count)

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0


class Accuracy:
    def __init__(self) -> None:
        self.reset()

    def update(self, logits: Tensor, target: np.ndarray) -> None:
        pred = logits.data.argmax(axis=-1).reshape(-1)
        labels = target.reshape(-1)
        self.correct += int((pred == labels).sum())
        self.count += labels.size

    def compute(self) -> float:
        return self.correct / max(1, self.count)

    def reset(self) -> None:
        self.correct = 0
        self.count = 0
