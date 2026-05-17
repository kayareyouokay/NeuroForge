from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from neuroforge.nn.losses import cross_entropy
from neuroforge.nn.module import Module
from neuroforge.optim.optimizer import Optimizer
from neuroforge.tensor import Tensor
from neuroforge.training.metrics import Accuracy, MeanMetric

Batch = tuple[np.ndarray, np.ndarray]


@dataclass
class TrainResult:
    loss: float
    accuracy: float | None


class Trainer:
    """Small training harness for NumPy-backed supervised experiments."""

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        loss_fn: Callable[[Tensor, np.ndarray], Tensor] = cross_entropy,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def fit_epoch(self, batches: Iterable[Batch], *, track_accuracy: bool = True) -> TrainResult:
        self.model.train()
        loss_metric = MeanMetric()
        acc = Accuracy() if track_accuracy else None
        for x_np, y_np in batches:
            logits = self.model(Tensor(x_np))
            loss = self.loss_fn(logits, y_np)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_metric.update(loss.item(), len(y_np))
            if acc is not None:
                acc.update(logits, y_np)
        return TrainResult(loss_metric.compute(), acc.compute() if acc is not None else None)

    def evaluate(self, batches: Iterable[Batch]) -> TrainResult:
        self.model.eval()
        loss_metric = MeanMetric()
        acc = Accuracy()
        for x_np, y_np in batches:
            logits = self.model(Tensor(x_np))
            loss = self.loss_fn(logits, y_np)
            loss_metric.update(loss.item(), len(y_np))
            acc.update(logits, y_np)
        return TrainResult(loss_metric.compute(), acc.compute())
