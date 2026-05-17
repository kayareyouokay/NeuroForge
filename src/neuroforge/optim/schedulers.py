from __future__ import annotations

import math

from neuroforge.optim.optimizer import Optimizer


class StepLR:
    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1) -> None:
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.epoch = 0

    def step(self) -> float:
        self.epoch += 1
        if self.epoch % self.step_size == 0:
            self.optimizer.lr *= self.gamma
        return self.optimizer.lr


class ExponentialLR:
    def __init__(self, optimizer: Optimizer, gamma: float) -> None:
        self.optimizer = optimizer
        self.gamma = gamma

    def step(self) -> float:
        self.optimizer.lr *= self.gamma
        return self.optimizer.lr


class CosineAnnealingLR:
    def __init__(self, optimizer: Optimizer, max_steps: int, min_lr: float = 0.0) -> None:
        self.optimizer = optimizer
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.initial_lr = optimizer.lr
        self.t = 0

    def step(self) -> float:
        self.t += 1
        progress = min(self.t, self.max_steps) / self.max_steps
        self.optimizer.lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (
            1.0 + math.cos(math.pi * progress)
        )
        return self.optimizer.lr
