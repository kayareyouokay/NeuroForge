from __future__ import annotations

import numpy as np

from neuroforge.optim.optimizer import Optimizer
from neuroforge.tensor import Parameter


class SGD(Optimizer):
    """Stochastic gradient descent with optional momentum and weight decay."""

    def __init__(
        self,
        params: list[Parameter],
        lr: float = 1e-2,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.velocity = [np.zeros_like(p.data, dtype=np.float32) for p in self.params]

    def step(self) -> None:
        self.step_count += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad + self.weight_decay * p.data if self.weight_decay else p.grad
            if self.momentum:
                self.velocity[i] = self.momentum * self.velocity[i] + grad
                grad = (
                    grad + self.momentum * self.velocity[i] if self.nesterov else self.velocity[i]
                )
            p.data -= self.lr * grad

    def state_dict(self) -> dict[str, object]:
        state = super().state_dict()
        state.update({"momentum": self.momentum, "velocity": [v.copy() for v in self.velocity]})
        return state


class RMSProp(Optimizer):
    def __init__(
        self,
        params: list[Parameter],
        lr: float = 1e-3,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr)
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.square_avg = [np.zeros_like(p.data, dtype=np.float32) for p in self.params]

    def step(self) -> None:
        self.step_count += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad + self.weight_decay * p.data if self.weight_decay else p.grad
            self.square_avg[i] = self.alpha * self.square_avg[i] + (1.0 - self.alpha) * grad * grad
            p.data -= self.lr * grad / (np.sqrt(self.square_avg[i]) + self.eps)
