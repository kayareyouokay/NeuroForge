from __future__ import annotations

import math

from neuroforge.nn.module import Module
from neuroforge.tensor import Tensor


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


class GELU(Module):
    """Gaussian error linear unit with the tanh approximation."""

    def forward(self, x: Tensor) -> Tensor:
        return 0.5 * x * (1.0 + (math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x**3))).tanh())
