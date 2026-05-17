from __future__ import annotations

import numpy as np

from neuroforge.nn.init import xavier_uniform
from neuroforge.nn.module import Module
from neuroforge.tensor import Parameter, Tensor


class RNN(Module):
    """Single-layer tanh RNN."""

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(xavier_uniform((input_size, hidden_size)))
        self.weight_hh = Parameter(xavier_uniform((hidden_size, hidden_size)))
        self.bias = Parameter(np.zeros((hidden_size,), dtype=np.float32)) if bias else None

    def forward(self, x: Tensor, h0: Tensor | None = None) -> tuple[Tensor, Tensor]:
        batch, steps, _ = x.shape
        h = h0 if h0 is not None else Tensor(np.zeros((batch, self.hidden_size), dtype=np.float32))
        outputs: list[Tensor] = []
        for t in range(steps):
            bias = self.bias if self.bias is not None else 0.0
            h = (x[:, t, :] @ self.weight_ih + h @ self.weight_hh + bias).tanh()
            outputs.append(h)
        return Tensor.stack(outputs, axis=1), h


class LSTM(Module):
    """Single-layer LSTM with input, forget, candidate and output gates."""

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(xavier_uniform((input_size, 4 * hidden_size)))
        self.weight_hh = Parameter(xavier_uniform((hidden_size, 4 * hidden_size)))
        self.bias = Parameter(np.zeros((4 * hidden_size,), dtype=np.float32)) if bias else None

    def forward(
        self,
        x: Tensor,
        state: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        batch, steps, _ = x.shape
        if state is None:
            h = Tensor(np.zeros((batch, self.hidden_size), dtype=np.float32))
            c = Tensor(np.zeros((batch, self.hidden_size), dtype=np.float32))
        else:
            h, c = state
        outputs: list[Tensor] = []
        hs = self.hidden_size
        for t in range(steps):
            bias = self.bias if self.bias is not None else 0.0
            gates = x[:, t, :] @ self.weight_ih + h @ self.weight_hh + bias
            i = gates[:, :hs].sigmoid()
            f = gates[:, hs : 2 * hs].sigmoid()
            g = gates[:, 2 * hs : 3 * hs].tanh()
            o = gates[:, 3 * hs :].sigmoid()
            c = f * c + i * g
            h = o * c.tanh()
            outputs.append(h)
        return Tensor.stack(outputs, axis=1), (h, c)
