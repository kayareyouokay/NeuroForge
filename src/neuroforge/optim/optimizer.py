from __future__ import annotations

from neuroforge.tensor import Parameter


class Optimizer:
    """Base optimizer with gradient zeroing and state serialization."""

    def __init__(self, params: list[Parameter], lr: float) -> None:
        self.params = list(params)
        self.lr = lr
        self.step_count = 0

    def step(self) -> None:
        raise NotImplementedError

    def zero_grad(self) -> None:
        for parameter in self.params:
            parameter.zero_grad()

    def state_dict(self) -> dict[str, object]:
        return {"lr": self.lr, "step_count": self.step_count}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.lr = float(state["lr"])
        self.step_count = int(state["step_count"])
