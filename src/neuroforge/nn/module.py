from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from typing import Any

import numpy as np

from neuroforge.tensor import Parameter, Tensor


class Module:
    """Base class for trainable components."""

    training: bool

    def __init__(self) -> None:
        self.training = True

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def train(self) -> None:
        self.training = True
        for module in self.children():
            module.train()

    def eval(self) -> None:
        self.training = False
        for module in self.children():
            module.eval()

    def parameters(self) -> list[Parameter]:
        params: list[Parameter] = []
        self._walk(lambda _, value: params.append(value) if isinstance(value, Parameter) else None)
        return params

    def named_parameters(self, prefix: str = "") -> OrderedDict[str, Parameter]:
        found: OrderedDict[str, Parameter] = OrderedDict()

        def visit(name: str, value: Any) -> None:
            if isinstance(value, Parameter):
                found[prefix + name] = value

        self._walk(visit)
        return found

    def children(self) -> list[Module]:
        modules: list[Module] = []
        for value in self.__dict__.values():
            modules.extend(_extract_modules(value))
        return modules

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.zero_grad()

    def state_dict(self) -> OrderedDict[str, np.ndarray]:
        return OrderedDict((name, p.data.copy()) for name, p in self.named_parameters().items())

    def load_state_dict(self, state: dict[str, np.ndarray], strict: bool = True) -> None:
        own = self.named_parameters()
        missing = [name for name in own if name not in state]
        extra = [name for name in state if name not in own]
        if strict and (missing or extra):
            raise KeyError(f"state mismatch: missing={missing}, extra={extra}")
        for name, value in state.items():
            if name in own:
                own[name].data[...] = value

    def _walk(self, visitor: Any, prefix: str = "") -> None:
        for name, value in self.__dict__.items():
            if name.startswith("_") or name == "training":
                continue
            _walk_value(value, visitor, prefix + name)


def _walk_value(value: Any, visitor: Any, name: str) -> None:
    visitor(name, value)
    if isinstance(value, Module):
        for child_name, child_value in value.__dict__.items():
            if child_name.startswith("_") or child_name == "training":
                continue
            _walk_value(child_value, visitor, f"{name}.{child_name}")
    elif isinstance(value, (list, tuple)):
        for i, child in enumerate(value):
            _walk_value(child, visitor, f"{name}.{i}")
    elif isinstance(value, dict):
        for key, child in value.items():
            _walk_value(child, visitor, f"{name}.{key}")


def _extract_modules(value: Any) -> list[Module]:
    if isinstance(value, Module):
        return [value]
    if isinstance(value, (list, tuple)):
        return [m for item in value for m in _extract_modules(item)]
    if isinstance(value, dict):
        return [m for item in value.values() for m in _extract_modules(item)]
    return []


class Sequential(Module):
    """A module container that forwards through layers in order."""

    def __init__(self, *layers: Module) -> None:
        super().__init__()
        self.layers = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def __iter__(self) -> Iterable[Module]:
        return iter(self.layers)


class Residual(Module):
    """Residual wrapper: output = fn(x) + x."""

    def __init__(self, branch: Module) -> None:
        super().__init__()
        self.branch = branch

    def forward(self, x: Tensor) -> Tensor:
        return self.branch(x) + x
