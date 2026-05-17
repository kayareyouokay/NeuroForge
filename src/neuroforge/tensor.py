from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Callable

import numpy as np

_grad_enabled = True


def _coerce_array(data: Any, dtype: np.dtype | str | None = None) -> np.ndarray:
    if isinstance(data, Tensor):
        return data.data
    arr = np.asarray(data, dtype=dtype)
    if dtype is None and arr.dtype.kind in {"f", "c"}:
        return arr.astype(np.float32, copy=False)
    if dtype is None and arr.dtype.kind not in {"i", "u", "b"}:
        return arr.astype(np.float32, copy=False)
    return arr


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum broadcasted gradient dimensions back to an operand shape."""

    if grad.shape == shape:
        return grad
    while len(grad.shape) > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


@contextmanager
def no_grad() -> Iterator[None]:
    """Temporarily disable graph construction."""

    global _grad_enabled
    old = _grad_enabled
    _grad_enabled = False
    try:
        yield
    finally:
        _grad_enabled = old


class Tensor:
    """Autograd-enabled n-dimensional array.

    The implementation is eager and scalar-free: every operation records a small
    backward closure.  Backprop performs a reverse topological traversal over the
    dynamic computation graph.
    """

    __array_priority__ = 1000

    def __init__(
        self,
        data: Any,
        requires_grad: bool = False,
        *,
        dtype: np.dtype | str | None = None,
        _children: tuple[Tensor, ...] = (),
        _op: str = "",
    ) -> None:
        self.data = _coerce_array(data, dtype=dtype)
        self.requires_grad = bool(requires_grad and _grad_enabled)
        self.grad: np.ndarray | None = None
        self._prev = set(_children)
        self._op = _op
        self._backward: Callable[[], None] = lambda: None

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def T(self) -> Tensor:
        return self.transpose()

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, dtype={self.dtype}, requires_grad={self.requires_grad})"

    def __hash__(self) -> int:
        return id(self)

    def zero_grad(self) -> None:
        self.grad = None

    def detach(self) -> Tensor:
        return Tensor(self.data.copy(), requires_grad=False)

    def numpy(self) -> np.ndarray:
        return self.data

    def item(self) -> float:
        return float(self.data.item())

    def backward(self, grad: np.ndarray | None = None) -> None:
        """Backpropagate from this tensor through the recorded graph."""

        if grad is None:
            grad = np.ones_like(self.data, dtype=np.float32)
        self.grad = grad.astype(np.float32, copy=False)

        topo: list[Tensor] = []
        seen: set[Tensor] = set()

        def build(v: Tensor) -> None:
            if v in seen:
                return
            seen.add(v)
            for child in v._prev:
                build(child)
            topo.append(v)

        build(self)
        for node in reversed(topo):
            node._backward()

    def _binary(self, other: Any, op: str) -> Tensor:
        other = ensure_tensor(other)
        if op == "add":
            data = self.data + other.data
        elif op == "sub":
            data = self.data - other.data
        elif op == "mul":
            data = self.data * other.data
        elif op == "div":
            data = self.data / other.data
        else:
            raise ValueError(op)
        out = Tensor(
            data,
            self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op=op,
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                if op in {"add", "sub"}:
                    g = out.grad
                elif op == "mul":
                    g = out.grad * other.data
                else:
                    g = out.grad / other.data
                self._accumulate(_unbroadcast(g, self.shape))
            if other.requires_grad:
                if op == "add":
                    g = out.grad
                elif op == "sub":
                    g = -out.grad
                elif op == "mul":
                    g = out.grad * self.data
                else:
                    g = -out.grad * self.data / (other.data**2)
                other._accumulate(_unbroadcast(g, other.shape))

        out._backward = _backward
        return out

    def _accumulate(self, grad: np.ndarray) -> None:
        grad = grad.astype(np.float32, copy=False)
        self.grad = grad if self.grad is None else self.grad + grad

    def __add__(self, other: Any) -> Tensor:
        return self._binary(other, "add")

    def __radd__(self, other: Any) -> Tensor:
        return self + other

    def __sub__(self, other: Any) -> Tensor:
        return self._binary(other, "sub")

    def __rsub__(self, other: Any) -> Tensor:
        return ensure_tensor(other) - self

    def __mul__(self, other: Any) -> Tensor:
        return self._binary(other, "mul")

    def __rmul__(self, other: Any) -> Tensor:
        return self * other

    def __truediv__(self, other: Any) -> Tensor:
        return self._binary(other, "div")

    def __rtruediv__(self, other: Any) -> Tensor:
        return ensure_tensor(other) / self

    def __neg__(self) -> Tensor:
        return self * -1.0

    def __pow__(self, power: float) -> Tensor:
        out = Tensor(self.data**power, self.requires_grad, _children=(self,), _op="pow")

        def _backward() -> None:
            if out.grad is not None and self.requires_grad:
                self._accumulate(out.grad * power * (self.data ** (power - 1)))

        out._backward = _backward
        return out

    def matmul(self, other: Any) -> Tensor:
        other = ensure_tensor(other)
        out = Tensor(
            np.matmul(self.data, other.data),
            self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="matmul",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                g = np.einsum("...mn,...kn->...mk", out.grad, other.data)
                self._accumulate(_unbroadcast(g, self.shape))
            if other.requires_grad:
                g = np.einsum("...mk,...mn->...kn", self.data, out.grad)
                other._accumulate(_unbroadcast(g, other.shape))

        out._backward = _backward
        return out

    def __matmul__(self, other: Any) -> Tensor:
        return self.matmul(other)

    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            self.requires_grad,
            _children=(self,),
            _op="sum",
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            grad = out.grad
            if axis is not None and not keepdims:
                axes = (axis,) if isinstance(axis, int) else axis
                axes = tuple(a if a >= 0 else a + self.ndim for a in axes)
                for ax in sorted(axes):
                    grad = np.expand_dims(grad, ax)
            self._accumulate(np.ones_like(self.data, dtype=np.float32) * grad)

        out._backward = _backward
        return out

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        if axis is None:
            denom = self.data.size
        else:
            axes = (axis,) if isinstance(axis, int) else axis
            axes = tuple(a if a >= 0 else a + self.ndim for a in axes)
            denom = int(np.prod([self.shape[a] for a in axes]))
        return self.sum(axis=axis, keepdims=keepdims) / float(denom)

    def reshape(self, *shape: int | tuple[int, ...]) -> Tensor:
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        out = Tensor(self.data.reshape(shape), self.requires_grad, _children=(self,), _op="reshape")

        def _backward() -> None:
            if out.grad is not None and self.requires_grad:
                self._accumulate(out.grad.reshape(self.shape))

        out._backward = _backward
        return out

    def transpose(self, axes: tuple[int, ...] | None = None) -> Tensor:
        out = Tensor(
            self.data.transpose(axes),
            self.requires_grad,
            _children=(self,),
            _op="transpose",
        )

        def _backward() -> None:
            if out.grad is not None and self.requires_grad:
                inv = None if axes is None else np.argsort(axes)
                self._accumulate(out.grad.transpose(inv))

        out._backward = _backward
        return out

    def permute(self, *axes: int) -> Tensor:
        return self.transpose(tuple(axes))

    def swapaxes(self, axis1: int, axis2: int) -> Tensor:
        axes = list(range(self.ndim))
        axes[axis1], axes[axis2] = axes[axis2], axes[axis1]
        return self.transpose(tuple(axes))

    def exp(self) -> Tensor:
        data = np.exp(self.data)
        out = Tensor(data, self.requires_grad, _children=(self,), _op="exp")

        def _backward() -> None:
            if out.grad is not None and self.requires_grad:
                self._accumulate(out.grad * data)

        out._backward = _backward
        return out

    def log(self) -> Tensor:
        out = Tensor(np.log(self.data), self.requires_grad, _children=(self,), _op="log")

        def _backward() -> None:
            if out.grad is not None and self.requires_grad:
                self._accumulate(out.grad / self.data)

        out._backward = _backward
        return out

    def tanh(self) -> Tensor:
        data = np.tanh(self.data)
        out = Tensor(data, self.requires_grad, _children=(self,), _op="tanh")

        def _backward() -> None:
            if out.grad is not None and self.requires_grad:
                self._accumulate(out.grad * (1.0 - data**2))

        out._backward = _backward
        return out

    def sigmoid(self) -> Tensor:
        data = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(data, self.requires_grad, _children=(self,), _op="sigmoid")

        def _backward() -> None:
            if out.grad is not None and self.requires_grad:
                self._accumulate(out.grad * data * (1.0 - data))

        out._backward = _backward
        return out

    def relu(self) -> Tensor:
        out = Tensor(np.maximum(self.data, 0.0), self.requires_grad, _children=(self,), _op="relu")

        def _backward() -> None:
            if out.grad is not None and self.requires_grad:
                self._accumulate(out.grad * (self.data > 0))

        out._backward = _backward
        return out

    def __getitem__(self, index: Any) -> Tensor:
        out = Tensor(self.data[index], self.requires_grad, _children=(self,), _op="slice")

        def _backward() -> None:
            if out.grad is not None and self.requires_grad:
                g = np.zeros_like(self.data, dtype=np.float32)
                np.add.at(g, index, out.grad)
                self._accumulate(g)

        out._backward = _backward
        return out

    @staticmethod
    def concat(tensors: list[Tensor], axis: int = 0) -> Tensor:
        arrays = [t.data for t in tensors]
        out = Tensor(
            np.concatenate(arrays, axis=axis),
            any(t.requires_grad for t in tensors),
            _children=tuple(tensors),
            _op="concat",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            offset = 0
            for t in tensors:
                width = t.shape[axis]
                slc = [slice(None)] * out.ndim
                slc[axis] = slice(offset, offset + width)
                if t.requires_grad:
                    t._accumulate(out.grad[tuple(slc)])
                offset += width

        out._backward = _backward
        return out

    @staticmethod
    def stack(tensors: list[Tensor], axis: int = 0) -> Tensor:
        expanded = [t.reshape(*t.shape[:axis], 1, *t.shape[axis:]) for t in tensors]
        return Tensor.concat(expanded, axis=axis)


class Parameter(Tensor):
    """Trainable tensor owned by a Module."""

    def __init__(self, data: Any, *, dtype: np.dtype | str | None = np.float32) -> None:
        super().__init__(data, requires_grad=True, dtype=dtype)


def ensure_tensor(value: Any) -> Tensor:
    return value if isinstance(value, Tensor) else Tensor(value)
