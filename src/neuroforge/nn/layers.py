from __future__ import annotations

import numpy as np

from neuroforge.nn.init import he_normal, normal, xavier_uniform
from neuroforge.nn.module import Module
from neuroforge.tensor import Parameter, Tensor


class Linear(Module):
    """Affine projection over the last dimension."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(xavier_uniform((in_features, out_features)))
        self.bias = Parameter(np.zeros((out_features,), dtype=np.float32)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        return out + self.bias if self.bias is not None else out


class Embedding(Module):
    """Lookup table with sparse-style gradient accumulation via indexed writes."""

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(normal((num_embeddings, embedding_dim), std=0.02))

    def forward(self, indices: np.ndarray | Tensor | list[int]) -> Tensor:
        if isinstance(indices, Tensor):
            indices = indices.data
        return self.weight[np.asarray(indices, dtype=np.int64)]


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError("dropout probability must be in [0, 1)")
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        mask = (np.random.random(x.shape) >= self.p).astype(np.float32) / (1.0 - self.p)
        return x * Tensor(mask)


class LayerNorm(Module):
    def __init__(self, normalized_shape: int | tuple[int, ...], eps: float = 1e-5) -> None:
        super().__init__()
        self.normalized_shape = (
            (normalized_shape,) if isinstance(normalized_shape, int) else normalized_shape
        )
        self.eps = eps
        self.weight = Parameter(np.ones(self.normalized_shape, dtype=np.float32))
        self.bias = Parameter(np.zeros(self.normalized_shape, dtype=np.float32))

    def forward(self, x: Tensor) -> Tensor:
        axes = tuple(range(x.ndim - len(self.normalized_shape), x.ndim))
        mean = x.mean(axis=axes, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=axes, keepdims=True)
        return (x - mean) / ((var + self.eps) ** 0.5) * self.weight + self.bias


class BatchNorm1D(Module):
    """Batch normalization for rank-2 activations."""

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(np.ones((num_features,), dtype=np.float32))
        self.bias = Parameter(np.zeros((num_features,), dtype=np.float32))
        self.running_mean = np.zeros((num_features,), dtype=np.float32)
        self.running_var = np.ones((num_features,), dtype=np.float32)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mean = x.mean(axis=0, keepdims=True)
            var = ((x - mean) ** 2).mean(axis=0, keepdims=True)
            self.running_mean = (
                1.0 - self.momentum
            ) * self.running_mean + self.momentum * mean.data.squeeze(0)
            self.running_var = (
                1.0 - self.momentum
            ) * self.running_var + self.momentum * var.data.squeeze(0)
        else:
            mean = Tensor(self.running_mean.reshape(1, -1))
            var = Tensor(self.running_var.reshape(1, -1))
        return (x - mean) / ((var + self.eps) ** 0.5) * self.weight + self.bias


class Conv2D(Module):
    """NCHW 2D convolution with explicit backward kernels.

    It is intentionally simple but gradient-correct.  The loops are isolated so
    a future im2col/CUDA backend can replace them without changing Module APIs.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        kh, kw = _pair(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kh, kw)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.weight = Parameter(he_normal((out_channels, in_channels, kh, kw)))
        self.bias = Parameter(np.zeros((out_channels,), dtype=np.float32)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        return conv2d(x, self.weight, self.bias, self.stride, self.padding)


def conv2d(
    x: Tensor,
    weight: Parameter,
    bias: Parameter | None,
    stride: tuple[int, int],
    padding: tuple[int, int],
) -> Tensor:
    n, c, h, w = x.shape
    out_c, in_c, kh, kw = weight.shape
    if c != in_c:
        raise ValueError(f"expected {in_c} input channels, got {c}")
    sh, sw = stride
    ph, pw = padding
    x_pad = np.pad(x.data, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1
    out_data = np.empty((n, out_c, out_h, out_w), dtype=np.float32)
    for i in range(out_h):
        hs = i * sh
        for j in range(out_w):
            ws = j * sw
            window = x_pad[:, :, hs : hs + kh, ws : ws + kw]
            out_data[:, :, i, j] = np.tensordot(
                window,
                weight.data,
                axes=((1, 2, 3), (1, 2, 3)),
            )
    if bias is not None:
        out_data += bias.data.reshape(1, -1, 1, 1)

    parents: tuple[Tensor, ...] = (x, weight) if bias is None else (x, weight, bias)
    out = Tensor(
        out_data,
        any(p.requires_grad for p in parents),
        _children=parents,
        _op="conv2d",
    )

    def _backward() -> None:
        if out.grad is None:
            return
        dx_pad = np.zeros_like(x_pad, dtype=np.float32)
        dw = np.zeros_like(weight.data, dtype=np.float32)
        for i in range(out_h):
            hs = i * sh
            for j in range(out_w):
                ws = j * sw
                g = out.grad[:, :, i, j]
                window = x_pad[:, :, hs : hs + kh, ws : ws + kw]
                if weight.requires_grad:
                    dw += np.tensordot(g, window, axes=((0), (0)))
                if x.requires_grad:
                    dx_pad[:, :, hs : hs + kh, ws : ws + kw] += np.tensordot(
                        g,
                        weight.data,
                        axes=((1), (0)),
                    )
        if x.requires_grad:
            x._accumulate(dx_pad[:, :, ph : ph + h, pw : pw + w])
        if weight.requires_grad:
            weight._accumulate(dw)
        if bias is not None and bias.requires_grad:
            bias._accumulate(out.grad.sum(axis=(0, 2, 3)))

    out._backward = _backward
    return out


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    return value if isinstance(value, tuple) else (value, value)
