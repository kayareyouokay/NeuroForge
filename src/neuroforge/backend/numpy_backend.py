from __future__ import annotations

import numpy as np

from neuroforge.backend.base import Backend


class NumPyBackend(Backend):
    """Eager CPU backend implemented on top of NumPy."""

    def __init__(self) -> None:
        super().__init__(name="numpy")

    def array(self, value: object, dtype: np.dtype | str | None = None) -> np.ndarray:
        return np.asarray(value, dtype=dtype)

    def zeros(self, shape: tuple[int, ...], dtype: np.dtype | str = np.float32) -> np.ndarray:
        return np.zeros(shape, dtype=dtype)

    def ones(self, shape: tuple[int, ...], dtype: np.dtype | str = np.float32) -> np.ndarray:
        return np.ones(shape, dtype=dtype)

    def randn(self, shape: tuple[int, ...], dtype: np.dtype | str = np.float32) -> np.ndarray:
        return np.random.standard_normal(shape).astype(dtype)


default_backend = NumPyBackend()
