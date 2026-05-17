from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class ArrayLike(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...


@dataclass(frozen=True)
class Backend:
    """Minimal backend contract used by the engine.

    New backends should keep Tensor semantics identical and differ only in array
    storage plus kernel dispatch.  That makes autograd tests backend-agnostic.
    """

    name: str

    def array(self, value: object, dtype: np.dtype | str | None = None) -> np.ndarray:
        raise NotImplementedError

    def zeros(self, shape: tuple[int, ...], dtype: np.dtype | str = np.float32) -> np.ndarray:
        raise NotImplementedError

    def ones(self, shape: tuple[int, ...], dtype: np.dtype | str = np.float32) -> np.ndarray:
        raise NotImplementedError

    def randn(self, shape: tuple[int, ...], dtype: np.dtype | str = np.float32) -> np.ndarray:
        raise NotImplementedError
