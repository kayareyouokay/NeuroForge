from __future__ import annotations

import math

import numpy as np


def zeros(shape: tuple[int, ...]) -> np.ndarray:
    return np.zeros(shape, dtype=np.float32)


def ones(shape: tuple[int, ...]) -> np.ndarray:
    return np.ones(shape, dtype=np.float32)


def normal(shape: tuple[int, ...], mean: float = 0.0, std: float = 1.0) -> np.ndarray:
    return np.random.normal(mean, std, size=shape).astype(np.float32)


def uniform(shape: tuple[int, ...], low: float = -1.0, high: float = 1.0) -> np.ndarray:
    return np.random.uniform(low, high, size=shape).astype(np.float32)


def xavier_uniform(shape: tuple[int, ...], gain: float = 1.0) -> np.ndarray:
    fan_in, fan_out = _fans(shape)
    limit = gain * math.sqrt(6.0 / (fan_in + fan_out))
    return uniform(shape, -limit, limit)


def he_normal(shape: tuple[int, ...]) -> np.ndarray:
    fan_in, _ = _fans(shape)
    return normal(shape, std=math.sqrt(2.0 / fan_in))


def _fans(shape: tuple[int, ...]) -> tuple[int, int]:
    if len(shape) < 2:
        return 1, 1
    if len(shape) == 2:
        return shape[0], shape[1]
    receptive = int(np.prod(shape[2:]))
    return shape[1] * receptive, shape[0] * receptive
