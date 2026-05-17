from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from neuroforge.nn.module import Module
from neuroforge.optim.optimizer import Optimizer


def save_checkpoint(
    path: str | Path,
    model: Module,
    optimizer: Optimizer | None = None,
    **metadata: Any,
) -> None:
    payload: dict[str, Any] = {"model": model.state_dict(), "metadata": metadata}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, payload=np.array(payload, dtype=object))


def load_checkpoint(
    path: str | Path,
    model: Module,
    optimizer: Optimizer | None = None,
) -> dict[str, Any]:
    payload = np.load(path, allow_pickle=True)["payload"].item()
    model.load_state_dict(payload["model"], strict=False)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return dict(payload.get("metadata", {}))
