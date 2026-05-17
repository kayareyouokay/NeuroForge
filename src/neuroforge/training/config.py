from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    """Thin YAML-backed config object for reproducible experiments."""

    values: dict[str, Any]

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        with Path(path).open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle) or {}
        return cls(values)

    def get(self, key: str, default: Any = None) -> Any:
        node: Any = self.values
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node
