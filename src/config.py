from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass(frozen=True)
class Config:
    """Lightweight configuration wrapper with helper lookup."""

    data: Dict[str, Any]

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.data
        for key in keys:
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                return default
        return node


def load_config(path: str | Path = "config/default.yaml", *, override: Optional[Dict[str, Any]] = None) -> Config:
    """Load YAML configuration and optionally apply overrides."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if override:
        data.update(override)
    return Config(data=data)
