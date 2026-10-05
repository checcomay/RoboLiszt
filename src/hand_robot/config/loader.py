from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def default_config_path() -> Path:
    """Return the repository's default robot configuration path."""
    return Path(__file__).with_name("robot.yaml")
