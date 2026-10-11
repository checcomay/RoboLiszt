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
    return Path(__file__).resolve().parents[3] / "src" / "hand_robot" / "config" / "robot.yaml"


def joint_limits(config: dict):
    """Return (q_min, q_max) in rad for the 20 joints (default + optional overrides)."""
    import numpy as np
    lim = config.get("limits", {})
    lo = np.full(20, np.deg2rad(float(lim.get("default_min_deg", -90.0))))
    hi = np.full(20, np.deg2rad(float(lim.get("default_max_deg", 90.0))))
    for j, (a, b) in (lim.get("overrides") or {}).items():
        lo[int(j) - 1], hi[int(j) - 1] = np.deg2rad(a), np.deg2rad(b)
    return lo, hi
