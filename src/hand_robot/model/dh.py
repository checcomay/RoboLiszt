from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DHParameter:
    """Standard DH parameters for one revolute joint."""
    a: float
    alpha: float
    d: float
    theta_offset: float = 0.0


def dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """Build the standard DH homogeneous transform."""
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)

    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
