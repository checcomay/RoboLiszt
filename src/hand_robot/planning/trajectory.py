from __future__ import annotations

import numpy as np


def quintic(tau: np.ndarray | float) -> np.ndarray | float:
    """Minimum-jerk time scaling s(tau), tau in [0,1]: zero velocity/acceleration at both ends."""
    tau = np.clip(tau, 0.0, 1.0)
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def time_grid(duration: float, dt: float) -> np.ndarray:
    n = max(int(round(duration / dt)), 1)
    return np.linspace(0.0, duration, n + 1)


def segment(x0: np.ndarray, x1: np.ndarray, duration: float, dt: float) -> np.ndarray:
    """Point-to-point minimum-jerk segment (N x dim, includes both end points)."""
    s = quintic(time_grid(duration, dt) / max(duration, dt))
    x0, x1 = np.asarray(x0, float), np.asarray(x1, float)
    return x0 + s[:, None] * (x1 - x0)


def follow_path(path: np.ndarray, duration: float, dt: float) -> np.ndarray:
    """Time-scale a dense waypoint path (M x dim) with a quintic law, linear interpolation."""
    path = np.asarray(path, float)
    s = quintic(time_grid(duration, dt) / max(duration, dt)) * (len(path) - 1)
    idx = np.clip(np.floor(s).astype(int), 0, len(path) - 2)
    frac = (s - idx)[:, None]
    return path[idx] * (1 - frac) + path[idx + 1] * frac


def derivative(x: np.ndarray, dt: float) -> np.ndarray:
    return np.gradient(x, dt, axis=0)


def concat(segments: list[np.ndarray]) -> np.ndarray:
    """Concatenate segments dropping the duplicated junction sample."""
    out = [segments[0]]
    for s in segments[1:]:
        out.append(s[1:])
    return np.vstack(out)
