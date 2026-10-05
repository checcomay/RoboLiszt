from __future__ import annotations

import numpy as np


class JointPDController:
    """Joint-space PD controller with gravity compensation."""

    def __init__(self, dynamics, kp: float, kd: float):
        self.dynamics = dynamics
        self.kp = float(kp)
        self.kd = float(kd)

    def compute(self, q: np.ndarray, qd: np.ndarray, target_q: np.ndarray | None) -> np.ndarray:
        target = np.zeros(20) if target_q is None else self.dynamics.robot.validate_q(target_q)
        q = self.dynamics.robot.validate_q(q)
        qd = self.dynamics.robot.validate_q(qd)

        error = target - q
        tau = self.kp * error - self.kd * qd + self.dynamics.gravity(q)
        return tau
