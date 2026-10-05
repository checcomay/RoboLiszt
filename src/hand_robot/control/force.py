from __future__ import annotations

import numpy as np

from hand_robot.kinematics.jacobian import translational_jacobian


class ForceController:
    """Simple Cartesian force controller for a selected fingertip.

    It maps a desired endpoint force into joint torques with J^T F and
    adds a position-restoring term. This is a simulation controller.
    """

    def __init__(self, robot, dynamics, branch: str, kp: float, force_limit: float):
        self.robot = robot
        self.dynamics = dynamics
        self.branch = branch
        self.kp = float(kp)
        self.force_limit = float(force_limit)

    def desired_force(self, measured_force: float, desired_force: float) -> float:
        error = desired_force - measured_force
        force = self.kp * error + desired_force
        return float(np.clip(force, -self.force_limit, self.force_limit))

    def torque_from_force(self, q: np.ndarray, force_vector: np.ndarray) -> np.ndarray:
        J = translational_jacobian(self.robot, q, self.branch)
        return J.T @ np.asarray(force_vector, dtype=float).reshape(3)
