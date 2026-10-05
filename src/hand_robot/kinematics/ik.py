from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .jacobian import translational_jacobian
from hand_robot.model.robot import HandRobot


@dataclass
class IKResult:
    q: np.ndarray
    success: bool
    iterations: int
    error_norm: float
    history: np.ndarray


class InverseKinematicsSolver:
    """Damped-least-squares position IK for the branched 20-DOF hand."""

    def __init__(self, robot: HandRobot, config: dict):
        self.robot = robot
        c = config["control"]
        self.step = float(c["ik_step"])
        self.tolerance = float(c["ik_tolerance"])
        self.max_iterations = int(c["ik_max_iterations"])
        self.damping = float(c["ik_damping"])

    def solve(
        self,
        branch: str,
        target_position: np.ndarray,
        q0: np.ndarray | None = None,
    ) -> IKResult:
        q = np.zeros(20) if q0 is None else self.robot.validate_q(q0).copy()
        target = np.asarray(target_position, dtype=float).reshape(3)

        history = []
        endpoint = self.robot.branches[branch][-1]

        for iteration in range(1, self.max_iterations + 1):
            current = self.robot.forward_kinematics(q)[endpoint][:3, 3]
            error = target - current
            norm = float(np.linalg.norm(error))
            history.append(norm)

            if norm < self.tolerance:
                return IKResult(q, True, iteration, norm, np.asarray(history))

            J = translational_jacobian(self.robot, q, branch)
            A = J @ J.T + (self.damping ** 2) * np.eye(3)
            dq = J.T @ np.linalg.solve(A, error)

            q += self.step * dq

        final_error = history[-1] if history else np.inf
        return IKResult(q, False, self.max_iterations, final_error, np.asarray(history))
