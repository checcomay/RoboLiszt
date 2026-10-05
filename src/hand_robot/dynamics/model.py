from __future__ import annotations

import numpy as np

from hand_robot.model.robot import HandRobot
from hand_robot.kinematics.jacobian import geometric_jacobian


class DynamicsModel:
    """Configurable simulation-level dynamics approximation.

    Equation:
        M(q) qdd + B qd + g(q) = tau + J^T F_ext

    This is a development model. Replace parameters with identified/CAD data
    before physical deployment.
    """

    def __init__(self, robot: HandRobot, config: dict):
        self.robot = robot
        c = config["dynamics"]
        self.inertia = float(c["joint_inertia"])
        self.damping = float(c["joint_damping"])
        self.gravity_gain = float(c["gravity_gain"])

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        self.robot.validate_q(q)
        return self.inertia * np.eye(20)

    def coriolis(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        self.robot.validate_q(q)
        return self.damping * np.asarray(qd, dtype=float)

    def gravity(self, q: np.ndarray) -> np.ndarray:
        q = self.robot.validate_q(q)
        return self.gravity_gain * np.sin(q)

    def external_joint_torque(
        self,
        q: np.ndarray,
        branch: str,
        force: np.ndarray,
    ) -> np.ndarray:
        J = geometric_jacobian(self.robot, q, branch)
        wrench = np.zeros(6)
        wrench[:3] = np.asarray(force, dtype=float).reshape(3)
        return J.T @ wrench

    def acceleration(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        tau: np.ndarray,
        external_forces: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        q = self.robot.validate_q(q)
        qd = self.robot.validate_q(qd)
        tau = self.robot.validate_q(tau)

        tau_ext = np.zeros(20)
        for branch, force in (external_forces or {}).items():
            tau_ext += self.external_joint_torque(q, branch, force)

        rhs = tau + tau_ext - self.coriolis(q, qd) - self.gravity(q)
        return np.linalg.solve(self.mass_matrix(q), rhs)

    def energy(
        self,
        q: np.ndarray,
        qd: np.ndarray,
    ) -> float:
        q = self.robot.validate_q(q)
        qd = self.robot.validate_q(qd)
        kinetic = 0.5 * qd @ self.mass_matrix(q) @ qd
        potential = self.gravity_gain * np.sum(1.0 - np.cos(q))
        return float(kinetic + potential)
