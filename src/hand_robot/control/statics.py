"""Kineto-statics: tau = J^T F (virtual work), joint torques that balance a tip force."""
from __future__ import annotations

import numpy as np

from hand_robot.kinematics.jacobian import geometric_jacobian


def joint_torque_from_tip_force(robot, q, branch, force, T=None) -> np.ndarray:
    """tau = J_P^T F  (20,) -- torque the joints would feel under a tip force F."""
    J = geometric_jacobian(robot, q, branch, T)[:3]
    return J.T @ np.asarray(force, float).reshape(3)


def compensation_torque(robot, q, branch, contact_force) -> np.ndarray:
    """Actuator torque that holds the finger against an external (contact) force: tau = -J^T F_ext."""
    return -joint_torque_from_tip_force(robot, q, branch, contact_force)
