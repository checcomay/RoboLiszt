from __future__ import annotations

import numpy as np

from hand_robot.model.robot import HandRobot


def translational_jacobian(
    robot: HandRobot,
    q: np.ndarray,
    branch: str,
    eps: float = 1e-6,
) -> np.ndarray:
    """Finite-difference 3x20 translational Jacobian."""
    q = robot.validate_q(q)
    endpoint = robot.branches[branch][-1]
    J = np.zeros((3, 20))

    for k in range(20):
        qp = q.copy()
        qm = q.copy()
        qp[k] += eps
        qm[k] -= eps

        pp = robot.forward_kinematics(qp)[endpoint][:3, 3]
        pm = robot.forward_kinematics(qm)[endpoint][:3, 3]
        J[:, k] = (pp - pm) / (2 * eps)

    return J


def geometric_jacobian(
    robot: HandRobot,
    q: np.ndarray,
    branch: str,
) -> np.ndarray:
    """Return a 6x20 geometric Jacobian using joint axes in frame 0.

    All modeled joints are revolute. For joint i:
        Jv_i = z_i-1 x (p_e - p_i-1)
        Jw_i = z_i-1
    """
    q = robot.validate_q(q)
    T = robot.forward_kinematics(q)
    endpoint = robot.branches[branch][-1]
    p_e = T[endpoint][:3, 3]

    J = np.zeros((6, 20))

    active = set(robot.branches[branch]) | {1, 2}
    for joint in active:
        parent = robot.parent[joint]
        z = T[parent][:3, 2]
        p = T[parent][:3, 3]
        J[:3, joint - 1] = np.cross(z, p_e - p)
        J[3:, joint - 1] = z

    return J
