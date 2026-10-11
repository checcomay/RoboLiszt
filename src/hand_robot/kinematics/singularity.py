from __future__ import annotations

import numpy as np

from .jacobian import position_jacobian
from hand_robot.model.robot import HandRobot


def metrics(J: np.ndarray, tol: float = 1e-6) -> dict:
    """SVD-based indices of a (m x n) Jacobian."""
    s = np.linalg.svd(J, compute_uv=False)
    return {
        "sigma": s,
        "sigma_min": float(s[-1]),
        "sigma_max": float(s[0]),
        "w": float(np.prod(s)),                       # Yoshikawa manipulability sqrt(det(J J^T))
        "cond": float(s[0] / max(s[-1], 1e-12)),
        "rank": int(np.sum(s > tol * max(s[0], 1e-12))),
    }


def active_joints(robot: HandRobot, branch: str, include_wrist: bool = True) -> list[int]:
    return ([1, 2] if include_wrist else []) + list(robot.branches[branch])


def branch_jacobian(robot: HandRobot, q, branch: str, include_wrist: bool = True, T=None) -> np.ndarray:
    """3 x n_active position Jacobian (zero columns of the other fingers removed)."""
    J = position_jacobian(robot, q, branch, T)
    cols = [j - 1 for j in active_joints(robot, branch, include_wrist)]
    return J[:, cols]


def analyze_branch(robot: HandRobot, q, branch: str, include_wrist: bool = True) -> dict:
    J = branch_jacobian(robot, q, branch, include_wrist)
    m = metrics(J)
    m["n_joints"] = J.shape[1]
    m["redundancy"] = J.shape[1] - 3                  # n - m for the position task
    if J.shape[0] == J.shape[1]:
        m["det"] = float(np.linalg.det(J))
    return m


def stacked_metrics(robot: HandRobot, q, branches: list[str]) -> dict:
    """Metrics of the stacked (3r x 20) Jacobian used for chords."""
    J = np.vstack([position_jacobian(robot, q, b) for b in branches])
    J = J[:, np.any(np.abs(J) > 1e-12, axis=0)]       # drop joints that move nobody
    m = metrics(J)
    m["shape"] = J.shape
    return m


def along_trajectory(robot: HandRobot, q_traj: np.ndarray, branches: list[str], include_wrist=True) -> dict:
    """sigma_min and w of each finger along a joint trajectory (N x 20)."""
    out = {b: {"sigma_min": np.zeros(len(q_traj)), "w": np.zeros(len(q_traj))} for b in branches}
    for k, q in enumerate(q_traj):
        T = robot.forward_kinematics(q)
        for b in branches:
            m = metrics(branch_jacobian(robot, q, b, include_wrist, T))
            out[b]["sigma_min"][k] = m["sigma_min"]
            out[b]["w"][k] = m["w"]
    return out


def distal_sweep(robot: HandRobot, q, branch: str, joint: int, values: np.ndarray) -> dict:
    """Sweep one joint and return det(J_finger) (finger joints only): zero = boundary singularity."""
    det, smin = [], []
    for v in values:
        qq = np.array(q, float).copy()
        qq[joint - 1] = v
        m = analyze_branch(robot, qq, branch, include_wrist=False)
        det.append(m.get("det", np.nan))
        smin.append(m["sigma_min"])
    return {"values": np.asarray(values), "det": np.asarray(det), "sigma_min": np.asarray(smin)}
