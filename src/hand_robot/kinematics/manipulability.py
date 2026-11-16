"""Manipulability ellipsoids (velocity, force, dynamic) of a fingertip.

All ellipsoids are computed for the 3 x n position Jacobian restricted to the *active* joints of a finger
(wrist optionally included). Unit-norm bounds are used: |qd| <= 1 rad/s, |tau| <= 1 N*m.
"""
from __future__ import annotations

import numpy as np

from .singularity import branch_jacobian, active_joints


def svd_axes(J: np.ndarray):
    """Principal directions U (3x3) and singular values s of J (rank-3 case)."""
    U, s, _ = np.linalg.svd(J, full_matrices=False)
    return U, s


def velocity_ellipsoid(J):
    """Semi-axes sigma_i along U: set of tip velocities for |qd| <= 1."""
    return svd_axes(J)


def force_ellipsoid(J):
    """Semi-axes 1/sigma_i along U: set of tip forces sustainable with |tau| <= 1."""
    U, s = svd_axes(J)
    return U, 1.0 / np.maximum(s, 1e-12)


def directional_velocity(J, u) -> float:
    """Largest tip speed along the unit vector u for |qd| <= 1  (= 1/sqrt(u^T (J J^T)^-1 u))."""
    u = np.asarray(u, float) / np.linalg.norm(u)
    return float(1.0 / np.sqrt(u @ np.linalg.inv(J @ J.T) @ u))


def directional_force(J, u) -> float:
    """Largest force along u for |tau| <= 1  (= 1/sqrt(u^T J J^T u))."""
    u = np.asarray(u, float) / np.linalg.norm(u)
    return float(1.0 / np.sqrt(u @ (J @ J.T) @ u))


def dynamic_matrix(robot, rbd, q, branch, include_wrist=True, tau_max=None):
    """G = J_a M_aa^-1 diag(tau_max): tip acceleration for normalised active torques |tau_i/tau_max_i| <= 1
    (other joints locked). With tau_max=None unit torques (1 N*m) are used."""
    T = robot.forward_kinematics(q)
    J = branch_jacobian(robot, q, branch, include_wrist, T)
    act = [j - 1 for j in active_joints(robot, branch, include_wrist)]
    M = rbd.mass_matrix(q)[np.ix_(act, act)]
    Tm = np.eye(len(act)) if tau_max is None else np.diag(np.asarray(tau_max, float))
    return J @ np.linalg.inv(M) @ Tm, J, M


def directional_acceleration(G, u) -> float:
    """Largest tip acceleration along u for |tau| <= 1  (= 1/sqrt(u^T (G G^T)^-1 u))."""
    u = np.asarray(u, float) / np.linalg.norm(u)
    return float(1.0 / np.sqrt(u @ np.linalg.inv(G @ G.T) @ u))


def ellipsoid_surface(center, U, radii, n=24):
    """Points of the ellipsoid  center + U diag(radii) x,  |x| = 1  (for plotting)."""
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n // 2 + 1)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    P = U @ (np.asarray(radii)[:, None] * np.vstack([x.ravel(), y.ravel(), z.ravel()]))
    return (P.T + np.asarray(center)).T.reshape(3, *x.shape)
