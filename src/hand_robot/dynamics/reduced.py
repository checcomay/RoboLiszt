"""Reduced rigid-body model: wrist (joints 1, 2) + the joints of ONE finger are free, every other joint is locked.

The locked links are rigid: their mass, centre of mass and inertia are merged into the nearest free ancestor
(composite rigid body, constant in that link frame because the locked joints do not move). The result is a
serial chain of 4-5 free joints, whose Newton-Euler recursion is ~10x cheaper than the 20-joint tree and is
*exactly* the full model restricted to the free joints (verified against ``RigidBodyDynamics`` in the tests).

    M_a(q_a) qdd_a + h_a(q_a, qd_a) + g_a(q_a) + B qd_a = tau_a + J_a^T F_tip

Used by the Task 5 touch simulation (plant) and by the operational-space controller (model).
"""
from __future__ import annotations

import numpy as np

from hand_robot.dynamics.rigid_body import RigidBodyDynamics
from hand_robot.model.dh import dh_transform
from hand_robot.model.robot import HandRobot


class ReducedRigidBody:
    def __init__(self, rbd: RigidBodyDynamics, branch: str, q_locked: np.ndarray):
        self.rbd, self.robot, self.branch = rbd, rbd.robot, branch
        rb = self.robot
        self.active = [1, 2] + list(rb.branches[branch])           # free joints (1-based, tree order)
        self.n = len(self.active)
        self.idx = np.array([j - 1 for j in self.active])
        self.q_locked = rb.validate_q(q_locked).copy()
        self.B = rbd.B
        self.g = rbd.g.copy()
        self.par = {j: rb.parent[j] for j in self.active}          # parent is always 0, 1, 2 or a free finger joint
        self.par[1] = 0
        # _anc[k][i] = True if free joint i is an ancestor of (or equal to) free link k
        self._anc = np.zeros((self.n, self.n), dtype=bool)
        for k, j in enumerate(self.active):
            cur = j
            while cur in self.active:
                self._anc[k, self.active.index(cur)] = True
                cur = self.par[cur]
        self._build_composites()

    # ------------------------------------------------------------------ composites
    def _nearest_active(self, i: int) -> int:
        while i not in self.active:
            i = self.robot.parent[i]
        return i

    def _build_composites(self):
        rb, rbd = self.robot, self.rbd
        T = rb.forward_kinematics(self.q_locked)
        parts = {j: [(rbd.mass[j], rbd.com[j], rbd.inertia[j])] for j in self.active}
        for i in range(1, 21):
            if i in self.active:
                continue
            a = self._nearest_active(i)
            Trel = np.linalg.inv(T[a]) @ T[i]                      # constant: no free joint between a and i
            R, p = Trel[:3, :3], Trel[:3, 3]
            parts[a].append((rbd.mass[i], R @ rbd.com[i] + p, R @ rbd.inertia[i] @ R.T))
        self.mass, self.com, self.inertia = {}, {}, {}
        for j, lst in parts.items():
            M = sum(x[0] for x in lst)
            C = sum(x[0] * x[1] for x in lst) / M
            J = np.zeros((3, 3))
            for mi, ci, Ii in lst:
                d = ci - C
                J += Ii + mi * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
            self.mass[j], self.com[j], self.inertia[j] = M, C, J

    # ------------------------------------------------------------------ kinematics
    def full_q(self, qa: np.ndarray) -> np.ndarray:
        q = self.q_locked.copy()
        q[self.idx] = qa
        return q

    def kin(self, qa: np.ndarray) -> dict:
        """Frames ^0T_j of the free chain (and frame 2)."""
        rb = self.robot
        T = {0: np.eye(4)}
        for k, j in enumerate(self.active):
            p = rb.dh[j]
            T[j] = T[self.par[j]] @ dh_transform(p.a, p.alpha, p.d, qa[k] + p.theta_offset)
        return T

    def tip(self, T: dict) -> np.ndarray:
        return (T[self.active[-1]] @ self.robot._T_tip)[:3, 3]

    def jacobian(self, T: dict) -> np.ndarray:
        """3 x n position Jacobian of the fingertip with respect to the free joints."""
        p_tip = self.tip(T)
        J = np.zeros((3, self.n))
        for k, j in enumerate(self.active):
            Tp = T[self.par[j]]
            J[:, k] = np.cross(Tp[:3, 2], p_tip - Tp[:3, 3])
        return J

    # ------------------------------------------------------------------ dynamics
    def _rnea(self, T, qd, qdd, g):
        par = self.par
        w, al, ao = {0: np.zeros(3)}, {0: np.zeros(3)}, {0: np.zeros(3)}
        F, N, cw, ax = {}, {}, {}, {}
        for k, j in enumerate(self.active):
            p = par[j]
            a = T[p][:3, 2]
            ax[j] = a
            w[j] = w[p] + a * qd[k]
            al[j] = al[p] + a * qdd[k] + np.cross(w[p], a) * qd[k]
            r = T[j][:3, 3] - T[p][:3, 3]
            ao[j] = ao[p] + np.cross(al[j], r) + np.cross(w[j], np.cross(w[j], r))
            Rj = T[j][:3, :3]
            rc = Rj @ self.com[j]
            cw[j] = T[j][:3, 3] + rc
            ac = ao[j] + np.cross(al[j], rc) + np.cross(w[j], np.cross(w[j], rc))
            F[j] = self.mass[j] * (ac - g)
            Iw = Rj @ self.inertia[j] @ Rj.T
            N[j] = Iw @ al[j] + np.cross(w[j], Iw @ w[j])
        fs = {j: np.zeros(3) for j in [0] + self.active}
        ns = {j: np.zeros(3) for j in [0] + self.active}
        tau = np.zeros(self.n)
        for k in range(self.n - 1, -1, -1):
            j = self.active[k]
            p = par[j]
            Op, Oj = T[p][:3, 3], T[j][:3, 3]
            f = F[j] + fs[j]
            n = N[j] + np.cross(cw[j] - Op, F[j]) + ns[j] + np.cross(Oj - Op, fs[j])
            tau[k] = ax[j] @ n
            fs[p] += f
            ns[p] += n
        return tau

    def mass_matrix_rnea(self, T: dict) -> np.ndarray:
        """M by n RNEA calls (slow, used as reference in the tests)."""
        M = np.zeros((self.n, self.n))
        z = np.zeros(self.n)
        zero_g = np.zeros(3)
        for k in range(self.n):
            e = np.zeros(self.n)
            e[k] = 1.0
            M[:, k] = self._rnea(T, z, e, zero_g)
        return 0.5 * (M + M.T)

    def mass_matrix(self, T: dict) -> np.ndarray:
        """M = sum_j  m_j Jv_j^T Jv_j + Jw_j^T (R I R^T) Jw_j  with the COM Jacobians of the free links."""
        n = self.n
        Z = np.array([T[self.par[j]][:3, 2] for j in self.active])          # joint axes (n, 3)
        O = np.array([T[self.par[j]][:3, 3] for j in self.active])          # joint origins
        M = np.zeros((n, n))
        for k, j in enumerate(self.active):
            Rj = T[j][:3, :3]
            c = T[j][:3, 3] + Rj @ self.com[j]
            mask = self._anc[k]                                             # joints that move link j
            Jw = np.zeros((3, n))
            Jv = np.zeros((3, n))
            Jw[:, mask] = Z[mask].T
            Jv[:, mask] = np.cross(Z[mask], c - O[mask]).T
            Iw = Rj @ self.inertia[j] @ Rj.T
            M += self.mass[j] * Jv.T @ Jv + Jw.T @ Iw @ Jw
        return 0.5 * (M + M.T)

    def bias(self, T: dict, qd: np.ndarray) -> np.ndarray:
        """h + g (no friction)."""
        return self._rnea(T, qd, np.zeros(self.n), self.g)

    def gravity_torque(self, T: dict) -> np.ndarray:
        return self._rnea(T, np.zeros(self.n), np.zeros(self.n), self.g)


class ReducedPlant:
    """Time stepping of the reduced model with a linearly-implicit (viscous) semi-implicit Euler scheme.

    (M + dt B) qdd = tau + J^T F - h - g - B qd ;   qd <- qd + dt qdd ;   q <- q + dt qd.
    The viscous term is treated implicitly because the distal inertias (1e-8 .. 1e-7 kg m^2) make the explicit
    scheme unstable for dt > 2 I / B (about 0.02-0.2 ms)."""

    def __init__(self, model: ReducedRigidBody):
        self.m = model

    def state(self, qa, qd):
        T = self.m.kin(qa)
        J = self.m.jacobian(T)
        return T, self.m.tip(T), J, J @ qd

    def step(self, qa, qd, tau, F_tip, dt):
        """``F_tip``: external force on the fingertip (3,) or a callable ``(tip_position, tip_velocity) -> (3,)``
        evaluated at the start of the step (contact model). Returns (q, qd, qdd, F_applied)."""
        m = self.m
        T = m.kin(qa)
        M = m.mass_matrix(T)
        J = m.jacobian(T)
        F = F_tip(m.tip(T), J @ qd) if callable(F_tip) else np.asarray(F_tip, float)
        rhs = tau + J.T @ F - m.bias(T, qd) - m.B * qd
        qdd = np.linalg.solve(M + dt * m.B * np.eye(m.n), rhs)
        qd = qd + dt * qdd
        qa = qa + dt * qd
        return qa, qd, qdd, F
