"""Rigid-body dynamics of the branched hand (recursive Newton-Euler in the base frame).

    M(q) qdd + h(q, qd) + g(q) + B qd = tau  (+ J^T F_ext)

Every DH link i is a uniform thin rod joining O_{i-1} to O_i (plus tip / palm point masses); its inertial
parameters are derived from the DH geometry and from the ``rigid_body`` section of robot.yaml.
They are *design assumptions* meant to be replaced with CAD data.
"""
from __future__ import annotations

import numpy as np

from hand_robot.model.robot import HandRobot


def _rx(alpha: float) -> np.ndarray:
    c, s = np.cos(alpha), np.sin(alpha)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def _skew_inertia(m: float, size) -> np.ndarray:
    a, b, c = size
    return m / 12.0 * np.diag([b * b + c * c, a * a + c * c, a * a + b * b])


class RigidBodyDynamics:
    def __init__(self, robot: HandRobot, config: dict):
        self.robot = robot
        c = config["rigid_body"]
        self.rho, self.R = float(c["linear_density"]), float(c["link_radius"])
        self.m_min, self.m_tip = float(c["min_mass"]), float(c["tip_mass"])
        self.m_palm = float(c["palm_mass"])
        self.palm_com = np.asarray(c["palm_com"], float)
        self.palm_size = np.asarray(c["palm_size"], float)
        self.B = float(c["viscous_friction"])
        self.g = np.asarray(c["gravity"], float)
        self.mass = {}
        self.com = {}       # COM in the link frame
        self.inertia = {}   # inertia about the COM, link frame
        for i in range(1, 21):
            self._build_link(i)

    # ------------------------------------------------------------ parameters
    def _build_link(self, i: int):
        p = self.robot.dh[i]
        r = _rx(-p.alpha) @ np.array([-p.a, 0.0, -p.d])          # O_{i-1} expressed in frame i (constant)
        L = float(np.linalg.norm(r))
        m = max(self.rho * L, self.m_min)
        if L > 1e-9:
            u = r / L
            Ip, Ia = m * (L * L / 12.0 + self.R ** 2 / 4.0), 0.5 * m * self.R ** 2
            I = Ip * (np.eye(3) - np.outer(u, u)) + Ia * np.outer(u, u)
        else:
            I = 0.5 * m * self.R ** 2 * np.eye(3)
        com = 0.5 * r
        parts = [(m, com, I)]
        if i in {b[-1] for b in self.robot.branches.values()}:      # fingertip pad
            parts.append((self.m_tip, np.array([self.robot.L_TIP, 0.0, 0.0]), np.zeros((3, 3))))
        if i == 2:                                                    # palm
            parts.append((self.m_palm, self.palm_com, _skew_inertia(self.m_palm, self.palm_size)))
        M = sum(x[0] for x in parts)
        C = sum(x[0] * x[1] for x in parts) / M
        J = np.zeros((3, 3))
        for mi, ci, Ii in parts:                                      # parallel-axis theorem
            d = ci - C
            J += Ii + mi * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
        self.mass[i], self.com[i], self.inertia[i] = M, C, J

    # ---------------------------------------------------------------- RNEA
    def inverse_dynamics(self, q, qd=None, qdd=None, gravity: bool = True, friction: bool = True,
                         T: dict | None = None, base_acc=None, return_base_wrench: bool = False):
        """tau = RNEA(q, qd, qdd).

        ``base_acc``: acceleration of the hand base (carriage) in the base frame [m/s^2]; it is equivalent to a
        change of the gravity vector, g_eff = g - a_base. With ``return_base_wrench`` the force and moment (about
        O_0) that the carriage exerts on the hand are returned too."""
        rb = self.robot
        q = rb.validate_q(q)
        qd = np.zeros(20) if qd is None else rb.validate_q(qd)
        qdd = np.zeros(20) if qdd is None else rb.validate_q(qdd)
        T = rb.forward_kinematics(q) if T is None else T
        g = (self.g if gravity else np.zeros(3)).copy()
        if base_acc is not None:
            g = g - np.asarray(base_acc, float)
        par = rb.parent
        w, al, ao = {0: np.zeros(3)}, {0: np.zeros(3)}, {0: np.zeros(3)}
        F, N, cw, ax = {}, {}, {}, {}
        for i in range(1, 21):
            p = par[i]
            a = T[p][:3, 2]
            ax[i] = a
            w[i] = w[p] + a * qd[i - 1]
            al[i] = al[p] + a * qdd[i - 1] + np.cross(w[p], a) * qd[i - 1]
            r = T[i][:3, 3] - T[p][:3, 3]
            ao[i] = ao[p] + np.cross(al[i], r) + np.cross(w[i], np.cross(w[i], r))
            Ri = T[i][:3, :3]
            rc = Ri @ self.com[i]
            cw[i] = T[i][:3, 3] + rc
            ac = ao[i] + np.cross(al[i], rc) + np.cross(w[i], np.cross(w[i], rc))
            F[i] = self.mass[i] * (ac - g)
            Iw = Ri @ self.inertia[i] @ Ri.T
            N[i] = Iw @ al[i] + np.cross(w[i], Iw @ w[i])
        fs = {i: np.zeros(3) for i in range(0, 21)}
        ns = {i: np.zeros(3) for i in range(0, 21)}
        tau = np.zeros(20)
        for i in range(20, 0, -1):
            p = par[i]
            Op, Oi = T[p][:3, 3], T[i][:3, 3]
            f = F[i] + fs[i]
            n = N[i] + np.cross(cw[i] - Op, F[i]) + ns[i] + np.cross(Oi - Op, fs[i])
            tau[i - 1] = ax[i] @ n
            fs[p] += f
            ns[p] += n
        if friction:
            tau += self.B * qd
        if return_base_wrench:
            return tau, fs[0].copy(), ns[0].copy()
        return tau

    # --------------------------------------------------------------- matrices
    def mass_matrix(self, q) -> np.ndarray:
        T = self.robot.forward_kinematics(q)
        M = np.zeros((20, 20))
        for j in range(20):
            e = np.zeros(20)
            e[j] = 1.0
            M[:, j] = self.inverse_dynamics(q, None, e, gravity=False, friction=False, T=T)
        return 0.5 * (M + M.T)

    def gravity_torque(self, q) -> np.ndarray:
        return self.inverse_dynamics(q, None, None, gravity=True, friction=False)

    def bias(self, q, qd) -> np.ndarray:
        """h(q,qd) + g(q) (without friction)."""
        return self.inverse_dynamics(q, qd, None, gravity=True, friction=False)

    def coriolis_torque(self, q, qd) -> np.ndarray:
        return self.bias(q, qd) - self.gravity_torque(q)

    def forward_dynamics(self, q, qd, tau, external_torque=None) -> np.ndarray:
        rhs = np.asarray(tau, float) - self.bias(q, qd) - self.B * np.asarray(qd, float)
        if external_torque is not None:
            rhs = rhs + external_torque
        return np.linalg.solve(self.mass_matrix(q), rhs)

    # ----------------------------------------------------------------- energy
    def potential_energy(self, q) -> float:
        T = self.robot.forward_kinematics(q)
        return float(sum(-self.mass[i] * self.g @ (T[i][:3, :3] @ self.com[i] + T[i][:3, 3]) for i in range(1, 21)))

    def kinetic_energy(self, q, qd) -> float:
        qd = np.asarray(qd, float)
        return float(0.5 * qd @ self.mass_matrix(q) @ qd)

    def total_mass(self) -> float:
        return float(sum(self.mass.values()))
