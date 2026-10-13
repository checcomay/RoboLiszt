"""Baseline controllers used in the control chapter (comparison with the hybrid controller of hybrid.py).

* JointPDController          : PD on the joint angles q_ref(t) (obtained by IK) + gravity/damping compensation.
* OperationalSpaceController : Cartesian spring-damper on the fingertip,  F = K (p_ref - p) + D (v_ref - v) [+ ki * int],
                               tau = J^T F + null-space posture + compensation.
  - "position control"  : stiff K, D critically damped for the free-space tip dynamics;
  - "impedance control" : low K, D chosen for the *contact* (K + k_key), inertia not shaped (M_d = Lambda), so that no
                          force measurement is needed:  Lambda (xdd_d - xdd) + D (xd_d - xd) + K (x_d - x) = F_ext.
All use the simplified dynamics model (uniform joint inertia) with a ``model_mismatch`` error on gravity/damping compensation.
"""
from __future__ import annotations

import numpy as np

from hand_robot.kinematics.jacobian import position_jacobian


class JointPDController:
    def __init__(self, robot, dynamics, q_ref_fn, kp=2.0, kd=0.3, model_mismatch=0.15):
        self.robot, self.dyn, self.q_ref_fn = robot, dynamics, q_ref_fn
        self.kp, self.kd, self.comp = kp, kd, 1.0 - model_mismatch
        self.log = {"mode": [], "z_ref": [], "F_des": []}

    def compute(self, t, q, qd, meas):
        q_ref, qd_ref = self.q_ref_fn(t)
        tau = self.kp * (q_ref - q) + self.kd * (qd_ref - qd)
        tau += self.comp * (self.dyn.gravity(q) + self.dyn.coriolis(q, qd))
        for k, v in (("mode", 0), ("z_ref", 0.0), ("F_des", 0.0)):
            self.log[k].append(v)
        return tau


class OperationalSpaceController:
    def __init__(self, robot, dynamics, config, branch, base, q_ref, ref_fn, K, D=None, zeta=1.0,
                 k_env=0.0, ki=0.0, model_mismatch=0.15, zero_vref_on_contact=False):
        """ref_fn(t) -> (p_ref (3,), v_ref (3,)) in the keyboard frame.
        K: stiffness (3,) [N/m]; D: damping (3,) or None -> D = 2 zeta sqrt((K + k_env) Lambda_ii) computed online."""
        c = config["control"]
        self.robot, self.dyn, self.branch = robot, dynamics, branch
        self.base, self.q_ref, self.ref_fn = np.asarray(base, float), np.asarray(q_ref, float), ref_fn
        self.K = np.asarray(K, float) * np.ones(3)
        self.D = None if D is None else np.asarray(D, float) * np.ones(3)
        self.zeta, self.k_env, self.ki = zeta, k_env, ki
        self.zero_vref_on_contact = zero_vref_on_contact      # contact-aware reference: no velocity feed-forward once in contact
        self.comp = 1.0 - model_mismatch
        self.kpq, self.kdq = float(c.get("kp_posture", 1.0)), float(c.get("kd_posture", 0.25))
        self.cols = np.zeros(20, dtype=bool)
        for j in [1, 2] + list(robot.branches[branch]):
            self.cols[j - 1] = True
        self.inertia = dynamics.inertia
        self.integ = np.zeros(3)
        self.t_prev = None
        self.log = {"mode": [], "z_ref": [], "F_des": [], "Fcmd_z": [], "Dz": []}

    def compute(self, t, q, qd, meas):
        robot = self.robot
        T = robot.forward_kinematics(q)
        p = (T[robot.branches[self.branch][-1]] @ robot._T_tip)[:3, 3] + self.base
        J = position_jacobian(robot, q, self.branch, T) * self.cols
        v = J @ qd
        p_ref, v_ref = self.ref_fn(t)
        if self.zero_vref_on_contact and float(meas.get("Fz", 0.0)) > 0.05:
            v_ref = np.zeros(3)
        dt = 0.0 if self.t_prev is None else t - self.t_prev
        self.t_prev = t
        Lam = self.inertia * np.linalg.inv(J @ J.T + 1e-9 * np.eye(3))         # operational-space inertia (M = I*inertia)
        D = self.D if self.D is not None else 2 * self.zeta * np.sqrt((self.K + self.k_env) * np.maximum(np.diag(Lam), 1e-6))
        e = p_ref - p
        self.integ = np.clip(self.integ + e * dt, -0.05, 0.05)
        F = self.K * e + D * (v_ref - v) + self.ki * self.integ
        tau = J.T @ F
        Ja = J[:, self.cols]
        Na = np.eye(Ja.shape[1]) - Ja.T @ np.linalg.solve(Ja @ Ja.T + 1e-6 * np.eye(3), Ja)
        tau[self.cols] += Na @ (-self.kpq * (q - self.q_ref)[self.cols] - self.kdq * qd[self.cols])
        tau[~self.cols] += (-self.kpq * (q - self.q_ref) - self.kdq * qd)[~self.cols]
        tau += self.comp * (self.dyn.gravity(q) + self.dyn.coriolis(q, qd))
        for k, val in (("mode", 0), ("z_ref", p_ref[2]), ("F_des", 0.0), ("Fcmd_z", F[2]), ("Dz", D[2])):
            self.log[k].append(val)
        return tau
