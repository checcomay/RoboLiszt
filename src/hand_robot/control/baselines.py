"""Baseline motion controllers on the reduced rigid-body model (comparison with the hybrid controller of opspace.py).

* ``JointSpaceController``  : computed-torque PD on the active joints (wrist + finger), gains as natural frequencies:
                              tau = M (qdd_ref + 2 zeta wn (qd_ref - qd) + wn^2 (q_ref - q)) + (1 - mismatch)(h + g)
* ``MotionController``      : Cartesian spring-damper on the fingertip with dynamically consistent null space.
    form="accel" : a = wn^2 e + 2 zeta wn de [+ integral],  F = Lambda a       (gains independent of the tip inertia)
    form="force" : F = K e + D de [+ ki int e],  D = 2 zeta sqrt((K + k_env) Lambda_ii)      (K in N/m: "impedance")
    ``zero_vref_on_contact``: contact-aware reference (no velocity feed-forward once the contact force exceeds a threshold).
Both use the same torque limits, compensation mismatch and anti-windup as ``OperationalSpaceTouchController``.
"""
from __future__ import annotations

import numpy as np

from hand_robot.dynamics.reduced import ReducedRigidBody


def _limits(model: ReducedRigidBody, config: dict, tau_max_finger=None, tau_max_wrist=None):
    act = config["actuators"]
    tw = float(act["tau_max_wrist"]) if tau_max_wrist is None else float(tau_max_wrist)
    tf = float(act["tau_max_finger"]) if tau_max_finger is None else float(tau_max_finger)
    return np.array([tw if j in (1, 2) else tf for j in model.active])


class JointSpaceController:
    def __init__(self, model: ReducedRigidBody, config: dict, q_ref_fn, wn=40.0, zeta=1.0, model_mismatch=0.15,
                 tau_max_finger=None, tau_max_wrist=None):
        self.m, self.q_ref_fn, self.wn, self.zeta = model, q_ref_fn, wn, zeta
        self.comp = 1.0 - model_mismatch
        self.tau_max = _limits(model, config, tau_max_finger, tau_max_wrist)
        self.n_sat = self.n_steps = 0
        self.log = {"mode": [], "z_ref": [], "F_des": [], "sat": []}

    def compute(self, t, qa, qd, meas):
        m = self.m
        T = m.kin(qa)
        M, hg = m.mass_matrix(T), m.bias(T, qd)
        q_ref, qd_ref, qdd_ref = self.q_ref_fn(t)
        a = qdd_ref + 2 * self.zeta * self.wn * (qd_ref - qd) + self.wn ** 2 * (q_ref - qa)
        tau = self.comp * hg + M @ a
        sat = np.abs(tau) > self.tau_max
        tau = np.clip(tau, -self.tau_max, self.tau_max)
        self.n_steps += 1
        self.n_sat += int(sat.any())
        for k, v in (("mode", 0), ("z_ref", 0.0), ("F_des", 0.0), ("sat", bool(sat.any()))):
            self.log[k].append(v)
        return tau


class MotionController:
    def __init__(self, model: ReducedRigidBody, config: dict, base, q_ref_a, ref_fn, *, form="force",
                 K=100.0, wn=40.0, zeta=1.0, k_env=0.0, ki=0.0, zero_vref_on_contact=False,
                 contact_threshold=0.05, wn_post=20.0, zeta_post=1.0, model_mismatch=0.15,
                 tau_max_finger=None, tau_max_wrist=None):
        """ki (form 'force'): integral gain [N/(m s)]; ki (form 'accel'): -1 = no integral, 0 = default 0.25 wn^3, >0 = value."""
        self.m, self.base, self.q_ref, self.ref_fn = model, np.asarray(base, float), np.asarray(q_ref_a, float), ref_fn
        self.form, self.K, self.wn, self.zeta, self.k_env, self.ki = form, np.asarray(K, float) * np.ones(3), wn, zeta, k_env, ki
        self.zero_vref, self.thr = zero_vref_on_contact, contact_threshold
        self.wn_post, self.zeta_post, self.comp = wn_post, zeta_post, 1.0 - model_mismatch
        self.tau_max = _limits(model, config, tau_max_finger, tau_max_wrist)
        self.t_prev = None
        self.ipos = np.zeros(3)
        self.n_sat = self.n_steps = 0
        self.log = {"mode": [], "z_ref": [], "F_des": [], "sat": [], "Dz": []}

    def compute(self, t, qa, qd, meas):
        m = self.m
        T = m.kin(qa)
        J, M, hg = m.jacobian(T), m.mass_matrix(T), m.bias(T, qd)
        pos = m.tip(T) + self.base
        v = J @ qd
        p_ref, v_ref = self.ref_fn(t)
        if self.zero_vref and float(meas.get("Fz", 0.0)) > self.thr:
            v_ref = np.zeros(3)
        dt = 0.0 if self.t_prev is None else t - self.t_prev
        self.t_prev = t
        Minv = np.linalg.inv(M)
        Lam = np.linalg.inv(J @ Minv @ J.T + 1e-9 * np.eye(3))
        Jbar = Minv @ J.T @ Lam
        e = p_ref - pos
        if self.form == "accel":
            kp = self.wn ** 2 * np.ones(3)
            kd = 2 * self.zeta * self.wn * np.ones(3)
            if self.ki < 0:
                ai = 0.0
            else:
                ai = (self.ki if self.ki > 0 else 0.25 * self.wn ** 3) * self.ipos
            F = Lam @ (kp * e + kd * (v_ref - v) + ai)
            Dz = kd[2] * Lam[2, 2]
        else:
            D = 2 * self.zeta * np.sqrt((self.K + self.k_env) * np.maximum(np.diag(Lam), 1e-9))
            F = self.K * e + D * (v_ref - v) + self.ki * self.ipos
            Dz = D[2]
        a0 = self.wn_post ** 2 * (self.q_ref - qa) - 2 * self.zeta_post * self.wn_post * qd
        tau_null = (np.eye(m.n) - J.T @ Jbar.T) @ (M @ a0)
        tau = self.comp * hg + J.T @ F + tau_null
        sat = np.abs(tau) > self.tau_max
        tau = np.clip(tau, -self.tau_max, self.tau_max)
        self.n_steps += 1
        if sat.any():
            self.n_sat += 1
        else:
            self.ipos = np.clip(self.ipos + e * dt, -0.05, 0.05)
        for k, val in (("mode", 0), ("z_ref", p_ref[2]), ("F_des", 0.0), ("sat", bool(sat.any())), ("Dz", Dz)):
            self.log[k].append(val)
        return tau
