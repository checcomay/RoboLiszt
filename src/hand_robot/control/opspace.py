"""Operational-space hybrid position/force touch controller on the rigid-body model (Task 5, redesign).

Same task logic as ``HybridTouchController`` (APPROACH -> FORCE -> RETRACT, selection of x, y in position and z
in force) but with a dynamically consistent control law instead of joint-space PDs tuned on a toy inertia:

    Lambda = (J M^-1 J^T)^-1                       operational-space (tip) inertia, 3x3, from the rigid-body model
    F      = Lambda a_cmd                          free directions and approach/retract (a_cmd: PD + integral, in m/s^2)
    F_z    = -(Fd + kf e + ki int e) - c vz        contact: force PI + velocity damping (e = Fd - Fm)
    tau    = (1-mismatch)(h + g) + J^T F + N^T M a_post       N^T = I - J^T Jbar^T,  Jbar = M^-1 J^T Lambda

The gains of the motion loops are natural frequencies (rad/s), so they are independent of the (tiny and very
different) joint inertias. The term J_dot qd is neglected (tip speed <= 0.25 m/s).
Torques are clamped to the actuator limits; integrators are frozen while any actuator saturates (anti-windup).
``model_mismatch`` scales the model-based compensation of h + g (default 15 %, as in the simplified-model run).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hand_robot.control.hybrid import APPROACH, FORCE, RETRACT, TouchProfile
from hand_robot.dynamics.reduced import ReducedRigidBody
from hand_robot.planning.trajectory import quintic


@dataclass
class OpSpaceGains:
    wn_xy: float = 40.0           # rad/s
    zeta_xy: float = 1.0
    wn_z: float = 40.0            # rad/s (approach / retract)
    zeta_z: float = 1.0
    ki_pos: float = 0.0           # 1/s^3 acceleration units; 0 -> 0.25 * wn^3
    kf: float = 0.8               # proportional force gain
    ki_force: float = 3.0         # integral force gain
    c_force: float = 15.0         # N s/m velocity damping injected while in contact
    wn_post: float = 20.0         # rad/s null-space posture loop
    zeta_post: float = 1.0
    f_limit: float = 15.0         # N
    contact_threshold: float = 0.05
    free_gain: float = 1.0        # extra stiffness factor of the z loop in free space
    model_mismatch: float = 0.15

    @classmethod
    def from_config(cls, config: dict) -> "OpSpaceGains":
        c = config.get("touch_rigid", {})
        names = cls.__dataclass_fields__.keys()
        return cls(**{k: float(v) for k, v in c.items() if k in names})


class OperationalSpaceTouchController:
    def __init__(self, model: ReducedRigidBody, config: dict, base: np.ndarray, q_ref_a: np.ndarray,
                 xy_ref: np.ndarray, z_hover: float, z_floor: float, profile: TouchProfile,
                 gains: OpSpaceGains | None = None, tau_max_finger: float | None = None,
                 tau_max_wrist: float | None = None):
        self.m = model
        self.g = gains or OpSpaceGains.from_config(config)
        self.base = np.asarray(base, float)
        self.q_ref = np.asarray(q_ref_a, float)
        self.xy_ref = np.asarray(xy_ref, float)
        self.z_hover, self.z_floor, self.p = z_hover, z_floor, profile
        act = config["actuators"]
        tw = float(act["tau_max_wrist"]) if tau_max_wrist is None else float(tau_max_wrist)
        tf = float(act["tau_max_finger"]) if tau_max_finger is None else float(tau_max_finger)
        self.tau_max = np.array([tw if j in (1, 2) else tf for j in model.active])
        self.comp = 1.0 - self.g.model_mismatch
        self.reset()

    def reset(self):
        self.mode, self.t_prev, self.t_c, self.t_r = APPROACH, None, None, None
        self.z_ref = None
        self.integ_f = 0.0
        self.ipos = np.zeros(3)
        self.n_sat = 0
        self.n_steps = 0
        self.log = {"mode": [], "z_ref": [], "F_des": [], "sat": []}

    # ------------------------------------------------------------------------------------------
    def compute(self, t, qa, qd, meas) -> np.ndarray:
        g, p, m = self.g, self.p, self.m
        T = m.kin(qa)
        J = m.jacobian(T)
        M = m.mass_matrix(T)
        hg = m.bias(T, qd)
        pos = m.tip(T) + self.base
        v = J @ qd
        Fz = float(meas["Fz"])
        dt = 0.0 if self.t_prev is None else t - self.t_prev
        self.t_prev = t
        if self.z_ref is None:
            self.z_ref = pos[2]

        vz_ref, F_des = 0.0, 0.0
        if self.mode == APPROACH:
            vz_ref = -p.v_approach * float(quintic(t / 0.15))
            self.z_ref = max(self.z_ref + vz_ref * dt, self.z_floor)
            if Fz > g.contact_threshold:
                self.mode, self.t_c, self.integ_f = FORCE, t, 0.0
        if self.mode == FORCE:
            tc = t - self.t_c
            if tc < p.t_attack:
                F_des = p.f_hold * float(quintic(tc / p.t_attack))
            elif tc < p.t_attack + p.t_hold:
                F_des = p.f_hold
            elif tc < p.t_attack + p.t_hold + p.t_release:
                F_des = p.f_hold * (1.0 - float(quintic((tc - p.t_attack - p.t_hold) / p.t_release)))
            else:
                self.mode, self.t_r, self.z_ref = RETRACT, t, pos[2]
                self.ipos[2] = 0.0
        if self.mode == RETRACT:
            vz_ref = p.v_retract * float(quintic((t - self.t_r) / 0.15))
            self.z_ref = min(self.z_ref + vz_ref * dt, self.z_hover)

        # operational-space inertia, dynamically consistent null space
        Minv = np.linalg.inv(M)
        Lam = np.linalg.inv(J @ Minv @ J.T + 1e-9 * np.eye(3))
        Jbar = Minv @ J.T @ Lam

        # motion loops (acceleration commands)
        e_pos = np.array([self.xy_ref[0] - pos[0], self.xy_ref[1] - pos[1], self.z_ref - pos[2]])
        wxy, wz = g.wn_xy, g.wn_z * np.sqrt(g.free_gain)
        kp = np.array([wxy ** 2, wxy ** 2, wz ** 2])
        kd = np.array([2 * g.zeta_xy * wxy, 2 * g.zeta_xy * wxy, 2 * g.zeta_z * wz])
        if g.ki_pos > 0:
            ki = g.ki_pos * np.ones(3)
        else:                                       # default: a quarter of the stability limit 2 zeta wn^3
            ki = 0.25 * np.array([wxy ** 3, wxy ** 3, wz ** 3])
        vref = np.array([0.0, 0.0, vz_ref])
        a_cmd = kp * e_pos + kd * (vref - v) + ki * self.ipos
        F = Lam @ a_cmd
        in_force = self.mode == FORCE
        err_f = F_des - Fz
        if in_force:
            f_down = np.clip(F_des + g.kf * err_f + g.ki_force * self.integ_f, -g.f_limit, g.f_limit)
            F_xy = Lam[:2, :2] @ a_cmd[:2]
            F = np.array([F_xy[0], F_xy[1], -f_down - g.c_force * v[2]])

        # null-space posture (acceleration units, scaled by M)
        a0 = g.wn_post ** 2 * (self.q_ref - qa) - 2 * g.zeta_post * g.wn_post * qd
        tau_null = (np.eye(m.n) - J.T @ Jbar.T) @ (M @ a0)

        tau = self.comp * hg + J.T @ F + tau_null
        sat = np.abs(tau) > self.tau_max
        tau = np.clip(tau, -self.tau_max, self.tau_max)
        self.n_steps += 1
        if sat.any():
            self.n_sat += 1
        else:                                       # anti-windup: integrate only when no actuator saturates
            self.ipos[:2] = np.clip(self.ipos[:2] + e_pos[:2] * dt, -0.05, 0.05)
            if in_force:
                self.integ_f = float(np.clip(self.integ_f + err_f * dt, -3.0, 3.0))
                self.ipos[2] = 0.0
            else:
                self.ipos[2] = float(np.clip(self.ipos[2] + e_pos[2] * dt, -0.05, 0.05))
        for key, val in (("mode", self.mode), ("z_ref", self.z_ref), ("F_des", F_des), ("sat", bool(sat.any()))):
            self.log[key].append(val)
        return tau
