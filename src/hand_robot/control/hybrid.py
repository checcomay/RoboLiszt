from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hand_robot.kinematics.jacobian import position_jacobian
from hand_robot.planning.trajectory import quintic

APPROACH, FORCE, RETRACT = 0, 1, 2


@dataclass
class TouchProfile:
    """How the key is touched: approach speed sets the 'dynamics' (loudness), force the hold level."""
    name: str
    v_approach: float      # m/s, descent speed before contact
    f_hold: float          # N, desired normal force once in contact
    t_attack: float        # s, force ramp-up after contact
    t_hold: float = 0.25   # s
    t_release: float = 0.15
    v_retract: float = 0.10


class HybridTouchController:
    """Hybrid position/force controller of one fingertip (operational space, selection matrix S).

        x, y : position control   F = Kp e - Kd v
        z    : approach/retract -> position control with velocity feed-forward
               contact          -> force control  F_down = Fd + kf (Fd - F) + ki \u222b(Fd - F)
        tau  = J^T F  +  N (posture PD)  +  gravity / damping compensation

    ``J`` contains the wrist and the joints of the controlled finger (the contact force loads the wrist
    much more than the finger joints); the remaining redundancy is resolved with a null-space posture PD,
    the other fingers are held by a joint-level PD. Gravity and joint damping are compensated
    with the (imperfect) model: ``model_mismatch`` is the relative error of that compensation.
    """

    def __init__(self, robot, dynamics, config: dict, branch: str, base: np.ndarray, q_ref: np.ndarray,
                 xy_ref: np.ndarray, z_hover: float, z_floor: float, profile: TouchProfile,
                 model_mismatch: float = 0.15, free_gain: float = 4.0):
        c = config["control"]
        self.robot, self.dyn, self.branch = robot, dynamics, branch
        self.base = np.asarray(base, float)
        self.q_ref = np.asarray(q_ref, float)
        self.xy_ref = np.asarray(xy_ref, float)
        self.z_hover, self.z_floor = z_hover, z_floor
        self.p = profile
        self.kp, self.kd = float(c["kp_position"]), float(c["kd_position"])
        self.kf, self.ki = float(c["kp_force"]), float(c.get("ki_force", 0.0))
        self.f_limit = float(c["force_limit"])
        self.kip = float(c.get("ki_position", 0.0))
        self.kpq, self.kdq = float(c.get("kp_posture", 1.0)), float(c.get("kd_posture", 0.25))
        self.comp = 1.0 - model_mismatch
        self.free_gain = free_gain
        self.cols = np.zeros(20, dtype=bool)                # active joints: wrist (1, 2) + the controlled finger
        for j in [1, 2] + list(robot.branches[branch]):
            self.cols[j - 1] = True
        self.reset()

    def reset(self):
        self.mode, self.t_prev, self.t_c, self.t_r = APPROACH, None, None, None
        self.z_ref, self.integ = None, 0.0
        self.ipos = np.zeros(3)
        self.log = {"mode": [], "z_ref": [], "F_des": []}

    # ---------------------------------------------------------------------
    def compute(self, t, q, qd, meas) -> np.ndarray:
        robot, p = self.robot, self.p
        T = robot.forward_kinematics(q)
        pos = (T[robot.branches[self.branch][-1]] @ robot._T_tip)[:3, 3] + self.base
        J = position_jacobian(robot, q, self.branch, T) * self.cols
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
            if Fz > 0.05:                                          # contact detected -> switch to force control
                self.mode, self.t_c, self.integ = FORCE, t, 0.0
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
        if self.mode == RETRACT:
            vz_ref = p.v_retract * float(quintic((t - self.t_r) / 0.15))
            self.z_ref = min(self.z_ref + vz_ref * dt, self.z_hover)

        F = np.zeros(3)
        e_xy = self.xy_ref - pos[:2]
        self.ipos[:2] = np.clip(self.ipos[:2] + e_xy * dt, -0.05, 0.05)
        F[:2] = self.kp * e_xy + self.kip * self.ipos[:2] - self.kd * v[:2]
        if self.mode != FORCE:                      # integral action of the z position loop only outside contact
            self.ipos[2] = float(np.clip(self.ipos[2] + (self.z_ref - pos[2]) * dt, -0.05, 0.05))
        else:
            self.ipos[2] = 0.0
        if self.mode == FORCE:
            err = F_des - Fz
            self.integ = float(np.clip(self.integ + err * dt, -3.0, 3.0))
            f_down = np.clip(F_des + self.kf * err + self.ki * self.integ, -self.f_limit, self.f_limit)
            F[2] = -f_down - 0.5 * self.kd * v[2]                  # extra damping of the finger motion
        else:
            g = self.free_gain                              # stiffer vertical loop in free space (contact-free)
            F[2] = g * self.kp * (self.z_ref - pos[2]) + self.kip * self.ipos[2] + np.sqrt(g) * self.kd * (vz_ref - v[2])

        tau = J.T @ F
        # null-space posture (active joints only) + joint-level PD on the passive joints (other fingers)
        Ja = J[:, self.cols]
        Na = np.eye(Ja.shape[1]) - Ja.T @ np.linalg.solve(Ja @ Ja.T + 1e-6 * np.eye(3), Ja)
        e_q = (q - self.q_ref)[self.cols]
        tau[self.cols] += Na @ (-self.kpq * e_q - self.kdq * qd[self.cols])
        tau[~self.cols] += (-self.kpq * (q - self.q_ref) - self.kdq * qd)[~self.cols]
        tau += self.comp * (self.dyn.gravity(q) + self.dyn.coriolis(q, qd))
        for key, val in (("mode", self.mode), ("z_ref", self.z_ref), ("F_des", F_des)):
            self.log[key].append(val)
        return tau
