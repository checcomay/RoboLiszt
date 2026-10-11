"""ROS-independent core of the Task 5 touch node (joint-state in, joint-effort out).

``gazebo/roboliszt_gazebo/task5_touch_node.py`` is a thin rclpy wrapper around ``TouchNodeCore``; everything that can be
tested without ROS lives here (tests/test_touch_node_core.py closes the loop with the reduced rigid-body plant).

Phases
  PRE   all 20 joints follow a quintic joint-space ramp from the first measured posture to the touch start posture
        (inertia-scaled PD + full-model gravity feed-forward), ``t_pre`` seconds;
  RUN   the operational-space controller drives wrist + the touching finger (5 joints); the other 15 joints keep the start
        posture with the same inertia-scaled PD + gravity feed-forward (in the Python simulation they are ideally locked).

Contact force: reconstructed from the key joint, F = max(k d + c d_dot, 0), d = key depression [m]. This is exact while the
key is above its end stop (pp: 3.5 mm of 8 mm travel); at the end stop (mf, ff) the Gazebo joint limit is a rigid
constraint whose force is NOT visible in the joint state, so F is underestimated there (a force/torque sensor is needed).
"""
from __future__ import annotations

import numpy as np

from hand_robot.control.hybrid import TouchProfile
from hand_robot.control.opspace import OperationalSpaceTouchController, OpSpaceGains
from hand_robot.dynamics.reduced import ReducedRigidBody
from hand_robot.dynamics.rigid_body import RigidBodyDynamics
from hand_robot.planning.trajectory import quintic
from hand_robot.tasks import task5_rigid as r5
from hand_robot.tasks.common import Scene

PRE, RUN = "pre", "run"


def key_joint_name(kb, note: str) -> str:
    """Name of the prismatic key joint in worlds/piano.sdf (key_00 = C3 ... key_21 = C6, C4 = key_07)."""
    return f"key_{kb.white_index(note) - kb.white_index('C4') + 7:02d}_joint"


class TouchNodeCore:
    def __init__(self, scene: Scene, finger: str, profile: TouchProfile, gains: OpSpaceGains | None = None,
                 tau_max_finger: float | None = None, t_pre: float = 2.0, wn_hold: float = 20.0,
                 gravity_every: int = 20):
        self.scene, self.kb, self.profile = scene, scene.kb, profile
        self.q_start, self.note, self.branch, self.base = r5.start_posture(scene, finger)
        self.rbd = RigidBodyDynamics(scene.robot, scene.config)
        self.model = ReducedRigidBody(self.rbd, self.branch, self.q_start)
        kb = self.kb
        self.ctrl = OperationalSpaceTouchController(
            self.model, scene.config, self.base, self.q_start[self.model.idx],
            xy_ref=kb.surface_point(self.note, finger)[:2], z_hover=kb.z_surface + kb.safe_height,
            z_floor=kb.z_surface - kb.press_depth - 0.002, profile=profile, gains=gains, tau_max_finger=tau_max_finger)
        act = scene.config["actuators"]
        tf = float(act["tau_max_finger"]) if tau_max_finger is None else float(tau_max_finger)
        self.tau_lim = np.full(20, tf)
        self.tau_lim[[0, 1]] = float(act["tau_max_wrist"])
        # per-joint inertia of the held joints -> PD gains that are stable whatever the (tiny, very different) inertias
        self.inertia = np.maximum(np.diag(self.rbd.mass_matrix(self.q_start)), 1e-9)
        self.kp_hold = self.inertia * wn_hold ** 2
        self.kd_hold = 2.0 * self.inertia * wn_hold
        self.t_pre, self.gravity_every = t_pre, gravity_every
        self.key_joint = key_joint_name(kb, self.note)
        self.reset()

    def reset(self):
        self.phase, self.t0, self.q_init, self._g, self._n = PRE, None, None, None, 0
        self.ctrl.reset()
        self.log = {k: [] for k in ("t", "q", "qd", "tau", "Fz", "z", "vz", "F_des", "mode", "sat")}

    # ------------------------------------------------------------------ sensing
    def key_force(self, key_pos: float, key_vel: float) -> float:
        d, dd = -float(key_pos), -float(key_vel)             # depression, depression rate (key joint is 0 at rest, < 0 pressed)
        return max(self.kb.k_key * d + self.kb.c_key * dd, 0.0) if d > 0 else 0.0

    # ------------------------------------------------------------------ control
    def _gravity(self, q):
        if self._g is None or self._n % self.gravity_every == 0:
            self._g = self.rbd.gravity_torque(q)
        return self._g

    def _hold(self, q, qd, q_ref, qd_ref):
        return self.kp_hold * (q_ref - q) + self.kd_hold * (qd_ref - qd) + self._gravity(q)

    def step(self, t: float, q: np.ndarray, qd: np.ndarray, key_pos: float, key_vel: float) -> np.ndarray:
        """One control step. q, qd: the 20 hand joints (j1..j20); returns the 20 joint efforts [N m]."""
        q, qd = np.asarray(q, float), np.asarray(qd, float)
        if self.t0 is None:
            self.t0, self.q_init = t, q.copy()
        ts = t - self.t0
        Fz = self.key_force(key_pos, key_vel)
        ran_ctrl = False
        if self.phase == PRE:
            s = float(quintic(ts / self.t_pre))
            q_ref = self.q_init + (self.q_start - self.q_init) * s
            qd_ref = np.zeros(20)
            tau = self._hold(q, qd, q_ref, qd_ref)
            if ts >= self.t_pre:
                self.phase, self.t_run = RUN, t
        else:
            tr = t - self.t_run
            tau = self._hold(q, qd, self.q_start, np.zeros(20))
            ia = self.model.idx
            tau[ia] = self.ctrl.compute(tr, q[ia], qd[ia], {"Fz": Fz})
            ran_ctrl = True
        self._n += 1
        tau = np.clip(tau, -self.tau_lim, self.tau_lim)
        if ran_ctrl:
            T = self.model.kin(q[self.model.idx])
            self.log["t"].append(t - self.t_run)
            for k, v in (("q", q[self.model.idx].copy()), ("qd", qd[self.model.idx].copy()),
                         ("tau", tau[self.model.idx].copy()), ("Fz", Fz),
                         ("z", self.model.tip(T)[2] + self.base[2]),
                         ("vz", float((self.model.jacobian(T) @ qd[self.model.idx])[2])),
                         ("F_des", self.ctrl.log["F_des"][-1]), ("mode", self.ctrl.log["mode"][-1]),
                         ("sat", self.ctrl.log["sat"][-1])):
                self.log[k].append(v)
        return tau

    def metrics(self) -> dict:
        lg = {k: np.asarray(v) for k, v in self.log.items()}
        return r5.metrics_from_log(lg, self.profile, self.kb, self.scene.q_min, self.scene.q_max, self.model.idx)
