"""Task 5 - controllo dinamico del tocco e interazione ibrida (posizione x,y / forza z) con contatto tasto."""
from __future__ import annotations

import numpy as np

from hand_robot.control.hybrid import HybridTouchController, TouchProfile
from hand_robot.dynamics.model import DynamicsModel
from hand_robot.dynamics.simulation import simulate_closed_loop
from hand_robot.kinematics.jacobian import position_jacobian
from .common import Scene, TaskResult, FINGERS
from .task2_single_key import DEFAULT_NOTES

PROFILES = [
    TouchProfile("pp", v_approach=0.05, f_hold=1.0, t_attack=0.20),
    TouchProfile("mf", v_approach=0.12, f_hold=2.5, t_attack=0.12),
    TouchProfile("ff", v_approach=0.25, f_hold=5.0, t_attack=0.06),
]


class KeyContactEnv:
    """Fingertip <-> key contact: unilateral spring-damper with a hard stop at the key bottom."""

    def __init__(self, scene: Scene, branch: str, base):
        self.s, self.branch, self.base = scene, branch, np.asarray(base, float)

    def __call__(self, t, q, qd):
        r = self.s.robot
        T = r.forward_kinematics(q)
        p = (T[r.branches[self.branch][-1]] @ r._T_tip)[:3, 3] + self.base
        v = position_jacobian(r, q, self.branch, T) @ qd
        Fn = self.s.kb.contact_force(p[2], v[2])
        return {self.branch: np.array([0.0, 0.0, Fn])}, {"Fz": Fn, "z": p[2], "vz": v[2], "x": p[0], "y": p[1]}


def start_posture(scene: Scene, finger: str = "middle") -> np.ndarray:
    """Posture the touching finger starts from: the press posture of task 2 (wrist already pitched, neighbours above
    the keys), with the finger raised to the safe height above its key and everything else frozen."""
    notes = DEFAULT_NOTES
    note, branch = notes[finger], FINGERS[finger]
    base, kb = scene.base_home, scene.kb
    q_ready = scene.ready_posture(notes)
    qs, phase = scene.press_sequence(notes, (finger,), q_ready, base, wrist_weight=10.0)
    q_hold = qs[[i for i, p in enumerate(phase) if p == "hold"][0]]
    fj = scene.robot.branches[branch]
    return scene.solve_robust({branch: kb.hover_point(note, kb.safe_height, finger)}, q_hold, base,
                              locked=tuple(j for j in range(1, 21) if j not in fj)).q


def _metrics(res_sim, ctrl, prof, kb, dt) -> dict:
    t, F, z, vz = res_sim["t"][1:], res_sim["Fz"], res_sim["z"], res_sim["vz"]
    mode = np.array(ctrl.log["mode"])
    i_c = int(np.argmax(F > 0.05)) if (F > 0.05).any() else 0
    hold = slice(i_c + int((prof.t_attack + 0.5 * prof.t_hold) / dt), i_c + int((prof.t_attack + prof.t_hold) / dt))
    f_ss = float(F[hold].mean()) if F[hold].size else float("nan")
    f_des = np.array(ctrl.log["F_des"])
    rms = float(np.sqrt(np.mean((F[hold] - f_des[hold]) ** 2))) if F[hold].size else float("nan")
    peak = float(F[i_c:].max())
    i90 = int(np.argmax(F[i_c:] >= 0.9 * prof.f_hold)) if (F[i_c:] >= 0.9 * prof.f_hold).any() else -1
    return {
        "v_impact_m_s": float(-vz[i_c]),
        "F_peak_N": peak,
        "overshoot_pct": float(100 * max(peak - prof.f_hold, 0) / prof.f_hold),
        "F_steady_N": f_ss,
        "steady_error_pct": float(100 * abs(f_ss - prof.f_hold) / prof.f_hold),      # error of the MEAN force
        "F_rms_error_pct": float(100 * rms / prof.f_hold),                          # RMS error (includes oscillations)
        "rise_time_90_ms": float(1e3 * i90 * dt) if i90 >= 0 else float("nan"),
        "max_penetration_mm": float(1e3 * (kb.z_surface - z.min())),
        "key_bottomed": bool(kb.z_surface - z.min() > kb.press_depth),
        "contact_time_s": float(t[i_c]),
    }


def run(scene: Scene, finger: str = "middle", profiles=None, duration: float = 2.6,
        model_mismatch: float = 0.15) -> list[TaskResult]:
    notes = DEFAULT_NOTES
    note, branch = notes[finger], FINGERS[finger]
    base = scene.base_home
    kb = scene.kb
    dyn = DynamicsModel(scene.robot, scene.config)

    q_start = start_posture(scene, finger)
    out = []
    for prof in (profiles or PROFILES):
        env = KeyContactEnv(scene, branch, base)
        ctrl = HybridTouchController(scene.robot, dyn, scene.config, branch, base, q_start,
                                     xy_ref=kb.surface_point(note, finger)[:2],
                                     z_hover=kb.z_surface + kb.safe_height,
                                     z_floor=kb.z_surface - kb.press_depth - 0.002,
                                     profile=prof, model_mismatch=model_mismatch)
        sim = simulate_closed_loop(dyn, ctrl, q_start, duration, scene.dt, env)
        mode = np.array(ctrl.log["mode"])
        phases = ["approach", "force", "retract"]
        phase = [phases[m] for m in np.r_[mode, mode[-1:]]]
        res = TaskResult(f"task5_touch_{prof.name}", sim["t"], sim["q"], np.tile(base, (len(sim["t"]), 1)), phase)
        res.metrics = {"duration_s": duration, **_metrics(sim, ctrl, prof, kb, scene.dt)}
        res.extras.update(Fz=sim["Fz"], z=sim["z"], vz=sim["vz"], z_ref=np.array(ctrl.log["z_ref"]),
                          F_des=np.array(ctrl.log["F_des"]), mode=mode, tau=sim["tau"][:-1],
                          profile=prof, finger=finger, note=note)
        out.append(res)
    return out
