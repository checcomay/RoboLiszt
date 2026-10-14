"""Task 5 (redesign) - touch control on the rigid-body model with the operational-space controller.

Plant: reduced rigid-body model (wrist + the controlled finger free, all other joints locked: ideal hold),
linearly-implicit Euler at ``dt_plant``; the controller runs at ``dt_ctrl`` (zero-order hold of the torques)
and reads only q, qd and the contact force Fz (as it will from /joint_states and the key joint in Gazebo).
Contact: the same unilateral spring-damper with hard stop of the keyboard model.
"""
from __future__ import annotations

import numpy as np

from hand_robot.control.hybrid import TouchProfile
from hand_robot.control.opspace import OperationalSpaceTouchController, OpSpaceGains
from hand_robot.dynamics.reduced import ReducedPlant, ReducedRigidBody
from hand_robot.dynamics.rigid_body import RigidBodyDynamics
from .common import Scene, TaskResult, FINGERS
from .task2_single_key import DEFAULT_NOTES
from .task5_touch import PROFILES


def start_posture(scene: Scene, finger: str = "middle"):
    """Same initial posture as the simplified-model run: bottom posture of task 2, finger lifted to the safe height."""
    notes = DEFAULT_NOTES
    note, branch = notes[finger], FINGERS[finger]
    base, kb = scene.base_home, scene.kb
    q_ready = scene.ready_posture(notes)
    qs, phase = scene.press_sequence(notes, (finger,), q_ready, base, wrist_weight=10.0)
    q_hold = qs[[i for i, p in enumerate(phase) if p == "hold"][0]]
    fj = scene.robot.branches[branch]
    q_start = scene.solve_robust({branch: kb.hover_point(note, kb.safe_height, finger)}, q_hold, base,
                                 locked=tuple(j for j in range(1, 21) if j not in fj)).q
    return q_start, note, branch, base


def metrics_from_log(lg: dict, prof: TouchProfile, kb, q_min, q_max, active_idx) -> dict:
    t, F, z, vz, f_des = lg["t"], lg["Fz"], lg["z"], lg["vz"], lg["F_des"]
    dt = float(t[1] - t[0])
    in_c = F > 0.05
    i_c = int(np.argmax(in_c)) if in_c.any() else 0
    n_att, n_hold, n_rel = (int(round(x / dt)) for x in (prof.t_attack, prof.t_hold, prof.t_release))
    hold = slice(i_c + n_att + n_hold // 2, i_c + n_att + n_hold)
    full = slice(i_c, i_c + n_att + n_hold + n_rel)
    f_ss = float(F[hold].mean()) if F[hold].size else float("nan")
    rms_hold = float(np.sqrt(np.mean((F[hold] - f_des[hold]) ** 2))) if F[hold].size else float("nan")
    rms_full = float(np.sqrt(np.mean((F[full] - f_des[full]) ** 2))) if F[full].size else float("nan")
    peak = float(F[i_c:].max())
    ok90 = F[i_c:] >= 0.9 * prof.f_hold
    i90 = int(np.argmax(ok90)) if ok90.any() else -1
    # bounce: number of contact losses (F drops to 0) between the first contact and the end of the hold
    # (counted only once the key has really been pressed, i.e. penetration > 1 mm: the first ~10 ms of the impact
    # transient dither around the 0.05 N threshold with a sub-0.1 mm penetration and are not bounces)
    win = F[i_c:i_c + n_att + n_hold]
    armed = np.maximum.accumulate(z[i_c:i_c + n_att + n_hold] < kb.z_surface - 1e-3)
    seg = (win > 0.05)
    losses = int(np.sum(armed[1:] & seg[:-1] & ~seg[1:]))
    q = lg["q"]
    margin = float(np.degrees(np.min(np.minimum(q - q_min[active_idx], q_max[active_idx] - q))))
    return {
        "v_impact_m_s": float(-vz[max(i_c - 1, 0)]),          # last sample before the first contact force
        "F_peak_N": peak,
        "overshoot_pct": float(100 * max(peak - prof.f_hold, 0) / prof.f_hold),
        "F_steady_N": f_ss,
        "steady_error_pct": float(100 * abs(f_ss - prof.f_hold) / prof.f_hold),
        "F_rms_error_pct": float(100 * rms_hold / prof.f_hold),                     # hold window (as in the report)
        "F_rms_error_contact_pct": float(100 * rms_full / prof.f_hold),             # whole contact window (stricter)
        "rise_time_90_ms": float(1e3 * i90 * dt) if i90 >= 0 else float("nan"),
        "max_penetration_mm": float(1e3 * (kb.z_surface - z.min())),
        "key_bottomed": bool(kb.z_surface - z.min() > kb.press_depth),
        "contact_losses": losses,
        "contact_time_s": float(t[i_c]),
        "final_z_offset_mm": float(1e3 * (z[-1] - (kb.z_surface + kb.safe_height))),
        "tau_peak_wrist_mNm": float(1e3 * np.abs(lg["tau"][:, :2]).max()),
        "tau_peak_finger_mNm": float(1e3 * np.abs(lg["tau"][:, 2:]).max()),
        "saturated_pct": float(100 * np.mean(lg["sat"])),
        "qd_max_rad_s": float(np.abs(lg["qd"]).max()),
        "joint_limit_margin_deg": margin,
    }


def simulate_touch(scene: Scene, prof: TouchProfile, q_start, note, branch, base, finger="middle",
                   gains: OpSpaceGains | None = None, duration: float = 2.6, dt_ctrl: float | None = None,
                   dt_plant: float | None = None, rbd: RigidBodyDynamics | None = None,
                   tau_max_finger: float | None = None, tau_max_wrist: float | None = None):
    kb = scene.kb
    tc = scene.config.get("touch_rigid", {})
    dt_ctrl = float(tc.get("dt_ctrl", 1e-3)) if dt_ctrl is None else dt_ctrl
    dt_plant = float(tc.get("dt_plant", 5e-4)) if dt_plant is None else dt_plant
    rbd = rbd or RigidBodyDynamics(scene.robot, scene.config)
    model = ReducedRigidBody(rbd, branch, q_start)
    plant = ReducedPlant(model)
    ctrl = OperationalSpaceTouchController(model, scene.config, base, q_start[model.idx],
                                           xy_ref=kb.surface_point(note, finger)[:2],
                                           z_hover=kb.z_surface + kb.safe_height,
                                           z_floor=kb.z_surface - kb.press_depth - 0.002,
                                           profile=prof, gains=gains, tau_max_finger=tau_max_finger,
                                           tau_max_wrist=tau_max_wrist)
    n_sub = int(round(dt_ctrl / dt_plant))
    steps = int(round(duration / dt_ctrl))
    base = np.asarray(base, float)
    qa = q_start[model.idx].copy()
    qd = np.zeros(model.n)
    keys = ["t", "q", "qd", "Fz", "z", "vz", "tau", "F_des", "z_ref", "mode", "sat"]
    lg = {k: [] for k in keys}

    def contact(tip, v):
        return np.array([0.0, 0.0, kb.contact_force(tip[2] + base[2], v[2])])

    diverged = False
    for k in range(steps):
        t = k * dt_ctrl
        T = model.kin(qa)
        tip = model.tip(T) + base
        v = model.jacobian(T) @ qd
        Fz = kb.contact_force(tip[2], v[2])
        tau = ctrl.compute(t, qa, qd, {"Fz": Fz})
        for s in range(n_sub):
            ts = t + s * dt_plant
            qa, qd, qdd, F = plant.step(qa, qd, tau, contact, dt_plant)
            T = model.kin(qa)
            tp = model.tip(T) + base
            vp = model.jacobian(T) @ qd
            for key, val in (("t", ts + dt_plant), ("q", qa.copy()), ("qd", qd.copy()), ("Fz", F[2]),
                             ("z", tp[2]), ("vz", vp[2]), ("tau", tau.copy()), ("F_des", ctrl.log["F_des"][-1]),
                             ("z_ref", ctrl.log["z_ref"][-1]), ("mode", ctrl.log["mode"][-1]),
                             ("sat", ctrl.log["sat"][-1])):
                lg[key].append(val)
        if not np.all(np.isfinite(qa)) or np.abs(qd).max() > 1e3:
            diverged = True
            break
    lg = {k: np.asarray(v) for k, v in lg.items()}
    return lg, ctrl, model, diverged


def run(scene: Scene, finger: str = "middle", profiles=None, gains: OpSpaceGains | None = None,
        duration: float = 2.6, dt_ctrl: float | None = None, dt_plant: float | None = None,
        tau_max_finger: float | None = None, tau_max_wrist: float | None = None,
        tag: str = "") -> list[TaskResult]:
    q_start, note, branch, base = start_posture(scene, finger)
    rbd = RigidBodyDynamics(scene.robot, scene.config)
    out = []
    for prof in (profiles or PROFILES):
        lg, ctrl, model, diverged = simulate_touch(scene, prof, q_start, note, branch, base, finger, gains,
                                                   duration, dt_ctrl, dt_plant, rbd, tau_max_finger, tau_max_wrist)
        if diverged:
            raise RuntimeError(f"task5 rigid ({prof.name}): numerical divergence")
        q_full = np.tile(q_start, (len(lg["t"]), 1))
        q_full[:, model.idx] = lg["q"]
        phases = ["approach", "force", "retract"]
        res = TaskResult(f"task5rb_{prof.name}{tag}", lg["t"], q_full, np.tile(base, (len(lg["t"]), 1)),
                         [phases[m] for m in lg["mode"]])
        res.metrics = {"duration_s": duration,
                       **metrics_from_log(lg, prof, scene.kb, scene.q_min, scene.q_max, model.idx)}
        res.extras.update(Fz=lg["Fz"], z=lg["z"], vz=lg["vz"], z_ref=lg["z_ref"], F_des=lg["F_des"],
                          mode=lg["mode"], tau=lg["tau"], qd=lg["qd"], profile=prof, finger=finger, note=note)
        out.append(res)
    return out


def static_force_limit(scene: Scene, finger: str = "middle", tau_max_finger: float | None = None,
                       tau_max_wrist: float | None = None) -> dict:
    """Largest static fingertip force (vertical, pressing down) the actuators can sustain in the key-bottom posture.

    The joint torques that balance a tip force F are tau = g(q) - J^T F (no freedom: they follow from statics),
    so |tau_i| <= tau_max_i bounds F joint by joint; the smallest bound is the limit of the hand in that posture."""
    notes = DEFAULT_NOTES
    branch = FINGERS[finger]
    base = scene.base_home
    q_ready = scene.ready_posture(notes)
    qs, phase = scene.press_sequence(notes, (finger,), q_ready, base, wrist_weight=10.0)
    q_hold = qs[[i for i, p in enumerate(phase) if p == "hold"][0]]
    rbd = RigidBodyDynamics(scene.robot, scene.config)
    model = ReducedRigidBody(rbd, branch, q_hold)
    T = model.kin(q_hold[model.idx])
    J, g = model.jacobian(T), model.gravity_torque(T)
    act = scene.config["actuators"]
    tw = float(act["tau_max_wrist"]) if tau_max_wrist is None else tau_max_wrist
    tf = float(act["tau_max_finger"]) if tau_max_finger is None else tau_max_finger
    tmax = np.array([tw if j in (1, 2) else tf for j in model.active])
    fmax = []
    for i in range(model.n):
        if abs(J[2, i]) < 1e-9:
            fmax.append(float("inf"))
            continue
        bounds = sorted([(g[i] - tmax[i]) / J[2, i], (g[i] + tmax[i]) / J[2, i]])
        fmax.append(float(max(bounds[1], 0.0)))
    return {"joints": model.active, "Jz": J[2].tolist(), "gravity_torque_Nm": g.tolist(), "tau_max_Nm": tmax.tolist(),
            "F_max_per_joint_N": fmax, "F_max_N": float(min(fmax)),
            "limiting_joint": int(model.active[int(np.argmin(fmax))])}
