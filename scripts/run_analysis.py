"""Post-processing used in chapters 2.4-2.5, 3.5, 5.2-5.3:
manipulability ellipsoids, inverse dynamics along the task trajectories, kineto-statics of the key contact.

    python scripts/run_analysis.py      (needs data/trajectories/*.npz from scripts/run_tasks.py)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

from hand_robot.control.statics import compensation_torque
from hand_robot.dynamics.rigid_body import RigidBodyDynamics
from hand_robot.kinematics import manipulability as mp
from hand_robot.kinematics.singularity import branch_jacobian, active_joints
from hand_robot.tasks.common import FINGERS, Scene
from hand_robot.visualization.figures import FigureManager

TRAJ = ROOT / "data" / "trajectories"
NOTE = {"thumb": "C4", "index": "D4", "middle": "E4", "ring": "F4", "little": "G4"}
Z = np.array([0.0, 0.0, 1.0])


def load(name):
    d = np.load(TRAJ / f"{name}.npz", allow_pickle=True)
    return d, list(d["phase"])


def deriv(x, dt, order):
    return savgol_filter(x, 21, 4, deriv=order, delta=dt, axis=0)


def main():
    sc = Scene()
    rb = sc.robot
    rbd = RigidBodyDynamics(rb, sc.config)
    fm = FigureManager(ROOT / "docs" / "images")
    out = {"model": {"total_mass_kg": rbd.total_mass()}}

    # ------------------------------------------------------------------ ellipsoids (2.5, 3.5)
    act_cfg = sc.config["actuators"]
    def tmax(br):
        return [act_cfg["tau_max_wrist"]] * 2 + [act_cfg["tau_max_finger"]] * len(rb.branches[br])

    ell = {}
    for f in NOTE:
        d, ph = load(f"task2_press_{f}_{NOTE[f]}")
        k0, k1 = 0, ph.index("hold")
        br = FINGERS[f]
        row = {}
        for tag, k in (("hover", k0), ("pressed", k1)):
            q = d["q"][k]
            for wr in (True, False):
                J = branch_jacobian(rb, q, br, wr)
                U, s = mp.velocity_ellipsoid(J)
                key = f"{tag}_{'wrist' if wr else 'finger'}"
                row[key] = {
                    "sigma": s.tolist(), "w": float(np.prod(s)), "cond": float(s[0] / s[-1]),
                    "v_z": mp.directional_velocity(J, Z), "f_z": mp.directional_force(J, Z),
                }
            G, J, M = mp.dynamic_matrix(rb, rbd, q, br, True, tmax(br))
            sg = np.linalg.svd(G, compute_uv=False)
            row[f"{tag}_dyn"] = {"sigma": sg.tolist(), "a_z": mp.directional_acceleration(G, Z), "cond": float(sg[0] / sg[-1])}
        ell[f] = row
    out["ellipsoids"] = ell

    # figure: velocity / force / acceleration ellipsoids of the middle finger at hover and at the bottom of the stroke
    d, ph = load("task2_press_middle_E4")
    fig = plt.figure(figsize=(13, 4.4))
    scale = {"vel": 0.25, "force": 0.0022, "acc": 0.00005}
    titles = {"vel": "velocity ellipsoid\n(|q̇|≤1, scale 0.25 m per m/s)", "force": "force ellipsoid\n(|τ|≤1 N·m, scale 2.2 mm per N)",
              "acc": "dynamic (acceleration) ellipsoid\n(|τ_i/τ_max,i|≤1, scale 0.05 mm per m/s²)"}
    for pi, kind in enumerate(("vel", "force", "acc")):
        ax = fig.add_subplot(1, 3, pi + 1, projection="3d")
        for tag, k, col in (("hover", 0, "tab:blue"), ("pressed", ph.index("hold"), "tab:red")):
            q = d["q"][k]
            tip = rb.end_effectors(q)["finger_3"][:3, 3] + d["base"][k]
            J = branch_jacobian(rb, q, "finger_3", True)
            if kind == "vel":
                U, r = mp.velocity_ellipsoid(J)
            elif kind == "force":
                U, r = mp.force_ellipsoid(J)
            else:
                G, _, _ = mp.dynamic_matrix(rb, rbd, q, "finger_3", True, tmax("finger_3"))
                U, r = np.linalg.svd(G, full_matrices=False)[0], np.linalg.svd(G, compute_uv=False)
            X, Y, Zs = mp.ellipsoid_surface(tip, U, r * scale[kind])
            ax.plot_surface(X, Y, Zs, color=col, alpha=0.45, linewidth=0)
            ax.scatter(*tip, color=col, s=20)
            ax.quiver(*tip, *(U[:, 2] * r[2] * scale[kind] * 1.0), color="k", lw=1)    # weakest direction
        tips = [rb.end_effectors(d["q"][k])["finger_3"][:3, 3] + d["base"][k] for k in (0, ph.index("hold"))]
        c0, h = 0.5 * (tips[0] + tips[1]), 0.045
        ax.set_xlim(c0[0] - h, c0[0] + h); ax.set_ylim(c0[1] - h, c0[1] + h); ax.set_zlim(c0[2] - h, c0[2] + h)
        ax.set_title(titles[kind], fontsize=9)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_box_aspect([1, 1, 1]); ax.view_init(elev=22, azim=-55)
    fig.suptitle("Middle finger: hover (blue) and bottom of the stroke (red); black arrow = weakest principal direction", fontsize=10)
    fig.tight_layout()
    fm.save(fig, "ellipsoids_middle", "analysis")

    # figure: vertical (z) transmission of the five fingers
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    names = list(NOTE)
    x = np.arange(len(names)); wd = 0.38
    for a, (key, lab, unit) in zip(ax, (("v_z", "max vertical speed", "m/s per (rad/s)"), ("f_z", "max vertical force", "N per (N·m)"))):
        a.bar(x - wd / 2, [ell[f]["hover_wrist"][key] for f in names], wd, label="hover")
        a.bar(x + wd / 2, [ell[f]["pressed_wrist"][key] for f in names], wd, label="pressed")
        a.set_xticks(x); a.set_xticklabels(names, fontsize=8); a.set_ylabel(unit); a.set_title(lab, fontsize=9)
    ax[0].legend(fontsize=8)
    ax[2].bar(x - wd / 2, [ell[f]["hover_dyn"]["a_z"] for f in names], wd)
    ax[2].bar(x + wd / 2, [ell[f]["pressed_dyn"]["a_z"] for f in names], wd)
    ax[2].set_xticks(x); ax[2].set_xticklabels(names, fontsize=8); ax[2].set_ylabel("m/s² (|τ/τmax|≤1)"); ax[2].set_title("max vertical acceleration", fontsize=9)
    fig.tight_layout()
    fm.save(fig, "vertical_transmission", "analysis")

    # ------------------------------------------------------------------ inverse dynamics (5.2)
    def run_id(name, fingers_for_active, use_base=False):
        d, ph = load(name)
        dt = float(d["t"][1] - d["t"][0])
        q = d["q"]
        qd, qdd = deriv(q, dt, 1), deriv(q, dt, 2)
        ba = deriv(d["base"], dt, 2) if use_base else np.zeros((len(q), 3))
        comp = {k: np.zeros_like(q) for k in ("total", "inertial", "coriolis", "gravity", "friction")}
        fb = np.zeros((len(q), 3))
        for k in range(len(q)):
            T = rb.forward_kinematics(q[k])
            tau, f0, _ = rbd.inverse_dynamics(q[k], qd[k], qdd[k], T=T, base_acc=ba[k], return_base_wrench=True)
            comp["total"][k] = tau
            comp["inertial"][k] = rbd.inverse_dynamics(q[k], None, qdd[k], gravity=False, friction=False, T=T, base_acc=ba[k])
            comp["gravity"][k] = rbd.inverse_dynamics(q[k], None, None, gravity=True, friction=False, T=T)
            comp["coriolis"][k] = rbd.inverse_dynamics(q[k], qd[k], None, gravity=False, friction=False, T=T)
            comp["friction"][k] = rbd.B * qd[k]
            fb[k] = f0
        act = sorted({j - 1 for f in fingers_for_active for j in active_joints(rb, FINGERS[f])})
        rms = {c: float(np.sqrt(np.mean(comp[c][:, act] ** 2))) for c in comp}
        peak = np.abs(comp["total"]).max(0)
        top = [int(i) for i in np.argsort(-peak)[:5]]
        res = {"rms_N_m": rms, "peak_total_N_m": {f"J{i+1}": float(peak[i]) for i in top},
               "gravity_share_pct": float(100 * rms["gravity"] / (rms["gravity"] + rms["inertial"] + rms["coriolis"] + 1e-12))}
        if use_base:
            res["carriage_force_peak_N"] = [float(np.abs(fb[:, i]).max()) for i in range(3)]
            res["carriage_force_static_z_N"] = float(rbd.total_mass() * 9.81)
        return d, ph, dt, comp, fb, res

    ids = {}
    for key, name, fing, ub in (("task2_middle", "task2_press_middle_E4", ["middle"], False),
                                ("task3_triad", "task3_chord_C4E4G4", ["thumb", "middle", "little"], False),
                                ("task4_octave", "task4_octave_jump_7", ["thumb", "middle", "little"], True)):
        d, ph, dt, comp, fb, res = run_id(name, fing, ub)
        ids[key] = (d, ph, dt, comp, fb)
        out.setdefault("inverse_dynamics", {})[key] = res

    # figure 5.2: torque decomposition (triad) + carriage force (octave)
    d, ph, dt, comp, _ = ids["task3_triad"][:5]
    t = d["t"]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, j, ttl in ((axs[0], 2, "wrist pitch  J2"), (axs[1], 11, "middle  J11"), (axs[2], 5, "little  J5")):
        for c, ls in (("total", "-"), ("inertial", "--"), ("coriolis", ":"), ("gravity", "-.")):
            ax.plot(t, 1e3 * comp[c][:, j - 1], ls, label=c)
        ax.set_title(ttl, fontsize=9); ax.set_xlabel("time [s]"); ax.set_ylabel("torque [mN·m]")
    axs[0].legend(fontsize=7)
    fig.suptitle("Task 3 (triad): inverse dynamics, torque components", fontsize=10)
    fig.tight_layout()
    fm.save(fig, "id_task3", "analysis")

    d, ph, dt, comp, fb = ids["task4_octave"]
    t = d["t"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for i, lab in enumerate("xyz"):
        axs[0].plot(t, fb[:, i], label=f"F{lab}")
    axs[0].set_xlabel("time [s]"); axs[0].set_ylabel("force on the hand [N]"); axs[0].set_title("carriage force", fontsize=9); axs[0].legend(fontsize=8)
    axs[1].plot(t, 1e3 * comp["total"][:, 1], label="J2 total")
    axs[1].plot(t, 1e3 * comp["inertial"][:, 1], "--", label="J2 inertial")
    axs[1].plot(t, 1e3 * comp["gravity"][:, 1], "-.", label="J2 gravity")
    axs[1].set_xlabel("time [s]"); axs[1].set_ylabel("mN·m"); axs[1].legend(fontsize=8); axs[1].set_title("wrist pitch", fontsize=9)
    fig.suptitle("Task 4 (octave jump): inverse dynamics with base acceleration", fontsize=10)
    fig.tight_layout()
    fm.save(fig, "id_task4", "analysis")

    # ------------------------------------------------------------------ kineto-statics (5.3)
    stat = {}
    fig, axs = plt.subplots(2, 3, figsize=(13, 6), sharex="col")
    for col, prof in enumerate(("pp", "mf", "ff")):
        d, ph = load(f"task5_touch_{prof}")
        q, Fz = d["q"][:-1], d["x_Fz"]
        t = d["t"][:-1]
        act = [j - 1 for j in active_joints(rb, "finger_3")]
        tc = np.array([compensation_torque(rb, q[k], "finger_3", [0, 0, Fz[k]]) for k in range(len(q))])
        tg = np.array([rbd.gravity_torque(q[k]) for k in range(0, len(q), 10)])
        axs[0, col].plot(t, Fz, "k"); axs[0, col].set_title(f"touch {prof}"); axs[0, col].set_ylabel("contact force [N]" if col == 0 else "")
        for j in act:
            axs[1, col].plot(t, 1e3 * tc[:, j], label=f"J{j+1}")
        axs[1, col].set_xlabel("time [s]"); axs[1, col].set_ylabel("compensation torque [mN·m]" if col == 0 else "")
        kpk = int(np.argmax(Fz))
        hold = slice(int(np.argmax(Fz > 0.05)) + 40, int(np.argmax(Fz > 0.05)) + 60)
        peak = np.abs(tc).max(0)
        stat[prof] = {"peak_force_N": float(Fz.max()),
                      "peak_comp_torque_mNm": {f"J{j+1}": float(1e3 * peak[j]) for j in act},
                      "wrist_pitch_share_pct_of_norm": float(100 * np.abs(tc[kpk, 1]) / np.linalg.norm(tc[kpk, act])),
                      "max_gravity_torque_active_mNm": float(1e3 * np.abs(tg[:, act]).max())}
    axs[1, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("Kineto-statics of the key contact: τ = −Jᵀ F  (active joints: J1, J2 and the middle finger J10–J12)", fontsize=10)
    fig.tight_layout()
    fm.save(fig, "statics_touch", "analysis")
    out["statics"] = stat

    # ------------------------------------------------------------------ IK validation (5.1)
    order = list(rb.branches)
    track = {}
    for key, name in (("task2_middle", "task2_press_middle_E4"), ("task2_little", "task2_press_little_G4"),
                      ("task3_triad", "task3_chord_C4E4G4")):
        d, ph = load(name)
        tg = d["x_target_tips"]
        tips = np.array([[rb.end_effectors(q)[n][:3, 3] for n in order] for q in d["q"]]) + d["base"][:, None, :]
        e = np.linalg.norm(tips - tg, axis=2)
        e = e[:, ~np.all(np.isnan(e), axis=0)]
        track[key] = {"mean_mm": float(1e3 * np.nanmean(e)), "rms_mm": float(1e3 * np.sqrt(np.nanmean(e ** 2))),
                      "max_mm": float(1e3 * np.nanmax(e)), "n_samples": int(len(e))}
    out["tracking_error"] = track
    import time
    rng = np.random.default_rng(11)
    ok_n, its, tms = 0, [], []
    for _ in range(100):
        qt = rng.uniform(0.8 * sc.q_min, 0.8 * sc.q_max)
        tips = rb.end_effectors(qt)
        tg = {b: tips[b][:3, 3] for b in ("finger_5", "finger_3", "finger_1")}
        t0 = time.time()
        r = sc.ik.solve_multi(tg, np.zeros(20), max_iterations=600)
        tms.append(time.time() - t0); ok_n += int(r.success); its.append(r.iterations)
    out["ik_random_targets"] = {"trials": 100, "success": ok_n, "mean_iterations": float(np.mean(its)),
                                "mean_time_ms": float(1e3 * np.mean(tms)), "max_iterations_setting": 600}

    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data" / "analysis.json").write_text(json.dumps(out, indent=2))



if __name__ == "__main__":
    main()
