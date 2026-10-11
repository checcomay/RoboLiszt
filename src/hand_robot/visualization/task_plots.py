from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

PHASE_COLORS = {"lift": "#fde9c9", "descend": "#cfe5f7", "hold": "#d6f0d2", "release": "#cfe5f7", "return": "#fde9c9",
                "raise": "#fde9c9", "wave": "#d6f0d2", "lower": "#fde9c9", "jump": "#e8d6f0", "approach": "#cfe5f7",
                "force": "#d6f0d2", "retract": "#fde9c9"}
NAMES = {"finger_1": "little", "finger_2": "ring", "finger_3": "middle", "finger_4": "index", "finger_5": "thumb"}


def _shade(ax, res):
    phase, t = res.phase, res.t
    start = 0
    for i in range(1, len(phase) + 1):
        if i == len(phase) or phase[i] != phase[start]:
            c = PHASE_COLORS.get(phase[start])
            if c:
                ax.axvspan(t[start], t[min(i, len(t) - 1)], color=c, alpha=0.6, lw=0)
            start = i


def plot_tips_height(scene, res, manager, name=None):
    tips = scene.tip_traj(res)
    fig, ax = plt.subplots(figsize=(9, 4))
    _shade(ax, res)
    for i, n in enumerate(scene.robot.branches):
        ax.plot(res.t, 1e3 * tips[:, i, 2], label=NAMES[n])
    ax.axhline(0, color="k", lw=1)
    ax.axhline(-1e3 * scene.kb.press_depth, color="k", lw=0.8, ls="--")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("fingertip height above key surface [mm]")
    ax.set_title(res.name)
    ax.legend(ncol=5, fontsize=8, loc="upper right")
    fig.tight_layout()
    return manager.save(fig, name or f"{res.name}_tips_z", "tasks")


def plot_hand_on_keyboard(scene, q, base, manager, name, notes=None, pressed=(), n_keys=(-3, 12)):
    kb = scene.kb
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    base = np.asarray(base, float)
    for k in range(n_keys[0], n_keys[1]):
        y0 = (k - 0.5) * kb.width
        x0, x1 = kb.front_x, kb.front_x + kb.depth
        zt = -kb.press_depth if (notes and any(kb.white_index(notes[f]) - kb.white_index("C4") == k for f in pressed)) else 0.0
        v = [[x0, y0 + 0.0004, zt], [x1, y0 + 0.0004, zt], [x1, y0 + kb.width - 0.0004, zt], [x0, y0 + kb.width - 0.0004, zt]]
        ax.add_collection3d(Poly3DCollection([v], facecolor="#b5651d" if zt < 0 else "#f4f4f0", edgecolor="k", lw=0.4))
    base_pts, finger_pts, _ = scene.robot.chains(q)
    cmap = plt.get_cmap("tab10")
    ax.plot(*(base_pts + base).T, color="k", lw=3, marker="o")
    for i, (n, pts) in enumerate(finger_pts.items()):
        ax.plot(*(pts + base).T, color=cmap(i), lw=2, marker="o", ms=3, label=NAMES[n])
        ax.scatter(*(pts[-1] + base), color=cmap(i), marker="*", s=80)
    ax.set_xlim(0, 0.25); ax.set_ylim(-0.08, 0.17); ax.set_zlim(-0.02, 0.18)
    ax.set_box_aspect([0.25, 0.25, 0.20]); ax.view_init(elev=24, azim=-58)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(name)
    return manager.save(fig, name, "tasks")


def plot_wave(scene, res, manager):
    tips = scene.tip_traj(res)
    fig = plt.figure(figsize=(11, 4.2))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    for i, n in enumerate(scene.robot.branches):
        ax.plot(tips[:, i, 0], tips[:, i, 1], tips[:, i, 2], label=NAMES[n], lw=1)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z"); ax.set_title("fingertip paths [m]")
    ax.legend(fontsize=6)
    ax2 = fig.add_subplot(1, 3, 2)
    _shade(ax2, res)
    ax2.plot(res.t, np.rad2deg(res.q[:, 0]), label="q1 (deviation)")
    ax2.plot(res.t, np.rad2deg(res.q[:, 1]), label="q2 (pitch)")
    ax2.set_xlabel("time [s]"); ax2.set_ylabel("deg"); ax2.legend(fontsize=8); ax2.set_title("wrist")
    ax3 = fig.add_subplot(1, 3, 3)
    sm = res.extras["sigma_min_middle"]
    ax3.plot(res.t[::5][:len(sm)], sm); ax3.set_xlabel("time [s]"); ax3.set_ylabel(r"$\sigma_{min}(J)$ middle")
    ax3.set_title("singularity index")
    fig.tight_layout()
    return manager.save(fig, "task1_wave", "tasks")


def plot_octave(scene, res, manager):
    tips = scene.tip_traj(res)
    ex = res.extras
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    ax = axs[0]; _shade(ax, res)
    ax.plot(res.t, 1e3 * res.base[:, 1], label="carriage y")
    ax.plot(res.t, 1e3 * res.base[:, 2], label="carriage z")
    ax.set_xlabel("time [s]"); ax.set_ylabel("mm"); ax.legend(fontsize=8); ax.set_title("carriage motion")
    ax = axs[1]; _shade(ax, res)
    for i, n in enumerate(scene.robot.branches):
        ax.plot(res.t, 1e3 * tips[:, i, 1], label=NAMES[n])
    ax.set_xlabel("time [s]"); ax.set_ylabel("fingertip y [mm]"); ax.legend(fontsize=7, ncol=2); ax.set_title("lateral position")
    ax = axs[2]
    ax.bar(ex["reach_steps"], ex["reach_error_mm"], color=["tab:green" if e < 0.1 else "tab:red" for e in ex["reach_error_mm"]])
    ax.set_yscale("log"); ax.axhline(0.1, color="k", ls="--", lw=0.8)
    ax.set_xlabel("white keys from C4 (wrist rotation only)"); ax.set_ylabel("IK error [mm]")
    ax.set_title("reach without carriage")
    fig.tight_layout()
    return manager.save(fig, "task4_octave_jump", "tasks")


def plot_touch(scene, results, manager):
    fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for r in results:
        ex = r.extras
        l, = axs[0].plot(r.t[:-1], ex["Fz"], label=f"{ex['profile'].name} (measured)")
        axs[0].plot(r.t[:-1], ex["F_des"], "--", color=l.get_color(), lw=0.9)
        axs[1].plot(r.t[:-1], 1e3 * ex["z"], color=l.get_color(), label=ex["profile"].name)
    axs[1].axhline(0, color="k", lw=1); axs[1].axhline(-1e3 * scene.kb.press_depth, color="k", ls="--", lw=0.8)
    axs[0].set_ylabel("contact force [N]"); axs[1].set_ylabel("fingertip height [mm]"); axs[1].set_xlabel("time [s]")
    axs[0].legend(fontsize=8); axs[0].set_title("Task 5 - hybrid position/force touch (dashed: desired force)")
    fig.tight_layout()
    return manager.save(fig, "task5_touch", "tasks")
