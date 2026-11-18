"""Visual check of Task 2 (single key press) BEFORE Gazebo: numerical checks, summary figure, 3-D animation.

    python scripts/view_task2.py                         # middle finger on E4, interactive window
    python scripts/view_task2.py --finger index          # any of: thumb index middle ring little
    python scripts/view_task2.py --save                  # writes docs/images/tasks/task2_*.png and task2_press_<finger>.gif
    python scripts/view_task2.py --no-anim               # only the checks + summary figure

Uses only: Scene(), task2_single_key.run(), scene.tip_traj, scene.robot.chains/branches, scene.kb (keyboard).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FINGERS = {"little": "finger_1", "ring": "finger_2", "middle": "finger_3", "index": "finger_4", "thumb": "finger_5"}
NAMES = {"finger_1": "mignolo", "finger_2": "anulare", "finger_3": "medio", "finger_4": "indice", "finger_5": "pollice"}
PHASE_COLORS = {"lift": "#fde9c9", "descend": "#cfe5f7", "hold": "#d6f0d2", "release": "#cfe5f7", "return": "#fde9c9",
                "idle": "#f0f0f0", "home_return": "#e8d6f0"}
FLOOR_HEIGHT_MM = 8.0        # press_sequence(mode="floor", floor_height=0.008): non-pressed fingers must stay >= 8 mm above the keys


# ------------------------------------------------------------------------------------------------ data helpers
def padded_targets(res, n):
    """extras['target_tips'] padded with NaN up to n samples (the return-to-ready tail has no Cartesian target)."""
    tg = res.extras.get("target_tips")
    if tg is None:
        return np.full((n, 5, 3), np.nan)
    if len(tg) < n:
        tg = np.concatenate([tg, np.full((n - len(tg),) + tg.shape[1:], np.nan)])
    return tg[:n]


def shade(ax, res):
    ph, t, start = list(res.phase), res.t, 0
    for i in range(1, len(ph) + 1):
        if i == len(ph) or ph[i] != ph[start]:
            c = PHASE_COLORS.get(ph[start])
            if c:
                ax.axvspan(t[start], t[min(i, len(t) - 1)], color=c, alpha=0.7, lw=0)
            start = i


def checks(scene, res, finger, notes, tips, targets):
    """Numerical sanity checks of the trajectory that is going to be executed. Returns {name: (value, ok)}."""
    kb, names = scene.kb, list(scene.robot.branches)
    i = names.index(FINGERS[finger])
    out = {}
    z = 1e3 * tips[:, i, 2]
    out["pressed finger: min tip z [mm] (target -%.1f)" % (1e3 * kb.press_depth)] = (z.min(), abs(z.min() + 1e3 * kb.press_depth) < 0.5)
    err = np.linalg.norm(tips[:, i] - targets[:, i], axis=1) * 1e3
    err = err[~np.isnan(err)]
    out["pressed finger: max tracking error [mm]"] = (err.max(), err.max() < 0.5)
    out["pressed finger: rms tracking error [mm]"] = (np.sqrt((err ** 2).mean()), True)
    others = [k for k, n in enumerate(names) if n != FINGERS[finger]]
    clr = 1e3 * tips[:, others, 2].min()
    out["other fingers: min clearance [mm] (floor %.0f)" % FLOOR_HEIGHT_MM] = (clr, clr > FLOOR_HEIGHT_MM - 0.1)
    margin = np.rad2deg(np.minimum(res.q - scene.q_min, scene.q_max - res.q).min())
    out["min joint margin [deg]"] = (margin, margin > 1.0)
    qd = np.gradient(res.q, scene.dt, axis=0)
    out["max joint speed [rad/s] (URDF limit 12)"] = (np.abs(qd).max(), np.abs(qd).max() < 12.0)
    q0 = res.extras.get("q_ready", res.q[0])
    out["wrist deviation from ready [deg]"] = (np.rad2deg(np.abs(res.q[:, :2] - q0[:2]).max()), True)
    last = max(k for k, p in enumerate(res.phase) if p != "home_return")
    gap = np.rad2deg(np.abs(res.q[last] - q0).max())
    out["posture drift before the return move [deg]"] = (gap, gap < 5.0)
    out["end-of-task error vs ready [deg]"] = (np.rad2deg(np.abs(res.q[-1] - q0).max()), np.abs(res.q[-1] - q0).max() < 1e-6)
    out["duration [s]"] = (res.t[-1], True)
    return out


def print_checks(ch):
    print("\n--- Task 2 checks ---")
    for k, (v, ok) in ch.items():
        print(f"  [{'OK   ' if ok else 'CHECK'}] {k:<52s} {v:10.3f}")
    print()


# ------------------------------------------------------------------------------------------------ static summary
def plot_summary(scene, res, finger, tips, targets):
    kb, names = scene.kb, list(scene.robot.branches)
    i = names.index(FINGERS[finger])
    t = res.t
    fig, ax = plt.subplots(2, 2, figsize=(13, 7.5))
    a = ax[0, 0]; shade(a, res)
    for k, n in enumerate(names):
        a.plot(t, 1e3 * tips[:, k, 2], lw=2.2 if k == i else 1.0, label=NAMES[n])
    a.axhline(0, color="k", lw=1); a.axhline(-1e3 * kb.press_depth, color="k", ls="--", lw=0.8)
    a.axhline(FLOOR_HEIGHT_MM, color="gray", ls=":", lw=0.8)
    a.set(xlabel="t [s]", ylabel="fingertip height over the keys [mm]", title="all fingers (shaded: phases)"); a.legend(ncol=5, fontsize=7)
    a = ax[0, 1]; shade(a, res)
    a.plot(t, 1e3 * tips[:, i, 2], label="actual"); a.plot(t, 1e3 * targets[:, i, 2], "--", label="target")
    a.set(xlabel="t [s]", ylabel="z [mm]", title=f"pressed finger ({NAMES[FINGERS[finger]]}): z target vs actual"); a.legend(fontsize=8)
    a = ax[1, 0]; shade(a, res)
    a.plot(t, np.rad2deg(res.q[:, 0]), label="q1 wrist dev."); a.plot(t, np.rad2deg(res.q[:, 1]), label="q2 wrist flex.")
    for j in scene.robot.branches[FINGERS[finger]]:
        a.plot(t, np.rad2deg(res.q[:, j - 1]), lw=1, label=f"q{j}")
    a.set(xlabel="t [s]", ylabel="[deg]", title="wrist and pressed-finger joints"); a.legend(ncol=3, fontsize=7)
    a = ax[1, 1]; shade(a, res)
    err = 1e3 * np.linalg.norm(tips[:, i] - targets[:, i], axis=1)
    a.semilogy(t, np.maximum(err, 1e-6)); a.axhline(0.1, color="k", ls="--", lw=0.8)
    a.set(xlabel="t [s]", ylabel="||tip - target|| [mm]", title="IK tracking error (dashed: ik_tolerance 0.1 mm)")
    fig.suptitle(res.name); fig.tight_layout()
    return fig


# ------------------------------------------------------------------------------------------------ animation
def animate(scene, res, finger, notes, tips, step=3):
    kb, robot, names = scene.kb, scene.robot, list(scene.robot.branches)
    t, q, base, phase = res.t, res.q, res.base, list(res.phase)
    ip = names.index(FINGERS[finger])
    y_c4, w = kb.key_y("C4"), kb.width
    for f, n in notes.items():                                # the key layout used for drawing must agree with kb.key_y
        k = kb.white_index(n) - kb.white_index("C4")
        if abs(kb.key_y(n) - (y_c4 + k * w)) > 1e-6:
            print(f"[warn] key_y({n}) != C4 + {k}*width: the drawn keys may be offset")
    ylo, yhi = tips[:, :, 1].min() - 0.05, tips[:, :, 1].max() + 0.05
    ks = list(range(int(np.floor((ylo - y_c4) / w)), int(np.ceil((yhi - y_c4) / w)) + 1))
    kp = ks.index(kb.white_index(notes[finger]) - kb.white_index("C4"))
    x0, x1 = kb.front_x, kb.front_x + kb.depth

    def key_verts(k, depth):
        y0 = y_c4 + (k - 0.5) * w
        return [[x0, y0 + 4e-4, -depth], [x1, y0 + 4e-4, -depth], [x1, y0 + w - 4e-4, -depth], [x0, y0 + w - 4e-4, -depth]]

    fig = plt.figure(figsize=(14, 6.2))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    keys = Poly3DCollection([key_verts(k, 0.0) for k in ks], facecolor="#f4f4f0", edgecolor="k", lw=0.4)
    ax.add_collection3d(keys)
    zmax = max(base[:, 2].max() + 0.04, tips[:, :, 2].max() + 0.01)
    ax.set_xlim(0, x1 + 0.01); ax.set_ylim(ylo, yhi); ax.set_zlim(-0.02, zmax)
    ax.set_box_aspect([x1 + 0.01, yhi - ylo, zmax + 0.02]); ax.view_init(elev=24, azim=-58)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    cmap = plt.get_cmap("tab10")
    wl, = ax.plot([], [], [], "k-o", lw=3, ms=4)
    fl = [ax.plot([], [], [], "-o", color=cmap(k), lw=2.4 if k == ip else 1.4, ms=3, label=NAMES[n])[0] for k, n in enumerate(names)]
    tp = [ax.plot([], [], [], "*", color=cmap(k), ms=11)[0] for k in range(len(names))]
    ax.legend(loc="upper left", fontsize=7)
    ttl = fig.suptitle("")

    ax2 = fig.add_subplot(1, 2, 2); shade(ax2, res)
    for k, n in enumerate(names):
        ax2.plot(t, 1e3 * tips[:, k, 2], lw=2.2 if k == ip else 1.0, color=cmap(k))
    ax2.axhline(0, color="k", lw=1); ax2.axhline(-1e3 * kb.press_depth, color="k", ls="--", lw=0.8)
    ax2.set(xlabel="t [s]", ylabel="tip height [mm]", title="fingertip heights")
    cur = ax2.axvline(0, color="r", lw=1.2)

    def update(f):
        bp, fp, _ = robot.chains(q[f])
        off = base[f]
        wl.set_data_3d(*(bp + off).T)
        for k, n in enumerate(names):
            p = fp[n] + off
            fl[k].set_data_3d(*p.T)
            tp[k].set_data_3d(*p[-1:].T)
        depth = float(np.clip(-tips[f, ip, 2], 0.0, kb.press_depth))
        keys.set_verts([key_verts(k, depth if j == kp else 0.0) for j, k in enumerate(ks)])
        keys.set_facecolor(["#b5651d" if (j == kp and depth > 5e-4) else "#f4f4f0" for j in range(len(ks))])
        cur.set_xdata([t[f], t[f]])
        ttl.set_text(f"{res.name}   t = {t[f]:.2f} s   [{phase[f]}]   key depth {1e3 * depth:.1f} mm")

    frames = range(0, len(t), step)
    anim = FuncAnimation(fig, update, frames=frames, interval=1e3 * step * scene.dt, blit=False)
    return fig, anim, update


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finger", default="middle", choices=list(FINGERS))
    ap.add_argument("--save", action="store_true", help="save figures/GIF under docs/images/tasks instead of showing them")
    ap.add_argument("--no-anim", action="store_true")
    ap.add_argument("--step", type=int, default=3, help="animation: simulation samples per frame")
    a = ap.parse_args()
    if a.save:
        import matplotlib
        matplotlib.use("Agg")

    from hand_robot.tasks.common import Scene                  # <-- adapt the package path if your tasks live elsewhere
    from hand_robot.tasks import task2_single_key as task2     # <-- idem
    scene = Scene()
    res = task2.run(scene, finger=a.finger)[0]
    notes = res.extras["notes"]
    tips = scene.tip_traj(res)
    targets = padded_targets(res, len(res.t))

    print(f"{res.name}: {len(res.t)} samples, phases: {sorted(set(res.phase), key=res.phase.index)}")
    print("metrics from run():", {k: round(float(v), 4) for k, v in res.metrics.items()})
    print_checks(checks(scene, res, a.finger, notes, tips, targets))

    fig = plot_summary(scene, res, a.finger, tips, targets)
    out = ROOT / "docs" / "images" / "tasks"
    if a.save:
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / f"task2_{a.finger}_summary.png", dpi=150, bbox_inches="tight")
    if not a.no_anim:
        fig2, anim, _ = animate(scene, res, a.finger, notes, tips, a.step)
        if a.save:
            anim.save(out / f"task2_press_{a.finger}.gif", writer="pillow", fps=int(round(1 / (a.step * scene.dt))))
            print("saved:", out)
    if not a.save:
        plt.show()


if __name__ == "__main__":
    main()