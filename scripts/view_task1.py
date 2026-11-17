import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hand_robot.tasks import task1_wave
from hand_robot.tasks.common import Scene
from hand_robot.visualization.figures import FigureManager, plot_static_robot, VIEW_PRESETS

scene = Scene()
res = task1_wave.run(scene)[0]     # or run(scene, wrist_pitch_deg=-65, wave_amp_deg=35, wave_freq_hz=1.0)
robot, q, t = scene.robot, res.q, res.t
phase = np.array(res.phase)
manager = FigureManager()


# ---------- 1) static snapshots (saved in docs/images/kinematics/) ----------
w = np.where(phase == "wave")[0]
key = {"ready": 0,
       "raised": np.where(phase == "raise")[0][-1],
       "wave_min": w[np.argmin(q[w, 0])],
       "wave_max": w[np.argmax(q[w, 0])]}
for tag, i in key.items():
    for view in ("side", "front"):
        plot_static_robot(robot, q[i], manager, name=f"task1_{tag}_{view}", view=view,
                          show_frames=False, show_joint_labels=False)

# ---------- 2) animation: iso / side / front, same fixed limits ----------
def pose(f):
    bp, fp, _ = robot.chains(q[f])
    off = res.base[f]
    return bp + off, [p + off for p in fp.values()]

step = 4
frames = range(0, len(t), step)
allp = np.vstack([np.vstack([pose(f)[0]] + pose(f)[1]) for f in frames])
mid = (allp.max(0) + allp.min(0)) / 2
half = (allp.max(0) - allp.min(0)).max() / 2 * 1.05

fig = plt.figure(figsize=(15, 5))
artists = []                                   # (wrist_line, [finger_lines]) per axes
for k, view in enumerate(("iso", "side", "front")):
    ax = fig.add_subplot(1, 3, k + 1, projection="3d")
    ax.view_init(*VIEW_PRESETS[view]); ax.set_proj_type("ortho")
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)
    ax.set_box_aspect([1, 1, 1]); ax.set_title(view)
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
    wl, = ax.plot([], [], [], "k-o", lw=3, ms=4)
    fl = [ax.plot([], [], [], "-o", lw=2, ms=3, color=plt.get_cmap("tab10")(i))[0]
          for i in range(len(robot.branches))]
    artists.append((wl, fl))
title = fig.suptitle("")

def update(f):
    bp, fps = pose(f)
    for wl, fl in artists:
        wl.set_data_3d(*bp.T)
        for line, p in zip(fl, fps):
            line.set_data_3d(*p.T)
    title.set_text(f"t = {t[f]:.2f} s   [{phase[f]}]")

anim = FuncAnimation(fig, update, frames=frames, interval=1e3 * step * scene.dt)
# anim.save("docs/images/tasks/task1_wave.gif", writer="pillow", fps=int(1 / (step * scene.dt)))
plt.show()