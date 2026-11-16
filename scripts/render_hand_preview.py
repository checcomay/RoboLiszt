"""Offline preview of the hand meshes + piano in the real task postures (matplotlib, no Gazebo needed).

    python scripts/render_hand_preview.py      -> docs/images/hand/{ready,press,chord}_*.png

Uses exactly the transforms of the URDF: link frame L_i = ^0T_i . F_i^-1, tip link frame = ^0T_last . Tx(L_TIP).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from hand_robot.model.dh import dh_transform
from hand_robot.piano import layout as lay
from hand_robot.tasks.common import Scene
from hand_robot.tasks.task2_single_key import DEFAULT_NOTES
from hand_robot.visualization import hand_mesh as hm

OUT = ROOT / "docs" / "images" / "hand"
LIGHT = np.array([-0.45, -0.55, 0.70]) / np.linalg.norm([-0.45, -0.55, 0.70])


def link_frames(robot, q, base):
    T = robot.forward_kinematics(q)
    B = np.eye(4)
    B[:3, 3] = base
    fr = {"base_link": B}
    for i in range(1, 21):
        p = robot.dh[i]
        fr[f"link{i}"] = B @ T[i] @ np.linalg.inv(dh_transform(p.a, p.alpha, p.d, 0.0))
    for b, br in robot.branches.items():
        Tt = np.eye(4)
        Tt[0, 3] = robot.L_TIP
        fr[f"tip_{b}"] = B @ T[br[-1]] @ Tt
    return fr


def box(center, size):
    c, s = np.asarray(center, float), np.asarray(size, float) / 2
    v = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * s + c
    f = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
    return [v[k] for k in f]


def piano_polys(sc, cfg):
    kb, p = sc.kb, cfg["piano"]
    bw, bl, bh = float(p["black_key_width"]), float(p["black_key_length"]), float(p["black_key_height"])
    polys, cols = [], []
    for m in lay.keys():
        y = lay.key_y(m, kb.width)
        if lay.is_black(m):
            polys += box((kb.front_x + kb.depth - bl / 2, y, bh - 0.011), (bl, bw, 0.022)); cols += [(0.06, 0.06, 0.07)] * 6
        else:
            polys += box((kb.front_x + kb.depth / 2, y, -0.012), (kb.depth, kb.width - 0.001, 0.024)); cols += [(0.95, 0.94, 0.88)] * 6
    polys += box((kb.front_x + kb.depth / 2, 0.5 * (lay.key_y(48, kb.width) + lay.key_y(84, kb.width)), -0.05),
                 (kb.depth + 0.06, 0.55, 0.076)); cols += [(0.30, 0.17, 0.10)] * 6
    return polys, cols


def draw(sc, q, base, path, views, title):
    robot, cfg = sc.robot, sc.config
    meshes = hm.build_hand_meshes(robot)
    fr = link_frames(robot, q, base)
    fig = plt.figure(figsize=(6.2 * len(views), 5.2))
    for k, (elev, azim, ctr, span) in enumerate(views):
        ax = fig.add_subplot(1, len(views), k + 1, projection="3d", computed_zorder=False)
        ax.set_proj_type("ortho")
        polys, cols = piano_polys(sc, cfg)
        pc = Poly3DCollection(polys, facecolors=cols, edgecolors=(0, 0, 0, 0.25), linewidths=0.2)
        ax.add_collection3d(pc)
        for name, parts in meshes.items():
            Tm = fr[name]
            for part in parts:
                V = part.V @ Tm[:3, :3].T + Tm[:3, 3]
                N = part.N @ Tm[:3, :3].T
                tri = V[part.F]
                fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
                fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-15)
                lam = np.clip(fn @ LIGHT, 0, 1) * 0.6 + 0.4
                base_c = np.array(hm.MATERIALS[part.material][:3])
                ax.add_collection3d(Poly3DCollection(tri, facecolors=np.clip(base_c * lam[:, None], 0, 1), linewidths=0))
        c = np.asarray(ctr)
        ax.set_xlim(c[0] - span, c[0] + span); ax.set_ylim(c[1] - span, c[1] + span); ax.set_zlim(c[2] - span, c[2] + span)
        ax.set_box_aspect((1, 1, 1)); ax.view_init(elev, azim); ax.set_axis_off()
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sc = Scene()
    base = sc.base_home
    q_ready = sc.ready_posture(DEFAULT_NOTES)
    ctr = np.array([0.20, 0.0, 0.03])
    views = [(28, -62, ctr, 0.14), (89, -90, ctr, 0.15), (2, -90, ctr, 0.15)]
    draw(sc, q_ready, base, OUT / "ready_views.png", views, "postura ready: 3/4, dall'alto, di lato")
    d = np.load(ROOT / "data" / "trajectories" / "task2_press_middle_E4.npz", allow_pickle=True)
    ih = [i for i, p in enumerate(d["phase"]) if p == "hold"][0]
    draw(sc, d["q"][ih], d["base"][ih], OUT / "press_views.png", views, "Task 2: pressione del tasto E4 con il medio")
    dc = np.load(ROOT / "data" / "trajectories" / "task3_chord_C4E4G4.npz", allow_pickle=True)
    ic = [i for i, p in enumerate(dc["phase"]) if p == "hold"][0]
    draw(sc, dc["q"][ic], dc["base"][ic], OUT / "chord_views.png", views, "Task 3: accordo C4-E4-G4 (1-3-5)")
    print("written to", OUT)


if __name__ == "__main__":
    main()
