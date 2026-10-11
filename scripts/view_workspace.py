"""Spazio dei giunti, spazio di lavoro e spazio raggiungibile della mano a 20 DOF.

Uso:
    python scripts/view_workspace.py                # 30000 campioni, mostra le figure
    python scripts/view_workspace.py --n 100000 --no-show
    python scripts/view_workspace.py --no-wrist     # q1, q2 fissi a 0 (solo dita)
"""
from pathlib import Path
import argparse
import sys

import numpy as np

import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    pass
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hand_robot.config.loader import default_config_path, load_yaml
from hand_robot.model.robot import HandRobot
from hand_robot.visualization.figures import FigureManager


# ----------------------------------------------------------------- giunti
def joint_limits(config: dict) -> np.ndarray:
    """(20, 2) limiti [rad]. Default da `limits`; override opzionale con
    limits.joints: {5: {min_deg: -10, max_deg: 90}, ...}"""
    lim = config.get("limits", {})
    lo = np.full(20, np.deg2rad(lim.get("default_min_deg", -90.0)))
    hi = np.full(20, np.deg2rad(lim.get("default_max_deg", 90.0)))
    for k, v in lim.get("joints", {}).items():
        lo[int(k) - 1] = np.deg2rad(v["min_deg"])
        hi[int(k) - 1] = np.deg2rad(v["max_deg"])
    return np.column_stack([lo, hi])


def sample_joint_space(limits, n, rng, use_wrist=True):
    q = rng.uniform(limits[:, 0], limits[:, 1], size=(n, 20))
    if not use_wrist:
        q[:, :2] = 0.0
    return q


# ------------------------------------------- cinematica diretta vettorizzata
def _batch_dh(a, alpha, d, theta):
    """Trasformata DH standard per un vettore di theta -> (N, 4, 4)."""
    n = theta.shape[0]
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    T = np.zeros((n, 4, 4))
    T[:, 0, 0] = ct;  T[:, 0, 1] = -st * ca; T[:, 0, 2] = st * sa;  T[:, 0, 3] = a * ct
    T[:, 1, 0] = st;  T[:, 1, 1] = ct * ca;  T[:, 1, 2] = -ct * sa; T[:, 1, 3] = a * st
    T[:, 2, 1] = sa;  T[:, 2, 2] = ca;       T[:, 2, 3] = d
    T[:, 3, 3] = 1.0
    return T


def batch_fingertips(robot: HandRobot, Q: np.ndarray) -> dict[str, np.ndarray]:
    """Posizione del fingertip di ogni dito per N configurazioni -> {dito: (N, 3)}."""
    n = Q.shape[0]

    def local(j):
        p = robot.dh[j]
        return _batch_dh(p.a, p.alpha, p.d, Q[:, j - 1] + p.theta_offset)

    T = {1: local(1)}
    T[2] = T[1] @ local(2)
    tips = {}
    for name, branch in robot.branches.items():
        for j in branch:
            T[j] = T[robot.parent[j]] @ local(j)
        tips[name] = (T[branch[-1]] @ robot._T_tip)[:, :3, 3]
    return tips


def check_vectorized(robot, rng):
    """Verifica che la FK vettorizzata coincida con quella del modello."""
    q = rng.uniform(-1.0, 1.0, size=(1, 20))
    ref = robot.end_effectors(q[0])
    fast = batch_fingertips(robot, q)
    for name in ref:
        assert np.allclose(ref[name][:3, 3], fast[name][0], atol=1e-12), name


# -------------------------------------------------------------- utilità plot
def _cubic_box(ax, pts):
    mins, maxs = pts.min(0), pts.max(0)
    mid, half = (mins + maxs) / 2, max(np.max(maxs - mins) / 2, 1e-3) * 1.05
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)
    ax.set_box_aspect([1, 1, 1])
    ax.set_proj_type("ortho")
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")


def _hull(points):
    h = ConvexHull(points)
    return h, points[h.simplices]


# ------------------------------------------------------------------- figure
def fig_joint_space(robot, limits, manager):
    """Range di ogni giunto (in gradi) e angolo DH effettivo = q + theta_offset."""
    cmap = plt.get_cmap("tab10")
    color = {}
    for k, (name, br) in enumerate(robot.branches.items()):
        for j in br:
            color[j] = cmap(k)
    color[1] = color[2] = "k"

    fig, ax = plt.subplots(figsize=(9, 7))
    for j in range(1, 21):
        lo, hi = np.rad2deg(limits[j - 1])
        off = np.rad2deg(robot.dh[j].theta_offset)
        ax.barh(j, hi - lo, left=lo, color=color[j], alpha=0.35)                # range di q
        ax.barh(j, hi - lo, left=lo + off, color="none", edgecolor=color[j],    # range di theta DH
                linewidth=1.5, hatch="//")
        ax.plot([0, 0], [j - 0.4, j + 0.4], "k-", lw=0.8)
    ax.set_yticks(range(1, 21))
    ax.set_yticklabels([f"J{j}" for j in range(1, 21)])
    ax.invert_yaxis()
    ax.set_xlabel("angolo [deg]")
    ax.set_title("Spazio dei giunti: range di q (pieno) e θ = q + offset (tratteggiato)")
    ax.grid(axis="x", alpha=0.3)
    return manager.save(fig, "joint_space_limits", "workspace")


def fig_joint_space_3d(robot, Q, tips, manager):
    """Per ogni dito: primi 3 giunti campionati, colore = distanza fingertip-origine."""
    names = list(robot.branches)
    fig = plt.figure(figsize=(15, 9))
    m = min(len(Q), 4000)
    for k, name in enumerate(names):
        br = robot.branches[name]
        ax = fig.add_subplot(2, 3, k + 1, projection="3d")
        idx = [j - 1 for j in br[:3]]
        dist = np.linalg.norm(tips[name][:m], axis=1)
        sc = ax.scatter(*np.rad2deg(Q[:m, idx]).T, c=dist, s=3, cmap="viridis")
        ax.set_xlabel(f"q{br[0]} [deg]"); ax.set_ylabel(f"q{br[1]} [deg]"); ax.set_zlabel(f"q{br[2]} [deg]")
        ax.set_title(name, fontsize=10)
        fig.colorbar(sc, ax=ax, shrink=0.6, label="|p_tip| [m]")
    fig.suptitle("Spazio dei giunti (primi 3 giunti di ogni dito, colore = distanza del fingertip)")
    fig.tight_layout()
    return manager.save(fig, "joint_space_3d", "workspace")


def fig_workspace(robot, tips, manager, show_hull=True):
    """Nuvola dello spazio di lavoro (configurazioni campionate) + inviluppo raggiungibile."""
    cmap = plt.get_cmap("tab10")
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")
    allpts = []
    for k, (name, p) in enumerate(tips.items()):
        ax.scatter(*p.T, s=1, color=cmap(k), alpha=0.15, label=f"{name} (workspace)")
        if show_hull:
            h, tri = _hull(p)
            ax.add_collection3d(Poly3DCollection(tri, facecolor=cmap(k), edgecolor="none", alpha=0.12))
            print(f"  {name:9s} volume inviluppo = {h.volume * 1e6:9.1f} cm^3")
        allpts.append(p)
    allpts = np.vstack(allpts)
    ax.scatter(0, 0, 0, color="k", marker="s", s=40, label="base (frame 0)")
    _cubic_box(ax, allpts)
    ax.set_title("Spazio di lavoro (punti) e spazio raggiungibile (inviluppo convesso)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, markerscale=6)
    fig.tight_layout()
    return manager.save(fig, "workspace_reachable", "workspace")


def fig_projections(tips, manager):
    """Proiezioni XY / XZ / YZ con contorno dell'inviluppo 2D di ogni dito."""
    cmap = plt.get_cmap("tab10")
    pairs = [(0, 1, "X", "Y"), (0, 2, "X", "Z"), (1, 2, "Y", "Z")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, (i, j, li, lj) in zip(axes, pairs):
        for k, (name, p) in enumerate(tips.items()):
            ax.scatter(p[:, i], p[:, j], s=1, color=cmap(k), alpha=0.12)
            h = ConvexHull(p[:, [i, j]])
            v = np.append(h.vertices, h.vertices[0])
            ax.plot(p[v, i], p[v, j], color=cmap(k), lw=1.5, label=name)
        ax.plot(0, 0, "ks")
        ax.set_xlabel(f"{li} [m]"); ax.set_ylabel(f"{lj} [m]")
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_title(f"Proiezione {li}-{lj}")
    axes[0].legend(fontsize=8)
    fig.suptitle("Spazio di lavoro: proiezioni e contorno raggiungibile")
    fig.tight_layout()
    return manager.save(fig, "workspace_projections", "workspace")


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000, help="numero di configurazioni campionate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-wrist", action="store_true", help="blocca q1, q2 a 0")
    ap.add_argument("--no-hull", action="store_true")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    config = load_yaml(default_config_path())
    robot = HandRobot(config)
    rng = np.random.default_rng(args.seed)
    manager = FigureManager(ROOT / "docs" / "images")

    check_vectorized(robot, rng)
    limits = joint_limits(config)
    Q = sample_joint_space(limits, args.n, rng, use_wrist=not args.no_wrist)
    tips = batch_fingertips(robot, Q)

    print("Volumi degli inviluppi raggiungibili:")
    paths = [
        fig_joint_space(robot, limits, manager),
        fig_joint_space_3d(robot, Q, tips, manager),
        fig_workspace(robot, tips, manager, show_hull=not args.no_hull),
        fig_projections(tips, manager),
    ]
    for p in paths:
        print("Salvata:", p)

    if not args.no_show:
        # le figure sono chiuse dal FigureManager: le ridisegno in modo interattivo
        for img in paths:
            im = plt.imread(img)
            f = plt.figure(figsize=(12, 8))
            plt.imshow(im); plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()