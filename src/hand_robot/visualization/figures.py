from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


class FigureManager:
    """Centralized figure output manager.

    Every generated figure is saved under docs/images/.
    """

    def __init__(self, root: str | Path = "docs/images"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, fig, name: str, category: str = "general", dpi: int = 160) -> Path:
        directory = self.root / category
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{name}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return path


def plot_manipulability(robot, q, jacobian_fn, branch, manager: FigureManager) -> Path:
    J = jacobian_fn(robot, q, branch)
    A = J @ J.T
    U, S, _ = np.linalg.svd(A)
    radii = np.sqrt(np.maximum(S, 1e-14))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    pts = U @ np.vstack([x.ravel(), y.ravel(), z.ravel()])
    ax.plot_surface(
        pts[0].reshape(x.shape),
        pts[1].reshape(x.shape),
        pts[2].reshape(x.shape),
        alpha=0.45,
    )
    ax.set_title(f"Manipulability ellipsoid — {branch}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    return manager.save(fig, f"manipulability_{branch}", "kinematics")


def plot_operational_workspace(points: np.ndarray, manager: FigureManager) -> Path:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=5)
    ax.set_title("Operational workspace")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    return manager.save(fig, "operational_workspace", "kinematics")


def plot_dynamics(t, q, qd, tau, energy, manager: FigureManager) -> Path:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t, q[:, 0], label="q1")
    ax.plot(t, q[:, 1], label="q2")
    ax.set_title("Joint dynamics")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("joint angle [rad]")
    ax.legend()
    return manager.save(fig, "joint_dynamics", "dynamics")


def plot_motion_force_control(t, position, desired_position, force, desired_force, manager: FigureManager) -> Path:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t, position, label="position")
    ax.plot(t, desired_position, "--", label="desired position")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("position [m]")
    ax.set_title("Motion control")
    ax.legend()
    path = manager.save(fig, "motion_control", "control")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t, force, label="force")
    ax.plot(t, desired_force, "--", label="desired force")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("force [N]")
    ax.set_title("Force control")
    ax.legend()
    manager.save(fig, "force_control", "control")
    return path


def plot_task_trajectory(t, positions, branch, manager: FigureManager) -> Path:
    p = positions[branch]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(p[:, 0], p[:, 1], p[:, 2])
    ax.scatter(p[0, 0], p[0, 1], p[0, 2], label="start")
    ax.scatter(p[-1, 0], p[-1, 1], p[-1, 2], label="end")
    ax.set_title(f"Task trajectory — {branch}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend()
    return manager.save(fig, f"task_{branch}", "tasks")


# (elev, azim) presets. Frame convention: x = along the hand, y = lateral, z = up.
VIEW_PRESETS = {
    "iso": (25.0, -60.0),
    "top": (90.0, -90.0),    # looking down the z axis (x vertical on screen)
    "side": (0.0, -90.0),    # looking along y (x horizontal, z vertical)
    "front": (0.0, 0.0),     # looking along x (y horizontal, z vertical)
}


def _draw_hand(ax, robot, q, *, show_frames, show_joint_labels, frame_scale, elev, azim):
    """
    Draw skeleton, joints, fingertips, labels and frames on a 3-D axes.

    :return the array of every plotted point (used to build a cubic bounding box).
    """
    base_pts, finger_pts, T = robot.chains(q)

    # Common base chain 0 -> 1 -> 2
    ax.plot(*base_pts.T, color="k", linewidth=3, marker="o", markersize=4, label="wrist")

    # Branches (exact 20-DOF topology) ending on the fingertip pad
    cmap = plt.get_cmap("tab10")
    for k, (name, pts) in enumerate(finger_pts.items()):
        color = cmap(k)
        ax.plot(*pts.T, color=color, linewidth=2, marker="o", markersize=4, label=name)
        ax.scatter(*pts[-1], color=color, marker="*", s=90, zorder=5)  # fingertip

    joint_pts = np.array([T[i][:3, 3] for i in range(21)])
    tip_pts = np.array([pts[-1] for pts in finger_pts.values()])

    if show_joint_labels:
        # Coincident joints (J3/J4, J16/J17/J18, ...) share ONE label instead of overlapping.
        groups = defaultdict(list)
        for i in range(1, 21):
            groups[tuple(np.round(joint_pts[i], 5))].append(i)
        for pos, ids in groups.items():
            ax.text(*pos, " " + "/".join(f"J{i}" for i in ids), fontsize=7)

    if show_frames:
        axis_colors = ("tab:red", "tab:green", "tab:blue")  # x, y, z
        for i in range(21):
            R, o = T[i][:3, :3], T[i][:3, 3]
            for k in range(3):
                v = R[:, k] * frame_scale
                ax.quiver(*o, *v, color=axis_colors[k], arrow_length_ratio=0.2, linewidth=0.8)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.view_init(elev=elev, azim=azim)
    ax.set_proj_type("ortho")                 # no perspective distortion
    ax.tick_params(labelsize=7)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(5))  # avoids the overlapping tick labels
    if abs(elev) > 89.0:                      # pure top view: the z axis is degenerate, hide it
        ax.set_zticks([])
        ax.set_zlabel("")
    if abs(elev) < 1.0 and abs(azim + 90.0) < 1.0:   # side view: y axis is degenerate
        ax.set_yticks([])
        ax.set_ylabel("")
    if abs(elev) < 1.0 and abs(azim) < 1.0:          # front view: x axis is degenerate
        ax.set_xticks([])
        ax.set_xlabel("")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)

    all_pts = np.vstack([joint_pts, tip_pts])

    # Isometric rendering: cubic bounding box (1:1:1)
    mins, maxs = all_pts.min(axis=0), all_pts.max(axis=0)
    mid = (maxs + mins) / 2.0
    half = max(np.max(maxs - mins) / 2.0, 1e-3) * 1.05
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)
    ax.set_box_aspect([1, 1, 1])
    return all_pts


def plot_static_robot(
        robot,
        q,
        manager,
        name: str = "robot_static",
        show_frames: bool = True,
        show_joint_labels: bool = True,
        show: bool = False,
        elev: float = 25.0,
        azim: float = -60.0,
        view: str | None = None,
        frame_scale: float = 0.012,
) -> Path:
    """Plot the robot in a fixed configuration using Matplotlib 3-D.

    ``view`` ('iso', 'top', 'side', 'front') overrides ``elev``/``azim``.
    The same drawing routine is used for the saved figure and the interactive one.
    """
    q = robot.validate_q(q)
    if view is not None:
        elev, azim = VIEW_PRESETS[view]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    _draw_hand(ax, robot, q, show_frames=show_frames, show_joint_labels=show_joint_labels,
               frame_scale=frame_scale, elev=elev, azim=azim)
    ax.set_title("20-DOF robotic hand — static configuration")
    fig.tight_layout()

    path = manager.save(fig, name, "kinematics")
    plt.close(fig)  # free memory: the original code kept every saved figure open

    if show:
        fig_live = plt.figure(figsize=(10, 8))
        ax_live = fig_live.add_subplot(111, projection="3d")
        _draw_hand(ax_live, robot, q, show_frames=show_frames, show_joint_labels=show_joint_labels,
                   frame_scale=frame_scale, elev=elev, azim=azim)
        ax_live.set_title("Rotate with the mouse: drag in the 3-D axes")
        fig_live.tight_layout()
        plt.show()

    return path