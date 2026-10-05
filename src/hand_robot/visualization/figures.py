from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


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
