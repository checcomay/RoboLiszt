from __future__ import annotations

import numpy as np

from .config.loader import default_config_path, load_yaml
from .model.robot import HandRobot
from .visualization.figures import FigureManager, plot_task_trajectory
from .tasks.piano import PianoKeyboard, PianoTaskPlanner


def main() -> None:
    """Run a small end-to-end task-planning demonstration."""
    config = load_yaml(default_config_path())
    robot = HandRobot(config)
    keyboard = PianoKeyboard(config)
    planner = PianoTaskPlanner(keyboard)

    q = np.zeros(20)
    print(f"Robot: {config['robot']['name']}")
    print(f"Joints: {len(robot.dh)}")
    print(f"Branches: {robot.branches}")

    task = planner.single_key("finger_1", 60)
    manager = FigureManager("docs/images")
    path = plot_task_trajectory(task.t, task.positions, "finger_1", manager)
    print(f"Saved task figure: {path}")


if __name__ == "__main__":
    main()
