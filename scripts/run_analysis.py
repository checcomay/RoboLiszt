from __future__ import annotations

import numpy as np

from hand_robot.config.loader import default_config_path, load_yaml
from hand_robot.model.robot import HandRobot
from hand_robot.kinematics.jacobian import translational_jacobian
from hand_robot.visualization.figures import FigureManager, plot_manipulability, plot_operational_workspace

config = load_yaml(default_config_path())
robot = HandRobot(config)
manager = FigureManager("docs/images")
q0 = np.zeros(20)

# Manipulability ellipsoid.
plot_manipulability(robot, q0, translational_jacobian, "finger_1", manager)

# Operational workspace sampling.
rng = np.random.default_rng(7)
points = []
for _ in range(2500):
    q = rng.uniform(np.deg2rad(-70), np.deg2rad(70), size=20)
    T = robot.end_effectors(q)["finger_1"]
    points.append(T[:3, 3])

plot_operational_workspace(np.asarray(points), manager)
print("Analysis figures saved under docs/images/kinematics/")
