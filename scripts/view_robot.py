from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hand_robot.config.loader import default_config_path, load_yaml
from hand_robot.model.robot import HandRobot
from hand_robot.visualization.figures import FigureManager, plot_static_robot

import matplotlib
matplotlib.use('TkAgg')


config = load_yaml(default_config_path())
robot = HandRobot(config)

q = np.zeros(20)

manager = FigureManager(ROOT / "docs" / "images")
path = plot_static_robot(robot, q, manager, show=True)
print(f"Static figure saved to: {path}")
