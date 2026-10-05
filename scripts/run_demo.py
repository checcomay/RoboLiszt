from hand_robot.config.loader import default_config_path, load_yaml
from hand_robot.model.robot import HandRobot
import numpy as np

config = load_yaml(default_config_path())
robot = HandRobot(config)
q = np.zeros(20)

print("20-DOF hand")
print("Frames:", sorted(robot.forward_kinematics(q)))
for name, T in robot.end_effectors(q).items():
    print(f"\n{name}:\n{T}")
