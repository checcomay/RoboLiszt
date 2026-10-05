import numpy as np

from hand_robot.config.loader import default_config_path, load_yaml
from hand_robot.model.robot import HandRobot
from hand_robot.kinematics.ik import InverseKinematicsSolver


def test_ik_returns_valid_20dof_vector():
    config = load_yaml(default_config_path())
    robot = HandRobot(config)
    solver = InverseKinematicsSolver(robot, config)

    q0 = np.zeros(20)
    target = robot.end_effectors(q0)["finger_1"][:3, 3] + np.array([0.001, 0.0, 0.0])
    result = solver.solve("finger_1", target, q0)

    assert result.q.shape == (20,)
    assert result.history.size > 0
