import numpy as np

from hand_robot.config.loader import default_config_path, load_yaml
from hand_robot.model.robot import HandRobot


def make_robot():
    return HandRobot(load_yaml(default_config_path()))


def test_exactly_20_joints():
    robot = make_robot()
    assert len(robot.dh) == 20


def test_branched_topology():
    robot = make_robot()
    assert robot.parent[1] == 0
    assert robot.parent[2] == 1
    assert robot.branches["finger_1"] == [3, 4, 5, 6]
    assert robot.branches["finger_2"] == [7, 8, 9]
    assert robot.branches["finger_3"] == [10, 11, 12]
    assert robot.branches["finger_4"] == [13, 14, 15]
    assert robot.branches["finger_5"] == [16, 17, 18, 19, 20]


def test_fk_contains_all_frames():
    robot = make_robot()
    T = robot.forward_kinematics(np.zeros(20))
    assert set(T) == set(range(21))
    assert all(M.shape == (4, 4) for M in T.values())


def test_common_joints_affect_all_fingers():
    robot = make_robot()
    q0 = np.zeros(20)
    q1 = np.zeros(20)
    q1[0] = 0.1

    p0 = robot.end_effectors(q0)
    p1 = robot.end_effectors(q1)

    for branch in p0:
        assert not np.allclose(p0[branch][:3, 3], p1[branch][:3, 3])


def test_other_branch_does_not_move_finger_one():
    robot = make_robot()
    from hand_robot.kinematics.jacobian import translational_jacobian

    J = translational_jacobian(robot, np.zeros(20), "finger_1")
    assert np.allclose(J[:, 6], 0.0, atol=1e-8)
