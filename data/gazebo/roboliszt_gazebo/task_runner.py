"""Send a task trajectory (data/trajectories/*.npz) to the hand controller  -- NOT EXECUTED.

    ros2 run roboliszt_gazebo task_runner data/trajectories/task3_chord_C4E4G4.npz --rate 100

The npz holds q (N, 20) and base (N, 3) = carriage position in the keyboard frame; both are sent as one
FollowJointTrajectory goal on /hand_controller/follow_joint_trajectory (joints j1..j20, carriage_x/y/z).
Tasks 1-4 only (position mode). Task 5 is task5_touch_node (effort mode).
"""
import argparse

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectoryPoint

JOINTS = [f"j{i}" for i in range(1, 21)] + ["carriage_x", "carriage_y", "carriage_z"]


class Runner(Node):
    def __init__(self):
        super().__init__("roboliszt_task_runner")
        self.client = ActionClient(self, FollowJointTrajectory, "/hand_controller/follow_joint_trajectory")

    def send(self, npz_path: str, rate: float):
        d = np.load(npz_path, allow_pickle=True)
        t, q, base = d["t"], d["q"], d["base"]
        step = max(int(round((1.0 / rate) / (t[1] - t[0]))), 1)
        idx = np.arange(0, len(t), step)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINTS
        qd = np.gradient(np.hstack([q, base]), t, axis=0)
        for k in idx:
            p = JointTrajectoryPoint()
            p.positions = [float(x) for x in np.r_[q[k], base[k]]]
            p.velocities = [float(x) for x in qd[k]]
            sec = float(t[k]) + 1.0                                   # 1 s to reach the first sample
            p.time_from_start = Duration(sec=int(sec), nanosec=int((sec % 1) * 1e9))
            goal.trajectory.points.append(p)
        self.client.wait_for_server()
        fut = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut)
        res_fut = fut.result().get_result_async()
        rclpy.spin_until_future_complete(self, res_fut)
        self.get_logger().info(f"finished, error_code = {res_fut.result().result.error_code}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--rate", type=float, default=100.0)
    a = ap.parse_args()
    rclpy.init()
    Runner().send(a.npz, a.rate)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
