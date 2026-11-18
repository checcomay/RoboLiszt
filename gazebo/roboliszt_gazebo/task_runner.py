"""Send a task trajectory (data/trajectories/*.npz) to the hand controller  -- NOT EXECUTED.

    ros2 run roboliszt_gazebo task_runner data/trajectories/task3_chord_C4E4G4.npz --rate 100

The npz holds q (N, 20) and base (N, 3) = carriage position in the keyboard frame; both are sent as one
FollowJointTrajectory goal on /hand_controller/follow_joint_trajectory (joints j1..j20, carriage_x/y/z).
Tasks 1-4 only (position mode). Task 5 is task5_touch_node (effort mode).

Changes with respect to the first version (a goal that is not executed used to leave NO trace in this terminal):
  - wait_for_server has a timeout and says what to check;
  - the goal acceptance is checked (a goal rejected by the controller is reported here, not only in the launch terminal);
  - the LAST sample is always sent, with zero velocity at both ends (JointTrajectoryController rejects a trajectory whose
    last point has a non-zero velocity unless allow_nonzero_velocity_at_trajectory_end is true);
  - lead time before the first sample computed from the current /joint_states (the robot spawns at q = 0, the task starts
    from the ready posture), parameter --lead;
  - simulation time (use_sim_time), result status and error string are printed.
"""
import argparse
import sys
import time

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

JOINTS = [f"j{i}" for i in range(1, 21)] + ["carriage_x", "carriage_y", "carriage_z"]


class Runner(Node):
    def __init__(self):
        super().__init__("roboliszt_task_runner", parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        self.client = ActionClient(self, FollowJointTrajectory, "/hand_controller/follow_joint_trajectory")
        self.cur = None
        self.create_subscription(JointState, "/joint_states", lambda m: setattr(self, "cur", dict(zip(m.name, m.position))), 10)

    def send(self, npz_path: str, rate: float, lead: float) -> int:
        log = self.get_logger()
        d = np.load(npz_path, allow_pickle=True)
        t, q, base = d["t"], d["q"], d["base"]
        Q = np.hstack([q, base])
        step = max(int(round((1.0 / rate) / (t[1] - t[0]))), 1)
        idx = np.unique(np.r_[np.arange(0, len(t), step), len(t) - 1])          # the last sample is always sent
        qd = np.gradient(Q, t, axis=0)
        qd[0, :] = 0.0
        qd[-1, :] = 0.0

        if not self.client.wait_for_server(timeout_sec=30.0):
            log.error("action server /hand_controller/follow_joint_trajectory NOT available after 30 s: the controller is not "
                      "active (check  ros2 control list_controllers ; the spawners start 8 s after the launch) or Gazebo is not running")
            return 1
        t_end = time.time() + 5.0
        while self.cur is None and time.time() < t_end:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.cur is not None and all(j in self.cur for j in JOINTS):
            dq = float(np.abs(Q[0, :20] - np.array([self.cur[j] for j in JOINTS[:20]])).max())
            lead = max(lead, 1.9 * dq / 4.0)                                    # keep the first move below ~4 rad/s
            log.info(f"largest joint displacement to the first sample: {np.rad2deg(dq):.0f} deg -> lead time {lead:.1f} s")
        else:
            log.warn("no /joint_states received: lead time is the --lead value")

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINTS                                    # header.stamp = 0: start now (sim time)
        for k in idx:
            p = JointTrajectoryPoint()
            p.positions = [float(x) for x in Q[k]]
            p.velocities = [float(x) for x in qd[k]]
            sec = float(t[k] - t[0]) + lead
            p.time_from_start = Duration(sec=int(sec), nanosec=int((sec % 1) * 1e9))
            goal.trajectory.points.append(p)
        total = float(t[-1] - t[0]) + lead

        fut = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)
        gh = fut.result() if fut.done() else None
        if gh is None or not gh.accepted:
            log.error("goal REJECTED (or no answer in 10 s). The reason is printed by hand_controller in the terminal of the launch "
                      "file: joint names/order, non-increasing times, non-zero final velocity ...")
            return 1
        log.info(f"goal accepted: {len(goal.trajectory.points)} points, {total:.1f} s")
        res_fut = gh.get_result_async()
        rclpy.spin_until_future_complete(self, res_fut, timeout_sec=total + 20.0)
        if not res_fut.done():
            log.error("no result before the timeout: is the simulation much slower than real time (RTF)?")
            return 1
        r = res_fut.result()
        log.info(f"finished: status = {r.status}, error_code = {r.result.error_code} {r.result.error_string}")
        return 0 if r.result.error_code == 0 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--rate", type=float, default=100.0)
    ap.add_argument("--lead", type=float, default=3.0, help="s before the first sample (minimum; increased if the first move is large)")
    a = ap.parse_args()
    rclpy.init()
    rc = Runner().send(a.npz, a.rate, a.lead)
    rclpy.shutdown()
    sys.exit(rc)


if __name__ == "__main__":
    main()