"""Send a task trajectory (data/trajectories/*.npz) to the hand controller  -- NOT EXECUTED.

    ros2 run roboliszt_gazebo task_runner data/trajectories/task2_press_middle_E4.npz --rate 100

The npz holds q (N, 20) and base (N, 3) = carriage position in the keyboard frame; both are sent as one
FollowJointTrajectory goal on /hand_controller/follow_joint_trajectory (joints j1..j20, carriage_x/y/z).
Tasks 1-4 only (position mode). Task 5 is task5_touch_node (effort mode).

v2:
  - the robot spawns at q = 0 (straight fingers) and the task starts from the ready posture: a straight joint-space move
    swings the fingertips through the keys (unwanted notes).  The approach is now staged: RAISE the carriage, move the
    joints to the ready posture in the air, DESCEND to the playing height;
  - waits as long as needed (--timeout, wall seconds), prints the progress and the measured real-time factor;
  - checks the goal acceptance, always sends the last sample with zero velocity, uses simulation time.
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
Z_MAX = 0.25                                                       # carriage_z upper limit in the URDF


def quintic(tau):
    return 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5


def dquintic(tau):
    return 30 * tau ** 2 - 60 * tau ** 3 + 30 * tau ** 4


class Runner(Node):
    def __init__(self):
        super().__init__("roboliszt_task_runner", parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        self.client = ActionClient(self, FollowJointTrajectory, "/hand_controller/follow_joint_trajectory")
        self.cur = None
        self.create_subscription(JointState, "/joint_states", lambda m: setattr(self, "cur", dict(zip(m.name, m.position))), 10)

    @staticmethod
    def _point(tt, pos, vel):
        p = JointTrajectoryPoint()
        p.positions = [float(x) for x in pos]
        p.velocities = [float(x) for x in vel]
        sec = int(tt)
        p.time_from_start = Duration(sec=sec, nanosec=int((tt - sec) * 1e9))
        return p

    def _stage(self, pts, t0, p0, p1, T):
        n, d = max(int(np.ceil(T / 0.05)), 2), p1 - p0
        for k in range(1, n + 1):
            tau = k / n
            pts.append(self._point(t0 + tau * T, p0 + d * quintic(tau), d * dquintic(tau) / T))
        return t0 + T

    def send(self, npz_path: str, rate: float, lead_min: float, timeout: float) -> int:
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
                      "active (check  ros2 control list_controllers) or Gazebo is not running")
            return 1
        t_end = time.time() + 5.0
        while self.cur is None and time.time() < t_end:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.cur is None or not all(j in self.cur for j in JOINTS):
            log.error("no /joint_states with the 23 joints: cannot plan the approach")
            return 1
        cur = np.array([self.cur[j] for j in JOINTS])

        # ---- staged approach: raise -> reconfigure in the air -> descend
        pts, tt = [], 0.0
        z_safe = min(Z_MAX - 0.01, max(0.20, float(Q[0, 22]) + 0.15))
        p = cur.copy()
        if cur[22] < z_safe - 1e-3:
            nxt = p.copy(); nxt[22] = z_safe
            tt = self._stage(pts, tt, p, nxt, max(0.8, 1.9 * (z_safe - cur[22]) / 0.4)); p = nxt
        nxt = Q[0].copy(); nxt[22] = max(z_safe, cur[22])
        dh = float(np.abs(nxt[:20] - p[:20]).max()); dxy = float(np.abs(nxt[20:22] - p[20:22]).max())
        tt = self._stage(pts, tt, p, nxt, max(lead_min, 1.9 * dh / 3.0, 1.9 * dxy / 0.4)); p = nxt
        if abs(p[22] - Q[0, 22]) > 1e-3:
            tt = self._stage(pts, tt, p, Q[0], max(0.8, 1.9 * abs(p[22] - Q[0, 22]) / 0.4))
        lead = tt
        log.info(f"approach: {np.rad2deg(dh):.0f} deg largest joint move, carriage via z = {z_safe:.2f} m, {lead:.1f} s")
        for k in idx[1:]:
            pts.append(self._point(lead + float(t[k] - t[0]), Q[k], qd[k]))
        total = lead + float(t[-1] - t[0])

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINTS                                    # header.stamp = 0: start now (sim time)
        goal.trajectory.points = pts
        last_print = [0.0]

        def feedback(fb):
            if time.time() - last_print[0] > 2.0:
                last_print[0] = time.time()
                s = fb.feedback.desired.time_from_start
                log.info(f"  trajectory time {s.sec + 1e-9 * s.nanosec:5.1f} / {total:.1f} s")

        fut = self.client.send_goal_async(goal, feedback_callback=feedback)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)
        gh = fut.result() if fut.done() else None
        if gh is None or not gh.accepted:
            log.error("goal REJECTED (or no answer in 10 s): see the message of hand_controller in the launch terminal")
            return 1
        sim0, wall0 = self.get_clock().now().nanoseconds * 1e-9, time.time()
        log.info(f"goal accepted: {len(pts)} points, {total:.1f} s of simulated time")
        res_fut = gh.get_result_async()
        rclpy.spin_until_future_complete(self, res_fut, timeout_sec=timeout)
        rtf = (self.get_clock().now().nanoseconds * 1e-9 - sim0) / max(time.time() - wall0, 1e-6)
        log.info(f"measured real-time factor ~ {rtf:.2f}")
        if not res_fut.done():
            log.error(f"no result after {timeout:.0f} s of wall time (the goal is still running in Gazebo)")
            return 1
        r = res_fut.result()
        log.info(f"finished: status = {r.status}, error_code = {r.result.error_code} {r.result.error_string}")
        return 0 if r.result.error_code == 0 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--rate", type=float, default=100.0)
    ap.add_argument("--lead", type=float, default=2.0, help="s, minimum duration of the in-the-air reconfiguration")
    ap.add_argument("--timeout", type=float, default=900.0, help="wall-clock seconds to wait for the result")
    a = ap.parse_args()
    rclpy.init()
    rc = Runner().send(a.npz, a.rate, a.lead, a.timeout)
    rclpy.shutdown()
    sys.exit(rc)


if __name__ == "__main__":
    main()