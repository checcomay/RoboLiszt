"""Task 5 touch node (ROS 2 Jazzy)  -- NOT EXECUTED: no ROS 2 / Gazebo was available where this was written.

    ROBOLISZT_ROOT=/path/to/roboliszt ros2 run roboliszt_gazebo task5_touch_node --ros-args \
        -p use_sim_time:=true -p finger:=middle -p profile:=pp

Thin rclpy wrapper around hand_robot.control.touch_node_core.TouchNodeCore (the core IS tested offline, tests/test_touch_node_core.py).
Needs the effort URDF + controllers_effort.yaml (ros2 launch roboliszt_gazebo roboliszt.launch.py mode:=effort).

  /joint_states                                   -> q, qd of j1..j20
  /world/piano_world/model/piano/joint_state      -> key joint position/velocity  (bridged by the launch file)
  /effort_controller/commands   (Float64MultiArray, 20 efforts)  <- control output
  /carriage_controller/joint_trajectory           <- carriage held at the play pose (base_home)

The control loop is driven by the simulation time stamps of /joint_states (one control step per message), so it runs at the
ros2_control update rate (1000 Hz in controllers_effort.yaml). The python core needs ~2-5 ms per step: run Gazebo at a
real-time factor <= 0.25 (generate.py --rtf 0.25) or the node cannot keep up. Efforts are limited to the actuator limits of
config/robot.yaml: the URDF effort limit of the touching finger must be at least as large (0.25 N m for the ff profile).
"""
import json
import os
import sys
from pathlib import Path

root = os.environ.get("ROBOLISZT_ROOT")
if root:
    sys.path.insert(0, str(Path(root) / "src"))

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from hand_robot.control.touch_node_core import TouchNodeCore
from hand_robot.tasks.common import Scene
from hand_robot.tasks.task5_touch import PROFILES

HAND = [f"j{i}" for i in range(1, 21)]
CARRIAGE = ["carriage_x", "carriage_y", "carriage_z"]


class TouchNode(Node):
    def __init__(self):
        super().__init__("roboliszt_task5_touch")
        self.declare_parameter("finger", "middle")
        self.declare_parameter("profile", "pp")
        self.declare_parameter("tau_max_finger", 0.0)            # 0 -> config value
        self.declare_parameter("t_pre", 3.0)                     # s, joint-space approach to the start posture
        self.declare_parameter("run_time", 2.6)                  # s of touch before the log is written
        self.declare_parameter("out_prefix", "task5_gazebo")
        self.declare_parameter("key_topic", "/world/piano_world/model/piano/joint_state")
        P = {p.name: p for p in PROFILES}
        prof = P[self.get_parameter("profile").value]
        tmf = float(self.get_parameter("tau_max_finger").value) or None
        self.scene = Scene()
        self.core = TouchNodeCore(self.scene, self.get_parameter("finger").value, prof, tau_max_finger=tmf,
                                  t_pre=float(self.get_parameter("t_pre").value))
        self.run_time = float(self.get_parameter("run_time").value)
        self.out_prefix = self.get_parameter("out_prefix").value
        self.pub = self.create_publisher(Float64MultiArray, "/effort_controller/commands", 10)
        self.car_pub = self.create_publisher(JointTrajectory, "/carriage_controller/joint_trajectory", 10)
        self.create_subscription(JointState, "/joint_states", self.on_hand, 100)
        self.create_subscription(JointState, self.get_parameter("key_topic").value, self.on_keys, 100)
        self.key, self.key_prev, self.carriage_sent, self.done = (0.0, 0.0), None, False, False
        self.get_logger().info(f"touch node ready: key joint {self.core.key_joint}, profile {prof.name}")

    @staticmethod
    def _stamp(msg) -> float:
        return msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec

    def on_keys(self, msg: JointState):
        if self.core.key_joint not in msg.name:
            return
        i = msg.name.index(self.core.key_joint)
        t, pos = self._stamp(msg), float(msg.position[i])
        if len(msg.velocity) > i:
            vel = float(msg.velocity[i])
        elif self.key_prev is not None and t > self.key_prev[0]:
            vel = (pos - self.key_prev[1]) / (t - self.key_prev[0])
        else:
            vel = 0.0
        self.key_prev, self.key = (t, pos), (pos, vel)

    def _send_carriage(self):
        tr = JointTrajectory()
        tr.joint_names = CARRIAGE
        p = JointTrajectoryPoint()
        p.positions = [float(x) for x in self.core.base]
        p.time_from_start = Duration(sec=2, nanosec=0)
        tr.points.append(p)
        self.car_pub.publish(tr)
        self.carriage_sent = True

    def on_hand(self, msg: JointState):
        idx = {n: k for k, n in enumerate(msg.name)}
        if not all(j in idx for j in HAND):
            return
        if not self.carriage_sent:
            self._send_carriage()
        q = np.array([msg.position[idx[j]] for j in HAND])
        qd = np.array([msg.velocity[idx[j]] if len(msg.velocity) > idx[j] else 0.0 for j in HAND])
        t = self._stamp(msg)
        tau = self.core.step(t, q, qd, *self.key)
        self.pub.publish(Float64MultiArray(data=[float(x) for x in tau]))
        if not self.done and self.core.log["t"] and self.core.log["t"][-1] >= self.run_time:
            self.finish()

    def finish(self):
        self.done = True
        lg = {k: np.asarray(v) for k, v in self.core.log.items()}
        np.savez(self.out_prefix + ".npz", **lg)
        m = self.core.metrics()
        json.dump(m, open(self.out_prefix + ".json", "w"), indent=1, default=float)
        self.get_logger().info("touch finished: " + json.dumps({k: m[k] for k in ("F_peak_N", "F_rms_error_pct", "key_bottomed")}))


def main():
    rclpy.init()
    rclpy.spin(TouchNode())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
