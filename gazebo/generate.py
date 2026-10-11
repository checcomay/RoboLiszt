"""Generate URDF, Gazebo world and ros2_control configuration from config/robot.yaml.

    python gazebo/generate.py [--controllers-path /abs/path/controllers.yaml]

URDF mapping of the standard-DH convention  T_i = Rz(theta) Tz(d) Tx(a) Rx(alpha)  (rotation FIRST, then the
fixed part F_i = Tz(d)Tx(a)Rx(alpha)), whereas URDF applies  <origin> THEN the joint rotation:

    joint i origin (relative to the parent URDF link)  =  F_parent(i) . Rz(theta_0,i)        (F_0 = I)
    URDF link frame  L_i = DH frame i-1'  after the rotation,   DH frame i = L_i . F_i
    fingertip fixed joint origin  =  F_last . Tx(L_TIP)

The hand base (DH frame 0) is carried by a 3-axis prismatic carriage (x, y, z) in the keyboard frame.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from xml.dom import minidom
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hand_robot.config.loader import default_config_path, joint_limits, load_yaml
from hand_robot.dynamics.rigid_body import RigidBodyDynamics
from hand_robot.model.robot import HandRobot
from hand_robot.piano.keyboard import Keyboard

JOINT_NAMES = [f"j{i}" for i in range(1, 21)] + ["carriage_x", "carriage_y", "carriage_z"]


def F(a, alpha, d) -> np.ndarray:
    """Fixed part of the DH transform: Tz(d) Tx(a) Rx(alpha)."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("x", alpha).as_matrix()
    T[:3, 3] = [a, 0.0, d]
    return T


def rz(theta) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("z", theta).as_matrix()
    return T


def origin_attrs(T) -> dict:
    rpy = Rotation.from_matrix(T[:3, :3]).as_euler("xyz")           # URDF rpy: R = Rz(y) Ry(p) Rx(r)
    return {"xyz": " ".join(f"{v:.9g}" for v in T[:3, 3]), "rpy": " ".join(f"{v:.9g}" for v in rpy)}


def _sub(parent, tag, **attrs):
    e = ET.SubElement(parent, tag)
    for k, v in attrs.items():
        e.set(k, str(v))
    return e


def build_urdf(config: dict, effort: bool = False) -> str:
    robot = HandRobot(config)
    rbd = RigidBodyDynamics(robot, config)
    qmin, qmax = joint_limits(config)
    act = config["actuators"]
    car = config.get("carriage", {})
    kb = Keyboard(config)
    b0 = kb.base_home()
    B = float(config["rigid_body"]["viscous_friction"])

    root = ET.Element("robot", name=config["robot"]["name"])
    _sub(root, "link", name="world")

    # ---------------------------------------------------------------- carriage (keyboard frame)
    lim = {"x": (-0.20, 0.25), "y": (-0.20, 0.60), "z": (0.0, 0.25)}
    prev = "world"
    bl = _sub(root, "link", name="base_link")                     # DH frame 0 (hand base); mass is in the links
    i0 = _sub(bl, "inertial"); _sub(i0, "mass", value="0.001")
    _sub(i0, "inertia", ixx="1e-9", iyy="1e-9", izz="1e-9", ixy="0", ixz="0", iyz="0")
    for ax, vec in (("x", "1 0 0"), ("y", "0 1 0"), ("z", "0 0 1")):
        child = f"carriage_{ax}_link" if ax != "z" else "base_link"
        if child != "base_link":
            l = _sub(root, "link", name=child)
            i = _sub(l, "inertial"); _sub(i, "mass", value="0.001")
            _sub(i, "inertia", ixx="1e-9", iyy="1e-9", izz="1e-9", ixy="0", ixz="0", iyz="0")
        j = _sub(root, "joint", name=f"carriage_{ax}", type="prismatic")
        _sub(j, "parent", link=prev); _sub(j, "child", link=child)
        _sub(j, "origin", xyz="0 0 0", rpy="0 0 0"); _sub(j, "axis", xyz=vec)
        _sub(j, "limit", lower=lim[ax][0], upper=lim[ax][1], effort=200, velocity=float(car.get("max_speed", 1.0)))
        _sub(j, "dynamics", damping="5.0", friction="0")
        prev = child
    # base_link carries the (fixed) base frame DH0; its inertia is zero, the hand mass is in the links

    link_name = {0: "base_link"}
    Fm = {0: np.eye(4)}
    for i in range(1, 21):
        p = robot.dh[i]
        Fm[i] = F(p.a, p.alpha, p.d)
        link_name[i] = f"link{i}"
    cfg_t = {}
    for i in range(1, 21):
        p = robot.dh[i]
        par = robot.parent[i]
        # --- link (frame L_i)
        l = _sub(root, "link", name=link_name[i])
        Rf = Fm[i][:3, :3]
        c_L = Rf @ rbd.com[i] + Fm[i][:3, 3]
        I_L = Rf @ rbd.inertia[i] @ Rf.T
        inn = _sub(l, "inertial")
        _sub(inn, "origin", xyz=" ".join(f"{v:.9g}" for v in c_L), rpy="0 0 0")
        _sub(inn, "mass", value=f"{rbd.mass[i]:.9g}")
        _sub(inn, "inertia", ixx=f"{I_L[0,0]:.9g}", iyy=f"{I_L[1,1]:.9g}", izz=f"{I_L[2,2]:.9g}",
             ixy=f"{I_L[0,1]:.9g}", ixz=f"{I_L[0,2]:.9g}", iyz=f"{I_L[1,2]:.9g}")
        seg = Fm[i][:3, 3]                                         # O_{i-1} -> O_i in L_i
        v = _sub(l, "visual")
        if np.linalg.norm(seg) > 1e-6:
            u = seg / np.linalg.norm(seg)
            R = Rotation.align_vectors([u], [[0, 0, 1]])[0].as_matrix()
            T = np.eye(4); T[:3, :3] = R; T[:3, 3] = seg / 2
            _sub(v, "origin", **origin_attrs(T))
            g = _sub(v, "geometry"); _sub(g, "cylinder", radius=rbd.R * 0.8, length=f"{np.linalg.norm(seg):.9g}")
        else:
            _sub(v, "origin", xyz="0 0 0", rpy="0 0 0")
            g = _sub(v, "geometry"); _sub(g, "sphere", radius=rbd.R)
        # --- joint
        j = _sub(root, "joint", name=f"j{i}", type="revolute")
        _sub(j, "parent", link=link_name[par]); _sub(j, "child", link=link_name[i])
        _sub(j, "origin", **origin_attrs(Fm[par] @ rz(p.theta_offset)))
        _sub(j, "axis", xyz="0 0 1")
        tmax = act["tau_max_wrist"] if i <= 2 else act["tau_max_finger"]
        _sub(j, "limit", lower=f"{qmin[i-1]:.9g}", upper=f"{qmax[i-1]:.9g}", effort=tmax, velocity=12.0)
        _sub(j, "dynamics", damping=B, friction="0")

    # --- fingertips: fixed link with a contact sphere
    for name, br in robot.branches.items():
        last = br[-1]
        Tt = np.eye(4); Tt[0, 3] = robot.L_TIP
        tip = f"tip_{name}"
        l = _sub(root, "link", name=tip)
        inn = _sub(l, "inertial"); _sub(inn, "mass", value="1e-6")
        _sub(inn, "inertia", ixx="1e-10", iyy="1e-10", izz="1e-10", ixy="0", ixz="0", iyz="0")
        for kind in ("visual", "collision"):
            e = _sub(l, kind); g = _sub(e, "geometry"); _sub(g, "sphere", radius=0.0075)
        j = _sub(root, "joint", name=f"{tip}_fixed", type="fixed")
        _sub(j, "parent", link=link_name[last]); _sub(j, "child", link=tip)
        _sub(j, "origin", **origin_attrs(Fm[last] @ Tt))

    # --- ros2_control
    rc = _sub(root, "ros2_control", name="RoboLisztSystem", type="system")
    hw = _sub(rc, "hardware"); pl = _sub(hw, "plugin"); pl.text = "gz_ros2_control/GazeboSimSystem"
    for jn in JOINT_NAMES:
        e = _sub(rc, "joint", name=jn)
        if effort and jn.startswith("j"):
            _sub(e, "command_interface", name="effort")
        else:
            _sub(e, "command_interface", name="position")
        for s in ("position", "velocity", "effort"):
            _sub(e, "state_interface", name=s)
        if jn.startswith("carriage"):
            ax = jn[-1]
            _sub(e, "param", name="initial_position").text = f"{b0['xyz'.index(ax)]:.6f}"
    # gazebo reference + plugin
    gz = _sub(root, "gazebo")
    pg = _sub(gz, "plugin", filename="gz_ros2_control-system", name="gz_ros2_control::GazeboSimROS2ControlPlugin")
    _sub(pg, "parameters").text = "CONTROLLERS_YAML"
    for name in robot.branches:
        g2 = _sub(root, "gazebo", reference=f"tip_{name}")
        _sub(g2, "mu1").text = "0.8"; _sub(g2, "mu2").text = "0.8"
    xml = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
    return xml


def build_world(config: dict, rtf: float = 1.0) -> str:
    kb = Keyboard(config)
    w, depth, delta = kb.width, kb.depth, kb.press_depth
    keys, joints = [], []
    for k in range(-7, 15):                                        # C3 ... C6 (white keys)
        name = f"key_{k + 7:02d}"
        y = k * w
        keys.append(f"""
      <link name="{name}">
        <pose>{kb.front_x + depth / 2:.5f} {y:.5f} {-0.012:.5f} 0 0 0</pose>
        <inertial><mass>0.03</mass><inertia><ixx>1e-5</ixx><iyy>1e-5</iyy><izz>1e-5</izz></inertia></inertial>
        <collision name="c"><geometry><box><size>{depth:.4f} {w - 0.001:.4f} 0.024</size></box></geometry>
          <surface><friction><ode><mu>0.6</mu><mu2>0.6</mu2></ode></friction></surface></collision>
        <visual name="v"><geometry><box><size>{depth:.4f} {w - 0.001:.4f} 0.024</size></box></geometry>
          <material><ambient>0.95 0.95 0.92 1</ambient><diffuse>0.95 0.95 0.92 1</diffuse></material></visual>
      </link>""")
        joints.append(f"""
      <joint name="{name}_joint" type="prismatic">
        <parent>base</parent><child>{name}</child>
        <axis><xyz>0 0 1</xyz>
          <limit><lower>{-delta}</lower><upper>0</upper></limit>
          <dynamics><spring_reference>0</spring_reference><spring_stiffness>{kb.k_key}</spring_stiffness><damping>{kb.c_key}</damping></dynamics>
        </axis>
      </joint>""")
    jps = "".join(f"<joint_name>key_{k + 7:02d}_joint</joint_name>" for k in range(-7, 15))
    return f"""<?xml version="1.0"?>
<sdf version="1.9">
  <world name="piano_world">
    <physics name="p" type="dart"><max_step_size>{config['simulation']['dt'] / 5:.5f}</max_step_size><real_time_factor>{rtf:g}</real_time_factor></physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <gravity>0 0 -9.81</gravity>
    <light type="directional" name="sun"><direction>-0.3 0.2 -1</direction><diffuse>1 1 1 1</diffuse><cast_shadows>false</cast_shadows></light>
    <model name="piano">
      <static>false</static>
      <link name="base"><pose>{kb.front_x + depth / 2:.5f} {3.5 * w:.5f} -0.05 0 0 0</pose>
        <inertial><mass>50</mass><inertia><ixx>1</ixx><iyy>1</iyy><izz>1</izz></inertia></inertial>
        <collision name="c"><geometry><box><size>{depth + 0.06:.3f} {23 * w:.3f} 0.076</size></box></geometry></collision>
        <visual name="v"><geometry><box><size>{depth + 0.06:.3f} {23 * w:.3f} 0.076</size></box></geometry>
          <material><ambient>0.15 0.1 0.08 1</ambient><diffuse>0.15 0.1 0.08 1</diffuse></material></visual></link>
      <joint name="fix" type="fixed"><parent>world</parent><child>base</child></joint>{''.join(keys)}{''.join(joints)}
      <plugin filename="gz-sim-joint-state-publisher-system" name="gz::sim::systems::JointStatePublisher">{jps}</plugin>
    </model>
  </world>
</sdf>
"""


def build_controllers(config: dict, effort: bool = False) -> str:
    """ros2_control parameters.

    position (tasks 1-4): joint_state_broadcaster + hand_controller (JointTrajectoryController on j1..j20 and the carriage),
        update_rate 200 Hz.
    effort (task 5): joint_state_broadcaster + carriage_controller (position, 3 prismatic joints) + effort_controller
        (ForwardCommandController on j1..j20), update_rate 1000 Hz: the operational-space touch controller degrades at
        2 ms and is unstable at 5 ms (data/task5_rigid_rate_sweep.json), so 200 Hz is NOT enough for task 5.
    """
    hand = "\n".join(f"      - {j}" for j in JOINT_NAMES)
    eff = "\n".join(f"      - {j}" for j in JOINT_NAMES[:20])
    car = "\n".join(f"      - {j}" for j in JOINT_NAMES[20:])
    if not effort:
        return f"""controller_manager:
  ros__parameters:
    update_rate: 200
    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster
    hand_controller:
      type: joint_trajectory_controller/JointTrajectoryController

hand_controller:
  ros__parameters:
    joints:
{hand}
    command_interfaces: [position]
    state_interfaces: [position, velocity]
    allow_partial_joints_goal: false
    constraints:
      stopped_velocity_tolerance: 0.1
      goal_time: 0.5
"""
    return f"""controller_manager:
  ros__parameters:
    update_rate: 1000
    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster
    carriage_controller:
      type: joint_trajectory_controller/JointTrajectoryController
    effort_controller:
      type: forward_command_controller/ForwardCommandController

carriage_controller:
  ros__parameters:
    joints:
{car}
    command_interfaces: [position]
    state_interfaces: [position, velocity]
    allow_partial_joints_goal: false

effort_controller:
  ros__parameters:
    joints:
{eff}
    interface_name: effort
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rtf", type=float, default=1.0,
                    help="world real-time factor; use ~0.25 for task 5, whose python node needs ~2-5 ms per 1 ms step")
    ap.add_argument("--controllers-dir", default=None,
                    help="directory that holds controllers*.yaml on the machine that runs Gazebo; "
                         "default: keep the token @CONTROLLERS_DIR@, which launch/roboliszt.launch.py replaces with the "
                         "installed package share directory")
    a = ap.parse_args()
    cfg = load_yaml(default_config_path())
    out = ROOT / "gazebo"
    cdir = a.controllers_dir or "@CONTROLLERS_DIR@"
    for name, effort in (("roboliszt.urdf", False), ("roboliszt_effort.urdf", True)):
        yml = "controllers_effort.yaml" if effort else "controllers.yaml"
        (out / "urdf" / name).write_text(build_urdf(cfg, effort).replace("CONTROLLERS_YAML", f"{cdir}/{yml}"))
    (out / "worlds" / "piano.sdf").write_text(build_world(cfg, a.rtf))
    (out / "config" / "controllers.yaml").write_text(build_controllers(cfg, effort=False))
    (out / "config" / "controllers_effort.yaml").write_text(build_controllers(cfg, effort=True))
    print("written:", *[str(p.relative_to(ROOT)) for p in sorted(out.rglob("*")) if p.is_file() and p.suffix in (".urdf", ".sdf", ".yaml")], sep="\n  ")


if __name__ == "__main__":
    main()
