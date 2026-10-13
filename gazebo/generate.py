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
from hand_robot.visualization import hand_mesh as hm

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


def _mesh_visuals(parent, parts):
    for p in parts:
        v = _sub(parent, "visual")
        g = _sub(v, "geometry"); _sub(g, "mesh", filename=f"@MESH_DIR@/{p.name}.obj")
        _sub(v, "material", name=p.material)


def build_urdf(config: dict, effort: bool = False) -> str:
    robot = HandRobot(config)
    meshes = hm.build_hand_meshes(robot)
    r_tip = float(config["piano"].get("tip_collision_radius", 0.0055))
    rbd = RigidBodyDynamics(robot, config)
    qmin, qmax = joint_limits(config)
    act = config["actuators"]
    car = config.get("carriage", {})
    kb = Keyboard(config)
    b0 = kb.base_home()
    B = float(config["rigid_body"]["viscous_friction"])

    root = ET.Element("robot", name=config["robot"]["name"])
    for mname, rgba in hm.MATERIALS.items():
        mat = _sub(root, "material", name=mname)
        _sub(mat, "color", rgba=" ".join(f"{v:g}" for v in rgba))
    _sub(root, "link", name="world")

    # ---------------------------------------------------------------- carriage (keyboard frame)
    lim = {"x": (-0.20, 0.25), "y": (-0.20, 0.60), "z": (0.0, 0.25)}
    prev = "world"
    bl = _sub(root, "link", name="base_link")                     # DH frame 0 (hand base); mass is in the links
    i0 = _sub(bl, "inertial"); _sub(i0, "mass", value="0.001")
    _sub(i0, "inertia", ixx="1e-9", iyy="1e-9", izz="1e-9", ixy="0", ixz="0", iyz="0")
    _mesh_visuals(bl, meshes["base_link"])
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
        _mesh_visuals(l, meshes[f"link{i}"])                         # shells + hinge cylinders (visual only)
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
        _mesh_visuals(l, meshes[tip])
        col = _sub(l, "collision", name="tip_contact")
        _sub(col, "origin", xyz=f"{-r_tip:.6f} 0 0", rpy="0 0 0")      # tip frame origin = TCP; the sphere surface reaches it along +x
        g = _sub(col, "geometry"); _sub(g, "sphere", radius=r_tip)
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


def _box_link(name, pose, size, rgba, mass=None, collision=True, spec=0.1, mu=None):
    x, y, z = pose
    sx, sy, sz = size
    r, gc, bl, al = rgba
    inertial = ""
    if mass is not None:
        ixx, iyy, izz = (mass / 12) * (sy ** 2 + sz ** 2), (mass / 12) * (sx ** 2 + sz ** 2), (mass / 12) * (sx ** 2 + sy ** 2)
        inertial = (f"<inertial><mass>{mass}</mass><inertia><ixx>{ixx:.3e}</ixx><iyy>{iyy:.3e}</iyy><izz>{izz:.3e}</izz>"
                    f"</inertia></inertial>")
    surf = (f"<surface><friction><ode><mu>{mu}</mu><mu2>{mu}</mu2></ode></friction></surface>" if mu else "")
    col = (f'<collision name="c"><geometry><box><size>{sx:.5f} {sy:.5f} {sz:.5f}</size></box></geometry>{surf}</collision>'
           if collision else "")
    return f"""
      <link name="{name}">
        <pose>{x:.5f} {y:.5f} {z:.5f} 0 0 0</pose>{inertial}{col}
        <visual name="v"><geometry><box><size>{sx:.5f} {sy:.5f} {sz:.5f}</size></box></geometry>
          <material><ambient>{r} {gc} {bl} {al}</ambient><diffuse>{r} {gc} {bl} {al}</diffuse>
            <specular>{spec} {spec} {spec} 1</specular></material></visual>
      </link>"""


def build_world(config: dict, rtf: float = 1.0) -> str:
    """Piano world: 22 white + 15 black keys (C3..C6, each a spring-loaded prismatic key), keybed, cheeks, key slip,
    back panel, legs, floor. Keys are named by MIDI number (key_60 / key_60_joint = C4)."""
    from hand_robot.piano import layout as lay
    kb = Keyboard(config)
    p = config["piano"]
    w, depth, delta = kb.width, kb.depth, kb.press_depth
    bw, bl, bh = float(p.get("black_key_width", 0.0137)), float(p.get("black_key_length", 0.060)), float(p.get("black_key_height", 0.010))
    rng = tuple(lay.midi(n) for n in p.get("world_range", ["C3", "C6"]))
    ms = lay.keys(rng)
    x0 = kb.front_x
    ys = [lay.key_y(m, w) for m in ms]
    y_lo, y_hi = min(ys) - w / 2, max(ys) + w / 2                    # outer edges of the keyboard
    yc, span = 0.5 * (y_lo + y_hi), y_hi - y_lo
    keys, joints, names = [], [], []
    for m in ms:
        y = lay.key_y(m, w)
        name = lay.key_link_name(m)
        if lay.is_black(m):
            size, rgba, mass, spec = (bl, bw, 0.022), (0.05, 0.05, 0.055, 1), 0.02, 0.5
            pose = (x0 + depth - bl / 2, y, bh - 0.022 / 2)          # top face at +bh, bottom hidden inside the keybed
        else:
            size, rgba, mass, spec = (depth, w - 0.001, 0.024), (0.96, 0.95, 0.90, 1), 0.03, 0.15
            pose = (x0 + depth / 2, y, -0.012)                       # top face at z = 0
        keys.append(_box_link(name, pose, size, rgba, mass=mass, mu=0.6, spec=spec))
        joints.append(f"""
      <joint name="{lay.key_joint_name(m)}" type="prismatic">
        <parent>base</parent><child>{name}</child>
        <axis><xyz>0 0 1</xyz>
          <limit><lower>{-delta}</lower><upper>0</upper></limit>
          <dynamics><spring_reference>0</spring_reference><spring_stiffness>{kb.k_key}</spring_stiffness><damping>{kb.c_key}</damping></dynamics>
        </axis>
      </joint>""")
        names.append(f"<joint_name>{lay.key_joint_name(m)}</joint_name>")
    wood, dark = (0.30, 0.17, 0.10, 1), (0.04, 0.04, 0.045, 1)
    xc = x0 + depth / 2
    body = "".join([
        # keybed (the only collision of the case: the keys travel inside it, self-collision is off)
        _box_link("base", (xc, yc, -0.05), (depth + 0.06, span + 0.08, 0.076), wood, mass=50.0, spec=0.2),
        _box_link("cheek_l", (xc, y_lo - 0.025, -0.01), (depth + 0.06, 0.03, 0.14), wood, mass=5.0, collision=False, spec=0.2),
        _box_link("cheek_r", (xc, y_hi + 0.025, -0.01), (depth + 0.06, 0.03, 0.14), wood, mass=5.0, collision=False, spec=0.2),
        _box_link("key_slip", (x0 - 0.016, yc, -0.030), (0.028, span + 0.08, 0.040), dark, mass=3.0, collision=False, spec=0.6),
        _box_link("back_panel", (x0 + depth + 0.040, yc, 0.060), (0.020, span + 0.08, 0.200), dark, mass=8.0, collision=False, spec=0.6),
        _box_link("leg_l", (xc, y_lo - 0.01, -0.42), (0.05, 0.05, 0.66), wood, mass=4.0, collision=False, spec=0.2),
        _box_link("leg_r", (xc, y_hi + 0.01, -0.42), (0.05, 0.05, 0.66), wood, mass=4.0, collision=False, spec=0.2),
    ])
    fixed = "".join(f'\n      <joint name="fix_{n}" type="fixed"><parent>base</parent><child>{n}</child></joint>'
                    for n in ("cheek_l", "cheek_r", "key_slip", "back_panel", "leg_l", "leg_r"))
    return f"""<?xml version="1.0"?>
<sdf version="1.9">
  <world name="piano_world">
    <physics name="p" type="dart"><max_step_size>{config['simulation']['dt'] / 5:.5f}</max_step_size><real_time_factor>{rtf:g}</real_time_factor></physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <gravity>0 0 -9.81</gravity>
    <scene><ambient>0.45 0.45 0.47 1</ambient><background>0.62 0.70 0.80 1</background><shadows>true</shadows></scene>
    <light type="directional" name="sun"><pose>0 0 3 0 0 0</pose><direction>-0.4 0.25 -1</direction>
      <diffuse>0.9 0.88 0.85 1</diffuse><specular>0.3 0.3 0.3 1</specular><cast_shadows>true</cast_shadows></light>
    <light type="point" name="fill"><pose>-0.35 {yc:.3f} 0.45 0 0 0</pose><diffuse>0.35 0.37 0.42 1</diffuse>
      <specular>0.1 0.1 0.1 1</specular><attenuation><range>3</range><constant>0.6</constant><linear>0.2</linear><quadratic>0.1</quadratic></attenuation>
      <cast_shadows>false</cast_shadows></light>
    <model name="floor"><static>true</static><link name="l">
      <collision name="c"><pose>0 0 -0.76 0 0 0</pose><geometry><plane><normal>0 0 1</normal><size>10 10</size></plane></geometry></collision>
      <visual name="v"><pose>0 0 -0.76 0 0 0</pose><geometry><plane><normal>0 0 1</normal><size>10 10</size></plane></geometry>
        <material><ambient>0.35 0.36 0.38 1</ambient><diffuse>0.35 0.36 0.38 1</diffuse><specular>0.05 0.05 0.05 1</specular></material></visual></link></model>
    <model name="piano">
      <static>false</static>
      <self_collide>false</self_collide>
      <joint name="fix" type="fixed"><parent>world</parent><child>base</child></joint>{fixed}{body}{''.join(keys)}{''.join(joints)}
      <plugin filename="gz-sim-joint-state-publisher-system" name="gz::sim::systems::JointStatePublisher">{''.join(names)}</plugin>
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
    ap.add_argument("--mesh-dir", default=None,
                    help="URI prefix of the mesh directory (e.g. file:///abs/path/meshes); default: keep the token "
                         "@MESH_DIR@, replaced by the launch file with file://<package share>/meshes")
    ap.add_argument("--controllers-dir", default=None,
                    help="directory that holds controllers*.yaml on the machine that runs Gazebo; "
                         "default: keep the token @CONTROLLERS_DIR@, which launch/roboliszt.launch.py replaces with the "
                         "installed package share directory")
    a = ap.parse_args()
    cfg = load_yaml(default_config_path())
    out = ROOT / "gazebo"
    cdir = a.controllers_dir or "@CONTROLLERS_DIR@"
    mdir = a.mesh_dir or "@MESH_DIR@"
    (out / "meshes").mkdir(exist_ok=True)
    for old in (out / "meshes").glob("*.obj"):
        old.unlink()
    for parts in hm.build_hand_meshes(HandRobot(cfg)).values():
        for part in parts:
            hm.write_obj(out / "meshes" / f"{part.name}.obj", part)
    for name, effort in (("roboliszt.urdf", False), ("roboliszt_effort.urdf", True)):
        yml = "controllers_effort.yaml" if effort else "controllers.yaml"
        (out / "urdf" / name).write_text(build_urdf(cfg, effort).replace("CONTROLLERS_YAML", f"{cdir}/{yml}").replace("@MESH_DIR@", mdir))
    (out / "worlds" / "piano.sdf").write_text(build_world(cfg, a.rtf))
    (out / "config" / "controllers.yaml").write_text(build_controllers(cfg, effort=False))
    (out / "config" / "controllers_effort.yaml").write_text(build_controllers(cfg, effort=True))
    print("written:", *[str(p.relative_to(ROOT)) for p in sorted(out.rglob("*")) if p.is_file() and p.suffix in (".urdf", ".sdf", ".yaml")], sep="\n  ")


if __name__ == "__main__":
    main()
