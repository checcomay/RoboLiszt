"""Independent check of the generated URDF against the DH model (no ROS needed).

Parses the URDF with ElementTree, composes <origin>/<axis> transforms by itself and compares, for random
joint/carriage configurations: (1) the DH frame of every joint (L_i . F_i), (2) the fingertip poses,
(3) the total mass and the global centre of mass, (4) the world inertia tensor of every link.
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "gazebo"))

from hand_robot.config.loader import default_config_path, joint_limits, load_yaml
from hand_robot.dynamics.rigid_body import RigidBodyDynamics
from hand_robot.model.robot import HandRobot
from generate import F


def _vec(s):
    return np.array([float(x) for x in s.split()])


def _T(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", _vec(rpy)).as_matrix()   # URDF rpy: R = Rz Ry Rx
    T[:3, 3] = _vec(xyz)
    return T


class Urdf:
    def __init__(self, path):
        root = ET.parse(path).getroot()
        self.links = {l.get("name"): l for l in root.findall("link")}
        self.joints = {j.get("name"): j for j in root.findall("joint")}
        self.children = {}
        for j in self.joints.values():
            self.children.setdefault(j.find("parent").get("link"), []).append(j)

    def fk(self, q: dict) -> dict:
        T = {"world": np.eye(4)}
        stack = ["world"]
        while stack:
            p = stack.pop()
            for j in self.children.get(p, []):
                o = j.find("origin")
                To = _T(o.get("xyz"), o.get("rpy"))
                typ = j.get("type")
                Tj = np.eye(4)
                if typ == "revolute":
                    Tj[:3, :3] = Rotation.from_euler("z", q[j.get("name")]).as_matrix()      # axis is +z in this URDF
                elif typ == "prismatic":
                    Tj[:3, 3] = _vec(j.find("axis").get("xyz")) * q[j.get("name")]
                c = j.find("child").get("link")
                T[c] = T[p] @ To @ Tj
                stack.append(c)
        return T

    def inertial(self, link):
        i = self.links[link].find("inertial")
        m = float(i.find("mass").get("value"))
        o = i.find("origin")
        com = _vec(o.get("xyz")) if o is not None else np.zeros(3)
        n = i.find("inertia")
        I = np.array([[n.get("ixx"), n.get("ixy"), n.get("ixz")], [n.get("ixy"), n.get("iyy"), n.get("iyz")],
                      [n.get("ixz"), n.get("iyz"), n.get("izz")]], float)
        return m, com, I


def main(path=None):
    cfg = load_yaml(default_config_path())
    robot = HandRobot(cfg)
    rbd = RigidBodyDynamics(robot, cfg)
    u = Urdf(path or ROOT / "gazebo" / "urdf" / "roboliszt.urdf")
    lo, hi = joint_limits(cfg)
    rng = np.random.default_rng(7)
    err = dict(frame=0.0, frame_rot=0.0, tip=0.0, tip_rot=0.0, mass=0.0, com=0.0, inertia=0.0)
    for _ in range(25):
        q = rng.uniform(lo * 0.8, hi * 0.8)
        car = rng.uniform([0.0, 0.0, 0.03], [0.1, 0.3, 0.1])
        qd = {f"j{i+1}": q[i] for i in range(20)}
        qd.update(carriage_x=car[0], carriage_y=car[1], carriage_z=car[2])
        TU = u.fk(qd)
        T = robot.forward_kinematics(q)
        Tw = np.eye(4); Tw[:3, 3] = car                                   # carriage translation (base orientation = identity)
        mass_u, mcom_u, mass_d, mcom_d = 0.0, np.zeros(3), 0.0, np.zeros(3)
        for i in range(1, 21):
            p = robot.dh[i]
            Fi = F(p.a, p.alpha, p.d)
            Tdh_u = TU[f"link{i}"] @ Fi
            Tdh = Tw @ T[i]
            err["frame"] = max(err["frame"], np.linalg.norm(Tdh_u[:3, 3] - Tdh[:3, 3]))
            err["frame_rot"] = max(err["frame_rot"], np.abs(Tdh_u[:3, :3] - Tdh[:3, :3]).max())
            m, c, I = u.inertial(f"link{i}")
            Rl = TU[f"link{i}"][:3, :3]
            cw_u = TU[f"link{i}"][:3, 3] + Rl @ c
            cw_d = Tdh[:3, 3] + Tdh[:3, :3] @ rbd.com[i]
            err["com"] = max(err["com"], np.linalg.norm(cw_u - cw_d))
            Iu = Rl @ I @ Rl.T
            Id = Tdh[:3, :3] @ rbd.inertia[i] @ Tdh[:3, :3].T
            err["inertia"] = max(err["inertia"], np.abs(Iu - Id).max())
            mass_u += m; mcom_u += m * cw_u
            mass_d += rbd.mass[i]; mcom_d += rbd.mass[i] * cw_d
        err["mass"] = max(err["mass"], abs(mass_u - mass_d))
        err["com"] = max(err["com"], np.linalg.norm(mcom_u / mass_u - mcom_d / mass_d))
        for name, ee in robot.end_effectors(q).items():
            Tt = Tw @ ee
            Ttu = TU[f"tip_{name}"]
            err["tip"] = max(err["tip"], np.linalg.norm(Ttu[:3, 3] - Tt[:3, 3]))
            err["tip_rot"] = max(err["tip_rot"], np.abs(Ttu[:3, :3] - Tt[:3, :3]).max())
    for k, v in err.items():
        print(f"  max {k:10s} error: {v:.3e}")
    ok = err["frame"] < 1e-8 and err["tip"] < 1e-8 and err["frame_rot"] < 1e-8 and err["com"] < 1e-8
    print("URDF consistent with the DH model" if ok else "MISMATCH")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1] if len(sys.argv) > 1 else None) else 1)
