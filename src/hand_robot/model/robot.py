from __future__ import annotations

from typing import Dict, Tuple
import numpy as np

from .dh import DHParameter, dh_transform


class HandRobot:
    """20-DOF branched hand model (standard DH: Rz(theta) Tz(d) Tx(a) Rx(alpha)).

    The mechanical tree is:
        0 -> 1 -> 2
                 ├-> 3 -> 4 -> 5 -> 6
                 ├-> 7 -> 8 -> 9
                 ├-> 10 -> 11 -> 12
                 ├-> 13 -> 14 -> 15
                 └-> 16 -> 17 -> 18 -> 19 -> 20
    """

    N_JOINTS = 20
    L_TIP = 0.020

    def __init__(self, config: dict):
        self.config = config
        self.dh = self._load_dh(config["dh"])
        self.parent = {int(k): int(v) for k, v in config["topology"]["parent"].items()}
        self.branches = {
            name: [int(j) for j in joints]
            for name, joints in config["topology"]["branches"].items()
        }

        if len(self.dh) != self.N_JOINTS:
            raise ValueError("The robot must contain exactly 20 DH joints.")
        self._validate_topology()

        # Fixed transform from the last joint frame to the fingertip pad.
        self._T_tip = np.eye(4)
        self._T_tip[0, 3] = self.L_TIP

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _load_dh(rows: dict) -> Dict[int, DHParameter]:
        dh = {}
        for key, row in rows.items():
            joint = int(key)
            dh[joint] = DHParameter(
                a=float(row["a"]),
                alpha=np.deg2rad(float(row["alpha"])),
                d=float(row["d"]),
                theta_offset=np.deg2rad(float(row["theta_offset"])),
            )
        return dh

    def _validate_topology(self) -> None:
        """Fail early if parent/branches are inconsistent (typos in the YAML)."""
        seen = set()
        for name, branch in self.branches.items():
            if not branch:
                raise ValueError(f"Branch '{name}' is empty.")
            if self.parent.get(branch[0]) != 2:
                raise ValueError(f"Branch '{name}' must start from frame 2.")
            for prev, joint in zip(branch[:-1], branch[1:]):
                if self.parent.get(joint) != prev:
                    raise ValueError(f"Joint {joint} of '{name}' must have parent {prev}.")
            seen.update(branch)
        expected = set(range(3, self.N_JOINTS + 1))
        if seen != expected:
            raise ValueError(f"Branches must cover joints 3..20 exactly once (got {sorted(seen)}).")

    @staticmethod
    def validate_q(q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.size != 20:
            raise ValueError(f"Expected 20 joint coordinates, got {q.size}.")
        return q

    # -------------------------------------------------------------- kinematics
    def _local(self, joint: int, q: np.ndarray) -> np.ndarray:
        p = self.dh[joint]
        return dh_transform(p.a, p.alpha, p.d, q[joint - 1] + p.theta_offset)

    def local_transform(self, joint: int, q: np.ndarray) -> np.ndarray:
        """:return: ^{i-1}T_i."""
        return self._local(joint, self.validate_q(q))

    def forward_kinematics(self, q: np.ndarray) -> Dict[int, np.ndarray]:
        """:return: ^0T_i for all frames i=0,...,20."""
        q = self.validate_q(q)  # validated once, not 20 times
        T = {0: np.eye(4)}
        T[1] = T[0] @ self._local(1, q)
        T[2] = T[1] @ self._local(2, q)

        for branch in self.branches.values():
            for joint in branch:
                T[joint] = T[self.parent[joint]] @ self._local(joint, q)
        return T

    def end_effectors(self, q: np.ndarray) -> dict[str, np.ndarray]:
        """:return: the fingertip (TCP) pose of every finger."""
        T = self.forward_kinematics(q)
        return {name: T[branch[-1]] @ self._T_tip for name, branch in self.branches.items()}

    def positions(self, q: np.ndarray) -> dict[int | str, np.ndarray]:
        """:return: 3-D positions of all joints plus fingertips ('tip_<branch>')."""
        T = self.forward_kinematics(q)
        pos: dict[int | str, np.ndarray] = {i: T[i][:3, 3].copy() for i in T}
        for name, T_tip in self.end_effectors(q).items():
            pos[f"tip_{name}"] = T_tip[:3, 3].copy()
        return pos


    # ---------------------------------------------------------------- plotting
    def chains(self, q: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[int, np.ndarray]]:
        """
        Geometry helper for plotting / mesh export.
        :return:
            base_pts : (3, 3) array with the points of 0 -> 1 -> 2
            finger_pts : {branch_name: (n+2, 3)} polyline  2 -> joints -> fingertip
            T : all frames ^0T_i
        """
        T = self.forward_kinematics(q)
        tips = self.end_effectors(q)
        base_pts = np.array([T[i][:3, 3] for i in (0, 1, 2)])
        finger_pts = {}
        for name, branch in self.branches.items():
            pts = [T[2][:3, 3]] + [T[j][:3, 3] for j in branch] + [tips[name][:3, 3]]
            finger_pts[name] = np.array(pts)
        return base_pts, finger_pts, T
