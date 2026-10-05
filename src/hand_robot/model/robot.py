from __future__ import annotations

from typing import Dict
import numpy as np

from .dh import DHParameter, dh_transform


class HandRobot:
    """20-DOF branched hand model.

    The mechanical tree is:
        0 -> 1 -> 2
                 ├-> 3 -> 4 -> 5 -> 6
                 ├-> 7 -> 8 -> 9
                 ├-> 10 -> 11 -> 12
                 ├-> 13 -> 14 -> 15
                 └-> 16 -> 17 -> 18 -> 19 -> 20
    """

    N_JOINTS = 20

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

    @staticmethod
    def _load_dh(rows: dict) -> Dict[int, DHParameter]:
        dh = {}
        for key, row in rows.items():
            joint = int(key)
            dh[joint] = DHParameter(
                a=float(row["a"]),
                alpha=np.deg2rad(float(row["alpha"])),
                d=float(row["d"]),
                theta_offset=np.deg2rad(float(row["theta"])),
            )
        return dh

    @staticmethod
    def validate_q(q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.size != 20:
            raise ValueError(f"Expected 20 joint coordinates, got {q.size}.")
        return q

    def local_transform(self, joint: int, q: np.ndarray) -> np.ndarray:
        """Return ^{i-1}T_i."""
        q = self.validate_q(q)
        p = self.dh[joint]
        theta = q[joint - 1] + p.theta_offset
        return dh_transform(p.a, p.alpha, p.d, theta)

    def forward_kinematics(self, q: np.ndarray) -> Dict[int, np.ndarray]:
        """Return ^0T_i for all frames i=0,...,20."""
        q = self.validate_q(q)
        T = {0: np.eye(4)}

        T[1] = T[0] @ self.local_transform(1, q)
        T[2] = T[1] @ self.local_transform(2, q)

        for branch in self.branches.values():
            for joint in branch:
                T[joint] = T[self.parent[joint]] @ self.local_transform(joint, q)

        return T

    def end_effectors(self, q: np.ndarray) -> dict[str, np.ndarray]:
        T = self.forward_kinematics(q)
        return {name: T[branch[-1]] for name, branch in self.branches.items()}

    def positions(self, q: np.ndarray) -> dict[int, np.ndarray]:
        T = self.forward_kinematics(q)
        return {i: T[i][:3, 3].copy() for i in T}
