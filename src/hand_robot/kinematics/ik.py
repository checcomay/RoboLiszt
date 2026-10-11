from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .jacobian import position_jacobian
from hand_robot.model.robot import HandRobot


@dataclass
class IKResult:
    q: np.ndarray
    success: bool
    iterations: int
    error_norm: float
    history: np.ndarray
    errors: dict = field(default_factory=dict)      # per-branch final position error [m]


class InverseKinematicsSolver:
    """Damped-least-squares position IK for the branched 20-DOF hand.

    * single finger or any set of fingers solved together (stacked Jacobian, shared wrist);
    * weighted joints (``weights``: large weight = joint moves less);
    * ``locked`` joints (columns removed from the Jacobian);
    * null-space posture optimisation towards ``q_rest``;
    * joint limits enforced by projection at every iteration;
    * target components set to NaN are left free (e.g. a hovering finger constrained only in height);
    * the TCP is the fingertip pad (same point as ``HandRobot.end_effectors``).
    """

    def __init__(self, robot: HandRobot, config: dict, q_min=None, q_max=None):
        self.robot = robot
        c = config["control"]
        self.step = float(c["ik_step"])
        self.tolerance = float(c["ik_tolerance"])
        self.max_iterations = int(c["ik_max_iterations"])
        self.damping = float(c["ik_damping"])
        lim = config.get("limits", {})
        self.q_min = np.full(20, np.deg2rad(lim.get("default_min_deg", -90.0))) if q_min is None else np.asarray(q_min, float)
        self.q_max = np.full(20, np.deg2rad(lim.get("default_max_deg", 90.0))) if q_max is None else np.asarray(q_max, float)

    # ------------------------------------------------------------------ API
    def solve(self, branch: str, target_position: np.ndarray, q0: np.ndarray | None = None, **kw) -> IKResult:
        """Single-finger IK (kept for backward compatibility)."""
        return self.solve_multi({branch: target_position}, q0, **kw)

    def solve_multi(
        self,
        targets: dict[str, np.ndarray],
        q0: np.ndarray | None = None,
        *,
        weights: np.ndarray | None = None,
        locked: tuple[int, ...] = (),
        q_rest: np.ndarray | None = None,
        k_null: float = 0.3,
        step: float | None = None,
        max_iterations: int | None = None,
        tolerance: float | None = None,
        task_weights: dict[str, float] | None = None,
        min_iterations: int = 1,
        floors: dict[str, float] | None = None,
    ) -> IKResult:
        robot = self.robot
        q = np.zeros(20) if q0 is None else robot.validate_q(q0).copy()
        q = np.clip(q, self.q_min, self.q_max)
        step = self.step if step is None else step
        max_it = self.max_iterations if max_iterations is None else max_iterations
        tol = self.tolerance if tolerance is None else tolerance
        q_rest = q.copy() if q_rest is None else robot.validate_q(q_rest)
        w = np.ones(20) if weights is None else np.asarray(weights, float).reshape(20)
        winv = 1.0 / np.maximum(w, 1e-9)
        free = np.ones(20, dtype=bool)
        for j in locked:
            free[j - 1] = False
        winv = winv * free                                   # locked joints never move

        names = list(targets)
        tgt = {n: np.asarray(targets[n], float).reshape(3) for n in names}
        tw0 = {n: float((task_weights or {}).get(n, 1.0)) for n in names}   # <1: soft (compromise) task
        mask0 = {n: ~np.isnan(tgt[n]) for n in names}       # NaN component = axis left free
        tgt0 = {n: np.where(mask0[n], tgt[n], 0.0) for n in names}
        floors = floors or {}                                # unilateral constraints: tip z >= floor
        history = []
        range_ = np.maximum(self.q_max - self.q_min, 1e-6)

        for it in range(1, max_it + 1):
            T = robot.forward_kinematics(q)
            tips = {n: (T[robot.branches[n][-1]] @ robot._T_tip)[:3, 3] for n in robot.branches}
            names, tgt, mask, tw = list(tgt0), dict(tgt0), dict(mask0), dict(tw0)
            for n, zmin in floors.items():                   # active only when the floor is violated
                if n not in tgt0 and tips[n][2] < zmin:
                    names.append(n)
                    tgt[n] = np.array([0.0, 0.0, zmin])
                    mask[n] = np.array([False, False, True])
                    tw[n] = 1.0
            e = np.concatenate([tw[n] * (tgt[n] - tips[n])[mask[n]] for n in names])
            enorm = float(np.linalg.norm(e))
            history.append(enorm)
            per = {n: float(np.linalg.norm((tgt[n] - tips[n])[mask[n]])) for n in names}
            required = [n for n in names if tw[n] >= 1.0] or names      # soft tasks never block convergence
            required = [n for n in required]
            if it >= min_iterations and all(per[n] < tol for n in required):
                return IKResult(q, True, it, enorm, np.asarray(history), per)

            J = np.vstack([tw[n] * position_jacobian(robot, q, n, T)[mask[n]] for n in names])
            # adaptive damping: strong far from the target, vanishing close to it
            lam2 = 1e-6 + self.damping ** 2 * min(1.0, (enorm / 0.02) ** 2)
            # active-set handling of the joint limits: joints that would leave their range are frozen
            # and the step is recomputed with the remaining ones
            frozen = np.zeros(20, dtype=bool)
            for _ in range(6):
                wi = winv * ~frozen
                JW = J * wi
                A = JW @ J.T + lam2 * np.eye(J.shape[0])
                Jd = (wi[:, None] * J.T) @ np.linalg.inv(A)          # weighted damped pseudo-inverse
                dq = Jd @ e
                # null-space: pull towards the rest posture (normalised by joint range)
                dq0 = -k_null * (q - q_rest) * (range_.mean() / range_) ** 2
                dq += (np.eye(20) - Jd @ J) @ (dq0 * free * ~frozen)
                q_new = q + step * dq
                viol = ((q_new > self.q_max) & (dq > 0)) | ((q_new < self.q_min) & (dq < 0))
                viol &= ~frozen
                if not viol.any():
                    break
                frozen |= viol
            q = np.clip(q + step * dq, self.q_min, self.q_max)

        T = robot.forward_kinematics(q)
        names, tgt, mask, tw = list(tgt0), dict(tgt0), dict(mask0), dict(tw0)
        tips = {n: (T[robot.branches[n][-1]] @ robot._T_tip)[:3, 3] for n in names}
        per = {n: float(np.linalg.norm((tgt[n] - tips[n])[mask[n]])) for n in names}
        enorm = float(np.linalg.norm(np.concatenate([tw[n] * (tgt[n] - tips[n])[mask[n]] for n in names])))
        history.append(enorm)
        return IKResult(q, False, max_it, enorm, np.asarray(history), per)
