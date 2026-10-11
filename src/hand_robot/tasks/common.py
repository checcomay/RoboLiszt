from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from hand_robot.config.loader import default_config_path, joint_limits, load_yaml
from hand_robot.kinematics.ik import InverseKinematicsSolver, IKResult
from hand_robot.model.robot import HandRobot
from hand_robot.piano.keyboard import Keyboard
from hand_robot.planning import trajectory as tj

@dataclass
class PressTiming:
    lift: float = 0.30
    down: float = 0.30
    hold: float = 0.20
    up: float = 0.30
    ret: float = 0.25
    tail: float = 0.10


FINGERS = {"little": "finger_1", "ring": "finger_2", "middle": "finger_3", "index": "finger_4", "thumb": "finger_5"}
# right-hand fingering: thumb on the lowest note, little finger on the highest
FINGER_ORDER = ["thumb", "index", "middle", "ring", "little"]


@dataclass
class TaskResult:
    """Time-stamped output of a task: what the simulator / Gazebo executes."""
    name: str
    t: np.ndarray                      # (N,)
    q: np.ndarray                      # (N, 20) joint angles [rad]
    base: np.ndarray                   # (N, 3) carriage position in the keyboard frame [m]
    phase: list[str]                   # (N,) phase label
    metrics: dict = field(default_factory=dict)
    extras: dict = field(default_factory=dict)   # task specific arrays (forces, references ...)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, t=self.t, q=self.q, base=self.base, phase=np.array(self.phase),
                 **{f"x_{k}": v for k, v in self.extras.items() if isinstance(v, np.ndarray)})
        return path


class Scene:
    """Robot + keyboard + IK, with the hand base placed in the keyboard frame.

    The robot frame has the same orientation as the keyboard frame (x along the hand, y lateral,
    z up), so  p_world = p_robot + base.
    """

    def __init__(self, config_path: str | Path | None = None):
        self.config = load_yaml(config_path or default_config_path())
        self.robot = HandRobot(self.config)
        self.kb = Keyboard(self.config)
        self.q_min, self.q_max = joint_limits(self.config)
        self.ik = InverseKinematicsSolver(self.robot, self.config, self.q_min, self.q_max)
        self.dt = float(self.config["simulation"]["dt"])
        self.base_home = self.kb.base_home()                 # playing height
        self.base_rest = self.kb.base_home(play=False)      # rest height (home_z)
        self.carriage = self.config.get("carriage", {})

    # ----------------------------------------------------------- kinematics
    def tips_world(self, q, base) -> dict[str, np.ndarray]:
        return {n: T[:3, 3] + np.asarray(base) for n, T in self.robot.end_effectors(q).items()}

    def tip_traj(self, res: TaskResult) -> np.ndarray:
        """(N, 5, 3) fingertip positions in the keyboard frame (order of robot.branches)."""
        out = np.zeros((len(res.t), len(self.robot.branches), 3))
        for k in range(len(res.t)):
            tips = self.robot.end_effectors(res.q[k])
            for i, n in enumerate(self.robot.branches):
                out[k, i] = tips[n][:3, 3] + res.base[k]
        return out

    def solve(self, targets_world: dict[str, np.ndarray], q0, base, **kw) -> IKResult:
        base = np.asarray(base, float)
        if kw.get("floors"):                                  # floors are world heights -> robot frame
            kw["floors"] = {n: z - base[2] for n, z in kw["floors"].items()}
        return self.ik.solve_multi({n: np.asarray(p, float) - base for n, p in targets_world.items()}, q0, **kw)

    # -------------------------------------------------------------- robust IK
    def solve_robust(self, targets_world, q0, base, **kw) -> IKResult:
        """IK with a retry ladder (no null-space, more iterations) before giving up."""
        res = self.solve(targets_world, q0, base, **kw)
        if not res.success:
            res = self.solve(targets_world, q0, base, **{**kw, "k_null": 0.0, "max_iterations": 800})
        return res

    def ik_series(self, targets_t: list[dict], q0, base_t: np.ndarray, **kw) -> np.ndarray:
        """Warm-started IK sample by sample: targets_t[k] = {branch: world xyz}, base_t[k] = carriage."""
        qs, q = np.zeros((len(targets_t), 20)), np.asarray(q0, float).copy()
        for k, tg in enumerate(targets_t):
            res = self.solve_robust(tg, q, base_t[k], **kw)
            if not res.success:
                raise RuntimeError(f"IK failed at sample {k}/{len(targets_t)} "
                                   f"(err {res.error_norm:.2e} m, per-finger {res.errors})")
            q = res.q
            qs[k] = q
        return qs

    def ik_path(self, target_paths: dict[str, np.ndarray], q0, base, **kw) -> np.ndarray:
        """Warm-started IK along dense Cartesian paths {branch: (M,3)} -> (M,20) joint path."""
        m = len(next(iter(target_paths.values())))
        tg = [{n: p[k] for n, p in target_paths.items()} for k in range(m)]
        return self.ik_series(tg, q0, np.tile(np.asarray(base, float), (m, 1)), **kw)

    # ----------------------------------------------------------- press engine
    def press_sequence(self, notes: dict[str, str], pressed: tuple[str, ...], q0, base,
                       timing: "PressTiming | None" = None, delays: dict[str, float] | None = None,
                       wrist_weight: float = 3.0, w_hover: float = 0.25, mode: str = "floor", floor_height: float = 0.008, w_other: float = 1.0):
        """Lift -> descend -> hold -> release -> return for the fingers in ``pressed`` (the others hover).

        Each sample solves the stacked IK of the five fingers: pressed fingers are hard tasks,
        hovering ones soft tasks. ``delays[finger]`` (s) shifts the whole gesture of one finger
        (arpeggiated / rolled chord).  Returns (q (N,20), phase list)."""
        tm = timing or PressTiming()
        delays = delays or {}
        kb = self.kb
        z_h, z_s, z_p = kb.z_surface + kb.hover_height, kb.z_surface + kb.safe_height, kb.z_surface - kb.press_depth
        bounds = np.cumsum([tm.lift, tm.down, tm.hold, tm.up, tm.ret])
        total = bounds[-1] + max(delays.values(), default=0.0) + tm.tail
        t = np.arange(int(round(total / self.dt)) + 1) * self.dt

        def z_of(tt):
            if tt <= 0.0:
                return z_h
            if tt < bounds[0]:
                return z_h + (z_s - z_h) * tj.quintic(tt / tm.lift)
            if tt < bounds[1]:
                return z_s + (z_p - z_s) * tj.quintic((tt - bounds[0]) / tm.down)
            if tt < bounds[2]:
                return z_p
            if tt < bounds[3]:
                return z_p + (z_s - z_p) * tj.quintic((tt - bounds[2]) / tm.up)
            if tt < bounds[4]:
                return z_s + (z_h - z_s) * tj.quintic((tt - bounds[3]) / tm.ret)
            return z_h

        names = ["lift", "descend", "hold", "release", "return"]
        order = list(self.robot.branches)                       # desired tip positions (NaN = not constrained)
        self.last_target_tips = np.full((len(t), len(order), 3), np.nan)
        for k, tk in enumerate(t):
            for f in pressed:
                self.last_target_tips[k, order.index(FINGERS[f])] = [kb.x_for(f), kb.key_y(notes[f]), z_of(tk - delays.get(f, 0.0))]

        def phase_of(tt):
            return next((n for n, bd in zip(names, bounds) if tt < bd), "idle")

        targets = []
        for tk in t:
            tg = {}
            for f, note in notes.items():
                x, y = kb.x_for(f), kb.key_y(note)
                z = z_of(tk - delays.get(f, 0.0)) if f in pressed else z_h
                tg[FINGERS[f]] = np.array([x, y, z])
            targets.append(tg)
        base_t = np.tile(np.asarray(base, float), (len(t), 1))
        if mode in ("finger_only", "wrist_assist"):
            # only the pressed fingers move (plus the wrist in 'wrist_assist'); everything else is frozen at q0
            moving = {j for f in pressed for j in self.robot.branches[FINGERS[f]]}
            if mode == "wrist_assist":
                moving |= {1, 2}
            locked = tuple(j for j in range(1, 21) if j not in moving)
            targets = [{FINGERS[f]: tg[FINGERS[f]] for f in pressed} for tg in targets]
            w = np.ones(20)
            w[:2] = wrist_weight
            qs = self.ik_series(targets, q0, base_t, locked=locked, weights=w, q_rest=np.asarray(q0, float), k_null=0.2)
        elif mode == "floor":
            # pressed fingers: hard tasks. The others are free but must stay above the keys
            # (unilateral constraint tip z >= floor), so they follow the wrist without touching a key.
            w = np.ones(20)
            w[:2] = wrist_weight
            for f in notes:
                if f not in pressed:
                    for j in self.robot.branches[FINGERS[f]]:
                        w[j - 1] = w_other
            floors = {FINGERS[f]: kb.z_surface + floor_height for f in notes if f not in pressed}
            targets = [{FINGERS[f]: tg[FINGERS[f]] for f in pressed} for tg in targets]
            qs = self.ik_series(targets, q0, base_t, weights=w, q_rest=np.asarray(q0, float), k_null=0.2,
                                floors=floors)
        else:
            w = np.ones(20)
            w[:2] = wrist_weight
            tw = self.play_weights(notes, pressed, w_hover)
            qs = self.ik_series(targets, q0, base_t, weights=w, q_rest=np.asarray(q0, float), k_null=0.2,
                                task_weights=tw, min_iterations=25)
        return qs, [phase_of(tk) for tk in t]

    # ------------------------------------------------------------ ready pose
    def hover_targets(self, notes: dict[str, str], pressed: tuple[str, ...] = (), z_hover: float | None = None):
        """World targets: fingers in ``pressed`` on the key bottom, all others hovering above their key."""
        h = self.kb.hover_height if z_hover is None else z_hover
        out = {}
        for f, note in notes.items():
            out[FINGERS[f]] = (self.kb.pressed_point(note, f) if f in pressed else self.kb.hover_point(note, h, f))
        return out

    def play_targets(self, notes: dict[str, str], pressed: tuple[str, ...], z_hover: float | None = None):
        """Targets while playing: pressed fingers on the key bottom, the others hovering above their
        own key (these are *soft* tasks, see ``play_weights``)."""
        return self.hover_targets(notes, pressed=pressed, z_hover=z_hover)

    @staticmethod
    def play_weights(notes: dict[str, str], pressed: tuple[str, ...], w_hover: float = 0.25) -> dict[str, float]:
        return {FINGERS[f]: (1.0 if f in pressed else w_hover) for f in notes}

    def ready_posture(self, notes: dict[str, str] | None = None, base=None, wrist_cost: float = 0.05,
                      q_comfort: np.ndarray | None = None, k_null: float = 0.05,
                      wrist: tuple[float, float] | None = None) -> np.ndarray:
        """Posture with the five fingers hovering over five adjacent white keys (C4..G4 by default)."""
        notes = notes or {"thumb": "C4", "index": "D4", "middle": "E4", "ring": "F4", "little": "G4"}
        base = self.base_home if base is None else base
        tg = self.hover_targets(notes)
        q = np.zeros(20)
        locked = ()
        if wrist is not None:                                # wrist frozen at a chosen posture (rad)
            q[0], q[1] = wrist
            locked = (1, 2)
        w = np.ones(20)
        w[:2] = wrist_cost                                   # cheap wrist: the palm is pitched down over the keys
        for _ in range(3):                                   # a few restarts help the 5-finger problem
            res = self.solve(tg, q, base, weights=w, q_rest=q_comfort, k_null=k_null, max_iterations=600, step=0.6, locked=locked)
            q = res.q
            if res.success:
                return q
        raise RuntimeError(f"ready posture IK failed: {res.errors}")

    # ----------------------------------------------------------- assembling
    def assemble(self, name, parts: list[tuple]) -> TaskResult:
        """parts = [(phase (str or list), q_seg (N,20), base_seg (N,3)), ...]; junction samples merged."""
        q = tj.concat([p[1] for p in parts])
        b = tj.concat([p[2] for p in parts])
        def lab(p):
            return list(p[0]) if not isinstance(p[0], str) else [p[0]] * len(p[1])
        phase = lab(parts[0])
        for p in parts[1:]:
            phase += lab(p)[1:]
        t = np.arange(len(q)) * self.dt
        return TaskResult(name, t, q, b, phase)

    def const_base(self, n: int, base=None) -> np.ndarray:
        return np.tile(self.base_home if base is None else np.asarray(base, float), (n, 1))

    # ---------------------------------------------------------------- metrics
    def common_metrics(self, res: TaskResult, pressed: tuple[str, ...] = (), notes: dict | None = None) -> dict:
        from hand_robot.kinematics import singularity as sg
        qd = tj.derivative(res.q, self.dt)
        tips = self.tip_traj(res)
        names = list(self.robot.branches)
        m = {
            "duration_s": float(res.t[-1]),
            "max_joint_speed_rad_s": float(np.abs(qd).max()),
            "min_joint_margin_deg": float(np.rad2deg(np.minimum(res.q - self.q_min, self.q_max - res.q).min())),
        }
        sm = sg.along_trajectory(self.robot, res.q[::5], [FINGERS[f] for f in pressed] or names)
        m["sigma_min_min"] = float(min(v["sigma_min"].min() for v in sm.values()))
        if pressed and notes:
            hold = [i for i, p in enumerate(res.phase) if p == "hold"]
            if hold:
                k = hold[len(hold) // 2]
                m["press_error_mm"] = float(1e3 * max(
                    np.linalg.norm(tips[k, names.index(FINGERS[f])] - self.kb.pressed_point(notes[f], f)) for f in pressed))
            others = [names.index(FINGERS[f]) for f in notes if f not in pressed]
            if others:
                m["min_hover_clearance_mm"] = float(1e3 * tips[:, others, 2].min())
        return m


def comfort_posture(scene: "Scene", wrist_pitch_deg: float = 10.0, flex_deg: float = -30.0) -> np.ndarray:
    """Relaxed hand: wrist slightly pitched down, fingers gently curved (used as null-space target)."""
    q = np.zeros(20)
    q[1] = np.deg2rad(wrist_pitch_deg)
    for name, br in scene.robot.branches.items():
        if name != "finger_5":
            for j in br[1:]:
                q[j - 1] = np.deg2rad(flex_deg)
    return q
