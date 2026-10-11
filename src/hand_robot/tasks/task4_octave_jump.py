"""Task 4 - locomozione del polso e salto d'ottava.

The wrist only has 2 rotational DoF, so a lateral displacement of a whole octave is delivered by the
virtual carriage (3 prismatic joints x, y, z carrying the hand base). The module also quantifies how
far the hand can reach with wrist rotation alone, which motivates the carriage.
"""
from __future__ import annotations

import numpy as np

from hand_robot.planning import trajectory as tj
from .common import Scene, TaskResult, FINGERS
from .task2_single_key import DEFAULT_NOTES
from .task3_chord import TRIAD


def jump_time(distance: float, v_max: float, a_max: float, t_min: float = 0.35) -> float:
    """Shortest min-jerk duration respecting speed and acceleration limits (peaks: 1.875 d/T, 5.77 d/T^2)."""
    return max(t_min, 1.875 * distance / v_max, float(np.sqrt(5.7735 * distance / a_max)))


def carriage_jump(scene: Scene, b0: np.ndarray, d_y: float, lift: float, duration: float) -> np.ndarray:
    t = tj.time_grid(duration, scene.dt)
    tau = t / duration
    path = np.tile(np.asarray(b0, float), (len(t), 1))
    path[:, 1] += d_y * tj.quintic(tau)
    path[:, 2] += lift * np.sin(np.pi * tau) ** 2             # lift the hand over the keys while travelling
    return path


def wrist_only_reach(scene: Scene, q_ready, notes: dict, pressed=TRIAD, max_steps: int = 14) -> dict:
    """Triad pressed at +k white keys with the base FIXED (wrist rotations only): position error vs k."""
    kb, base = scene.kb, scene.base_home
    steps, err, wrist, q = [], [], [], np.asarray(q_ready, float).copy()
    for k in range(max_steps + 1):
        tg = {FINGERS[f]: kb.pressed_point(kb.transpose(notes[f], k), f) for f in pressed}
        r = scene.solve_robust(tg, q, base, q_rest=q_ready, k_null=0.05, max_iterations=600)
        q = r.q
        steps.append(k)
        err.append(max(r.errors.values()))
        wrist.append(np.rad2deg(r.q[:2]))
    return {"steps": np.array(steps), "error_mm": 1e3 * np.array(err), "wrist_deg": np.array(wrist)}


def run(scene: Scene, steps: int = 7, pressed: tuple[str, ...] = TRIAD) -> list[TaskResult]:
    notes0 = DEFAULT_NOTES
    notes1 = {f: scene.kb.transpose(n, steps) for f, n in notes0.items()}
    q_ready = scene.ready_posture(notes0)
    b0 = scene.base_home
    d_y = steps * scene.kb.width
    c = scene.carriage
    lift = float(c.get("lift_height", 0.035))
    T = jump_time(abs(d_y), float(c.get("max_speed", 1.0)), float(c.get("max_accel", 12.0)))

    qs1, ph1 = scene.press_sequence(notes0, pressed, q_ready, b0)
    path = carriage_jump(scene, b0, d_y, lift, T)
    b1 = path[-1].copy()
    q_fly = np.tile(qs1[-1], (len(path), 1))                  # rigid hand shape during the flight
    qs2, ph2 = scene.press_sequence(notes1, pressed, qs1[-1], b1)

    parts = [(ph1, qs1, scene.const_base(len(qs1), b0)),
             ("jump", q_fly, path),
             (ph2, qs2, scene.const_base(len(qs2), b1))]
    res = scene.assemble(f"task4_octave_jump_{steps}", parts)

    tips = scene.tip_traj(res)
    names = list(scene.robot.branches)
    fly = [i for i, p in enumerate(res.phase) if p == "jump"]
    v = np.linalg.norm(tj.derivative(path, scene.dt), axis=1)
    a = np.linalg.norm(tj.derivative(tj.derivative(path, scene.dt), scene.dt), axis=1)
    land = max(np.linalg.norm(tips[fly[-1], names.index(FINGERS[f])] - scene.kb.hover_point(notes1[f], scene.kb.hover_height, f))
               for f in notes0)
    reach = wrist_only_reach(scene, q_ready, notes0, pressed)
    ok = reach["error_mm"] < 0.1
    res.metrics = {
        **scene.common_metrics(res),
        "jump_distance_mm": 1e3 * d_y,
        "jump_time_s": T,
        "carriage_peak_speed_m_s": float(v.max()),
        "carriage_peak_accel_m_s2": float(a.max()),
        "landing_error_mm": 1e3 * float(land),
        "min_tip_height_in_flight_mm": float(1e3 * tips[fly][:, :, 2].min()),
        "wrist_only_max_reachable_steps": int(reach["steps"][ok].max() if ok.any() and ok[0] else 0),
    }
    res.extras.update(reach_steps=reach["steps"], reach_error_mm=reach["error_mm"], reach_wrist_deg=reach["wrist_deg"],
                      notes0=notes0, notes1=notes1, pressed=pressed, q_ready=q_ready)
    return [res]
