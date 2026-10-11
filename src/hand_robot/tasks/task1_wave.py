"""Task 1 - movimento in spazio libero (gesto di saluto): nessun contatto, controllo nello spazio dei giunti."""
from __future__ import annotations

import numpy as np

from hand_robot.kinematics import singularity as sg
from hand_robot.planning import trajectory as tj
from .common import Scene, TaskResult, FINGERS
from .task2_single_key import DEFAULT_NOTES


def open_palm(scene: Scene, wrist_pitch_deg: float = -55.0) -> np.ndarray:
    """Open hand, palm raised: flexion joints at theta = 0 (straight finger), thumb neutral."""
    q = np.zeros(20)
    q[1] = np.deg2rad(wrist_pitch_deg)
    for name, br in scene.robot.branches.items():
        if name == "finger_5":
            continue
        for j in br[1:]:
            q[j - 1] = -scene.robot.dh[j].theta_offset
    return np.clip(q, scene.q_min, scene.q_max)


def run(scene: Scene, wave_amp_deg: float = 25.0, wave_freq_hz: float = 1.5, n_cycles: int = 3,
        ripple_deg: float = 18.0, t_raise: float = 1.0, t_lower: float = 1.0) -> list[TaskResult]:
    dt = scene.dt
    q_ready = scene.ready_posture(DEFAULT_NOTES)
    q_wave = open_palm(scene)
    base = scene.base_home

    t_wave = n_cycles / wave_freq_hz
    ramp = 0.4
    t = tj.time_grid(t_wave, dt)
    env = tj.quintic(t / ramp) * tj.quintic((t_wave - t) / ramp)           # smooth start / stop of the oscillation
    qw = np.tile(q_wave, (len(t), 1))
    qw[:, 0] += np.deg2rad(wave_amp_deg) * env * np.sin(2 * np.pi * wave_freq_hz * t)      # radial-ulnar deviation
    # travelling "ripple" of the fingers, little -> index
    for k, name in enumerate(["finger_1", "finger_2", "finger_3", "finger_4"]):
        phase = -0.7 * k
        flex = 0.5 - 0.5 * np.cos(2 * np.pi * wave_freq_hz * 2 * t + phase)
        for j in scene.robot.branches[name][1:]:
            qw[:, j - 1] += -np.deg2rad(ripple_deg) * env * flex
    qw = np.clip(qw, scene.q_min, scene.q_max)

    up = tj.segment(q_ready, q_wave, t_raise, dt)
    down = tj.segment(qw[-1], q_ready, t_lower, dt)
    parts = [("raise", up, scene.const_base(len(up))), ("wave", qw, scene.const_base(len(qw))),
             ("lower", down, scene.const_base(len(down)))]
    res = scene.assemble("task1_wave", parts)

    tips = scene.tip_traj(res)
    names = list(scene.robot.branches)
    sm = sg.along_trajectory(scene.robot, res.q[::5], [FINGERS["middle"], FINGERS["thumb"]])
    qd = tj.derivative(res.q, dt)
    res.metrics = {
        "duration_s": float(res.t[-1]),
        "max_joint_speed_rad_s": float(np.abs(qd).max()),
        "min_joint_margin_deg": float(np.rad2deg(np.minimum(res.q - scene.q_min, scene.q_max - res.q).min())),
        "max_tip_height_mm": float(1e3 * tips[:, :, 2].max()),
        "middle_tip_lateral_sweep_mm": float(1e3 * np.ptp(tips[[i for i, p in enumerate(res.phase) if p == "wave"]][:, names.index("finger_3"), 1])),
        "sigma_min_min": float(min(v["sigma_min"].min() for v in sm.values())),
    }
    res.extras["sigma_min_middle"] = sm[FINGERS["middle"]]["sigma_min"]
    return [res]
