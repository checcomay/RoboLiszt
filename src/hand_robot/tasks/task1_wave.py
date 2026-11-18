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

    q_raw = q.copy()
    q = np.clip(q, scene.q_min, scene.q_max)
    if not np.isclose(q[1], q_raw[1]):
        print(f"[task1] wrist pitch clipped: requested {wrist_pitch_deg:.1f} deg, "
              f"got {np.rad2deg(q[1]):.1f} deg (limits {np.rad2deg(scene.q_min[1]):.1f}/{np.rad2deg(scene.q_max[1]):.1f})")
    return q


def run(scene, wave_amp_deg=35.0, wave_freq_hz=1.0, n_cycles=3,
        ripple_deg=0.0, wrist_pitch_deg=-65.0, t_raise=1.0, t_lower=1.0):
    dt = scene.dt
    q_ready = scene.ready_posture(DEFAULT_NOTES)
    q_wave = open_palm(scene, wrist_pitch_deg)
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

    wave_idx = [i for i, p in enumerate(res.phase) if p == "wave"]
    res.metrics.update({
        "wrist_pitch_min_deg": float(np.rad2deg(res.q[:, 1].min())),
        "wave_dev_ptp_deg": float(np.rad2deg(np.ptp(res.q[wave_idx, 0]))),
        "wave_dev_clipped": bool(np.rad2deg(np.ptp(res.q[wave_idx, 0])) < 2 * wave_amp_deg - 1e-6),
        "return_error_deg": float(np.rad2deg(np.abs(res.q[-1] - res.q[0]).max())),
        "return_base_error_mm": float(1e3 * np.abs(res.base[-1] - res.base[0]).max()),
    })
    return [res]
