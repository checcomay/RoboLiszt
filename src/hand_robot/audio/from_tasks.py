"""Sound of the Python task trajectories (no Gazebo): fingertip height over each key -> key depression -> detector -> events.

Point-contact model of the task planner: a fingertip (TCP) at x within a key, z < 0 presses the white key under it by -z
(clipped to the 8 mm travel). Task 5 files (task5rb_*.npz) carry the tip height in ``x_z`` and play E4 only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from hand_robot.audio.detector import KeyEvent, KeyPressDetector, pair_events
from hand_robot.audio.synth import PianoSynth, render_events, write_wav
from hand_robot.piano import layout as lay


def key_depressions_from_task(scene, d) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """-> (t, {midi: depression [m] per sample}) for a task npz (arrays t, q, base)."""
    kb = scene.kb
    t = np.asarray(d["t"], float)
    dep: dict[int, np.ndarray] = {}
    if "x_z" in d.files:                                              # task 5: single E4 key, TCP height given
        z = np.asarray(d["x_z"], float)
        t = t[-len(z):]
        dep[lay.midi("E4")] = np.clip(kb.z_surface - z, 0.0, kb.press_depth)
        return t, dep
    q, base = np.asarray(d["q"]), np.asarray(d["base"])
    whites = [m for m in lay.keys() if not lay.is_black(m)]
    ky = np.array([lay.key_y(m, kb.width) for m in whites])
    for n in range(len(t)):
        tips = scene.robot.end_effectors(q[n])
        for b, T in tips.items():
            p = T[:3, 3] + base[n]
            if p[2] >= kb.z_surface or not (kb.front_x <= p[0] <= kb.front_x + kb.depth):
                continue
            k = int(np.argmin(np.abs(ky - p[1])))
            if abs(ky[k] - p[1]) <= kb.width / 2:
                arr = dep.setdefault(whites[k], np.zeros(len(t)))
                arr[n] = max(arr[n], min(kb.z_surface - p[2], kb.press_depth))
    return t, dep


def events_from_depressions(t, dep, detector: KeyPressDetector) -> list[KeyEvent]:
    ev: list[KeyEvent] = []
    for n in range(len(t)):
        for m, arr in dep.items():
            ev += detector.update(float(t[n]), m, -float(arr[n]))
    return sorted(ev, key=lambda e: e.t)


def render_task_wav(scene, npz_path, wav_path=None, tail: float = 1.5):
    d = np.load(npz_path, allow_pickle=True)
    t, dep = key_depressions_from_task(scene, d)
    cfg = scene.config
    ev = events_from_depressions(t, dep, KeyPressDetector.from_config(cfg))
    pairs = pair_events(ev)
    s = cfg["sound"]
    synth = PianoSynth(int(s["sample_rate"]), float(s["a4"]))
    t0 = float(t[0])
    x = render_events([(a - t0, m, v, None if b is None else b - t0) for a, m, v, b in pairs], synth,
                      float(s["release_time"]), tail, float(s["master_gain"])) if pairs else np.zeros(1, np.float32)
    if wav_path is not None:
        Path(wav_path).parent.mkdir(parents=True, exist_ok=True)
        write_wav(wav_path, x, synth.sr)
    return pairs, x
