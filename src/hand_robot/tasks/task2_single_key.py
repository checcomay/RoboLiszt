"""Task 2 - pressione di un singolo tasto (cinematica inversa base)."""
from __future__ import annotations

import numpy as np

from .common import Scene, TaskResult, PressTiming

DEFAULT_NOTES = {"thumb": "C4", "index": "D4", "middle": "E4", "ring": "F4", "little": "G4"}


def run(scene: Scene, finger: str = "middle", notes: dict | None = None, timing: PressTiming | None = None,
        mode: str | None = None, wrist_weight: float = 10.0, t_return: float = 0.50) -> list[TaskResult]:
    notes = notes or DEFAULT_NOTES
    if finger not in notes:
        raise ValueError(f"finger '{finger}' is not in notes {list(notes)}")
    q0 = scene.ready_posture(notes)
    base = scene.base_home
    mode = mode or "floor"
    qs, phase = scene.press_sequence(notes, (finger,), q0, base, timing, mode=mode, wrist_weight=wrist_weight)
    n_press = len(qs)
    target = scene.last_target_tips                                   # (n_press, 5, 3), NaN = finger not constrained
    pre_return_gap = float(np.rad2deg(np.abs(qs[-1] - q0).max()))     # how far the posture drifted from q_ready

    res = scene.assemble(f"task2_press_{finger}_{notes[finger]}", [(phase, qs, scene.const_base(n_press, base))])
    res = scene.return_to_ready(res, q_ready=q0, duration=t_return)

    # metrics AFTER the return move: duration, joint speed, margins and hover clearance now cover the whole trajectory
    res.metrics = scene.common_metrics(res, (finger,), notes)
    res.metrics["wrist_deviation_deg"] = float(np.rad2deg(np.abs(res.q[:, :2] - q0[:2]).max()))
    res.metrics["pre_return_gap_deg"] = pre_return_gap
    pad = np.full((len(res.t) - n_press,) + target.shape[1:], np.nan)  # the return tail has no Cartesian target
    res.extras.update(notes=notes, pressed=(finger,), q_ready=q0, target_tips=np.concatenate([target, pad]))
    return [res]