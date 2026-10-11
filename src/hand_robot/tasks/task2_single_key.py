"""Task 2 - pressione di un singolo tasto (cinematica inversa base)."""
from __future__ import annotations

from .common import Scene, TaskResult, PressTiming

DEFAULT_NOTES = {"thumb": "C4", "index": "D4", "middle": "E4", "ring": "F4", "little": "G4"}


def run(scene: Scene, finger: str = "middle", notes: dict | None = None, timing: PressTiming | None = None,
        mode: str | None = None, wrist_weight: float = 10.0) -> list[TaskResult]:
    notes = notes or DEFAULT_NOTES
    q0 = scene.ready_posture(notes)
    base = scene.base_home
    mode = mode or "floor"
    qs, phase = scene.press_sequence(notes, (finger,), q0, base, timing, mode=mode, wrist_weight=wrist_weight)
    res = scene.assemble(f"task2_press_{finger}_{notes[finger]}", [(phase, qs, scene.const_base(len(qs), base))])
    res.metrics = scene.common_metrics(res, (finger,), notes)
    res.metrics["wrist_deviation_deg"] = float(abs(qs[:, :2] - q0[:2]).max() * 180 / 3.141592653589793)
    res.extras.update(notes=notes, pressed=(finger,), q_ready=q0, target_tips=scene.last_target_tips)
    return [res]
