"""Task 3 - esecuzione di un accordo (coordinazione multi-catena, IK impilata con polso condiviso)."""
from __future__ import annotations

from .common import Scene, TaskResult, PressTiming
from .task2_single_key import DEFAULT_NOTES

TRIAD = ("thumb", "middle", "little")          # fingering 1-3-5: C4 - E4 - G4


def run(scene: Scene, pressed: tuple[str, ...] = TRIAD, notes: dict | None = None,
        stagger: float = 0.0, timing: PressTiming | None = None, mode: str = "floor", wrist_weight: float = 3.0) -> list[TaskResult]:
    """``stagger`` > 0 gives a rolled chord: each pressed finger starts ``stagger`` s after the previous."""
    notes = notes or DEFAULT_NOTES
    q0 = scene.ready_posture(notes)
    base = scene.base_home
    delays = {f: i * stagger for i, f in enumerate(pressed)}
    qs, phase = scene.press_sequence(notes, pressed, q0, base, timing, delays, mode=mode, wrist_weight=wrist_weight)
    tag = "".join(notes[f] for f in pressed)
    res = scene.assemble(f"task3_chord_{tag}" + ("_roll" if stagger else "") + ("" if mode == "floor" else f"_{mode}"), [(phase, qs, scene.const_base(len(qs), base))])
    res.metrics = scene.common_metrics(res, pressed, notes)
    res.metrics["wrist_deviation_deg"] = float(abs(qs[:, :2] - q0[:2]).max() * 57.29577951308232)
    res.extras.update(notes=notes, pressed=pressed, q_ready=q0, target_tips=scene.last_target_tips)
    return [res]
