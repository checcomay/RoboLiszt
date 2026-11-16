"""Render the sound of the task trajectories to WAV (no Gazebo, no audio device needed).

    python scripts/render_task_audio.py                 -> data/audio/*.wav + data/audio/events.json
    python scripts/render_task_audio.py data/trajectories/task3_chord_C4E4G4.npz
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hand_robot.audio.from_tasks import render_task_wav
from hand_robot.piano.layout import note_name
from hand_robot.tasks.common import Scene

DEFAULT = ["task2_press_middle_E4", "task3_chord_C4E4G4", "task3_chord_C4E4G4_roll", "task4_octave_jump_7",
           "task5rb_pp", "task5rb_mf", "task5rb_ff_tau0.25"]


def main():
    sc = Scene()
    files = [Path(a) for a in sys.argv[1:]] or [ROOT / "data" / "trajectories" / f"{n}.npz" for n in DEFAULT]
    out = ROOT / "data" / "audio"
    summary = {}
    for f in files:
        pairs, x = render_task_wav(sc, f, out / f"{f.stem}.wav")
        summary[f.stem] = [{"t_on": round(a, 3), "note": note_name(m), "midi": m, "velocity": v,
                            "t_off": None if b is None else round(b, 3)} for a, m, v, b in pairs]
        print(f"{f.stem:28s} {len(pairs)} notes: " + ", ".join(f"{note_name(m)}@{a:.2f}s v{v}" for a, m, v, b in pairs))
    json.dump(summary, open(out / "events.json", "w"), indent=1)


if __name__ == "__main__":
    main()
