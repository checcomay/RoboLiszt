"""Run the five RoboLiszt tasks: trajectories (npz) + figures + metrics (json).

    python scripts/run_tasks.py             # all tasks
    python scripts/run_tasks.py --task 3    # only task 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from hand_robot.tasks.common import Scene
from hand_robot.tasks import task1_wave, task2_single_key, task3_chord, task4_octave_jump, task5_touch
from hand_robot.visualization.figures import FigureManager
from hand_robot.visualization import task_plots as tp


def _clean(d):
    return {k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer,)) else v)
            for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, choices=range(1, 6))
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()

    scene = Scene()
    fm = FigureManager(ROOT / "docs" / "images")
    out_dir = ROOT / "data" / "trajectories"
    report = {}

    def save(res):
        res.save(out_dir / f"{res.name}.npz")
        report[res.name] = _clean(res.metrics)
        print(f"[{res.name}]")
        for k, v in res.metrics.items():
            print(f"    {k:34s} {v:.4g}" if isinstance(v, (float, int, np.floating)) and not isinstance(v, bool) else f"    {k:34s} {v}")

    todo = [a.task] if a.task else [1, 2, 3, 4, 5]
    if 1 in todo:
        r = task1_wave.run(scene)[0]; save(r)
        if not a.no_plots: tp.plot_wave(scene, r, fm)
    if 2 in todo:
        for f in ("thumb", "index", "middle", "ring", "little"):
            r = task2_single_key.run(scene, finger=f)[0]; save(r)
            if not a.no_plots and f == "middle":
                tp.plot_tips_height(scene, r, fm)
                k = [i for i, p in enumerate(r.phase) if p == "hold"][0]
                tp.plot_hand_on_keyboard(scene, r.q[k], r.base[k], fm, "task2_pose_middle", r.extras["notes"], ("middle",))
    if 3 in todo:
        for kw in ({}, {"stagger": 0.04}, {"mode": "wrist_assist"}, {"pressed": ("thumb", "index", "middle", "ring", "little")}):
            r = task3_chord.run(scene, **kw)[0]; save(r)
            if not a.no_plots and not kw:
                tp.plot_tips_height(scene, r, fm)
                k = [i for i, p in enumerate(r.phase) if p == "hold"][0]
                tp.plot_hand_on_keyboard(scene, r.q[k], r.base[k], fm, "task3_pose_triad", r.extras["notes"], r.extras["pressed"])
    if 4 in todo:
        r = task4_octave_jump.run(scene)[0]; save(r)
        if not a.no_plots: tp.plot_octave(scene, r, fm)
    if 5 in todo:
        rs = task5_touch.run(scene)
        for r in rs: save(r)
        if not a.no_plots: tp.plot_touch(scene, rs, fm)

    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data" / "metrics.json").write_text(json.dumps(report, indent=2))
    print(f"\ntrajectories -> {out_dir}\nfigures      -> {ROOT / 'docs' / 'images' / 'tasks'}")


if __name__ == "__main__":
    main()
