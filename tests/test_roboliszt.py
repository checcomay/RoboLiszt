import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hand_robot.kinematics import singularity as sg
from hand_robot.kinematics.jacobian import position_jacobian
from hand_robot.tasks import (task1_wave, task2_single_key, task3_chord, task4_octave_jump, task5_touch)
from hand_robot.tasks.common import Scene


@pytest.fixture(scope="module")
def scene():
    return Scene()


def _fd_jacobian(robot, q, branch, eps=1e-6):
    J = np.zeros((3, 20))
    for k in range(20):
        a, b = q.copy(), q.copy()
        a[k] += eps
        b[k] -= eps
        J[:, k] = (robot.end_effectors(a)[branch][:3, 3] - robot.end_effectors(b)[branch][:3, 3]) / (2 * eps)
    return J


def test_analytic_jacobian_matches_finite_differences(scene):
    q = np.random.default_rng(1).uniform(-0.8, 0.8, 20)
    for b in scene.robot.branches:
        assert np.abs(position_jacobian(scene.robot, q, b) - _fd_jacobian(scene.robot, q, b)).max() < 1e-7


def test_ik_multi_converges_and_respects_limits(scene):
    rng = np.random.default_rng(2)
    qt = rng.uniform(-0.5, 0.5, 20)
    tips = scene.robot.end_effectors(qt)
    tg = {b: tips[b][:3, 3] for b in ("finger_1", "finger_3", "finger_5")}
    res = scene.ik.solve_multi(tg, np.zeros(20), max_iterations=600)
    assert res.success
    assert np.all(res.q >= scene.q_min - 1e-12) and np.all(res.q <= scene.q_max + 1e-12)


def test_boundary_singularity_of_distal_joint(scene):
    d = sg.distal_sweep(scene.robot, np.zeros(20), "finger_3", 12, np.deg2rad(np.linspace(-60, 60, 121)))
    assert abs(np.rad2deg(d["values"][np.argmin(np.abs(d["det"]))]) - 11.0) < 1.0     # theta_12 = 0 (straight finger)


def test_keyboard_geometry(scene):
    kb = scene.kb
    assert kb.key_y("C4") == 0.0
    assert np.isclose(kb.key_y("C5"), 7 * kb.width)
    assert kb.transpose("E4", 7) == "E5"
    with pytest.raises(ValueError):
        kb.key_y("C#4")


def test_task1(scene):
    r = task1_wave.run(scene)[0]
    assert r.metrics["min_joint_margin_deg"] > 5


@pytest.mark.parametrize("finger", ["thumb", "index", "middle", "ring", "little"])
def test_task2_press(scene, finger):
    r = task2_single_key.run(scene, finger=finger)[0]
    assert r.metrics["press_error_mm"] < 0.2
    assert r.metrics["min_hover_clearance_mm"] > 0


def test_task3_chord(scene):
    r = task3_chord.run(scene)[0]
    assert r.metrics["press_error_mm"] < 0.2 and r.metrics["min_hover_clearance_mm"] > 0


def test_task4_octave(scene):
    r = task4_octave_jump.run(scene)[0]
    assert r.metrics["landing_error_mm"] < 1.0
    assert r.metrics["carriage_peak_speed_m_s"] <= scene.carriage["max_speed"]
    assert r.metrics["wrist_only_max_reachable_steps"] < 7          # the carriage is needed


def test_task5_force_tracking(scene):
    for r in task5_touch.run(scene):
        assert r.metrics["steady_error_pct"] < 10      # mean force; the RMS error is larger for mf/ff (bouncing on the key stop)
