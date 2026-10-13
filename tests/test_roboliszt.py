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


# ---------------------------------------------------------------- dynamics, statics, URDF
def test_rigid_body_gravity_is_potential_gradient(scene):
    from hand_robot.dynamics.rigid_body import RigidBodyDynamics
    d = RigidBodyDynamics(scene.robot, scene.config)
    q = np.random.default_rng(5).uniform(-0.6, 0.6, 20)
    g_fd = np.array([(d.potential_energy(q + 1e-6 * e) - d.potential_energy(q - 1e-6 * e)) / 2e-6 for e in np.eye(20)])
    assert np.abs(g_fd - d.gravity_torque(q)).max() < 1e-8
    M = d.mass_matrix(q)
    assert np.abs(M - M.T).max() < 1e-12 and np.linalg.eigvalsh(M).min() > 0


def test_virtual_work_duality(scene):
    from hand_robot.control.statics import joint_torque_from_tip_force
    q = np.random.default_rng(6).uniform(-0.6, 0.6, 20)
    qd = np.random.default_rng(7).uniform(-1, 1, 20)
    F = np.array([0.3, -0.2, 1.5])
    v = position_jacobian(scene.robot, q, "finger_3") @ qd
    assert abs(joint_torque_from_tip_force(scene.robot, q, "finger_3", F) @ qd - F @ v) < 1e-12


def test_urdf_matches_dh_model():
    import importlib.util
    spec = importlib.util.spec_from_file_location("verify_urdf", Path(__file__).resolve().parents[1] / "gazebo" / "verify_urdf.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main()


# ------------------------------------------------------------------------------- control study
def _study():
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_control_study", Path(__file__).resolve().parents[1] / "scripts" / "run_control_study.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, mod.Study()


def test_contact_aware_impedance_reduces_impact_vs_stiff_position():
    mod, S = _study()
    Fd = 2.5
    d_eq = (Fd + S.kb.k_stop * S.kb.press_depth) / (S.kb.k_key + S.kb.k_stop)
    pos = S.os_controller(S.descend_ref(-(d_eq + Fd / 1000), 0.25), [1000] * 3, zeta=1.0)
    imp = S.os_controller(S.descend_ref(-(d_eq + Fd / 100), 0.25), [100] * 3, zeta=1.0, k_env=S.kb.k_key,
                          zero_vref_on_contact=True)
    Fp = mod.simulate_closed_loop(S.dyn, pos, S.q0, 2.4, S.dt, S.env(0.0))["Fz"].max()
    Fi = mod.simulate_closed_loop(S.dyn, imp, S.q0, 2.4, S.dt, S.env(0.0))["Fz"].max()
    assert Fi < 0.5 * Fp


def test_hybrid_keeps_force_and_lateral_position():
    from hand_robot.control.hybrid import TouchProfile
    mod, S = _study()
    prof = TouchProfile("x", 0.12, 2.5, 0.12, t_hold=1.2, t_release=0.15)
    res = {}
    for dz in (-0.002, 0.002):
        c = mod.HybridTouchController(S.sc.robot, S.dyn, S.sc.config, mod.BRANCH, S.base, S.q0, xy_ref=S.xy,
                                      z_hover=0.03, z_floor=-0.010, profile=prof, model_mismatch=0.15)
        o = mod.simulate_closed_loop(S.dyn, c, S.q0, 2.4, S.dt, S.env(dz))
        w = mod.plateau_window(c.log["F_des"], 2.5, S.dt)
        res[dz] = (o["Fz"][w].mean(), np.hypot(o["x"] - S.xy[0], o["y"] - S.xy[1]).max())
    assert abs(res[-0.002][0] - res[0.002][0]) < 0.1          # force insensitive to a +-2 mm key-height error
    assert max(r[1] for r in res.values()) < 0.005            # lateral error below 5 mm
