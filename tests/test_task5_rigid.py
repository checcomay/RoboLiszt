"""Tests of the reduced rigid-body model and of the operational-space touch controller (Task 5, redesign)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hand_robot.control.opspace import OpSpaceGains
from hand_robot.dynamics.reduced import ReducedPlant, ReducedRigidBody
from hand_robot.dynamics.rigid_body import RigidBodyDynamics
from hand_robot.kinematics.jacobian import position_jacobian
from hand_robot.tasks import task5_rigid as r5
from hand_robot.tasks.common import Scene
from hand_robot.tasks.task5_touch import PROFILES


@pytest.fixture(scope="module")
def scene():
    return Scene()


@pytest.fixture(scope="module")
def rbd(scene):
    return RigidBodyDynamics(scene.robot, scene.config)


def _setup(scene, rbd, branch, seed):
    q = np.random.default_rng(seed).uniform(-0.6, 0.6, 20)
    return q, ReducedRigidBody(rbd, branch, q)


@pytest.mark.parametrize("branch", ["finger_1", "finger_3", "finger_5"])
def test_reduced_mass_matrix_equals_full_model_block(scene, rbd, branch):
    q, red = _setup(scene, rbd, branch, 11)
    T = red.kin(q[red.idx])
    M_full = rbd.mass_matrix(q)[np.ix_(red.idx, red.idx)]
    assert np.abs(red.mass_matrix(T) - M_full).max() < 1e-12
    assert np.abs(red.mass_matrix(T) - red.mass_matrix_rnea(T)).max() < 1e-12


@pytest.mark.parametrize("branch", ["finger_1", "finger_3", "finger_5"])
def test_reduced_bias_and_jacobian_equal_full_model(scene, rbd, branch):
    q, red = _setup(scene, rbd, branch, 12)
    rng = np.random.default_rng(13)
    qd = np.zeros(20)
    qd[red.idx] = rng.uniform(-1, 1, red.n)                   # locked joints do not move
    T = red.kin(q[red.idx])
    h_full = rbd.inverse_dynamics(q, qd, np.zeros(20), gravity=True, friction=False)[red.idx]
    assert np.abs(red.bias(T, qd[red.idx]) - h_full).max() < 1e-10
    J_full = position_jacobian(scene.robot, q, branch)[:, red.idx]
    assert np.abs(red.jacobian(T) - J_full).max() < 1e-12
    p_full = scene.robot.end_effectors(q)[branch][:3, 3]
    assert np.abs(red.tip(T) - p_full).max() < 1e-12


def test_reduced_plant_equilibrium_under_gravity_compensation(scene, rbd):
    q, red = _setup(scene, rbd, "finger_3", 14)
    plant, qa, qd = ReducedPlant(red), q[red.idx].copy(), np.zeros(red.n)
    tau = red.gravity_torque(red.kin(qa))
    for _ in range(200):
        qa, qd, _, _ = plant.step(qa, qd, tau, np.zeros(3), 5e-4)
    assert np.abs(qa - q[red.idx]).max() < 1e-9 and np.abs(qd).max() < 1e-9


def test_opspace_gains_match_stored_configuration(scene):
    g = OpSpaceGains.from_config(scene.config)
    assert g.wn_z == 40.0 and g.kf == 0.8 and g.model_mismatch == 0.15
    assert float(scene.config["touch_rigid"]["tau_max_finger_sensitivity"]) == 0.25


def test_task5_rigid_pp_is_stable_and_tracks_force(scene):
    res = r5.run(scene, profiles=[PROFILES[0]])[0]
    m = res.metrics
    assert m["saturated_pct"] == 0.0
    assert m["F_rms_error_pct"] < 10.0                   # hold window
    assert m["F_peak_N"] < 1.3 * PROFILES[0].f_hold
    assert not m["key_bottomed"]
    assert abs(m["final_z_offset_mm"]) < 1.0              # gravity compensation + integral: no offset after release


def test_task5_rigid_mf_does_not_bounce_on_key_stop(scene):
    m = r5.run(scene, profiles=[PROFILES[1]])[0].metrics
    assert m["saturated_pct"] == 0.0 and m["F_rms_error_pct"] < 10.0 and m["contact_losses"] == 0
