"""Closed-loop test of the ROS-independent touch-node core (joint states in, efforts out) on the reduced plant.

This checks the plumbing the Gazebo node relies on (joint ordering, key-force reconstruction, held joints, torque
clamping), NOT Gazebo itself, which has never been run.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hand_robot.control.touch_node_core import RUN, TouchNodeCore
from hand_robot.dynamics.reduced import ReducedPlant
from hand_robot.tasks.common import Scene
from hand_robot.tasks.task5_touch import PROFILES


@pytest.fixture(scope="module")
def scene():
    return Scene()


def _close_loop(scene, profile, dt_ctrl=1e-3, dt_plant=5e-4, duration=2.2):
    core = TouchNodeCore(scene, "middle", profile, t_pre=0.05)
    kb, m = scene.kb, core.model
    plant = ReducedPlant(m)
    q = core.q_start.copy()                                   # all 20 joints; only the 5 active ones move
    qd = np.zeros(20)
    ia = m.idx
    contact = lambda tip, v: np.array([0.0, 0.0, kb.contact_force(tip[2] + core.base[2], v[2])])
    base = core.base
    taus = []
    for k in range(int(round(duration / dt_ctrl))):
        t = k * dt_ctrl
        T = m.kin(q[ia])
        z, vz = m.tip(T)[2] + base[2], (m.jacobian(T) @ qd[ia])[2]
        depress = min(max(kb.z_surface - z, 0.0), kb.press_depth)
        key_pos = -depress                                    # key joint: 0 at rest, negative when pressed
        key_vel = vz if depress > 0 else 0.0                  # = d(key_pos)/dt = -(depression rate) = vz while pressed
        tau = core.step(t, q, qd, key_pos, key_vel)
        taus.append(tau)
        for _ in range(int(round(dt_ctrl / dt_plant))):
            qa, qda, _, _ = plant.step(q[ia], qd[ia], tau[ia], contact, dt_plant)
            q[ia], qd[ia] = qa, qda
    return core, np.array(taus)


def test_key_joint_name_matches_world_file(scene):
    from hand_robot.control.touch_node_core import key_joint_name
    assert key_joint_name(scene.kb, "C4") == "key_60_joint"
    assert key_joint_name(scene.kb, "E4") == "key_64_joint"
    sdf = (Path(__file__).resolve().parents[1] / "gazebo" / "worlds" / "piano.sdf").read_text()
    assert 'joint name="key_64_joint"' in sdf


def test_held_joints_balance_gravity_at_rest(scene):
    core = TouchNodeCore(scene, "middle", PROFILES[0], t_pre=0.05)
    tau = core.step(0.0, core.q_start, np.zeros(20), 0.0, 0.0)
    held = [i for i in range(20) if i not in core.model.idx]
    g = core.rbd.gravity_torque(core.q_start)
    assert np.abs(tau[held] - g[held]).max() < 1e-12
    assert np.all(np.abs(tau) <= core.tau_lim + 1e-15)


def test_core_closed_loop_pp_tracks_force_like_the_python_simulation(scene):
    core, taus = _close_loop(scene, PROFILES[0])
    assert core.phase == RUN
    m = core.metrics()
    assert m["saturated_pct"] == 0.0
    assert m["F_rms_error_pct"] < 10.0                         # python simulation: 4.6 %
    assert 0.9 < m["F_peak_N"] < 1.4
    assert not m["key_bottomed"]
    assert np.all(np.abs(taus) <= core.tau_lim + 1e-12)
