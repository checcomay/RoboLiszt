"""Tests of the baseline motion controllers and of the control study (unified rigid-body dynamics)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("run_control_study", ROOT / "scripts" / "run_control_study.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def study(mod):
    return mod.Study()


def _steady_force(study, mod, ctrl, dz):
    lg = study.simulate(ctrl, dz, T=2.0)
    assert not lg["diverged"]
    return mod.steady(lg["Fz"], lg["t"], 1.4, 1.9)[0]


def test_position_control_force_depends_on_key_height_but_hybrid_does_not(study, mod):
    """+-2 mm of key-height error moves the force of a stiff position controller by more than 1.5 N, not that of the hybrid one."""
    f_pos = [_steady_force(study, mod, study.mk_position(0.12), dz) for dz in (-2e-3, 2e-3)]
    assert f_pos[1] - f_pos[0] > 1.5
    hyb = []
    for dz in (-2e-3, 2e-3):
        c = study.mk_hybrid(0.12, gains=mod.hybrid_gains(study))
        lg = study.simulate(c, dz, T=2.0)
        hyb.append(mod.hyb_plateau(study, c, lg)[0])
    assert abs(hyb[1] - hyb[0]) < 0.2


def test_contact_aware_impedance_has_lower_peak_than_plain_impedance(study, mod):
    plain = study.simulate(study.mk_impedance(0.25), 0.0, T=2.0)["Fz"].max()
    aware = study.simulate(study.mk_impedance(0.25, contact_aware=True), 0.0, T=2.0)["Fz"].max()
    assert aware < plain
