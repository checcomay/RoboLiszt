"""Piano sound: synthesis (pitch, loudness, brightness), key-press detection, offline mixing, task audio."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hand_robot.audio.detector import KeyPressDetector, pair_events
from hand_robot.audio.from_tasks import render_task_wav
from hand_robot.audio.mixer import PolyMixer
from hand_robot.audio.synth import PianoSynth, midi_velocity_from_speed, render_events
from hand_robot.piano.layout import freq
from hand_robot.tasks.common import Scene

SR = 22050


@pytest.fixture(scope="module")
def synth():
    return PianoSynth(SR)


def _peak_freq(x, sr):
    n = min(len(x), sr)                                              # first second, after the attack
    seg = x[int(0.05 * sr):n] * np.hanning(n - int(0.05 * sr))
    sp = np.abs(np.fft.rfft(seg, 8 * len(seg)))
    return np.argmax(sp) * sr / (8 * len(seg))


def _centroid(x, sr):
    sp = np.abs(np.fft.rfft(x[: sr] * np.hanning(sr)))
    f = np.fft.rfftfreq(sr, 1 / sr)
    return float((f * sp).sum() / sp.sum())


@pytest.mark.parametrize("m", [48, 60, 69, 72, 84])
def test_pitch_matches_equal_temperament(synth, m):
    x = synth.render(m, 80)
    assert np.isfinite(x).all() and np.abs(x).max() <= 1.0
    f = _peak_freq(x, SR)
    # the strongest partial may be the 2nd/3rd one on a piano: accept the fundamental or a (slightly inharmonic) low harmonic
    ratios = f / freq(m)
    assert min(abs(ratios - h) / h for h in (1, 2, 3)) < 0.01


def test_velocity_controls_loudness_and_brightness(synth):
    soft, loud = synth.render(60, 30), synth.render(60, 120)
    rms = lambda x: np.sqrt(np.mean(x[: SR] ** 2))
    assert rms(loud) > 4 * rms(soft)
    assert _centroid(loud, SR) > 1.2 * _centroid(soft, SR)


def test_speed_to_velocity_mapping_is_monotonic_and_covers_the_task5_touches():
    sp = np.geomspace(0.002, 0.5, 50)
    v = [midi_velocity_from_speed(s) for s in sp]
    assert all(b >= a for a, b in zip(v, v[1:])) and v[0] == 1 and v[-1] == 127
    assert midi_velocity_from_speed(0.015) == 20 and midi_velocity_from_speed(0.18) == 127


def _ramp(det, midi, v, t0=0.0, dt=1e-3, depth=0.008):
    """Press with constant speed v down to `depth`, hold, release at the same speed; returns events."""
    ev, t = [], t0
    n_down = int(depth / v / dt)
    for k in range(n_down + 1):
        ev += det.update(t0 + k * dt, midi, -min(v * k * dt, depth))
    for k in range(1, n_down + 2):
        ev += det.update(t0 + (n_down + k) * dt, midi, -max(depth - v * k * dt, 0.0))
    return ev


@pytest.mark.parametrize("v", [0.02, 0.05, 0.12, 0.25])
def test_detector_measures_the_descent_speed(v):
    det = KeyPressDetector()
    ev = _ramp(det, 64, v)
    on = [e for e in ev if e.on]
    off = [e for e in ev if not e.on]
    assert len(on) == 1 and len(off) == 1 and on[0].midi == 64
    assert abs(on[0].speed - v) / v < 0.03
    assert off[0].t > on[0].t


def test_detector_intensity_is_ordered_and_keys_are_independent():
    det = KeyPressDetector()
    ev = _ramp(det, 60, 0.03) + _ramp(det, 64, 0.15, t0=0.0)
    on = {e.midi: e for e in ev if e.on}
    assert set(on) == {60, 64} and on[64].velocity > on[60].velocity


def test_detector_ignores_noise_and_jitter_at_the_threshold():
    det = KeyPressDetector()
    rng = np.random.default_rng(0)
    ev = []
    for k in range(500):                                          # rest noise <= 0.2 mm: nothing
        ev += det.update(k * 1e-3, 62, -abs(rng.normal(0, 7e-5)))
    assert ev == []
    ev = []
    for k in range(26):                                           # a press down to 2.5 mm at 0.1 m/s ...
        ev += det.update(1.0 + k * 1e-3, 62, -0.1 * k * 1e-3)
    assert [e.on for e in ev] == [True]
    for k in range(200):                                          # ... then jitter of +-0.3 mm around 2.2 mm
        ev += det.update(2.0 + k * 1e-3, 62, -(0.0022 + 0.0003 * np.sin(k)))
    assert sum(e.on for e in ev) == 1 and not any(not e.on for e in ev)


def test_offline_mix_places_notes_at_their_times_and_damps_on_release(synth):
    x = render_events([(0.5, 60, 100, 0.8), (0.5, 64, 100, None)], synth, release_time=0.08, tail=0.5)
    assert np.abs(x[: int(0.49 * SR)]).max() < 1e-6 and np.abs(x[int(0.5 * SR): int(0.55 * SR)]).max() > 0.01
    y = render_events([(0.5, 60, 100, 0.8)], synth, 0.08, 1.0)
    assert np.sqrt(np.mean(y[int(1.1 * SR): int(1.2 * SR)] ** 2)) < 0.02 * np.sqrt(np.mean(y[int(0.6 * SR): int(0.7 * SR)] ** 2))


def test_realtime_mixer_matches_note_on_off(synth):
    mx = PolyMixer(synth)
    assert np.abs(mx.render(512)).max() == 0.0
    mx.note_on(60, 100)
    a = mx.render(2048)
    assert np.abs(a).max() > 0.05
    mx.note_off(60)
    for _ in range(40):
        mx.render(512)
    assert np.abs(mx.render(512)).max() < 1e-3 and not mx._v


@pytest.fixture(scope="module")
def scene():
    return Scene()


def test_task_audio_notes_and_intensity(scene):
    tr = ROOT / "data" / "trajectories"
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p, x = render_task_wav(scene, tr / "task3_chord_C4E4G4.npz", Path(td) / "c.wav")
        assert (Path(td) / "c.wav").stat().st_size > 10000
    assert sorted(m for _, m, _, _ in p) == [60, 64, 67] and max(a for a, *_ in p) - min(a for a, *_ in p) < 0.01
    p, _ = render_task_wav(scene, tr / "task3_chord_C4E4G4_roll.npz")
    assert [m for _, m, _, _ in sorted(p)] == [60, 64, 67]
    p, _ = render_task_wav(scene, tr / "task4_octave_jump_7.npz")
    assert sorted(m for _, m, _, _ in p) == [60, 64, 67, 72, 76, 79]
    vel = [render_task_wav(scene, tr / f"{n}.npz")[0][0][2] for n in ("task5rb_pp", "task5rb_mf", "task5rb_ff_tau0.25")]
    assert vel[0] < vel[1] < vel[2] and vel[2] - vel[0] > 40
