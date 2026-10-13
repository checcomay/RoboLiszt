"""Chromatic layout (piano/layout.py) and the generated Gazebo world (white + black keys)."""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hand_robot.piano import layout as lay
from hand_robot.tasks.common import Scene


@pytest.fixture(scope="module")
def scene():
    return Scene()


def test_midi_names_and_frequencies():
    assert lay.midi("C4") == 60 and lay.midi("A4") == 69 and lay.midi("F#3") == 54 and lay.midi("Bb2") == 46
    assert lay.note_name(61) == "C#4" and lay.note_name(48) == "C3"
    assert abs(lay.freq(69) - 440.0) < 1e-12 and abs(lay.freq(60) - 261.6256) < 1e-3
    assert lay.midi_from_joint("key_60_joint") == 60 and lay.midi_from_joint("j3") is None


def test_white_positions_match_keyboard_class(scene):
    kb = scene.kb
    for note in ("C3", "E4", "G4", "B4", "C6"):
        assert abs(lay.key_y(lay.midi(note), kb.width) - kb.key_y(note)) < 1e-12
    with pytest.raises(ValueError):
        lay.white_index(61)


def test_black_keys_lie_between_their_white_neighbours(scene):
    w = scene.kb.width
    for m in lay.keys():
        if lay.is_black(m):
            assert lay.key_y(m - 1, w) < lay.key_y(m, w) < lay.key_y(m + 1, w)


def test_world_has_all_keys_with_unique_joints():
    sdf = ET.parse(ROOT / "gazebo" / "worlds" / "piano.sdf").getroot()
    piano = [m for m in sdf.iter("model") if m.get("name") == "piano"][0]
    kj = [j for j in piano.findall("joint") if j.get("type") == "prismatic"]
    midis = sorted(lay.midi_from_joint(j.get("name")) for j in kj)
    assert midis == lay.keys() and len(midis) == 37
    assert sum(lay.is_black(m) for m in midis) == 15


def test_black_keys_do_not_touch_each_other_or_the_fingertips(scene):
    cfg, kb = scene.config, scene.kb
    p = cfg["piano"]
    bw, bl = float(p["black_key_width"]), float(p["black_key_length"])
    ys = sorted(lay.key_y(m, kb.width) for m in lay.keys() if lay.is_black(m))
    assert np.diff(ys).min() > bw + 0.002                           # >= 2 mm gap between black keys
    x_black_front = kb.front_x + kb.depth - bl
    r_tip = float(p["tip_collision_radius"])
    for finger in ("thumb", "index", "middle", "ring", "little"):
        assert kb.x_for(finger) + r_tip < x_black_front - 0.005     # pressing a white key never reaches a black key
