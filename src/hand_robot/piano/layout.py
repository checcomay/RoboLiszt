"""Full 12-tone keyboard layout (white AND black keys) used by the Gazebo world and the sound node.

``Keyboard`` (keyboard.py) stays white-keys-only because the five tasks only play white keys; this module adds the
chromatic layout without changing any task geometry. Keys are identified by their MIDI number (C4 = 60):
link ``key_60``, joint ``key_60_joint``.

Frame: the keyboard frame K of ``Keyboard`` (x along the key, y along the keyboard with C4 at 0, z up, white top at 0).
Black keys sit between the neighbouring white keys; ``BLACK_SHIFT`` moves them off the exact boundary the way a real
keyboard does (C#/F# toward the lower white key, D#/A# toward the upper one). These shifts are a design approximation,
not measured values.
"""
from __future__ import annotations

NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_WHITE_POS = {0: 0, 2: 1, 4: 2, 5: 3, 7: 4, 9: 5, 11: 6}
BLACK_PCS = (1, 3, 6, 8, 10)
BLACK_SHIFT = {1: -0.08, 3: 0.08, 6: -0.10, 8: 0.0, 10: 0.10}      # fraction of the white key width
DEFAULT_RANGE = (48, 84)                                            # C3 .. C6 (22 white + 15 black keys)


def midi(note) -> int:
    """'C4' -> 60, 'F#3' -> 54, 'Bb2' -> 46 (flats accepted); an int is returned unchanged."""
    if isinstance(note, int):
        return note
    s = note.strip()
    pc = _PC[s[0].upper()]
    i = 1
    while i < len(s) and s[i] in "#b":
        pc += 1 if s[i] == "#" else -1
        i += 1
    return 12 * (int(s[i:]) + 1) + pc


def note_name(m: int) -> str:
    return f"{NAMES[m % 12]}{m // 12 - 1}"


def is_black(m: int) -> bool:
    return m % 12 in BLACK_PCS


def freq(m: int, a4: float = 440.0) -> float:
    return a4 * 2.0 ** ((m - 69) / 12.0)


def white_index(m: int) -> int:
    """Index of a white key counted in white keys from C4 (C4 = 0, D4 = 1, B3 = -1)."""
    if is_black(m):
        raise ValueError(f"{note_name(m)} is a black key")
    return (m // 12 - 5) * 7 + _WHITE_POS[m % 12]


def key_y(m: int, white_width: float) -> float:
    """Lateral position [m] of the key centre."""
    if not is_black(m):
        return white_index(m) * white_width
    lo, hi = white_index(m - 1), white_index(m + 1)
    return (0.5 * (lo + hi) + BLACK_SHIFT[m % 12]) * white_width


def keys(rng=DEFAULT_RANGE) -> list[int]:
    return list(range(rng[0], rng[1] + 1))


def key_link_name(m: int) -> str:
    return f"key_{m}"


def key_joint_name(m) -> str:
    return f"key_{midi(m)}_joint"


def midi_from_joint(name: str) -> int | None:
    """'key_60_joint' -> 60 (None if the name is not a key joint)."""
    if name.startswith("key_") and name.endswith("_joint"):
        try:
            return int(name[4:-6])
        except ValueError:
            return None
    return None
