from __future__ import annotations

import re
import numpy as np

_STEPS = {"C": 0, "D": 1, "E": 2, "F": 3, "G": 4, "A": 5, "B": 6}
_NAMES = "CDEFGAB"


class Keyboard:
    """White-key keyboard geometry.

    Keyboard frame: x along the key (0 at the hand base, front edge at ``key_front_x``),
    y along the keyboard (0 = centre of C4, increasing with pitch), z up (0 = key surface).
    Only white keys are modelled (black keys are not described in robot.yaml).
    """

    def __init__(self, config: dict):
        p = config["piano"]
        self.width = float(p["key_width"])
        self.depth = float(p["white_key_depth"])
        self.z_surface = float(p["key_surface_z"])
        self.press_depth = float(p["press_depth"])
        self.safe_height = float(p["safe_height"])
        self.hover_height = float(p.get("hover_height", 0.02))
        self.home = np.array([p["home_x"], p["home_y"], p["home_z"]], dtype=float)
        self.play_z = float(p.get("play_z", p["home_z"]))
        self.front_x = float(p.get("key_front_x", 0.10))
        self.press_x = self.front_x + float(p.get("press_x_ratio", 0.5)) * self.depth
        # shorter fingers play closer to the front edge of the key
        ratios = {"thumb": 0.10, "little": 0.15, "ring": 0.35, "middle": 0.5, "index": 0.5}
        ratios.update(p.get("press_x_ratios", {}) or {})
        self.press_x_by_finger = {f: self.front_x + float(r) * self.depth for f, r in ratios.items()}
        self.home_note = str(p.get("home_note", "E4"))
        self.k_key = float(p.get("key_stiffness", 300.0))
        self.c_key = float(p.get("key_damping", 5.0))
        self.k_stop = float(p.get("key_stop_stiffness", 8000.0))

    # ----------------------------------------------------------------- notes
    @staticmethod
    def white_index(note: str) -> int:
        m = re.fullmatch(r"([A-Ga-g])([#b]?)(-?\d)", note.strip())
        if not m:
            raise ValueError(f"Invalid note '{note}' (expected e.g. 'C4').")
        if m.group(2):
            raise ValueError(f"'{note}': only white keys are modelled.")
        return int(m.group(3)) * 7 + _STEPS[m.group(1).upper()]

    @staticmethod
    def note_from_index(idx: int) -> str:
        return f"{_NAMES[idx % 7]}{idx // 7}"

    def transpose(self, note: str, white_steps: int) -> str:
        return self.note_from_index(self.white_index(note) + white_steps)

    # -------------------------------------------------------------- geometry
    def key_y(self, note: str) -> float:
        return (self.white_index(note) - self.white_index("C4")) * self.width

    def x_for(self, finger: str | None = None) -> float:
        return self.press_x_by_finger.get(finger, self.press_x) if finger else self.press_x

    def surface_point(self, note: str, finger: str | None = None) -> np.ndarray:
        return np.array([self.x_for(finger), self.key_y(note), self.z_surface])

    def hover_point(self, note: str, height: float | None = None, finger: str | None = None) -> np.ndarray:
        h = self.safe_height if height is None else height
        return np.array([self.x_for(finger), self.key_y(note), self.z_surface + h])

    def pressed_point(self, note: str, finger: str | None = None) -> np.ndarray:
        return np.array([self.x_for(finger), self.key_y(note), self.z_surface - self.press_depth])

    def base_home(self, play: bool = True) -> np.ndarray:
        """Hand-base (frame 0) position in the keyboard frame: playing height (``play_z``) or the
        rest height ``home_z`` (play=False)."""
        b = self.home.copy()
        b[1] += self.key_y(self.home_note)
        if play:
            b[2] = self.play_z
        return b

    # --------------------------------------------------------------- contact
    def contact_force(self, z: float, vz: float) -> float:
        """Normal force (>= 0) of a key on a fingertip at height z moving with vz."""
        d = self.z_surface - z                  # penetration
        if d <= 0.0:
            return 0.0
        f = self.k_key * d + self.c_key * (-vz)      # spring + damper (vz < 0 while pressing)
        if d > self.press_depth:
            f += self.k_stop * (d - self.press_depth)
        return max(f, 0.0)
