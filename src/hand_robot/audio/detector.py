"""Key-press detection from key joint states (position only: 0 at rest, negative when pressed, ~ -8 mm at the end stop).

State machine per key: ARMED -> (depression >= trigger_depth) NOTE ON -> PRESSED -> (depression <= release_depth) NOTE OFF -> ARMED.
The hysteresis (trigger > release) prevents re-triggering from contact jitter. The intensity is the mean descent speed
between ``motion_depth`` and ``trigger_depth`` (crossing times are linearly interpolated between samples, so the estimate
does not depend on the sample rate except through noise; at 200 Hz a ff press crosses this 1.7 mm in ~7 ms, i.e. ~1 sample:
use >= 500 Hz joint states for dynamics).
"""
from __future__ import annotations

from dataclasses import dataclass

from hand_robot.audio.synth import midi_velocity_from_speed


@dataclass
class KeyEvent:
    t: float
    midi: int
    on: bool
    velocity: int = 0
    speed: float = 0.0


class KeyPressDetector:
    def __init__(self, trigger_depth=0.0020, release_depth=0.0010, motion_depth=0.0003, speed_min=0.015, speed_max=0.18):
        assert release_depth < trigger_depth and motion_depth < trigger_depth
        self.trig, self.rel, self.mot = trigger_depth, release_depth, motion_depth
        self.vmin, self.vmax = speed_min, speed_max
        self._s: dict[int, dict] = {}

    @classmethod
    def from_config(cls, cfg: dict) -> "KeyPressDetector":
        s = cfg["sound"]
        return cls(s["trigger_depth"], s["release_depth"], s["motion_depth"], s["speed_min"], s["speed_max"])

    @staticmethod
    def _cross(t0, d0, t1, d1, level):
        return t1 if d1 == d0 else t0 + (level - d0) * (t1 - t0) / (d1 - d0)

    def update(self, t: float, midi: int, key_pos: float) -> list[KeyEvent]:
        d = max(-float(key_pos), 0.0)                                   # depression [m]
        st = self._s.setdefault(midi, {"pressed": False, "prev": (t, d), "t_mot": None})
        t0, d0 = st["prev"]
        ev: list[KeyEvent] = []
        if not st["pressed"]:
            if d <= self.mot:
                st["t_mot"] = None                                      # key back at rest: forget the previous motion start
            elif st["t_mot"] is None:                                   # first sample above the motion level
                st["t_mot"] = self._cross(t0, d0, t, d, self.mot) if d0 < self.mot else t0
            if d >= self.trig:
                t_on = self._cross(t0, d0, t, d, self.trig) if d0 < self.trig else t
                t_m = st["t_mot"] if st["t_mot"] is not None else t0
                dt = max(t_on - t_m, 1e-4)
                speed = (self.trig - self.mot) / dt
                ev.append(KeyEvent(t_on, midi, True, midi_velocity_from_speed(speed, self.vmin, self.vmax), speed))
                st["pressed"], st["t_mot"] = True, None
        elif d <= self.rel:
            ev.append(KeyEvent(self._cross(t0, d0, t, d, self.rel) if d0 > self.rel else t, midi, False))
            st["pressed"] = False
        st["prev"] = (t, d)
        return ev


def pair_events(events: list[KeyEvent]):
    """[(t_on, midi, velocity, t_off|None)] from a time-ordered stream of KeyEvents."""
    open_: dict[int, list] = {}
    out = []
    for e in events:
        if e.on:
            row = [e.t, e.midi, e.velocity, None]
            open_[e.midi] = row
            out.append(row)
        elif e.midi in open_:
            open_.pop(e.midi)[3] = e.t
    return [tuple(r) for r in out]
