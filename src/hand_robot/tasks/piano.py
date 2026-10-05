from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import numpy as np


@dataclass
class TaskTrajectory:
    """Time-parameterized Cartesian trajectory for one or more fingers."""
    t: np.ndarray
    positions: dict[str, np.ndarray]
    desired_force: dict[str, np.ndarray]


class PianoKeyboard:
    """Minimal geometric keyboard model for simulation and task planning."""

    NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

    def __init__(self, config: dict):
        p = config["piano"]
        self.width = float(p["key_width"])
        self.depth = float(p["white_key_depth"])
        self.z = float(p["key_surface_z"])
        self.press_depth = float(p["press_depth"])
        self.safe_height = float(p["safe_height"])
        self.home = np.array([p["home_x"], p["home_y"], p["home_z"]], dtype=float)

    def note_to_position(self, midi_note: int) -> np.ndarray:
        """Map a MIDI-like pitch number to a simplified linear keyboard."""
        x = (midi_note - 60) * self.width
        return np.array([self.home[0] + x, self.home[1], self.z])

    def key_position(self, midi_note: int, pressed: bool = False) -> np.ndarray:
        p = self.note_to_position(midi_note)
        if pressed:
            p = p.copy()
            p[2] -= self.press_depth
        return p


class PianoTaskPlanner:
    """Build high-level trajectories for the six requested tasks."""

    def __init__(self, keyboard: PianoKeyboard, dt: float = 0.01):
        self.keyboard = keyboard
        self.dt = float(dt)

    def _times(self, duration: float) -> np.ndarray:
        return np.arange(0.0, duration + self.dt, self.dt)

    @staticmethod
    def _smoothstep(s: np.ndarray) -> np.ndarray:
        return 3*s**2 - 2*s**3

    def _interpolate(self, p0, p1, duration) -> np.ndarray:
        t = self._times(duration)
        s = self._smoothstep(t / duration)
        return p0[None, :] + s[:, None] * (p1 - p0)[None, :]

    def free_space_greeting(self, branch: str, amplitude: float = 0.04) -> TaskTrajectory:
        """Task 1: open-space greeting wave."""
        t = self._times(2.0)
        base = self.keyboard.home + np.array([0.0, 0.0, 0.06])
        p = np.tile(base, (len(t), 1))
        p[:, 1] += amplitude * np.sin(2*np.pi*1.5*t)
        return TaskTrajectory(t, {branch: p}, {branch: np.zeros(len(t))})

    def single_key(self, branch: str, midi_note: int) -> TaskTrajectory:
        """Task 2: approach, press and release one key."""
        p = self.keyboard.key_position(midi_note, False)
        pp = self.keyboard.key_position(midi_note, True)

        segments = [
            self._interpolate(self.keyboard.home, p + np.array([0, 0, self.keyboard.safe_height]), 0.6),
            self._interpolate(p + np.array([0, 0, self.keyboard.safe_height]), p, 0.3),
            self._interpolate(p, pp, 0.15),
            self._interpolate(pp, p, 0.15),
            self._interpolate(p, p + np.array([0, 0, self.keyboard.safe_height]), 0.3),
        ]
        pos = np.vstack(segments)
        t = np.arange(len(pos)) * self.dt
        force = np.zeros(len(pos))
        force[len(force)//2] = 1.0
        return TaskTrajectory(t, {branch: pos}, {branch: force})

    def chord(self, branches: Iterable[str], notes: Iterable[int]) -> TaskTrajectory:
        """Task 3: execute several key presses simultaneously."""
        branches = list(branches)
        notes = list(notes)
        if len(branches) != len(notes):
            raise ValueError("Each finger must be paired with one chord note.")

        t = self._times(1.2)
        positions = {}
        forces = {}
        for branch, note in zip(branches, notes):
            p = self.keyboard.key_position(note, False)
            pp = self.keyboard.key_position(note, True)
            pos = np.tile(p, (len(t), 1))
            phase = np.clip((t - 0.4) / 0.2, 0, 1)
            phase = self._smoothstep(phase)
            release = np.clip((t - 0.75) / 0.2, 0, 1)
            release = self._smoothstep(release)
            pos[:, 2] -= self.keyboard.press_depth * (phase - release)
            positions[branch] = pos
            forces[branch] = phase - release
        return TaskTrajectory(t, positions, forces)

    def octave_jump(self, branch: str, midi_note: int) -> TaskTrajectory:
        """Task 4: move from a note to the note one octave above."""
        p0 = self.keyboard.key_position(midi_note, False) + np.array([0, 0, self.keyboard.safe_height])
        p1 = self.keyboard.key_position(midi_note + 12, False) + np.array([0, 0, self.keyboard.safe_height])
        pos = self._interpolate(p0, p1, 0.8)
        t = np.arange(len(pos)) * self.dt
        return TaskTrajectory(t, {branch: pos}, {branch: np.zeros(len(t))})

    def dynamic_key(self, branch: str, midi_note: int, force_level: float) -> TaskTrajectory:
        """Task 5: press one key with a commanded force profile."""
        p = self.keyboard.key_position(midi_note, False)
        pp = self.keyboard.key_position(midi_note, True)
        t = self._times(1.0)
        s = self._smoothstep(np.clip(t / 0.35, 0, 1))
        r = self._smoothstep(np.clip((t - 0.65) / 0.35, 0, 1))
        depth = self.keyboard.press_depth * (s - r)
        pos = np.tile(p, (len(t), 1))
        pos[:, 2] -= depth
        force = force_level * np.maximum(0.0, s - r)
        return TaskTrajectory(t, {branch: pos}, {branch: force})

    def piece(self, branch: str, notes: Iterable[int]) -> TaskTrajectory:
        """Task 6: combine single-key and octave-like transitions."""
        notes = list(notes)
        if not notes:
            raise ValueError("The piece needs at least one note.")

        chunks = []
        force_chunks = []
        for i, note in enumerate(notes):
            task = self.single_key(branch, note)
            chunks.append(task.positions[branch])
            force_chunks.append(task.desired_force[branch])
            if i < len(notes) - 1:
                jump = self.octave_jump(branch, note)
                chunks.append(jump.positions[branch])
                force_chunks.append(jump.desired_force[branch])

        pos = np.vstack(chunks)
        force = np.concatenate(force_chunks)
        t = np.arange(len(pos)) * self.dt
        return TaskTrajectory(t, {branch: pos}, {branch: force})
