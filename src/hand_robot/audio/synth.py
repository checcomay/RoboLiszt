"""Additive piano-like synthesis (numpy only). Not a sampled piano: a physically-flavoured approximation.

Per note: inharmonic partials f_n = n f0 sqrt(1 + B n^2), two slightly detuned 'strings' per partial (slow beating), faster
decay of the higher partials, hammer-position notch (strike at 1/8 of the string), a short filtered-noise hammer thump.
Velocity (1..127) sets loudness (~ (v/127)^1.5) AND brightness (spectral slope n^-p, p = 1.7 - v/127), as on a real piano.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from hand_robot.piano.layout import freq

T60_MAX = 8.0                          # s, longest rendered note (a held note is looped no further)


def midi_velocity_from_speed(speed: float, speed_min: float = 0.015, speed_max: float = 0.18) -> int:
    """Key descent speed [m/s] -> MIDI velocity 1..127 on a log scale (speed_min -> 20, speed_max -> 127; defaults = config/robot.yaml sound.*)."""
    if speed <= 0:
        return 1
    v = 20.0 + 107.0 * np.log(speed / speed_min) / np.log(speed_max / speed_min)
    return int(np.clip(round(v), 1, 127))


def release_fade(n: int, sr: int, tau: float) -> np.ndarray:
    return np.exp(-np.arange(n) / (tau * sr)).astype(np.float32)


class PianoSynth:
    def __init__(self, sr: int = 44100, a4: float = 440.0):
        self.sr, self.a4 = sr, a4
        self._cache: dict[tuple[int, int], np.ndarray] = {}

    @staticmethod
    def _layer(vel: int) -> int:
        return int(np.clip(round(vel / 8.0), 1, 16))

    def render(self, m: int, vel: int) -> np.ndarray:
        """Mono float32 note with peak <= gain(vel) <= 1, cached per (note, 16 velocity layers)."""
        key = (m, self._layer(vel))
        if key not in self._cache:
            self._cache[key] = self._render(m, min(self._layer(vel) * 8, 127))
        return self._cache[key]

    def _render(self, m: int, vel: int) -> np.ndarray:
        sr, v = self.sr, vel / 127.0
        f0 = freq(m, self.a4)
        t60 = float(np.clip(14.0 * 2.0 ** (-(m - 21) / 20.0), 1.2, 14.0))
        dur = min(1.15 * t60, T60_MAX)
        t = np.arange(int(dur * sr)) / sr
        B = float(np.clip(1.0e-4 * 2.0 ** ((m - 60) / 14.0), 2e-5, 6e-4))
        p = 1.7 - 1.0 * v
        tau1 = t60 / 6.908
        sig = np.zeros_like(t)
        rng = np.random.default_rng(m)
        for n in range(1, 33):
            fn = n * f0 * np.sqrt(1.0 + B * n * n)
            if fn > 0.45 * sr:
                break
            a = n ** -p * max(abs(np.sin(np.pi * n / 8.0)), 0.15)
            tau = tau1 / (1.0 + 0.45 * (n - 1) ** 0.8)
            ph = rng.uniform(0, 2 * np.pi, 2)
            d = 4e-4
            sig += a * np.exp(-t / tau) * 0.5 * (np.sin(2 * np.pi * fn * (1 + d) * t + ph[0]) + np.sin(2 * np.pi * fn * (1 - d) * t + ph[1]))
        sig *= 1.0 - np.exp(-t / 0.002)                                       # 2 ms attack
        # hammer thump: low-passed noise burst, louder and brighter with velocity
        alpha = 1.0 - np.exp(-2 * np.pi * (700.0 + 2500.0 * v) / sr)
        thump = lfilter([alpha], [1.0, -(1.0 - alpha)], rng.standard_normal(len(t))) * np.exp(-t / 0.008)
        sig = sig / max(np.abs(sig).max(), 1e-12) + 0.18 * v ** 1.5 * thump / max(np.abs(thump).max(), 1e-12)
        sig /= max(np.abs(sig).max(), 1e-12)
        return (sig * (0.06 + 0.94 * v ** 1.5)).astype(np.float32)


def render_events(events, synth: PianoSynth, release_time: float = 0.08, tail: float = 1.0, master_gain: float = 0.8):
    """Offline mix. events: [(t_on, midi, velocity, t_off or None)] -> float32 mono array (soft-limited)."""
    sr = synth.sr
    end = 0.0
    for t_on, m, vel, t_off in events:
        end = max(end, t_off + 6 * release_time if t_off is not None else t_on + len(synth.render(m, vel)) / sr)
    out = np.zeros(int((end + tail) * sr) + 1, dtype=np.float64)
    for t_on, m, vel, t_off in events:
        s = synth.render(m, vel).astype(np.float64)
        k = int(round(t_on * sr))
        if t_off is not None:
            ko = max(int((t_off - t_on) * sr), 0)
            s = s[: ko + int(6 * release_time * sr)].copy()
            s[ko:] *= release_fade(len(s) - ko, sr, release_time)
        out[k:k + len(s)] += s[: len(out) - k]
    return np.tanh(master_gain * out).astype(np.float32)


def write_wav(path, x: np.ndarray, sr: int):
    import wave
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm.tobytes())
