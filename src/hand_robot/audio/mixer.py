"""Real-time polyphonic mixer + audio backends (sounddevice, aplay). Pure python/numpy; backends import lazily."""
from __future__ import annotations

import subprocess
import threading
import time

import numpy as np

from hand_robot.audio.synth import PianoSynth


class PolyMixer:
    def __init__(self, synth: PianoSynth, release_time: float = 0.08, master_gain: float = 0.8):
        self.synth, self.sr, self.tau, self.gain = synth, synth.sr, release_time, master_gain
        self._v: list[dict] = []
        self._lock = threading.Lock()

    def note_on(self, m: int, vel: int):
        s = self.synth.render(m, vel)                     # cached after the first call (see prewarm)
        with self._lock:
            for v in self._v:
                if v["m"] == m:
                    v["rel"], v["tau"] = True, 0.02       # re-struck key: damp the old sound quickly
            self._v.append({"m": m, "s": s, "pos": 0, "rel": False, "g": 1.0, "tau": self.tau})

    def note_off(self, m: int):
        with self._lock:
            for v in self._v:
                if v["m"] == m and not v["rel"]:
                    v["rel"] = True

    def render(self, frames: int) -> np.ndarray:
        out = np.zeros(frames, dtype=np.float64)
        with self._lock:
            keep = []
            for v in self._v:
                n = min(frames, len(v["s"]) - v["pos"])
                if n <= 0:
                    continue
                seg = v["s"][v["pos"]:v["pos"] + n].astype(np.float64)
                if v["rel"]:
                    f = v["g"] * np.exp(-np.arange(1, n + 1) / (v["tau"] * self.sr))
                    seg = seg * f
                    v["g"] = float(f[-1])
                out[:n] += seg
                v["pos"] += n
                if v["pos"] < len(v["s"]) and v["g"] > 1e-3:
                    keep.append(v)
            self._v = keep
        return np.tanh(self.gain * out).astype(np.float32)

    def prewarm(self, midis, velocities=(40, 80, 120)):
        for m in midis:
            for vel in velocities:
                self.synth.render(m, vel)


class SoundDeviceBackend:
    def __init__(self, mixer: PolyMixer, blocksize: int = 512):
        import sounddevice as sd                          # pip install sounddevice (needs PortAudio)
        self.stream = sd.OutputStream(samplerate=mixer.sr, channels=1, dtype="float32", blocksize=blocksize,
                                      latency="low", callback=lambda out, n, t, s: out.__setitem__(slice(None), mixer.render(n)[:, None]))

    def start(self):
        self.stream.start()

    def stop(self):
        self.stream.stop(); self.stream.close()


class AplayBackend:
    """Linux/ALSA fallback: pipes float32 blocks to `aplay`; the pipe is shrunk to keep the latency low."""
    def __init__(self, mixer: PolyMixer, blocksize: int = 512):
        self.mixer, self.bs, self.run = mixer, blocksize, False
        self.p = subprocess.Popen(["aplay", "-q", "-t", "raw", "-f", "FLOAT_LE", "-r", str(mixer.sr), "-c", "1", "--buffer-time=40000"],
                                  stdin=subprocess.PIPE)
        try:
            import fcntl
            fcntl.fcntl(self.p.stdin.fileno(), 1031, 4096)   # F_SETPIPE_SZ
        except Exception:
            pass

    def start(self):
        self.run = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        try:
            while self.run:
                self.p.stdin.write(self.mixer.render(self.bs).tobytes())
        except (BrokenPipeError, ValueError):
            pass

    def stop(self):
        self.run = False
        time.sleep(0.05)
        try:
            self.p.stdin.close(); self.p.terminate()
        except Exception:
            pass


def make_backend(name: str, mixer: PolyMixer):
    if name == "sounddevice":
        return SoundDeviceBackend(mixer)
    if name == "aplay":
        return AplayBackend(mixer)
    raise ValueError(name)
