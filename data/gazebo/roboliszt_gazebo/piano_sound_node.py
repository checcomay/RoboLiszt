"""Piano sound node (ROS 2 Jazzy)  -- NOT EXECUTED: no ROS 2 / Gazebo / audio device where this was written.

    ROBOLISZT_ROOT=/path/to/roboliszt ros2 run roboliszt_gazebo piano_sound_node --ros-args -p use_sim_time:=true \
        -p backend:=sounddevice            # or aplay, or wav (renders out.wav from the SIM time stamps when the node stops)

Listens to the key joints (bridged from Gazebo, topic /world/piano_world/model/piano/joint_state), detects presses with
hand_robot.audio.detector (note = key, intensity = key descent speed -> MIDI velocity) and plays/renders the synthesized
piano. Backends: `sounddevice` (pip install sounddevice, needs PortAudio), `aplay` (ALSA), `wav` (offline, exact timing
even when Gazebo runs slower than real time: realtime backends sound at WALL-clock time, so with --rtf 0.25 the notes are
spaced 4x farther apart than in simulation). Dynamics need joint states at >= 500 Hz (ros2_control update_rate).
"""
import os
import sys
from pathlib import Path

root = os.environ.get("ROBOLISZT_ROOT")
if root:
    sys.path.insert(0, str(Path(root) / "src"))

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from hand_robot.audio.detector import KeyPressDetector, pair_events
from hand_robot.audio.mixer import PolyMixer, make_backend
from hand_robot.audio.synth import PianoSynth, render_events, write_wav
from hand_robot.config.loader import default_config_path, load_yaml
from hand_robot.piano.layout import keys, midi_from_joint


class PianoSound(Node):
    def __init__(self):
        super().__init__("roboliszt_piano_sound")
        self.declare_parameter("backend", "sounddevice")
        self.declare_parameter("out", "piano_session.wav")
        self.declare_parameter("key_topic", "/world/piano_world/model/piano/joint_state")
        cfg = load_yaml(default_config_path())
        s = cfg["sound"]
        self.det = KeyPressDetector.from_config(cfg)
        self.synth = PianoSynth(int(s["sample_rate"]), float(s["a4"]))
        self.rel_t, self.gain = float(s["release_time"]), float(s["master_gain"])
        self.backend_name = self.get_parameter("backend").value
        self.out = self.get_parameter("out").value
        self.events, self.backend = [], None
        if self.backend_name != "wav":
            self.mixer = PolyMixer(self.synth, self.rel_t, self.gain)
            self.get_logger().info("pre-rendering notes ...")
            self.mixer.prewarm(keys())
            self.backend = make_backend(self.backend_name, self.mixer)
            self.backend.start()
        self.create_subscription(JointState, self.get_parameter("key_topic").value, self.on_keys, 200)
        self.get_logger().info(f"piano sound ready, backend={self.backend_name}")

    def on_keys(self, msg: JointState):
        t = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        if t == 0.0:
            t = self.get_clock().now().nanoseconds * 1e-9
        for name, pos in zip(msg.name, msg.position):
            m = midi_from_joint(name)
            if m is None:
                continue
            for e in self.det.update(t, m, pos):
                self.events.append(e)
                if self.backend is not None:
                    (self.mixer.note_on(m, e.velocity) if e.on else self.mixer.note_off(m))
                if e.on:
                    self.get_logger().info(f"note {m} velocity {e.velocity} (key speed {e.speed:.3f} m/s)")

    def shutdown(self):
        if self.backend is not None:
            self.backend.stop()
        if self.backend_name == "wav" and self.events:
            ev = pair_events(self.events)
            t0 = min(e[0] for e in ev)
            x = render_events([(a - t0, m, v, None if b is None else b - t0) for a, m, v, b in ev], self.synth, self.rel_t, 1.5, self.gain)
            write_wav(self.out, x, self.synth.sr)
            self.get_logger().info(f"wrote {self.out} ({len(ev)} notes, {len(x) / self.synth.sr:.1f} s)")


def main():
    rclpy.init()
    node = PianoSound()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
