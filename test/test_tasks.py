from hand_robot.config.loader import default_config_path, load_yaml
from hand_robot.tasks.piano import PianoKeyboard, PianoTaskPlanner


def make_planner():
    config = load_yaml(default_config_path())
    return PianoTaskPlanner(PianoKeyboard(config))


def test_all_six_task_primitives_exist():
    p = make_planner()
    assert p.free_space_greeting("finger_1").positions
    assert p.single_key("finger_1", 60).positions
    assert p.chord(["finger_1", "finger_2"], [60, 64]).positions
    assert p.octave_jump("finger_1", 60).positions
    assert p.dynamic_key("finger_1", 60, 5.0).positions
    assert p.piece("finger_1", [60, 62, 64]).positions
