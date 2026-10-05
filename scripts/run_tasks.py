from hand_robot.config.loader import default_config_path, load_yaml
from hand_robot.tasks.piano import PianoKeyboard, PianoTaskPlanner
from hand_robot.visualization.figures import FigureManager, plot_task_trajectory

config = load_yaml(default_config_path())
keyboard = PianoKeyboard(config)
planner = PianoTaskPlanner(keyboard)
manager = FigureManager("docs/images")

tasks = {
    "greeting": planner.free_space_greeting("finger_1"),
    "single_key": planner.single_key("finger_1", 60),
    "chord": planner.chord(["finger_1", "finger_2", "finger_3"], [60, 64, 67]),
    "octave": planner.octave_jump("finger_1", 60),
    "dynamic": planner.dynamic_key("finger_1", 60, 8.0),
    "piece": planner.piece("finger_1", [60, 62, 64, 67, 69]),
}

for name, task in tasks.items():
    branch = next(iter(task.positions))
    path = plot_task_trajectory(task.t, task.positions, branch, manager)
    print(f"{name}: {path}")
