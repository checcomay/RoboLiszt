# RoboLiszt 🎹🤖: 20-DOF Hand Robot

An anthropomorphic robotic hand simulator built for complex piano playing tasks.

This version reorganizes the project into a `src/` layout and separates
configuration, mechanical model, kinematics, dynamics, control, task
planning, visualization, GUI, and I/O.

## Mechanical topology

The model is deliberately **branched**, not a 20-DOF serial chain:

```text
0 -> 1 -> 2
         ├── 3 -> 4 -> 5 -> 6
         ├── 7 -> 8 -> 9
         ├── 10 -> 11 -> 12
         ├── 13 -> 14 -> 15
         └── 16 -> 17 -> 18 -> 19 -> 20
```

Thus there are exactly 20 revolute joints and five terminal chains.

## Piano-oriented tasks

The task layer implements simulation-level planners for:

1. free-space greeting motion;
2. single-key press;
3. chord execution;
4. octave jump;
5. force-controlled key press;
6. a musical piece combining the previous task primitives.

The code separates *task generation* from *robot execution*. It therefore
does not assume that a real keyboard, actuator or sensor is already connected.

## Inverse kinematics

The IK solver uses damped least squares on the translational endpoint
position. For a branch endpoint:
$$
    q_{next} = q + J^\top (J J^\top + \lambda^2 I)^{-1} e
$$
where $J$ is the $3\times20$ translational Jacobian of the selected end-effector.
Because joints outside the branch have zero influence, the branched structure
is preserved.

The common joints J1 and J2 can influence all fingers.

## Dynamics and control

A lightweight simulation model is included:
$$
    M(q) qdd + C(q,qd) qd + g(q) = tau + J^T F_ext
$$
For the current research scaffold, `M`, damping and gravity are configurable
approximations. Replace them with CAD/identified inertial parameters before
using the controller on hardware.

The force-control layer includes a simple Cartesian impedance/admittance-style
press model. It is intended for simulation and controller development, not
direct deployment on a physical actuator without safety validation.

## Figures

All generated figures are routed through `FigureManager` and are saved under:

    docs/images/

Subfolders are used for architecture, kinematics, dynamics, control and tasks.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
python scripts/run_demo.py
python scripts/run_analysis.py
python scripts/run_tasks.py
```

## Configuration

Edit:

    src/hand_robot/config/robot.yaml

before replacing the placeholder geometric and dynamic parameters with the
final CAD and experimentally identified values.
