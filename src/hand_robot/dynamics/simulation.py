from __future__ import annotations

import numpy as np

from .model import DynamicsModel


def simulate(
    dynamics: DynamicsModel,
    controller,
    q0: np.ndarray,
    duration: float,
    dt: float,
    target_provider=None,
    force_provider=None,
) -> dict[str, np.ndarray]:
    """Forward-Euler simulation of the joint dynamics."""
    steps = int(np.ceil(duration / dt))
    q = dynamics.robot.validate_q(q0).copy()
    qd = np.zeros(20)

    t_log = np.zeros(steps + 1)
    q_log = np.zeros((steps + 1, 20))
    qd_log = np.zeros((steps + 1, 20))
    tau_log = np.zeros((steps + 1, 20))
    energy_log = np.zeros(steps + 1)

    q_log[0] = q
    qd_log[0] = qd
    energy_log[0] = dynamics.energy(q, qd)

    for k in range(steps):
        t = k * dt
        target = target_provider(t) if target_provider else None
        forces = force_provider(t) if force_provider else None

        tau = controller.compute(q, qd, target)
        qdd = dynamics.acceleration(q, qd, tau, forces)

        qd = qd + dt * qdd
        q = q + dt * qd

        t_log[k + 1] = t + dt
        q_log[k + 1] = q
        qd_log[k + 1] = qd
        tau_log[k + 1] = tau
        energy_log[k + 1] = dynamics.energy(q, qd)

    return {
        "t": t_log,
        "q": q_log,
        "qd": qd_log,
        "tau": tau_log,
        "energy": energy_log,
    }
