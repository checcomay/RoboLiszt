"""Control study for the control chapter (4.1-4.4): joint vs operational space, position control against a key,
impedance, hybrid. Simplified dynamics model + key contact of Task 5.

    python scripts/run_control_study.py        -> data/control_study.json, docs/images/control/*.png
"""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from hand_robot.control.hybrid import HybridTouchController, TouchProfile
from hand_robot.control.operational import JointPDController, OperationalSpaceController
from hand_robot.dynamics.model import DynamicsModel
from hand_robot.dynamics.simulation import simulate_closed_loop
from hand_robot.planning import trajectory as tj
from hand_robot.tasks.common import FINGERS, Scene
from hand_robot.tasks.task5_touch import KeyContactEnv, start_posture

FINGER, NOTE = "middle", "E4"
BRANCH = FINGERS[FINGER]


class Study:
    def __init__(self):
        self.sc = Scene()
        self.dyn = DynamicsModel(self.sc.robot, self.sc.config)
        self.base = self.sc.base_home
        self.kb = self.sc.kb
        self.q0 = start_posture(self.sc, FINGER)
        self.p0 = (self.sc.robot.end_effectors(self.q0)[BRANCH][:3, 3] + self.base)
        self.dt = self.sc.dt
        self.xy = self.kb.surface_point(NOTE, FINGER)[:2]

    # -------------------------------------------------------------- environments
    def env(self, dz=0.0):
        """Key contact with the key surface shifted by dz (unknown to the controller)."""
        kb2 = copy.copy(self.kb)
        kb2.z_surface = self.kb.z_surface + dz
        return KeyContactEnv(SimpleNamespace(robot=self.sc.robot, kb=kb2), BRANCH, self.base)

    # ------------------------------------------------------------------ references
    def descend_ref(self, z_target, v_a, ramp=0.15):
        """Vertical reference descending from the start height to z_target at speed v_a (smooth start), then holding."""
        z0 = self.p0[2]
        xy = self.xy

        def zt(t):
            """Analytic integral of v_a*quintic(t/ramp): 2.5 u^4 - 3 u^5 + u^6 on the ramp, then constant speed."""
            if t <= 0:
                return z0, 0.0
            if t < ramp:
                u = t / ramp
                d = v_a * ramp * (2.5 * u ** 4 - 3 * u ** 5 + u ** 6)
                v = v_a * float(tj.quintic(u))
            else:
                d = v_a * ramp * 0.5 + v_a * (t - ramp)
                v = v_a
            z = z0 - d
            return (z_target, 0.0) if z <= z_target else (z, -v)

        def fn(t):
            z, vz = zt(t)
            return np.array([xy[0], xy[1], z]), np.array([0.0, 0.0, vz])
        return fn

    def os_controller(self, ref_fn, K, **kw):
        return OperationalSpaceController(self.sc.robot, self.dyn, self.sc.config, BRANCH, self.base, self.q0,
                                          ref_fn, K, **kw)

    def run(self, ctrl, dz=0.0, T=2.4):
        out = simulate_closed_loop(self.dyn, ctrl, self.q0, T, self.dt, self.env(dz))
        out["pos"] = None
        return out

    def tip_traj(self, q):
        r = self.sc.robot
        return np.array([r.end_effectors(x)[BRANCH][:3, 3] + self.base for x in q])


def steady(F, t, a=1.4, b=1.9):
    m = (t[1:] >= a) & (t[1:] <= b)
    return float(F[m].mean()), float(F[m].std())


def plateau_window(F_des, f_hold, dt, length=0.3):
    idx = np.where(np.asarray(F_des) >= 0.999 * f_hold)[0]
    b = idx[-1]
    return slice(max(b - int(length / dt), idx[0]), b)


def quintic_ref(z0, z1, T):
    """Vertical min-jerk reference z(t), zdot(t) with analytic derivative."""
    def f(t):
        u = min(max(t / T, 0.0), 1.0)
        s, ds = 10 * u**3 - 15 * u**4 + 6 * u**5, (30 * u**2 - 60 * u**3 + 30 * u**4) / T if 0 < t < T else 0.0
        return z0 + s * (z1 - z0), ds * (z1 - z0)
    return f


def main():
    import matplotlib.pyplot as plt
    from hand_robot.visualization.figures import FigureManager
    S = Study()
    fm = FigureManager(ROOT / "docs" / "images")
    res = {}

    # ------------------------------------------------------------ E1: joint vs operational space (free space)
    z0, z1, T1 = S.p0[2], 0.012, 0.8
    zr = quintic_ref(z0, z1, T1)
    ref = lambda t: (np.array([S.xy[0], S.xy[1], zr(t)[0]]), np.array([0.0, 0.0, zr(t)[1]]))
    N = int(2.0 / S.dt) + 2
    tg = [{BRANCH: np.array([S.xy[0], S.xy[1], zr(k * S.dt)[0]])} for k in range(N)]
    locked = tuple(j for j in range(1, 21) if j not in S.sc.robot.branches[BRANCH])
    qr = S.sc.ik_series(tg, S.q0, np.tile(S.base, (N, 1)), locked=locked)
    qd_r = np.gradient(qr, S.dt, axis=0)
    qfn = lambda t: (qr[min(int(round(t / S.dt)), N - 1)], qd_r[min(int(round(t / S.dt)), N - 1)])
    cases = [("joint PD (kp=2, kd=0.3)", JointPDController(S.sc.robot, S.dyn, qfn)),
             ("operational PD, K=100", S.os_controller(ref, [100] * 3, zeta=1.0)),
             ("operational PD+I, K=100", S.os_controller(ref, [100] * 3, zeta=1.0, ki=40.0)),
             ("operational PD, K=1000", S.os_controller(ref, [1000] * 3, zeta=1.0))]
    e1, curves = {}, {}
    for name, c in cases:
        o = simulate_closed_loop(S.dyn, c, S.q0, 2.0, S.dt, S.env(0.0))
        tip = S.tip_traj(o["q"])
        tz = np.array([zr(t)[0] for t in o["t"]])
        e = tip[:, 2] - tz
        exy = np.linalg.norm(tip[:, :2] - S.xy, axis=1)
        curves[name] = (o["t"], 1e3 * e)
        e1[name] = {"z_rms_mm": float(1e3 * np.sqrt(np.mean(e**2))), "z_max_mm": float(1e3 * np.abs(e).max()),
                    "z_final_mm": float(1e3 * e[-1]), "xy_max_mm": float(1e3 * exy.max())}
    res["E1_free_space"] = e1
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    tt = np.linspace(0, 2.0, 400)
    ax[0].plot(tt, [1e3 * zr(t)[0] for t in tt], "k"); ax[0].set_xlabel("time [s]"); ax[0].set_ylabel("tip height reference [mm]")
    ax[0].set_title("reference (quintic, 30 → 12 mm)")
    for name, (t, e) in curves.items():
        ax[1].plot(t, e, label=name)
    ax[1].set_xlabel("time [s]"); ax[1].set_ylabel("tip height error [mm]"); ax[1].legend(fontsize=7); ax[1].set_title("tracking error in free space")
    fig.tight_layout(); fm.save(fig, "ctrl_freespace", "control")

    # ------------------------------------------------------------ E2: position control against the key
    depths = [0, 2, 4, 6, 8, 10, 12, 14]
    Ks = [100, 1000, 3000]
    e2 = {}
    for K in Ks:
        row = []
        for dmm in depths:
            c = S.os_controller(S.descend_ref(-dmm / 1e3, 0.05), [K] * 3, zeta=1.0)
            o = simulate_closed_loop(S.dyn, c, S.q0, 2.4, S.dt, S.env(0))
            row.append(steady(o["Fz"], o["t"], 1.6, 2.3)[0])
        e2[K] = row
    res["E2_steady_force_vs_depth"] = {"depth_mm": depths, **{f"K={K}": e2[K] for K in Ks}}
    ts = {}
    for dmm in (8, 10, 12):
        c = S.os_controller(S.descend_ref(-dmm / 1e3, 0.05), [3000] * 3, zeta=1.0)
        o = simulate_closed_loop(S.dyn, c, S.q0, 2.4, S.dt, S.env(0))
        ts[dmm] = (o["t"][1:], o["Fz"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for K in Ks:
        ax[0].plot(depths, e2[K], "o-", label=f"K = {K} N/m")
    ax[0].axvline(1e3 * S.kb.press_depth, color="k", ls="--", lw=0.8); ax[0].text(8.2, 0.5, "key bottom", fontsize=8)
    ax[0].set_xlabel("commanded depth below the key surface [mm]"); ax[0].set_ylabel("steady contact force [N]")
    ax[0].set_title("position control: force set by the position reference"); ax[0].legend(fontsize=8)
    for dmm, (t, F) in ts.items():
        ax[1].plot(t, F, label=f"{dmm} mm")
    ax[1].set_xlabel("time [s]"); ax[1].set_ylabel("contact force [N]"); ax[1].legend(title="commanded depth", fontsize=8)
    ax[1].set_title("K = 3000 N/m, approach 0.05 m/s")
    fig.tight_layout(); fm.save(fig, "ctrl_position_contact", "control")

    # ------------------------------------------------------------ E3: position vs impedance vs hybrid
    Fd = 2.5
    d_eq = (Fd + S.kb.k_stop * S.kb.press_depth) / (S.kb.k_key + S.kb.k_stop)           # penetration giving Fd
    def make_P(v):   return S.os_controller(S.descend_ref(-(d_eq + Fd / 1000), v), [1000] * 3, zeta=1.0)
    def make_I(v, zeta=1.0, zv=False):
        return S.os_controller(S.descend_ref(-(d_eq + Fd / 100), v), [100] * 3, zeta=zeta, k_env=S.kb.k_key,
                               zero_vref_on_contact=zv)
    def make_Ic(v, zeta=1.0): return make_I(v, zeta, True)           # contact-aware impedance
    def make_H(v):
        prof = TouchProfile("x", v, Fd, 0.12, t_hold=1.2, t_release=0.15)
        return HybridTouchController(S.sc.robot, S.dyn, S.sc.config, BRANCH, S.base, S.q0, xy_ref=S.xy,
                                     z_hover=S.kb.z_surface + S.kb.safe_height, z_floor=-S.kb.press_depth - 0.002,
                                     profile=prof, model_mismatch=0.15)
    def run_case(make, v, dz):
        c = make(v)
        o = simulate_closed_loop(S.dyn, c, S.q0, 2.4, S.dt, S.env(dz))
        if isinstance(c, HybridTouchController):
            w = plateau_window(c.log["F_des"], Fd, S.dt)
            m, sd = float(o["Fz"][w].mean()), float(o["Fz"][w].std())
        else:
            m, sd = steady(o["Fz"], o["t"], 1.4, 1.9)
        return o, c, m, sd
    dzs = [-2e-3, -1e-3, 0.0, 1e-3, 2e-3]
    e3 = {}
    MK = (("position K=1000", make_P), ("impedance K=100", make_I), ("impedance K=100, contact-aware", make_Ic), ("hybrid", make_H))
    for name, mk in MK:
        rows = []
        for dz in dzs:
            o, c, m, sd = run_case(mk, 0.12, dz)
            rows.append({"dz_mm": 1e3 * dz, "F_mean": m, "F_std": sd, "F_peak": float(o["Fz"].max())})
        e3[name] = rows
    res["E3_key_height_error"] = e3
    vs = [0.05, 0.10, 0.15, 0.20, 0.25]
    e3v = {}
    for name, mk in MK:
        e3v[name] = [float(run_case(mk, v, 0.0)[0]["Fz"].max()) for v in vs]
    res["E3_peak_vs_speed"] = {"v_a": vs, **e3v}
    zs, zp = {}, {}
    for zeta in (0.5, 1.0, 2.0):
        o, c, m, sd = run_case(lambda v, z=zeta: make_Ic(v, z), 0.12, 0.0)
        zs[zeta] = (o["t"][1:], o["Fz"], float(o["Fz"].max()), m)
        o, c, m, sd = run_case(lambda v, z=zeta: make_I(v, z), 0.12, 0.0)
        zp[zeta] = (float(o["Fz"].max()), m)
    res["E3_damping_ratio"] = {f"zeta={z}": {"F_peak_contact_aware": zs[z][2], "F_peak_plain": zp[z][0], "F_steady": zs[z][3]} for z in zs}

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for zeta, (t, F, pk, m) in zs.items():
        ax[0].plot(t, F, label=f"contact-aware, ζ = {zeta}")
    o, c, m, sd = run_case(make_I, 0.12, 0.0); ax[0].plot(o["t"][1:], o["Fz"], "k--", lw=1, label="plain impedance, ζ = 1")
    o, c, m, sd = run_case(make_P, 0.12, 0.0); ax[0].plot(o["t"][1:], o["Fz"], "k:", lw=1.2, label="position K=1000")
    ax[0].set_xlim(0.2, 1.6); ax[0].set_xlabel("time [s]"); ax[0].set_ylabel("contact force [N]"); ax[0].legend(fontsize=7)
    ax[0].set_title("impedance K=100, approach 0.12 m/s")
    for name, rows in e3.items():
        ax[1].plot([r["dz_mm"] for r in rows], [r["F_mean"] for r in rows], "o-", label=name)
    ax[1].axhline(Fd, color="k", ls=":", lw=0.8)
    ax[1].set_xlabel("key height error unknown to the controller [mm]"); ax[1].set_ylabel("steady contact force [N]")
    ax[1].set_title("robustness to key-height uncertainty"); ax[1].legend(fontsize=8)
    for name in e3v:
        ax[2].plot(vs, e3v[name], "o-", label=name)
    ax[2].set_xlabel("approach speed [m/s]"); ax[2].set_ylabel("peak contact force [N]"); ax[2].legend(fontsize=8)
    ax[2].set_title("impact: peak force vs approach speed")
    fig.tight_layout(); fm.save(fig, "ctrl_impedance", "control")

    # ------------------------------------------------------------ E4: hybrid - lateral position while pressing
    from hand_robot.tasks.task5_touch import PROFILES
    e4 = {}
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for prof in PROFILES:
        c = HybridTouchController(S.sc.robot, S.dyn, S.sc.config, BRANCH, S.base, S.q0, xy_ref=S.xy,
                                  z_hover=S.kb.z_surface + S.kb.safe_height, z_floor=-S.kb.press_depth - 0.002,
                                  profile=prof, model_mismatch=0.15)
        o = simulate_closed_loop(S.dyn, c, S.q0, 2.6, S.dt, S.env(0.0))
        exy = np.hypot(o["x"] - S.xy[0], o["y"] - S.xy[1])
        e4[prof.name] = {"lateral_max_mm": float(1e3 * exy.max()), "lateral_rms_mm": float(1e3 * np.sqrt(np.mean(exy**2)))}
        ax[0].plot(o["t"][1:], 1e3 * exy, label=prof.name)
        ax[1].plot(o["t"][1:], o["Fz"], label=prof.name)
    # same touch with the impedance controller (virtual spring, integral off) for the lateral error
    c = make_Ic(0.25)
    o = simulate_closed_loop(S.dyn, c, S.q0, 2.6, S.dt, S.env(0.0))
    exy = np.hypot(o["x"] - S.xy[0], o["y"] - S.xy[1])
    e4["impedance contact-aware (v=0.25)"] = {"lateral_max_mm": float(1e3 * exy.max()), "lateral_rms_mm": float(1e3 * np.sqrt(np.mean(exy**2)))}
    ax[0].plot(o["t"][1:], 1e3 * exy, "k--", label="impedance (contact-aware) v=0.25")
    ax[0].set_xlabel("time [s]"); ax[0].set_ylabel("lateral fingertip error [mm]"); ax[0].legend(fontsize=8); ax[0].set_title("x-y position held while pressing")
    ax[1].set_xlabel("time [s]"); ax[1].set_ylabel("contact force [N]"); ax[1].legend(fontsize=8); ax[1].set_title("vertical force (hybrid controller)")
    fig.tight_layout(); fm.save(fig, "ctrl_hybrid_lateral", "control")
    res["E4_hybrid_lateral"] = e4

    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data" / "control_study.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
