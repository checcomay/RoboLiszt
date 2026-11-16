"""Control study of chapter 4 on the UNIFIED dynamic model (reduced rigid-body: wrist + 1 finger, exact Newton-Euler
inertias, implicit viscous term). Joint space vs operational space; position vs impedance vs hybrid control.

Each experiment is a separate part (each fits in a few minutes) cached in data/_study_parts:

    python scripts/run_control_study.py --only E1,E2a       # run parts
    python scripts/run_control_study.py --merge --plot      # merge parts -> data/control_study.json + figures (Italian)

Same plant, contact and actuator limits as scripts/run_task5_rigid.py (1 kHz control, 2 kHz plant).
tau_max of the finger is STUDY_TAU_FINGER (0.25 N*m, as in the ff run of task 5) so that no actuator saturates.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import sys
import time
from functools import partial
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from scipy.interpolate import CubicSpline

from hand_robot.control.baselines import JointSpaceController, MotionController
from hand_robot.control.hybrid import TouchProfile
from hand_robot.control.opspace import OpSpaceGains, OperationalSpaceTouchController
from hand_robot.dynamics.reduced import ReducedPlant, ReducedRigidBody
from hand_robot.dynamics.rigid_body import RigidBodyDynamics
from hand_robot.planning import trajectory as tj
from hand_robot.tasks import task5_rigid as t5
from hand_robot.tasks.common import Scene

STUDY_TAU_FINGER = 0.25
FD = 2.5
PARTS = ROOT / "data" / "_study_parts"


class Study:
    def __init__(self, finger="middle", dt_ctrl=1e-3, dt_plant=5e-4):
        self.sc = Scene()
        self.kb = self.sc.kb
        self.q_start, self.note, self.branch, self.base = t5.start_posture(self.sc, finger)
        self.finger = finger
        self.rbd = RigidBodyDynamics(self.sc.robot, self.sc.config)
        self.model = ReducedRigidBody(self.rbd, self.branch, self.q_start)
        self.qa0 = self.q_start[self.model.idx].copy()
        T = self.model.kin(self.qa0)
        self.p0 = self.model.tip(T) + self.base
        self.xy = self.kb.surface_point(self.note, finger)[:2]
        self.dt_c, self.dt_p = dt_ctrl, dt_plant
        self.d_eq = (FD + self.kb.k_stop * self.kb.press_depth) / (self.kb.k_key + self.kb.k_stop)

    # ---------------------------------------------------------------- references
    def descend_ref(self, z_target, v_a, ramp=0.15):
        z0, xy = self.p0[2], self.xy

        def zt(t):
            if t <= 0:
                return z0, 0.0
            if t < ramp:
                u = t / ramp
                d, v = v_a * ramp * (2.5 * u**4 - 3 * u**5 + u**6), v_a * float(tj.quintic(u))
            else:
                d, v = v_a * ramp * 0.5 + v_a * (t - ramp), v_a
            z = z0 - d
            return (z_target, 0.0) if z <= z_target else (z, -v)

        def fn(t):
            z, vz = zt(t)
            return np.array([xy[0], xy[1], z]), np.array([0.0, 0.0, vz])
        return fn

    # ---------------------------------------------------------------- simulation
    def simulate(self, ctrl, dz=0.0, T=2.4, k_scale=1.0):
        kb2 = copy.copy(self.kb)
        kb2.z_surface = self.kb.z_surface + dz              # key height unknown to the controller
        kb2.k_key = self.kb.k_key * k_scale
        model, base = self.model, self.base
        plant = ReducedPlant(model)
        n_sub = int(round(self.dt_c / self.dt_p))
        contact = lambda tip, v: np.array([0.0, 0.0, kb2.contact_force(tip[2] + base[2], v[2])])
        qa, qd = self.qa0.copy(), np.zeros(model.n)
        keys = ("t", "Fz", "x", "y", "z", "vz")
        lg = {k: [] for k in keys}
        for k in range(int(round(T / self.dt_c))):
            t = k * self.dt_c
            Tk = model.kin(qa)
            tip = model.tip(Tk) + base
            v = model.jacobian(Tk) @ qd
            tau = ctrl.compute(t, qa, qd, {"Fz": kb2.contact_force(tip[2], v[2])})
            for s in range(n_sub):
                qa, qd, qdd, F = plant.step(qa, qd, tau, contact, self.dt_p)
                Tp = model.kin(qa)
                tp = model.tip(Tp) + base
                vp = model.jacobian(Tp) @ qd
                for key, val in zip(keys, (t + (s + 1) * self.dt_p, F[2], tp[0], tp[1], tp[2], vp[2])):
                    lg[key].append(val)
            if not np.all(np.isfinite(qa)) or np.abs(qd).max() > 1e3:
                lg = {k: np.asarray(v) for k, v in lg.items()}
                lg["diverged"] = True
                return lg
        lg = {k: np.asarray(v) for k, v in lg.items()}
        lg["diverged"] = False
        return lg

    # ---------------------------------------------------------------- controllers
    def mk_position(self, v, K=1000.0):
        return MotionController(self.model, self.sc.config, self.base, self.qa0,
                                self.descend_ref(-(self.d_eq + FD / K), v), form="force", K=K, zeta=1.0,
                                tau_max_finger=STUDY_TAU_FINGER)

    def mk_impedance(self, v, zeta=1.0, K=100.0, contact_aware=False):
        return MotionController(self.model, self.sc.config, self.base, self.qa0,
                                self.descend_ref(-(self.d_eq + FD / K), v), form="force", K=K, zeta=zeta,
                                k_env=self.kb.k_key, zero_vref_on_contact=contact_aware, tau_max_finger=STUDY_TAU_FINGER)

    def mk_hybrid(self, v, hold=1.2, F=FD, gains=None, mismatch=None):
        prof = TouchProfile("x", v, F, 0.12, t_hold=hold, t_release=0.15)
        if gains is None:
            gains = OpSpaceGains.from_config(self.sc.config)
        if mismatch is not None:
            gains.model_mismatch = mismatch
        return OperationalSpaceTouchController(self.model, self.sc.config, self.base, self.qa0, xy_ref=self.xy,
                                               z_hover=self.kb.z_surface + self.kb.safe_height,
                                               z_floor=self.kb.z_surface - self.kb.press_depth - 0.002,
                                               profile=prof, gains=gains, tau_max_finger=STUDY_TAU_FINGER)


def steady(F, t, a=1.4, b=1.9):
    m = (t >= a) & (t <= b)
    return float(F[m].mean()), float(F[m].std())


def plateau_window(F_des, f_hold, dt, length=0.3):
    idx = np.where(np.asarray(F_des) >= 0.999 * f_hold)[0]
    b = idx[-1]
    return slice(max(b - int(length / dt), idx[0]), b)


def hyb_plateau(S, ctrl, lg):
    """Mean/std of the force over the last 0.3 s of the force plateau (no release inside the window)."""
    w = plateau_window(ctrl.log["F_des"], ctrl.p.f_hold, S.dt_c)
    return steady(lg["Fz"], lg["t"], w.start * S.dt_c, w.stop * S.dt_c)


def hybrid_gains(S, wn_xy=None, mismatch=None):
    g = OpSpaceGains.from_config(S.sc.config)
    if wn_xy is not None:
        g.wn_xy = wn_xy
    if mismatch is not None:
        g.model_mismatch = mismatch
    return g


def make_cases(S):
    return {"position K=1000": lambda v: S.mk_position(v),
            "impedance K=100": lambda v: S.mk_impedance(v),
            "impedance K=100, contact-aware": lambda v: S.mk_impedance(v, contact_aware=True),
            "hybrid": lambda v: S.mk_hybrid(v, gains=hybrid_gains(S))}


def run_case(S, MK, name, v, dz=0.0, T=2.4, k_scale=1.0):
    c = MK[name](v)
    lg = S.simulate(c, dz, T, k_scale)
    m, sd = hyb_plateau(S, c, lg) if name == "hybrid" else steady(lg["Fz"], lg["t"], 1.4, 1.9)
    return lg, c, m, sd


# ---------------------------------------------------------------------------------------------- experiments
def exp_E1(S, log):
    z0, z1, T1 = S.p0[2], 0.012, 0.8

    def zr(t):
        u = min(max(t / T1, 0.0), 1.0)
        ds = (30 * u**2 - 60 * u**3 + 30 * u**4) / T1 if 0 < t < T1 else 0.0
        return z0 + (10 * u**3 - 15 * u**4 + 6 * u**5) * (z1 - z0), ds * (z1 - z0)
    ref = lambda t: (np.array([S.xy[0], S.xy[1], zr(t)[0]]), np.array([0.0, 0.0, zr(t)[1]]))
    tgrid = np.arange(0, 2.0 + 0.02, 0.005)
    tg = [{S.branch: np.array([S.xy[0], S.xy[1], zr(t)[0]])} for t in tgrid]
    lock = tuple(j for j in range(1, 21) if j not in S.sc.robot.branches[S.branch])
    qs = S.sc.ik_series(tg, S.q_start, np.tile(S.base, (len(tgrid), 1)), locked=lock)[:, S.model.idx]
    cs = CubicSpline(tgrid, qs)
    qfn = lambda t: (cs(min(t, 2.0)), cs(min(t, 2.0), 1), cs(min(t, 2.0), 2))
    tm = STUDY_TAU_FINGER
    cases = [("joint computed-torque PD (wn=40)", JointSpaceController(S.model, S.sc.config, qfn, wn=40.0, tau_max_finger=tm)),
             ("operational PD (wn=40)", MotionController(S.model, S.sc.config, S.base, S.qa0, ref, form="accel", wn=40.0, ki=-1, tau_max_finger=tm)),
             ("operational PD+I (wn=40)", MotionController(S.model, S.sc.config, S.base, S.qa0, ref, form="accel", wn=40.0, ki=0.0, tau_max_finger=tm)),
             ("operational PD (wn=100)", MotionController(S.model, S.sc.config, S.base, S.qa0, ref, form="accel", wn=100.0, ki=-1, tau_max_finger=tm))]
    e1, ser = {}, {"t_grid": tgrid, "z_ref": np.array([zr(t)[0] for t in tgrid])}
    for k, (name, c) in enumerate(cases):
        lg = S.simulate(c, 0.0, 2.0)
        e = lg["z"] - np.array([zr(t)[0] for t in lg["t"]])
        exy = np.hypot(lg["x"] - S.xy[0], lg["y"] - S.xy[1])
        e1[name] = {"z_rms_mm": 1e3 * float(np.sqrt(np.mean(e**2))), "z_max_mm": 1e3 * float(np.abs(e).max()),
                    "z_final_mm": 1e3 * float(e[-1]), "xy_max_mm": 1e3 * float(exy.max()), "saturation_steps": int(c.n_sat)}
        ser[f"c{k}__t"], ser[f"c{k}__e"] = lg["t"][::20], 1e3 * e[::20]
        log(f"E1 {name}: rms {e1[name]['z_rms_mm']:.3f} mm")
    ser["names"] = np.array([n for n, _ in cases])
    return {"E1_free_space": e1}, {"E1": ser}


def exp_E2(S, log, K=1000):
    depths = [0, 2, 4, 6, 8, 10, 12]
    row, ser = [], {"depths": np.array(depths)}
    for d in depths:
        c = MotionController(S.model, S.sc.config, S.base, S.qa0, S.descend_ref(-d / 1e3, 0.05), form="force", K=K,
                             zeta=1.0, tau_max_finger=STUDY_TAU_FINGER)
        lg = S.simulate(c, 0.0, 2.4)
        row.append(steady(lg["Fz"], lg["t"], 1.6, 2.3)[0])
        if K == 3000 and d in (8, 10, 12):
            ser[f"ts{d}__t"], ser[f"ts{d}__F"] = lg["t"][::10], lg["Fz"][::10]
    ser[f"K{K}"] = np.array(row)
    log(f"E2 K={K}: {np.round(row, 2).tolist()}")
    return {"E2_steady_force_vs_depth": {"depth_mm": depths, f"K={K}": row}}, {"E2": ser}


def exp_E3a(S, log, names=None):
    MK = {k: v for k, v in make_cases(S).items() if names is None or k in names}
    dzs = [-2e-3, -1e-3, 0.0, 1e-3, 2e-3]
    e3, ser = {}, {"dz": np.array(dzs) * 1e3}
    for name in MK:
        rows = []
        for dz in dzs:
            lg, c, m, sd = run_case(S, MK, name, 0.12, dz)
            rows.append({"dz_mm": 1e3 * dz, "F_mean": m, "F_std": sd, "F_peak": float(lg["Fz"].max())})
        e3[name] = rows
        ser[f"dz__{name}"] = np.array([r["F_mean"] for r in rows])
        log(f"E3a {name}: {[round(r['F_mean'], 2) for r in rows]} peaks {[round(r['F_peak'], 2) for r in rows]}")
    return {"E3_key_height_error": e3}, {"E3a": ser}


def exp_E3b(S, log, names=None):
    MK = {k: v for k, v in make_cases(S).items() if names is None or k in names}
    vs = [0.05, 0.10, 0.15, 0.20, 0.25]
    e3v = {name: [float(run_case(S, MK, name, v)[0]["Fz"].max()) for v in vs] for name in MK}
    ser = {"vs": np.array(vs), **{f"vs__{k}": np.array(v) for k, v in e3v.items()}}
    for name, pk in e3v.items():
        log(f"E3b peak vs speed {name}: {np.round(pk, 2).tolist()}")
    return {"E3_peak_vs_speed": {"v_a": vs, **e3v}}, {"E3b": ser}


def exp_E3c(S, log):
    zs, zp, ser = {}, {}, {}
    for zeta in (0.5, 1.0, 2.0):
        lg = S.simulate(S.mk_impedance(0.25, zeta, contact_aware=True), 0.0, 2.4)
        zs[zeta] = float(lg["Fz"].max())
        ser[f"zeta{zeta}__t"], ser[f"zeta{zeta}__F"] = lg["t"][::10], lg["Fz"][::10]
        zp[zeta] = float(S.simulate(S.mk_impedance(0.25, zeta), 0.0, 2.4)["Fz"].max())
    lgP = S.simulate(S.mk_position(0.25), 0.0, 2.4)
    lgI = S.simulate(S.mk_impedance(0.25), 0.0, 2.4)
    ser["pos__t"], ser["pos__F"] = lgP["t"][::10], lgP["Fz"][::10]
    ser["plain__t"], ser["plain__F"] = lgI["t"][::10], lgI["Fz"][::10]
    log(f"E3c damping contact-aware {zs} plain {zp}")
    return ({"E3_damping_ratio_v0.25": {f"zeta={z}": {"F_peak_contact_aware": zs[z], "F_peak_plain": zp[z]} for z in zs}},
            {"E3c": ser})


def exp_E4(S, log):
    e4, ser = {}, {}
    for wn in (40.0, 100.0):
        for prof in t5.PROFILES:
            c = OperationalSpaceTouchController(S.model, S.sc.config, S.base, S.qa0, xy_ref=S.xy,
                                                z_hover=S.kb.z_surface + S.kb.safe_height,
                                                z_floor=S.kb.z_surface - S.kb.press_depth - 0.002, profile=prof,
                                                gains=hybrid_gains(S, wn), tau_max_finger=STUDY_TAU_FINGER)
            lg = S.simulate(c, 0.0, 2.6)
            exy = np.hypot(lg["x"] - S.xy[0], lg["y"] - S.xy[1])
            e4[f"wn_xy={int(wn)} {prof.name}"] = {"lateral_max_mm": 1e3 * float(exy.max()),
                                                   "lateral_rms_mm": 1e3 * float(np.sqrt(np.mean(exy**2))),
                                                   "F_peak": float(lg["Fz"].max())}
            for key, val in (("t", lg["t"][::10]), ("e", 1e3 * exy[::10]), ("F", lg["Fz"][::10])):
                ser[f"{int(wn)}_{prof.name}__{key}"] = val
            log(f"E4 wn_xy={wn:.0f} {prof.name}: lateral {1e3*exy.max():.2f} mm, peak {lg['Fz'].max():.2f} N")
    MK = make_cases(S)
    lg, c, _, _ = run_case(S, MK, "impedance K=100, contact-aware", 0.25, T=2.6)
    exy = np.hypot(lg["x"] - S.xy[0], lg["y"] - S.xy[1])
    e4["impedance contact-aware v=0.25"] = {"lateral_max_mm": 1e3 * float(exy.max()), "lateral_rms_mm": 1e3 * float(np.sqrt(np.mean(exy**2)))}
    return {"E4_hybrid_lateral": e4}, {"E4": ser}


def exp_E5(S, log):
    """Robustness on conditions NOT used for tuning: gravity mismatch, key stiffness, other fingers."""
    e5 = {}

    def one(St, label, c, k_scale=1.0):
        lg = St.simulate(c, 0.0, 2.4, k_scale)
        m, sd = hyb_plateau(St, c, lg)
        e5[label] = {"F_mean": m, "F_std": sd, "F_peak": float(lg["Fz"].max()),
                     "lateral_max_mm": 1e3 * float(np.hypot(lg["x"] - St.xy[0], lg["y"] - St.xy[1]).max()),
                     "saturation_steps": int(c.n_sat)}
        log(f"E5 {label}: {m:.2f} N")
    for mm in (0.0, 0.30):
        one(S, f"gravity mismatch {int(mm*100)}%", S.mk_hybrid(0.12, gains=hybrid_gains(S, None, mm)))
    for ks in (0.7, 1.3):
        one(S, f"key stiffness x{ks}", S.mk_hybrid(0.12, gains=hybrid_gains(S)), k_scale=ks)
    for fng in ("index", "ring"):
        S2 = Study(finger=fng)
        one(S2, f"finger {fng} ({S2.note})", S2.mk_hybrid(0.12, gains=hybrid_gains(S2)))
    return {"E5_robustness_hybrid": e5}, {}


EXPERIMENTS = {"E1": exp_E1,
               "E2a": partial(exp_E2, K=100), "E2b": partial(exp_E2, K=1000), "E2c": partial(exp_E2, K=3000),
               "E3a1": partial(exp_E3a, names=("position K=1000", "impedance K=100")),
               "E3a2": partial(exp_E3a, names=("impedance K=100, contact-aware", "hybrid")),
               "E3b1": partial(exp_E3b, names=("position K=1000", "impedance K=100")),
               "E3b2": partial(exp_E3b, names=("impedance K=100, contact-aware", "hybrid")),
               "E3c": exp_E3c, "E4": exp_E4, "E5": exp_E5}


def run_parts(names, force=False):
    S = Study()
    PARTS.mkdir(parents=True, exist_ok=True)
    T0 = time.time()
    log = lambda m: print(f"[{time.time()-T0:6.0f}s] {m}", flush=True)
    for n in names:
        if (PARTS / f"{n}.pkl").exists() and not force:
            log(f"{n}: cached")
            continue
        r, s = EXPERIMENTS[n](S, log)
        pickle.dump((r, s), open(PARTS / f"{n}.pkl", "wb"))
        log(f"{n}: saved")


def _deep_merge(a, b):
    for k, v in b.items():
        if k in a and isinstance(a[k], dict) and isinstance(v, dict):
            _deep_merge(a[k], v)
        else:
            a[k] = v
    return a


def merge():
    res, series = {}, {}
    for n in EXPERIMENTS:
        r, s = pickle.load(open(PARTS / f"{n}.pkl", "rb"))
        _deep_merge(res, r)
        _deep_merge(series, s)
    (ROOT / "data" / "control_study.json").write_text(json.dumps(res, indent=2, default=float))
    pickle.dump(series, open(ROOT / "data" / "control_study_series.pkl", "wb"))
    return res, series


def plot_all():
    import matplotlib.pyplot as plt
    from hand_robot.visualization.figures import FigureManager
    ser = pickle.load(open(ROOT / "data" / "control_study_series.pkl", "rb"))
    fm = FigureManager(ROOT / "docs" / "images")
    it = {"joint computed-torque PD (wn=40)": "giunti, coppia calcolata (ωn=40 rad/s)",
          "operational PD (wn=40)": "operativo PD (ωn=40 rad/s)",
          "operational PD+I (wn=40)": "operativo PD+I (ωn=40 rad/s)",
          "operational PD (wn=100)": "operativo PD (ωn=100 rad/s)"}
    e1 = ser["E1"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(e1["t_grid"], 1e3 * e1["z_ref"], "k")
    ax[0].set_xlabel("tempo [s]"); ax[0].set_ylabel("quota di riferimento del polpastrello [mm]"); ax[0].set_title("riferimento (quintico, da 30 a 12 mm)")
    for k, n in enumerate(e1["names"]):
        ax[1].plot(e1[f"c{k}__t"], e1[f"c{k}__e"], label=it[str(n)])
    ax[1].set_xlabel("tempo [s]"); ax[1].set_ylabel("errore di quota [mm]"); ax[1].legend(fontsize=7); ax[1].set_title("errore di inseguimento in spazio libero")
    fig.tight_layout(); fm.save(fig, "ctrl_freespace", "control")
    e2 = ser["E2"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for K in (100, 1000, 3000):
        ax[0].plot(e2["depths"], e2[f"K{K}"], "o-", label=f"K = {K} N/m")
    ax[0].axvline(8, color="k", ls="--", lw=0.8); ax[0].text(8.2, 0.4, "fondo corsa", fontsize=8)
    ax[0].set_xlabel("profondità comandata sotto la superficie [mm]"); ax[0].set_ylabel("forza di contatto a regime [N]")
    ax[0].set_title("controllo di posizione: la forza è fissata dal riferimento"); ax[0].legend(fontsize=8)
    for d in (8, 10, 12):
        ax[1].plot(e2[f"ts{d}__t"], e2[f"ts{d}__F"], label=f"{d} mm")
    ax[1].set_xlabel("tempo [s]"); ax[1].set_ylabel("forza di contatto [N]"); ax[1].legend(title="profondità comandata", fontsize=8)
    ax[1].set_title("K = 3000 N/m, avvicinamento 0,05 m/s")
    fig.tight_layout(); fm.save(fig, "ctrl_position_contact", "control")
    names = {"position K=1000": "posizione K=1000", "impedance K=100": "impedenza K=100",
             "impedance K=100, contact-aware": "impedenza K=100, con rilevamento contatto", "hybrid": "ibrido"}
    a, b, c = ser["E3a"], ser["E3b"], ser["E3c"]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for z in (0.5, 1.0, 2.0):
        ax[0].plot(c[f"zeta{z}__t"], c[f"zeta{z}__F"], label=f"con rilevamento contatto, ζ = {z}")
    ax[0].plot(c["plain__t"], c["plain__F"], "k--", lw=1, label="impedenza semplice, ζ = 1")
    ax[0].plot(c["pos__t"], c["pos__F"], "k:", lw=1.2, label="posizione K=1000")
    ax[0].set_xlim(0.2, 1.6); ax[0].set_xlabel("tempo [s]"); ax[0].set_ylabel("forza di contatto [N]"); ax[0].legend(fontsize=7)
    ax[0].set_title("impedenza K=100, avvicinamento 0,25 m/s")
    for k, lab in names.items():
        ax[1].plot(a["dz"], a[f"dz__{k}"], "o-", label=lab)
    ax[1].axhline(FD, color="k", ls=":", lw=0.8)
    ax[1].set_xlabel("errore sull'altezza del tasto, ignoto al controllore [mm]"); ax[1].set_ylabel("forza di contatto a regime [N]")
    ax[1].set_title("robustezza all'incertezza sull'altezza del tasto"); ax[1].legend(fontsize=7)
    for k, lab in names.items():
        ax[2].plot(b["vs"], b[f"vs__{k}"], "o-", label=lab)
    ax[2].set_xlabel("velocità di avvicinamento [m/s]"); ax[2].set_ylabel("picco di forza di contatto [N]"); ax[2].legend(fontsize=7)
    ax[2].set_title("urto: picco di forza in funzione della velocità")
    fig.tight_layout(); fm.save(fig, "ctrl_impedance", "control")
    e4 = ser["E4"]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    for k, wn in enumerate((40, 100)):
        for prof in ("pp", "mf", "ff"):
            ax[k].plot(e4[f"{wn}_{prof}__t"], e4[f"{wn}_{prof}__e"], label=prof)
        ax[k].set_title(f"ibrido, banda laterale ωn = {wn} rad/s"); ax[k].set_xlabel("tempo [s]"); ax[k].legend(fontsize=8)
        ax[k].set_ylabel("errore laterale del polpastrello [mm]")
        ax[k].set_ylim(0, 12)
    for prof in ("pp", "mf", "ff"):
        ax[2].plot(e4[f"100_{prof}__t"], e4[f"100_{prof}__F"], label=prof)
    ax[2].set_title("forza verticale (ωn = 100 rad/s)"); ax[2].set_xlabel("tempo [s]"); ax[2].set_ylabel("forza di contatto [N]"); ax[2].legend(fontsize=8)
    fig.tight_layout(); fm.save(fig, "ctrl_hybrid_lateral", "control")
    print("figures ->", ROOT / "docs" / "images" / "control")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma separated: " + ",".join(EXPERIMENTS))
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.only:
        run_parts([x for x in a.only.split(",") if x], a.force)
    if a.merge:
        merge()
        print("merged ->", ROOT / "data" / "control_study.json")
    if a.plot:
        plot_all()
