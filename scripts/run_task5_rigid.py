"""Task 5 redesigned: operational-space touch control on the rigid-body model (reduced to wrist + one finger).

    python scripts/run_task5_rigid.py     -> data/trajectories/task5rb_*.npz, data/task5_rigid.json,
                                              docs/images/task5_rigid/*.png
Does not touch data/metrics.json (the simplified-model results of scripts/run_tasks.py stay as the baseline).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np

from hand_robot.control.opspace import OpSpaceGains
from hand_robot.dynamics.rigid_body import RigidBodyDynamics
from hand_robot.tasks import task5_rigid as r5
from hand_robot.tasks.common import Scene
from hand_robot.tasks.task5_touch import PROFILES

OUT_T = ROOT / "data" / "trajectories"
OUT_F = ROOT / "docs" / "images" / "task5_rigid"
COL = {"pp": "tab:blue", "mf": "tab:orange", "ff": "tab:red"}


def main():
    t0 = time.time()
    OUT_T.mkdir(parents=True, exist_ok=True)
    OUT_F.mkdir(parents=True, exist_ok=True)
    scene = Scene()
    cfg = scene.config
    gains = OpSpaceGains.from_config(cfg)
    tf_sens = float(cfg["touch_rigid"]["tau_max_finger_sensitivity"])
    P = {p.name: p for p in PROFILES}
    out = {"gains": gains.__dict__.copy(), "dt_ctrl": cfg["touch_rigid"]["dt_ctrl"], "dt_plant": cfg["touch_rigid"]["dt_plant"],
           "tau_max_nominal": dict(cfg["actuators"])}

    # 1) static feasibility under the nominal and the sensitivity actuator limits
    out["static_limit_nominal"] = r5.static_force_limit(scene)
    out["static_limit_finger_%.2f" % tf_sens] = r5.static_force_limit(scene, tau_max_finger=tf_sens)
    print("static limit (nominal): %.2f N, limiting joint %d" % (out["static_limit_nominal"]["F_max_N"],
                                                                 out["static_limit_nominal"]["limiting_joint"]))

    # 2) the three profiles with the nominal torque limits, and ff with the declared finger limit
    runs = {}
    for tag, kw in (("", {}), ("_tau%.2f" % tf_sens, {"tau_max_finger": tf_sens})):
        profs = PROFILES if tag == "" else [P["ff"]]
        for res in r5.run(scene, profiles=profs, gains=gains, tag=tag, **kw):
            res.save(OUT_T / f"{res.name}.npz")
            runs[res.name] = res
            out.setdefault("runs", {})[res.name] = res.metrics
            m = res.metrics
            print(f"{res.name:20s} Fpk={m['F_peak_N']:5.2f}N rmsHold={m['F_rms_error_pct']:5.1f}% "
                  f"rmsContact={m['F_rms_error_contact_pct']:5.1f}% sat={m['saturated_pct']:4.1f}% "
                  f"margin={m['joint_limit_margin_deg']:6.1f}deg", flush=True)

    # 3) sensitivity: model mismatch (mf) and contact damping (ff with the declared finger limit)
    sens = {}
    for mm in (0.0, 0.15, 0.30):
        g = OpSpaceGains(**{**gains.__dict__, "model_mismatch": mm})
        sens[f"mf_mismatch_{mm:.2f}"] = r5.run(scene, profiles=[P["mf"]], gains=g)[0].metrics
    for cf in (5.0, 15.0, 30.0):
        g = OpSpaceGains(**{**gains.__dict__, "c_force": cf})
        sens[f"ff_tau{tf_sens:.2f}_c_force_{cf:.0f}"] = r5.run(scene, profiles=[P["ff"]], gains=g,
                                                                tau_max_finger=tf_sens)[0].metrics
    out["sensitivity"] = sens

    # 4) baseline: simplified-model results of run_tasks.py (kept unchanged)
    mj = ROOT / "data" / "metrics.json"
    if mj.exists():
        base = json.load(open(mj))
        out["baseline_simplified_model"] = {k: v for k, v in base.items() if k.startswith("task5_touch_")}
    json.dump(out, open(ROOT / "data" / "task5_rigid.json", "w"), indent=1, default=float)

    # figures ---------------------------------------------------------------------------------------
    fig, ax = plt.subplots(2, 3, figsize=(13, 6.5), sharex="col")
    for k, name in enumerate(("pp", "mf", "ff")):
        rb = runs[f"task5rb_{name}" if name != "ff" else f"task5rb_ff_tau{tf_sens:.2f}"]
        old = np.load(OUT_T / f"task5_touch_{name}.npz", allow_pickle=True)
        t, F, Fd = rb.t[1:], rb.extras["Fz"], rb.extras["F_des"]
        i_c = int(np.argmax(F > 0.05))
        io = int(np.argmax(old["x_Fz"] > 0.05))
        sl = slice(max(i_c - 100, 0), i_c + int(0.75 / 5e-4))
        so = slice(max(io - 20, 0), io + 150)
        a = ax[0, k]
        a.plot(1e3 * (old["t"][1:][so] - old["t"][1:][io]), old["x_Fz"][so], color="gray", lw=1.0, label="modello semplificato (5 ms)")
        a.plot(1e3 * (t[sl] - t[i_c]), F[sl], color=COL[name], lw=1.4, label="corpi rigidi + spazio operativo")
        a.plot(1e3 * (t[sl] - t[i_c]), Fd[sl], "k--", lw=0.8, label="forza desiderata")
        a.set_title(f"{name}: v={rb.metrics['v_impact_m_s']:.2f} m/s, F={rb.extras['profile'].f_hold:g} N" +
                    (f" (limite dito {tf_sens:g} N m)" if name == "ff" else ""), fontsize=9)
        a.set_ylabel("forza sul tasto [N]")
        a.grid(alpha=0.3)
        if k == 0:
            a.legend(fontsize=7)
        b = ax[1, k]
        b.plot(1e3 * (t[sl] - t[i_c]), 1e3 * rb.extras["z"][sl], color=COL[name])
        b.axhline(-8, color="k", ls=":", lw=0.8)
        b.set_ylabel("quota punta [mm]")
        b.set_xlabel("tempo dal contatto [ms]")
        b.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_F / "task5_rigid_force_comparison.png", dpi=150)
    plt.close(fig)
    json_t = time.time() - t0
    print(f"done in {json_t:.0f}s -> data/task5_rigid.json, {OUT_F}")


if __name__ == "__main__":
    main()
