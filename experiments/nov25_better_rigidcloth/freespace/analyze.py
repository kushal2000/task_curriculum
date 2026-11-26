"""Compare the free-space probe runs (probe.py): summary JSON + PNG plots in outputs/freespace_compare/.

    python experiments/nov25_better_rigidcloth/freespace/analyze.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import signals  # noqa: E402

OUT = "outputs/freespace_compare"
CASES = {"physx": "PhysX (IL 2.3)", "newton": "Newton/MJWarp (IL 3)", "newton_sysid": "Newton + hand_sysid",
         "newton_noselfcoll": "Newton, robot_self_collision=false (diagnostic)"}
COLORS = {"target": "#666666", "physx": "#257de0", "newton": "#f47230", "newton_sysid": "#00b894", "newton_noselfcoll": "#6a4ce2"}
STEP_EVENTS = (1.0, 3.5, 6.0, 8.0)  # s; segment ends at the next event (or the end)


def load():
    runs = {}
    for k in CASES:
        p = f"{OUT}/{k}.npz"
        if os.path.exists(p):
            runs[k] = dict(np.load(p, allow_pickle=False))
    return runs


def step_metrics(t, q, tgt, t0, t1):
    """One step event on a joint: q, tgt (T,) policy-step samples; event at t0, segment to t1."""
    pre = (t >= t0 - 0.25) & (t < t0)
    post = (t >= t1 - 0.3) & (t < t1)
    seg = (t >= t0) & (t < t1)
    q0, qf = q[pre].mean(), q[post].mean()
    tf = tgt[post].mean()
    dq = qf - q0
    res = {"q_start": float(q0), "q_final": float(qf), "target_final": float(tf),
           "steady_state_error_mrad": float((tf - qf) * 1e3)}
    ts, qs, gs = t[seg] - t0, q[seg], tgt[seg]
    if abs(dq) < 1e-4:
        return res
    frac = (qs - q0) / dq
    gfrac = (gs - tgt[pre].mean()) / (tf - tgt[pre].mean())

    def cross(f, lvl):
        i = np.argmax(f >= lvl)
        return float(ts[i]) if f[i] >= lvl else float("nan")

    res["rise_time_10_90_s"] = cross(frac, 0.9) - cross(frac, 0.1)
    res["target_rise_time_10_90_s"] = cross(gfrac, 0.9) - cross(gfrac, 0.1)
    res["delay_50pct_vs_target_s"] = cross(frac, 0.5) - cross(gfrac, 0.5)
    res["overshoot_pct"] = float(max(0.0, (frac.max() - 1.0) * 100.0))
    out = np.abs(frac - 1.0) > 0.02
    res["settling_2pct_s"] = float(ts[np.where(out)[0][-1] + 1]) if out.any() and np.where(out)[0][-1] + 1 < len(ts) else (0.0 if not out.any() else float("nan"))
    return res


def sine_fit(t, x, f):
    """Amplitude and phase of x at frequency f (least squares on sin, cos, 1)."""
    A = np.stack([np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t), np.ones_like(t)], 1)
    c, *_ = np.linalg.lstsq(A, x, rcond=None)
    return np.hypot(c[0], c[1]), np.arctan2(c[1], c[0])


def main():
    runs = load()
    ref = runs["physx"]
    names = [str(n) for n in ref["joint_names"]]
    sig = [str(s) for s in ref["signal_names"]]
    dt = float(ref["policy_dt"])
    T = ref["q"].shape[0]
    t = (np.arange(T) + 1) * dt
    for k, r in runs.items():
        assert [str(n) for n in r["joint_names"]] == names
        assert np.allclose(r["actions"], ref["actions"], atol=1e-5), f"{k}: actions differ"
        assert np.allclose(r["lower"], ref["lower"], atol=1e-5) and np.allclose(r["upper"], ref["upper"], atol=1e-5), k

    summary = {"cases": {k: CASES[k] for k in runs}, "policy_dt": dt, "phys_dt": {k: float(r["phys_dt"]) for k, r in runs.items()},
               "apply_action_calls_per_policy_step": {k: r["phys_q"].shape[0] / r["q"].shape[0] for k, r in runs.items()},
               "info": {k: str(r["info"]) for k, r in runs.items()}, "signals": sig, "joint_names": names,
               "any_reset_during_run": {k: bool(r["done"].any()) for k, r in runs.items()},
               "min_fingertip_z_m": {k: float(r["tips"][..., 2].min()) for k, r in runs.items()},
               "table_top_z_m": 0.53,
               "target_max_abs_diff_vs_physx_rad": {k: float(np.abs(r["tgt"] - ref["tgt"]).max()) for k, r in runs.items()},
               "drive": {k: {d[6:]: dict(zip(names, map(float, r[d]))) for d in r if d.startswith("drive_")} for k, r in runs.items()}}

    # Per-joint RMS tracking error (q - target), per signal, policy and physics rate; and backend RMS diff.
    track, track_phys, diff, diff_vel = {}, {}, {}, {}
    after = t >= 1.0
    for k, r in runs.items():
        e = (r["q"] - r["tgt"])[after]          # (T', E, J)
        track[k] = {s: dict(zip(names, np.round(np.sqrt((e[:, i] ** 2).mean(0)) * 1e3, 3).tolist())) for i, s in enumerate(sig)}
        ep = r["phys_q"] - r["phys_tgt"]
        track_phys[k] = {s: dict(zip(names, np.round(np.sqrt((ep[int(round(ep.shape[0] / T * 60)):, i] ** 2).mean(0)) * 1e3, 3).tolist())) for i, s in enumerate(sig)}
        if k != "physx":
            d = r["q"] - ref["q"]
            diff[k] = {s: dict(zip(names, np.round(np.sqrt((d[:, i] ** 2).mean(0)) * 1e3, 3).tolist())) for i, s in enumerate(sig)}
            dv = r["qd"] - ref["qd"]
            diff_vel[k] = {s: dict(zip(names, np.round(np.sqrt((dv[:, i] ** 2).mean(0)), 4).tolist())) for i, s in enumerate(sig)}
            tipd = np.linalg.norm(r["tips"] - ref["tips"], axis=-1)  # (T, E, 5)
            summary.setdefault("fingertip_rms_diff_vs_physx_mm", {})[k] = {
                s: float(np.sqrt((tipd[:, i] ** 2).mean()) * 1e3) for i, s in enumerate(sig)}
            summary.setdefault("fingertip_max_diff_vs_physx_mm", {})[k] = {
                s: float(tipd[:, i].max() * 1e3) for i, s in enumerate(sig)}
    summary["rms_tracking_error_mrad"] = track
    summary["rms_tracking_error_physics_rate_mrad"] = track_phys
    summary["rms_q_diff_vs_physx_mrad"] = diff
    summary["rms_qd_diff_vs_physx_rad_s"] = diff_vel

    # Compact roll-ups: the signal that exercises each joint.
    hand = names[7:]
    roll = {}
    for k in runs:
        roll[k] = {
            "multisine_hand_rms_track_mrad_mean": float(np.mean([track[k]["multisine"][j] for j in hand])),
            "arm_open_arm_rms_track_mrad_mean": float(np.mean([track[k]["arm_open"][j] for j in names[:7]])),
            "arm_closed_arm_rms_track_mrad_mean": float(np.mean([track[k]["arm_closed"][j] for j in names[:7]])),
            "arm_open_hand_hold_rms_track_mrad_mean": float(np.mean([track[k]["arm_open"][j] for j in hand])),
        }
        if k != "physx":
            roll[k]["multisine_hand_rms_diff_vs_physx_mrad_mean"] = float(np.mean([diff[k]["multisine"][j] for j in hand]))
            roll[k]["arm_open_arm_rms_diff_vs_physx_mrad_mean"] = float(np.mean([diff[k]["arm_open"][j] for j in names[:7]]))
    summary["rollup"] = roll

    # Step responses.
    steps = {}
    for k, r in runs.items():
        steps[k] = {}
        for e, j in enumerate(signals.STEP_JOINTS):
            ji = names.index(j)
            ev = list(STEP_EVENTS) + [t[-1] + dt]
            steps[k][j] = [dict(event_t_s=t0, **step_metrics(t, r["q"][:, e, ji], r["tgt"][:, e, ji], t0, ev[n + 1]))
                           for n, t0 in enumerate(STEP_EVENTS)]
    summary["step_response"] = steps

    # Multisine: gain / phase of q relative to the target at each hand joint's own frequency.
    freqs = np.linspace(0.5, 3.0, len(hand))
    m = t >= 2.0
    fr = {}
    for k, r in runs.items():
        fr[k] = {}
        for i, j in enumerate(hand):
            ji = names.index(j)
            aq, pq = sine_fit(t[m], r["q"][m, 5, ji], freqs[i])
            ag, pg = sine_fit(t[m], r["tgt"][m, 5, ji], freqs[i])
            fr[k][j] = {"f_hz": round(float(freqs[i]), 3), "gain": round(float(aq / ag), 4),
                        "lag_ms": round(float(((pg - pq + np.pi) % (2 * np.pi) - np.pi) / (2 * np.pi * freqs[i]) * 1e3), 2)}
    summary["multisine_q_over_target"] = fr

    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    plots(runs, names, sig, t, track, diff)
    print(json.dumps(roll, indent=1))


def plots(runs, names, sig, t, track, diff):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "font.size": 9, "axes.edgecolor": "#999999"})

    def top_legend(fig, ax):
        h, l = ax.get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", ncol=len(l), frameon=False, fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.97))

    def trace(ax, e, j, key="q", zoom=None, err=False):
        ji = names.index(j)
        ref = runs["physx"]
        if not err:
            ax.plot(t, ref["tgt"][:, e, ji], color=COLORS["target"], lw=1.5, ls="--", label="target")
        for k, r in runs.items():
            y = r[key][:, e, ji] - (r["tgt"][:, e, ji] if err else 0)
            ax.plot(t, y * (1e3 if err else 1), color=COLORS[k], lw=1.5, label=CASES[k])
        ax.set_title(f"{sig[e]}: {j}", loc="left", fontsize=9)
        if zoom:
            ax.set_xlim(*zoom)

    # 1. Step responses (full, and zoomed on the first step).
    fig, axes = plt.subplots(4, 2, figsize=(12, 11), gridspec_kw={"width_ratios": [2.2, 1]})
    for e, j in enumerate(signals.STEP_JOINTS):
        trace(axes[e, 0], e, j)
        trace(axes[e, 1], e, j, zoom=(0.9, 1.9))
        axes[e, 0].set_ylabel("rad")
    axes[-1, 0].set_xlabel("time [s]"); axes[-1, 1].set_xlabel("time [s] (first step)")
    top_legend(fig, axes[0, 0]); fig.savefig(f"{OUT}/steps.png", dpi=130); plt.close(fig)

    # 2. Chirp: q vs target, and the tracking error.
    cj = ("r_thumb_cmc_abd", "r_thumb_cmc_flex", "r_index_finger_mcp_flex", "r_index_finger_pip")
    fig, axes = plt.subplots(4, 2, figsize=(13, 11))
    for i, j in enumerate(cj):
        trace(axes[i, 0], 4, j)
        trace(axes[i, 1], 4, j, err=True)
        axes[i, 0].set_ylabel("rad"); axes[i, 1].set_ylabel("q - target [mrad]")
    axes[-1, 0].set_xlabel("time [s] (chirp 0.5 Hz at 1 s -> 5 Hz at 10 s)"); axes[-1, 1].set_xlabel("time [s]")
    top_legend(fig, axes[0, 0]); fig.savefig(f"{OUT}/chirp.png", dpi=130); plt.close(fig)

    # 3. Arm: q vs target and error for joint2/4/6 with the hand open and closed, plus a held hand joint.
    aj = ("joint2", "joint4", "joint6")
    fig, axes = plt.subplots(4, 2, figsize=(13, 11))
    for c, e in enumerate((6, 7)):
        for i, j in enumerate(aj):
            trace(axes[i, c], e, j, err=True)
            axes[i, c].set_ylabel("q - target [mrad]")
        trace(axes[3, c], e, "r_thumb_cmc_abd", err=True)
        axes[3, c].set_ylabel("q - target [mrad]")
    for ax in axes[-1]:
        ax.set_xlabel("time [s]")
    top_legend(fig, axes[0, 0]); fig.savefig(f"{OUT}/arm.png", dpi=130); plt.close(fig)

    # 4. Per-joint RMS tracking error (multisine for the hand, arm_open for the arm).
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.5))
    ks = list(runs)
    w = 0.8 / len(ks)
    for ax, (s, js) in zip(axes, (("multisine", names[7:]), ("arm_open", names[:7]))):
        x = np.arange(len(js))
        for n, k in enumerate(ks):
            ax.bar(x + (n - (len(ks) - 1) / 2) * w, [track[k][s][j] for j in js], w * 0.92, color=COLORS[k], label=CASES[k])
        ax.set_xticks(x, [j.replace("r_", "").replace("_finger", "") for j in js], rotation=35, ha="right")
        ax.set_ylabel("RMS q - target [mrad]")
        ax.set_title(f"RMS tracking error per joint, signal '{s}' (t > 1 s)", loc="left", fontsize=9)
        ax.grid(axis="x", visible=False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/rms_tracking.png", dpi=130); plt.close(fig)

    # 5. Physics-step resolution of the first thumb_cmc_abd step.
    fig, ax = plt.subplots(figsize=(8, 4))
    ji = names.index("r_thumb_cmc_abd")
    for k, r in runs.items():
        # One sample per _apply_action call: every 1/120 s on PhysX; once per policy step on Newton,
        # whose physics manager runs the decimation inside one CUDA graph (DirectRLEnv.step).
        tp = np.arange(r["phys_q"].shape[0]) * float(r["policy_dt"]) * r["q"].shape[0] / r["phys_q"].shape[0]
        ax.plot(tp, r["phys_q"][:, 1, ji], color=COLORS[k], lw=1.2, marker=".", ms=3, label=CASES[k])
    r = runs["physx"]
    tp = np.arange(r["phys_tgt"].shape[0]) * float(r["phys_dt"])
    ax.plot(tp, r["phys_tgt"][:, 1, ji], color=COLORS["target"], ls="--", lw=1.2, label="target")
    ax.set_xlim(0.98, 1.4); ax.set_xlabel("time [s] (per _apply_action call: PhysX 1/120 s, Newton 1/60 s)"); ax.set_ylabel("rad")
    ax.set_title("r_thumb_cmc_abd first step, per physics step", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/step_physics_rate.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
