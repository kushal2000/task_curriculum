"""Score sysid_replay_physx.py rollouts against the real Wuji Hand 2 recordings (numpy + matplotlib).

    .venv_isaacsim/bin/python scripts/flexiv_wuji/sysid_replay_report.py --data_dir <dir with sysid-2026*/> \
        --replay docs/results/flexiv_right_sysid_replay/replay_dt120.npz [--replay ..._dt1000.npz] \
        --out docs/results/flexiv_right_sysid_replay

Window RMS per (trial, joint) = sqrt(sum of squared angle errors over every window and sample / count),
as sysid_fit.py reports `window_rms_mrad`; samples at the sim steps, the recording linearly
interpolated to the sim time (exact at dt 1 ms). Step steady-state error: on window 0 (starts at the
trial start), the mean target-minus-angle over the last 20 ms of each of the 4 step plateaus
(0.05 up, back, 0.3 up, back).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sysid_replay_data import FRAME_DT, JOINT_NAMES, interp_frames, load_trials  # noqa: E402

GROUPS = ("S1", "S2", "S3", "S4", "thumb_S1", "thumb_S2", "thumb_S3", "thumb_S4")


def gain_key(t):
    return f"{t.kp[t.joints[0]]:g}{'u' if t.mount == 'fingers-up' else 'd'}"


def score(rep, trials):
    """sse/count per (variant, trial, joint) and the sim trajectory on frames for the step analysis."""
    sim_q, dt, n_steps = rep["sim_q"], float(rep["dt"]), int(rep["n_steps"])
    V = sim_q.shape[0]
    sse = np.zeros((V, len(trials), 4))
    cnt = np.zeros((V, len(trials), 4))
    for wi, (ti, start, stop) in enumerate(zip(rep["win_trial"], rep["win_start"], rep["win_stop"])):
        t = trials[ti]
        f = start + np.arange(1, n_steps + 1) * dt / FRAME_DT
        real = interp_frames(t.q, f)  # (n_steps, 4)
        ok = np.isfinite(real) & (f <= stop)[:, None]
        for v in range(V):
            s = sim_q[v, wi, 1:]
            if not np.isfinite(s).all():
                continue
            e = np.where(ok, s - np.nan_to_num(real), 0.0)
            sse[v, ti] += (e ** 2).sum(0)
            cnt[v, ti] += ok.sum(0)
    return sse, cnt


def step_errors(rep, trials, v):
    """(trial, plateau 0..3, joint) steady-state target - angle for the real hand and variant v."""
    sim_q, dt = rep["sim_q"], float(rep["dt"])
    out_real = np.full((len(trials), 4, 4), np.nan)
    out_sim = np.full((len(trials), 4, 4), np.nan)
    for wi, (ti, start) in enumerate(zip(rep["win_trial"], rep["win_start"])):
        if start > 50:
            continue
        t = trials[ti]
        c = t.cmd[:, t.joints]
        change = np.flatnonzero(np.any(np.abs(np.diff(c[:1900], axis=0)) > 1e-6, axis=1)) + 1  # first frame of new target
        if len(change) < 5:
            continue
        ends = change[1:5]  # the plateaus 0.05 up / back / 0.3 up / back end where the next one starts
        tsim = start + np.arange(sim_q.shape[2]) * dt / FRAME_DT
        for p, end in enumerate(ends):
            fr = np.arange(end - 20, end)
            tgt = c[fr]
            real = t.q[fr]
            out_real[ti, p] = np.nanmean(tgt - real, axis=0)
            simf = np.stack([np.interp(fr, tsim, sim_q[v, wi, :, j]) for j in range(4)], axis=1)
            out_sim[ti, p] = (tgt - simf).mean(0)
    return out_real, out_sim


def group_of(t, j):
    return f"thumb_S{j + 1}" if t.finger == "thumb" else f"S{j + 1}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--replay", nargs="+", required=True)
    ap.add_argument("--fit_json", default="assets/flexiv_wuji/wuji_hand2_right_sysid_fit_20261009.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--plot_variants", default="V1,V2,V3")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    trials = load_trials(args.data_dir)
    fit = json.loads(Path(args.fit_json).read_text())
    mj = {}
    for finger, f in fit["fit"].items():
        for tr in f["trials"]:
            mj[tr["file"]] = np.array(tr["window_rms_mrad"])
    gains = sorted({gain_key(t) for t in trials}, key=lambda g: (float(g[:-1]), g[-1]))

    # columns: MuJoCo, then (variant @ dt) for every replay file
    cols = {"MuJoCo fit": np.stack([mj[t.file] for t in trials])}  # (trials, 4) mrad
    reps, all_reps = {}, []
    for path in args.replay:
        rep = dict(np.load(path, allow_pickle=False))
        all_reps.append(rep)
        rep["variants"] = json.loads(str(rep["variants"]))
        rep["report"] = json.loads(str(rep["report"]))
        tag = f"{1 / float(rep['dt']):.0f}"
        rep["tag"] = tag
        reps.setdefault(tag, rep)
        rep["keep"] = [f"{var['name']}@{tag}" not in cols for var in rep["variants"]]
        sse, cnt = score(rep, trials)
        rms = 1000 * np.sqrt(sse / np.maximum(cnt, 1))
        rms[cnt == 0] = np.nan
        for v, var in enumerate(rep["variants"]):
            name = f"{var['name']}@{tag}"
            if name in cols:  # a repeat in a second file at the same dt: check it is reproduced
                print(f"repeat {name}: max |diff| {np.nanmax(np.abs(cols[name] - rms[v])):.3f} mrad")
                continue
            cols[name] = rms[v]

    def pooled(arr, g, gain):
        vals = [arr[i, j] for i, t in enumerate(trials) for j in range(4)
                if group_of(t, j) == g and gain_key(t) == gain and np.isfinite(arr[i, j])]
        return float(np.sqrt(np.mean(np.square(vals)))) if vals else float("nan")

    table = {g: {gain: {c: round(pooled(a, g, gain), 2) for c, a in cols.items()} for gain in gains} for g in GROUPS}
    overall = {c: round(float(np.sqrt(np.nanmean(np.square(a)))), 2) for c, a in cols.items()}
    result = {"window_rms_mrad": table, "window_rms_mrad_all": overall,
              "gain_keys": "kp (A/rad) + u = fingers-up run, d = palm-down run",
              "pooling": "RMS over the fingers' per-trial window RMS (index/middle/ring/pinky for S1-S4)"}

    # ------------------------------------------------------------- step steady-state errors
    steps = {}
    for rep in all_reps:
        tag = rep["tag"]
        for v, var in enumerate(rep["variants"]):
            if not rep["keep"][v]:
                continue
            r, s = step_errors(rep, trials, v)
            name = f"{var['name']}@{tag}"
            if "real" not in steps:
                steps["real"] = r
            steps[name] = s
    plateaus = ("0.05 up", "0.05 back", "0.3 up", "0.3 back")
    step_tab = {}
    for g in GROUPS:
        for gain in gains:
            idx = [(i, j) for i, t in enumerate(trials) for j in range(4) if group_of(t, j) == g and gain_key(t) == gain]
            if not idx:
                continue
            row = {}
            for name, arr in steps.items():
                row[name] = {p: round(1000 * float(np.nanmean([abs(arr[i, k, j]) for i, j in idx])), 1)
                             for k, p in enumerate(plateaus)}
            step_tab.setdefault(g, {})[gain] = row
    result["step_steady_error_abs_mrad"] = step_tab
    (out / "replay_scores.json").write_text(json.dumps(result, indent=1) + "\n")

    # ------------------------------------------------------------- markdown tables
    lines = []
    names = list(cols)
    lines.append("## Window RMS (mrad), sim vs real, pooled over fingers\n")
    for g in GROUPS:
        lines.append(f"### {g}\n")
        lines.append("| kp | " + " | ".join(names) + " |")
        lines.append("|---" * (len(names) + 1) + "|")
        for gain in gains:
            lines.append(f"| {gain} | " + " | ".join(f"{table[g][gain][c]:.1f}" for c in names) + " |")
        lines.append("")
    lines.append("All trials/joints: " + ", ".join(f"{c} {v:.1f}" for c, v in overall.items()) + "\n")
    lines.append("## Step steady-state |target - q| (mrad), last 20 ms of each plateau\n")
    sn = list(steps)
    for g in GROUPS:
        if g not in step_tab:
            continue
        lines.append(f"### {g}\n")
        lines.append("| kp | plateau | " + " | ".join(sn) + " |")
        lines.append("|---" * (len(sn) + 2) + "|")
        for gain, row in step_tab[g].items():
            for p in plateaus:
                lines.append(f"| {gain} | {p} | " + " | ".join(f"{row[n][p]:.1f}" for n in sn) + " |")
        lines.append("")
    (out / "replay_scores.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ------------------------------------------------------------- plots (dt 1/120)
    if "120" in reps:
        plot(reps["120"], trials, args.plot_variants.split(","), out)


def plot(rep, trials, show, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"real": "#0b0b0b", "target": "#9a9994"}
    series = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
    vnames = [v["name"] for v in rep["variants"]]
    show = [s for s in show if s in vnames]
    dt, n = float(rep["dt"]), int(rep["n_steps"])
    picks = [  # (finger, kp, mount, window index within the trial, label)
        ("index", 1.0, "palm-down", 0, "steps"),
        ("index", 1.0, "palm-down", 7, "multisine"),
        ("middle", 3.0, "fingers-up", 0, "steps"),
    ]
    for finger, kp, mount, wk, label in picks:
        ti = next(i for i, t in enumerate(trials) if t.finger == finger and t.mount == mount
                  and np.isclose(t.kp[t.joints[0]], kp))
        wins = [w for w in range(len(rep["win_trial"])) if rep["win_trial"][w] == ti]
        wi = wins[wk]
        t = trials[ti]
        start = int(rep["win_start"][wi])
        fr = np.arange(start, start + 2001)
        ts = np.arange(n + 1) * dt
        fig, axes = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True)
        for ax, j, jn in ((axes[0], 0, "MCP flexion (S1)"), (axes[1], 2, "PIP (S3)")):
            ax.plot((fr - start) * FRAME_DT, t.cmd[fr, t.joints[j]], color=colors["target"], lw=1.2, ls="--",
                    label="target (2 ms lag)")
            ax.plot((fr - start) * FRAME_DT, t.q[fr, j], color=colors["real"], lw=2, label="real")
            for k, s in enumerate(show):
                ax.plot(ts, rep["sim_q"][vnames.index(s), wi, :, j], color=series[k], lw=1.5,
                        label=f"PhysX {s}")
            ax.set_ylabel(f"{jn} [rad]")
            ax.grid(alpha=0.25)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        axes[1].set_xlabel(f"time in window [s] (window starts {start / 1000:.1f} s into the trial)")
        axes[0].legend(loc="best", fontsize=8, frameon=False, ncol=3)
        fig.suptitle(f"{finger}, kp {kp:g} A/rad, {mount}, {label}: real vs PhysX RightReal at dt 1/120", fontsize=11)
        fig.tight_layout()
        p = out / f"replay_{finger}_kp{kp:g}_{mount}_{label}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        print("plot", p)


if __name__ == "__main__":
    main()
