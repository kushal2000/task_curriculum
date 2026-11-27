"""Collate the VBD-iteration convergence runs: hand-free benches (bench.py) and the hand runs (hand.sbatch)."""
import json, glob, os
import numpy as np

R = "outputs/vbd_convergence"
print("== hand-free benches (each vs its 192-iteration run)")
for f in sorted(glob.glob(f"{R}/*_*.jsonl")) + sorted(glob.glob(f"{R}/kd/*.jsonl")):
    rows = [json.loads(l) for l in open(f)]
    if not rows:
        continue
    ref = max(rows, key=lambda r: r["iters"])
    qref = np.asarray(ref["final_q"])
    print(f"-- {rows[0]['material']} / {rows[0]['test']}" + (f" / edge_kd {rows[0]['edge_kd']:g}" if "/kd/" in f else ""))
    for r in sorted(rows, key=lambda r: r["iters"]):
        d = np.linalg.norm(np.asarray(r["final_q"]) - qref, axis=1)
        extra = " ".join(f"{k} {r[k]}" for k in ("droop_mm", "droop_deg", "free_edge_z_mm", "height_mm") if k in r)
        print(f"   iters {r['iters']:3d}: final shape vs 192: mean {1000*d.mean():6.2f} mm, max {1000*d.max():6.2f} mm | "
              f"strain med {r['strain_median']:.3f} max {r['strain_max']:.3f} | residual {r['residual_speed_mm_s']:7.3f} mm/s | "
              f"settle {r['settle_s']} | {r['ms_per_step']} ms/step | {extra} | finite {r['finite']}")

print("== with the hand (stable.yaml, pinned start, deterministic, 4 envs): cloth vs the 192-iteration run")
runs = {}
for d in sorted(glob.glob(f"{R}/hand_it*")):
    it = int(d.split("hand_it")[1])
    eps = sorted(glob.glob(f"{d}/*.npz"))
    if eps:
        runs[it] = [np.load(e, allow_pickle=True) for e in eps]
if runs:
    ref = runs[max(runs)]
    for it in sorted(runs):
        errs, strains = [], []
        for a, b in zip(runs[it], ref):
            ca, cb = np.asarray(a["cloth"], float), np.asarray(b["cloth"], float)
            T = min(len(ca), len(cb), 60)  # the first second: before closed-loop divergence dominates
            errs.append(1000 * np.linalg.norm(ca[:T] - cb[:T], axis=2).mean())
            res = int(round(ca.shape[1] ** 0.5)); sp = 0.10 / (res - 1)
            g = ca.reshape(len(ca), res, res, 3)
            e = np.concatenate([np.linalg.norm(np.diff(g, axis=1), axis=3).reshape(len(ca), -1),
                                np.linalg.norm(np.diff(g, axis=2), axis=3).reshape(len(ca), -1)], 1) / sp
            strains.append(e.max(1).max())
        print(f"   iters {it:3d}: cloth vs 192 over the first second {np.mean(errs):6.2f} mm (per env {np.round(errs,1)}) | worst edge stretch {np.max(strains):.2f}x")
