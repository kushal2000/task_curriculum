"""Tabulate the contact-stiffness benchmark (scripts/cluster/bos14_contact_stiffness_bench.sh).

    .venv_isaaclab3/bin/python scripts/analysis/contact_stiffness_summary.py [--speed_job <id>]

Fold rate = ``done_fold_<shape>`` among finished episodes (held 10 steps at 1.5 cm), as
``fold_pool_eval_summary.py`` counts it; both eval seeds pooled. Also the outcome mix (fall /
timeout / nonfinite) and the median best fold error, so a stiffer contact that makes the plates
jitter or diverge shows up as more than a lower rate. Speed = mean rl_games ``fps step`` over the
second half of each speed run's epochs.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

CFGS = ("base", "stiff2", "stiff4")
SOLREF = {"base": "0.02", "stiff2": "0.01", "stiff4": "0.005 (4 substeps)"}


def _episodes(path: Path) -> tuple[list[dict], dict]:
    r = json.loads(path.read_text())
    extra = r.get("per_env_extra_done", {})
    shapes = sorted(k[len("shape_"):] for k in extra if k.startswith("shape_"))
    best = r.get("fold", {}).get("best_fold_err") or [None] * r["num_envs"]
    recs = []
    for e in range(r["num_envs"]):
        shape = next((s for s in shapes if extra[f"shape_{s}"][e]), None)
        if shape is not None:
            recs.append({"shape": shape, "folded": bool(extra[f"fold_{shape}"][e]), "best": best[e]})
    reasons = {k: (sum(v) if isinstance(v, list) else v) for k, v in r["termination_reasons"].items()}
    return recs, reasons


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", default="docs/results/contact_stiffness")
    p.add_argument("--speed_job", default=None, help="speed array job id (slurm_logs/cstiff_<id>_<cfg>.out)")
    args = p.parse_args()

    groups: dict[tuple, list[dict]] = defaultdict(list)
    reasons: dict[tuple, dict] = defaultdict(lambda: defaultdict(int))
    for f in sorted(Path(args.results).glob("*__*__*_s*.json")):
        pol, cfg, rest = f.stem.split("__")
        start = rest.rsplit("_s", 1)[0]
        fam, seed = pol.rsplit("_s", 1)
        recs, rs = _episodes(f)
        for key in ((fam, cfg, start, seed), (fam, cfg, start, "all")):
            groups[key] += recs
            for k, v in rs.items():
                reasons[key][k] += v

    fams = sorted({k[0] for k in groups})
    print("fold rate (finished episodes, 2 eval seeds x 320 envs per policy seed); per-seed rates in []")
    for fam in fams:
        for start in ("pinned", "rand"):
            print(f"\n{fam} / {start}")
            print(f"  {'config':8s} {'solref':20s} {'fold':>14s}  per seed           med best err  fall  timeout  nonfinite")
            for cfg in CFGS:
                allr = groups.get((fam, cfg, start, "all"))
                if not allr:
                    continue
                k, n = sum(r["folded"] for r in allr), len(allr)
                seeds = sorted({key[3] for key in groups if key[:3] == (fam, cfg, start) and key[3] != "all"})
                per = " ".join(f"{100 * sum(r['folded'] for r in groups[(fam, cfg, start, s)]) / max(len(groups[(fam, cfg, start, s)]), 1):4.0f}"
                               for s in seeds)
                errs = [r["best"] for r in allr if r["best"] is not None]
                med = f"{statistics.median(errs) * 1e3:5.1f} mm" if errs else "   -   "
                rs = reasons[(fam, cfg, start, "all")]
                print(f"  {cfg:8s} {SOLREF[cfg]:20s} {k:4d}/{n:<4d} {100 * k / n:5.1f}%  [{per}]  {med:>12s}  "
                      f"{rs.get('fall', 0):4d}  {rs.get('timeout', 0):7d}  {rs.get('nonfinite', 0):9d}")
        shapes = sorted({r["shape"] for key, v in groups.items() if key[0] == fam for r in v})
        print(f"\n{fam} per shape (pinned + rand pooled):")
        print("  " + " ".join(f"{s:>14s}" for s in ["config"] + shapes))
        for cfg in CFGS:
            row = []
            for s in shapes:
                rr = [r for st in ("pinned", "rand") for r in groups.get((fam, cfg, st, "all"), []) if r["shape"] == s]
                row.append(f"{100 * sum(r['folded'] for r in rr) / len(rr):13.1f}%" if rr else f"{'-':>14s}")
            print("  " + f"{cfg:>14s} " + " ".join(row))

    if args.speed_job:
        print("\nspeed (1536 envs, rl_games fps step, mean of the second half of the epochs)")
        for i, cfg in enumerate(CFGS):
            log = Path(f"slurm_logs/cstiff_{args.speed_job}_{i}.out")
            if not log.exists():
                continue
            fps = [float(x.replace(",", "")) for x in re.findall(r"fps step\s*:\s*([0-9,.]+)", log.read_text())]
            if fps:
                half = fps[len(fps) // 2:]
                print(f"  {cfg:8s} {SOLREF[cfg]:20s} {statistics.mean(half):9.0f} env-steps/s  ({len(fps)} epochs)")


if __name__ == "__main__":
    main()
