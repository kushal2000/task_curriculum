"""Tabulate fold-pool evaluations (bos14_fold_pool_eval.sh) by shape and by sheet size.

    .venv_isaaclab3/bin/python scripts/analysis/fold_pool_eval_summary.py --train_job 156603

Each eval JSON holds one episode per env. The pool env publishes ``done_shape_<shape>`` and
``done_fold_<shape>`` per episode, which ``isaacsimenvs.eval.episodes`` keeps as
``per_env_extra_done``, and the env's fixed sheet side as ``per_env_sheet_size``. A fold here is the
eval's own criterion (held 10 steps at the task's 1.5 cm). Censored envs -- still on their first
episode at the step budget -- published nothing and are excluded, and counted.

Besides the fold rate, the MEDIAN BEST FOLD ERROR per group: at a tight tolerance the rate alone
cannot tell "never close" from "just outside", and stage 1's 1.5 cm collapse was diagnosed exactly
from that number (a precision ceiling, not a failure to fold).
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

SIZE_BINS = ((0.100, 0.125), (0.125, 0.150), (0.150, 0.175), (0.175, 0.2001))


def _load(path: Path) -> list[dict]:
    """One record per finished episode: shape, size, folded, best fold error."""
    r = json.loads(path.read_text())
    extra = r.get("per_env_extra_done", {})
    shapes = sorted(k[len("shape_") :] for k in extra if k.startswith("shape_"))
    sizes = r.get("per_env_sheet_size") or [None] * r["num_envs"]
    best = r.get("fold", {}).get("best_fold_err") or [None] * r["num_envs"]
    out = []
    for e in range(r["num_envs"]):
        shape = next((s for s in shapes if extra[f"shape_{s}"][e]), None)
        if shape is None:
            continue  # censored: never finished, published nothing
        out.append(
            {
                "shape": shape,
                "size": sizes[e],
                "folded": bool(extra[f"fold_{shape}"][e]),
                "best_err": best[e],
            }
        )
    return out


def _row(recs: list[dict]) -> str:
    n = len(recs)
    if n == 0:
        return "   -   |   -   |   -   "
    k = sum(r["folded"] for r in recs)
    errs = [r["best_err"] for r in recs if r["best_err"] is not None]
    med = f"{statistics.median(errs) * 1e3:5.1f}" if errs else "  -  "
    return f"{k:3d}/{n:<3d} | {100.0 * k / n:5.1f}% | {med} mm"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train_job", required=True)
    p.add_argument("--results", default="docs/results")
    p.add_argument("--prefix", default="fp_eval_")
    args = p.parse_args()

    by_mode: dict[str, list[dict]] = defaultdict(list)
    by_seed_mode: dict[tuple[str, str], list[dict]] = defaultdict(list)
    files = sorted(Path(args.results).glob(f"{args.prefix}s*_{args.train_job}__*_s*.json"))
    if not files:
        raise SystemExit(f"no eval JSONs for train job {args.train_job} in {args.results}")
    censored = 0
    for f in files:
        # <prefix>s<seed>_<job>__<mode>_s<evalseed>.json
        stem = f.stem[len(args.prefix) :]
        seed = stem.split("_", 1)[0]
        mode = stem.split("__", 1)[1].rsplit("_s", 1)[0]
        recs = _load(f)
        censored += json.loads(f.read_text())["censored"]
        by_mode[mode] += recs
        by_seed_mode[(seed, mode)] += recs
    print(
        f"{len(files)} eval files, {sum(len(v) for v in by_mode.values())} finished episodes, "
        f"{censored} censored\n"
    )
    print("columns: held folds / episodes | rate | median best fold error\n")

    for mode, recs in sorted(by_mode.items()):
        print(f"=== {mode} (all policy seeds) ===")
        shapes = sorted({r["shape"] for r in recs})
        print(f"{'all':16s} {_row(recs)}")
        for s in shapes:
            print(f"{s:16s} {_row([r for r in recs if r['shape'] == s])}")
        if any(r["size"] is not None for r in recs):
            print("by sheet size:")
            for lo, hi in SIZE_BINS:
                sel = [r for r in recs if r["size"] is not None and lo <= r["size"] < hi]
                print(f"  {lo * 100:4.1f}-{min(hi, 0.2) * 100:4.1f} cm   {_row(sel)}")
        print()

    print("=== per policy seed ===")
    for (seed, mode), recs in sorted(by_seed_mode.items()):
        print(f"{seed} {mode:7s} {_row(recs)}")


if __name__ == "__main__":
    main()
