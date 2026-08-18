#!/usr/bin/env python3
"""Aggregate cloth-fold eval JSONs into the numbers the finetune is judged on.

Usage:
    python scripts/analysis/cloth_fold_summary.py docs/results/cloth_nohold_s*.json
    python scripts/analysis/cloth_fold_summary.py --label finetuned docs/results/cloth_ft_s*.json

Reports the two fold criteria separately, because the gap between them IS a diagnostic: many
first-entries with few holds means the flap reaches the target and springs back.

`success_tolerance` is the INHERITED rigid-tool criterion and must be pinned low enough that it can
never fire -- it is a second writer to `_successes` otherwise, and a sparse reward keyed on it would
be farmable without ever folding. This script asserts that rather than trusting it.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path


def _load(paths: list[Path]) -> list[dict]:
    out = []
    for p in paths:
        with open(p) as fh:
            out.append(json.load(fh))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--label", default="baseline")
    args = ap.parse_args()

    runs = _load(args.files)
    if not runs:
        print("no files")
        return 1

    n_eps = sum(r["completed"]["n"] for r in runs)
    fold = sum(r["termination_reasons"].get("fold", 0) for r in runs)
    held = sum(r["termination_reasons"].get("fold_held", 0) for r in runs)
    nonfinite = sum(r["termination_reasons"].get("nonfinite", 0) for r in runs)
    ejected = sum(r.get("ejected", 0) for r in runs)
    falls = sum(r["termination_reasons"].get("fall", 0) for r in runs)

    errs = [e for r in runs for e in r["fold"]["best_fold_err"]]
    tol = {r["fold"]["keypoint_tolerance"] for r in runs}
    inherited = {r["success_tolerance"] for r in runs}

    # Per-seed means, so the spread reported is ACROSS SEEDS rather than across episodes --
    # the former is what says whether a difference is reproducible.
    per_seed = [st.mean(r["fold"]["best_fold_err"]) for r in runs]

    print(f"=== {args.label}: {len(runs)} seeds, {n_eps} episodes ===")
    print(f"  folds (first entry) : {fold:4d} / {n_eps}  ({100*fold/n_eps:.2f}%)")
    print(f"  folds (held)        : {held:4d} / {n_eps}  ({100*held/n_eps:.2f}%)")
    print(f"  best_fold_err       : {st.mean(errs):.4f}  (episode sd {st.pstdev(errs):.4f})")
    print(f"  best_fold_err/seed  : {st.mean(per_seed):.4f} +- {st.pstdev(per_seed):.4f}  <- across seeds")
    print(f"  within tolerance    : {sum(e <= max(tol) for e in errs)} / {len(errs)} at {max(tol)}")
    print(f"  falls / ejected     : {falls} / {ejected}")
    print(f"  nonfinite resets    : {nonfinite}")

    if len(tol) != 1:
        print(f"  !! fold tolerance differs across files: {sorted(tol)} -- NOT comparable")
    # The inherited tool criterion must be unable to fire, or `_successes` has two writers.
    for t in sorted(inherited):
        if t > 0.02:
            print(f"  !! inherited success_tolerance={t} can fire; goals/episode may not be folds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
