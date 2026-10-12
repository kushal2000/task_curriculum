"""Large vs standard VBD cloth for the viewer's four policies: held folds out of 64 (32 envs x 2 seeds).

    python3 scripts/analysis/large_cloth_summary.py [--tag s20] [--md docs/reports/4_physx/large_cloth_menu_policies_s20.md]

Large: docs/results/large_cloth/<tag>/<policy>__<pinned|rand>__<4cm|1p5cm>_s<es>.json
(bos14_large_cloth_eval.sh). 10 cm: the same checkpoints' existing evals -- px_matrix for the
2026-10-06 runs, rc_eval_[tol15_]pxpool_pxs1_arm1e3_e6144_s1 for 157622. "?" = file missing.
"""

from __future__ import annotations

import argparse
import json
import os

POLICIES = {
    "run2_flat_s3": ("run 2 flat pool (s3)", "px_matrix/one_flat_s3__on_vbd__{start}__{cut}_s{es}.json"),
    "run1_initfold_s1": ("run 1 pool + folded starts (s1)", "px_matrix/one_initfold_s1__on_vbd__{start}__{cut}_s{es}.json"),
    "run3_box3_s3": ("run 3 box3 (s3)", "px_matrix/box3_s3__on_vbd__{start}__{cut}_s{es}.json"),
    "prev_best_157622_s1": ("previous best 157622 (s1)", "rc_eval_{tol}pxpool_pxs1_arm1e3_e6144_s1__on_vbd{rand}_c0_s{es}.json"),
}
CELLS = [("pinned", "4cm"), ("rand", "4cm"), ("pinned", "1p5cm"), ("rand", "1p5cm")]


def held(paths: list[str]) -> str:
    h = n = 0
    for p in paths:
        if not os.path.isfile(p):
            return "?"
        d = json.load(open(p))
        h += int(d["termination_reasons"].get("fold_held", 0))
        n += int(d["num_envs"])
    return f"{h}/{n} ({100 * h / n:.0f}%)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s20")
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    rows = ["| policy | cloth | " + " | ".join(f"{c} {s}" for s, c in CELLS) + " |",
            "|---|---|" + "---|" * len(CELLS)]
    for key, (label, small) in POLICIES.items():
        big = [held([f"docs/results/large_cloth/{a.tag}/{key}__{s}__{c}_s{es}.json" for es in (1, 2)])
               for s, c in CELLS]
        ref = []
        for s, c in CELLS:
            ref.append(held([
                "docs/results/" + small.format(start=s, cut=c, es=es, tol="tol15_" if c == "1p5cm" else "",
                                               rand="__rand" if s == "rand" else "")
                for es in (1, 2)
            ]))
        rows.append(f"| {label} | 10 cm | " + " | ".join(ref) + " |")
        rows.append(f"| | **{a.tag}** | " + " | ".join(f"**{x}**" for x in big) + " |")
    text = "\n".join(rows)
    print(text)
    if a.md:
        open(a.md, "w").write(text + "\n")


if __name__ == "__main__":
    main()
