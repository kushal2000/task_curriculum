"""Tabulate bos14_px_eval_matrix.sh: held folds per seed for every target, start and cutoff.

    python3 scripts/analysis/px_eval_matrix_summary.py one_flat one_initfold box3 [--md docs/reports/4_physx/eval_matrix.md]

Reads docs/results/px_matrix/<tag>_s<seed>__on_<target>__<pinned|rand>__<4cm|1p5cm>_s<es>.json and
prints, per cell, the held folds of each training seed (summed over the two eval seeds) and the share
over all seeds. A missing file shows as "?" so an unfinished or failed task cannot pass as a zero. A
target with no file at all for a tag (the 7-shape targets before 2026-10-12) is left out.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

TARGETS = [
    ("physx_pool", "PhysX pool"),
    ("physx_box3", "PhysX box3"),
    ("newton_pool", "Newton pool"),
    ("newton_box3", "Newton box3"),
    ("vbd", "VBD cloth"),
    ("physx_pool7", "PhysX 7-shape pool"),
    ("physx_pool7of", "PhysX 7-shape, folded starts"),
]
COLS = [(s, c) for c in ("4cm", "1p5cm") for s in ("pinned", "rand")]


def cell(tag: str, target: str, start: str, cut: str) -> str:
    parts, held_all, n_all = [], 0, 0
    for seed in (1, 2, 3):
        held = n = 0
        files = sorted(glob.glob(f"docs/results/px_matrix/{tag}_s{seed}__on_{target}__{start}__{cut}_s*.json"))
        for path in files:
            d = json.load(open(path))
            held += int(d["termination_reasons"].get("fold_held", 0))
            n += int(d["num_envs"])
        if len(files) < 2:
            parts.append("?")
        else:
            parts.append(str(held))
            held_all += held
            n_all += n
    share = f" ({100 * held_all / n_all:.0f}%)" if n_all else ""
    return ", ".join(parts) + share


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("tags", nargs="+")
    p.add_argument("--md", default=None, help="also write the tables as markdown here")
    a = p.parse_args()
    out = []
    for tag in a.tags:
        out.append(f"### {tag}\n")
        out.append("held folds per training seed (2 eval seeds summed; pool targets of 640, 7-shape of 896, others of 64)\n")
        out.append("| target | " + " | ".join(f"{c} {s}" for s, c in COLS) + " |")
        out.append("|---|" + "---|" * len(COLS))
        for key, name in TARGETS:
            if not glob.glob(f"docs/results/px_matrix/{tag}_s[123]__on_{key}__*.json"):
                continue
            out.append(f"| {name} | " + " | ".join(cell(tag, key, s, c) for s, c in COLS) + " |")
        out.append("")
    text = "\n".join(out)
    print(text)
    if a.md:
        os.makedirs(os.path.dirname(os.path.abspath(a.md)), exist_ok=True)
        open(a.md, "w").write(text + "\n")


if __name__ == "__main__":
    main()
