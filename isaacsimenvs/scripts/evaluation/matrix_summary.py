"""Tabulate experiments/eval.sh results: held folds per training seed in every cell, at 1.5 cm.

    python3 isaacsimenvs/scripts/evaluation/matrix_summary.py flat7RW_base rigidcloth_ftGateFric [--md out.md]

Reads docs/results/px_matrix/<tag>_s<seed>__on_<target>__<start>__1p5cm_s<es>.json. One row per tag;
per cell the held folds of each training seed (summed over the eval seeds) and the share over all
seeds. A missing file shows as "?" so an unfinished or failed task cannot pass as a zero; a cell with
no file for any seed (e.g. the VBD cells of a goal point-cloud student) shows "-"; the opt-in
rigidbody_folded column only appears when a tag has it. Results
under the pre-2026-11-22 target names (physx_pool7 / physx_pool7of: task defaults, not the policy's
own recipe) are not read.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

CELLS = [
    ("rigidbody_flat", "rand", "rigid flat rand"),
    ("rigidbody_folded", "rand", "rigid folded rand"),
    ("vbd", "pinned", "VBD pinned"),
    ("vbd", "rand", "VBD rand"),
]


def cell(tag: str, target: str, start: str) -> str:
    parts, held_all, n_all, any_file = [], 0, 0, False
    for seed in (1, 2, 3):
        held = n = 0
        files = sorted(glob.glob(f"docs/results/px_matrix/{tag}_s{seed}__on_{target}__{start}__1p5cm_s*.json"))
        any_file |= bool(files)
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
    if not any_file:
        return "-"
    while parts and parts[-1] == "?":      # seeds the tag never had
        parts.pop()
    share = f" ({100 * held_all / n_all:.0f}%)" if n_all else ""
    return ", ".join(parts or ["?"]) + share


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("tags", nargs="+")
    p.add_argument("--md", default=None, help="also write the table as markdown here")
    a = p.parse_args()
    rows = {tag: [cell(tag, t, s) for t, s, _ in CELLS] for tag in a.tags}
    # an opt-in cell (rigidbody_folded) only when some tag has it
    cols = [i for i, (t, _, _) in enumerate(CELLS) if t != "rigidbody_folded" or any(r[i] != "-" for r in rows.values())]
    out = [
        "held folds per training seed at 1.5 cm (2 eval seeds summed: rigid body of 896, VBD of 64)\n",
        "| tag | " + " | ".join(CELLS[i][2] for i in cols) + " |",
        "|---|" + "---|" * len(cols),
    ]
    for tag, r in rows.items():
        out.append(f"| {tag} | " + " | ".join(r[i] for i in cols) + " |")
    text = "\n".join(out)
    print(text)
    if a.md:
        os.makedirs(os.path.dirname(os.path.abspath(a.md)), exist_ok=True)
        open(a.md, "w").write(text + "\n")


if __name__ == "__main__":
    main()
