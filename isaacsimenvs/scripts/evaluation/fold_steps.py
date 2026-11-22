"""How many steps a fold takes: median / mean length of the FOLDED episodes in experiments/eval.sh
results.

    python3 isaacsimenvs/scripts/evaluation/fold_steps.py flat7RW_base rigidcloth_ftGateFric

Reads docs/results/px_matrix/<tag>_s<seed>__on_<target>__<start>__1p5cm_s<es>.json, pools every
training seed and eval seed, and keeps the episodes with a held fold (`per_env_goals` >= 1). An
episode ends 10 steps after its fold is first entered (`env.termination.success_steps=10`), so the
length is the steps to fold + ~10. Per cell: "median / mean (folded episodes)"; "-" if none.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics

from matrix_summary import CELLS  # run as a file, beside it


def lengths(tag: str, target: str, start: str) -> list[int]:
    out = []
    for path in glob.glob(f"docs/results/px_matrix/{tag}_s[123]__on_{target}__{start}__1p5cm_s*.json"):
        d = json.load(open(path))
        out += [n for n, g in zip(d["per_env_length"], d["per_env_goals"]) if g >= 1]
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("tags", nargs="+")
    a = p.parse_args()
    print("\n### 1.5 cm: steps of folded episodes, median / mean (n)\n")
    print("| tag | " + " | ".join(name for _, _, name in CELLS) + " |")
    print("|---|" + "---|" * len(CELLS))
    for tag in a.tags:
        row = []
        for target, start, _ in CELLS:
            n = lengths(tag, target, start)
            row.append(f"{statistics.median(n):.0f} / {statistics.mean(n):.0f} ({len(n)})" if n else "-")
        print(f"| {tag} | " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
