"""How many steps a fold takes: median / mean length of the FOLDED episodes in bos14_px_eval_matrix.sh
(and bos14_px_eval_trend.sh) results.

    python3 scripts/analysis/px_eval_fold_steps.py one_flat ft7lr_flat_last [--targets vbd physx_pool]

Reads docs/results/px_matrix/<tag>_s<seed>__on_<target>__<start>__<cut>_s<es>.json, pools every
training seed and eval seed, and keeps the episodes with a held fold (`per_env_goals` >= 1). An
episode ends 10 steps after its fold is first entered (`env.termination.success_steps=10`), so the
length is the steps to fold + ~10. Per cell: "median / mean (folded episodes)"; "-" if none.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics

TARGETS = ["physx_pool", "physx_box3", "newton_pool", "newton_box3", "vbd", "physx_pool7", "physx_pool7of"]


def lengths(tag: str, target: str, start: str, cut: str) -> list[int]:
    out = []
    for path in glob.glob(f"docs/results/px_matrix/{tag}_s[123]__on_{target}__{start}__{cut}_s*.json"):
        d = json.load(open(path))
        out += [n for n, g in zip(d["per_env_length"], d["per_env_goals"]) if g >= 1]
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("tags", nargs="+")
    p.add_argument("--targets", nargs="+", default=TARGETS)
    a = p.parse_args()
    for cut in ("4cm", "1p5cm"):
        for start in ("pinned", "rand"):
            print(f"\n### {cut} {start}: steps of folded episodes, median / mean (n)\n")
            print("| tag | " + " | ".join(a.targets) + " |")
            print("|---|" + "---|" * len(a.targets))
            for tag in a.tags:
                row = []
                for t in a.targets:
                    n = lengths(tag, t, start, cut)
                    row.append(f"{statistics.median(n):.0f} / {statistics.mean(n):.0f} ({len(n)})" if n else "-")
                print(f"| {tag} | " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
