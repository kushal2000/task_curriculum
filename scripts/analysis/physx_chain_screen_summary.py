"""Tabulate bos14_physx_chain_screen.sh: held folds and falls per physics setting, policy and start.

    python3 scripts/analysis/physx_chain_screen_summary.py [TAG ...]
"""

from __future__ import annotations

import glob
import json
import re
import sys
from collections import defaultdict

POLICIES = ("ref", "newton", "physx")
STARTS = ("pinned", "rand")


def main() -> None:
    cells = defaultdict(lambda: [0, 0, 0])  # (tag, policy, start) -> held, falls, episodes
    pat = re.compile(r"pxscreen_(.+)_(ref|newton|physx)_(pinned|rand)_s\d+\.json$")
    for path in sorted(glob.glob("docs/results/pxscreen_*.json")):
        m = pat.search(path)
        if not m:
            continue
        d = json.load(open(path))
        t = d["termination_reasons"]
        c = cells[m.groups()]
        c[0] += int(t.get("fold_held", 0))
        c[1] += int(t.get("fall", 0))
        c[2] += int(d["num_envs"])
    tags = sys.argv[1:] or sorted({k[0] for k in cells})
    head = "".join(f"{p + ' ' + s:>18s}" for p in POLICIES for s in STARTS)
    print(f"{'setting':<24s}{head}")
    print(f"{'':<24s}" + "".join(f"{'held  (fall)':>18s}" for _ in POLICIES for _ in STARTS))
    for tag in tags:
        row = f"{tag:<24s}"
        for p in POLICIES:
            for s in STARTS:
                held, fall, n = cells.get((tag, p, s), (0, 0, 0))
                row += f"{f'{held}/{n} ({fall})' if n else '-':>18s}"
        print(row)


if __name__ == "__main__":
    main()
