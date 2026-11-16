"""Table of the fabric eval: held-fold (or held-unfold) rate per policy x task x fabric, pooled over seeds.

    python experiments/nov15_fabric_eval/summarize.py [results_dir]

Success = termination reason ``fold_held``: the target (folded, or flat for unfold) held for
``success_steps`` steps. ``n`` counts completed episodes; censored envs (still running at the step
budget) are counted as failures in the rate and shown separately.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

FABRICS = ("big_napkin", "chiffon", "cotton", "denim", "napkin", "silk", "towel")
NAME = re.compile(r"^(?P<policy>[^_]+)__(?P<task>fold|unfold)__(?P<fabric>.+)_s(?P<seed>\d+)\.json$")


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "results"
    cells = defaultdict(lambda: {"held": 0, "envs": 0, "censored": 0, "nonfinite": 0, "seeds": 0})
    for f in sorted(root.glob("*.json")):
        m = NAME.match(f.name)
        if not m:
            continue
        r = json.loads(f.read_text())
        c = cells[(m["policy"], m["task"], m["fabric"])]
        c["held"] += r["termination_reasons"]["fold_held"]
        c["nonfinite"] += r["termination_reasons"].get("nonfinite", 0)
        c["envs"] += r["num_envs"]
        c["censored"] += r["censored"]
        c["seeds"] += 1
    if not cells:
        sys.exit(f"no result JSONs in {root}")
    for policy in sorted({k[0] for k in cells}):
        for task in ("fold", "unfold"):
            rows = [(fab, cells.get((policy, task, fab))) for fab in FABRICS]
            if not any(c for _, c in rows):
                continue
            print(f"\n{policy} / {task}")
            print(f"  {'fabric':<11} {'success':>8} {'held/envs':>10} {'seeds':>5} {'censored':>8} {'nan':>4}")
            for fab, c in rows:
                if not c:
                    print(f"  {fab:<11} {'-':>8}")
                    continue
                print(f"  {fab:<11} {c['held'] / c['envs']:>8.1%} {c['held']:>4}/{c['envs']:<5} "
                      f"{c['seeds']:>5} {c['censored']:>8} {c['nonfinite']:>4}")


if __name__ == "__main__":
    main()
