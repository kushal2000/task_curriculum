"""Tabulate bos14_physx_eval_on_newton.sh results: held folds per seed, per target and start.

    python3 scripts/analysis/physx_eval_summary.py px_arm1e3 pxpool_box3_arm1e3_e6144 ...

Each argument is a policy label (``px_<RUN_TAG>`` for chains, ``pxpool_<RUN_TAG>`` for pools). Reads
``docs/results/rc_eval_[tol15_]<label>_s<seed>__on_<target>[__rand]_c0_s<es>.json`` and prints the
held folds out of 64 (32 envs x 2 eval seeds) per seed, and the share over all seeds.
"""

from __future__ import annotations

import glob
import json
import re
import sys


def cell(prefix: str, label: str, target: str, rand: bool) -> str:
    per_seed = []
    for seed in (1, 2, 3):
        held = n = 0
        pat = f"docs/results/{prefix}{label}_s{seed}__on_{target}{'__rand' if rand else ''}_c0_s*.json"
        for path in glob.glob(pat):
            d = json.load(open(path))
            held += int(d["termination_reasons"].get("fold_held", 0))
            n += int(d["num_envs"])
        per_seed.append((held, n))
    if not any(n for _, n in per_seed):
        return "-"
    tot = sum(h for h, _ in per_seed)
    num = sum(n for _, n in per_seed)
    return ", ".join(str(h) if n else "?" for h, n in per_seed) + f" ({100 * tot / max(num, 1):.0f}%)"


def main() -> None:
    for label in sys.argv[1:]:
        print(label)
        if label.startswith("pxpool_"):
            rows = [("vbd 4cm", "rc_eval_", "vbd"), ("vbd 1.5cm", "rc_eval_tol15_", "vbd")]
        else:
            rows = [("newton chain", "rc_eval_", "box3-mid"), ("vbd cloth", "rc_eval_", "vbd")]
        for name, prefix, target in rows:
            print(f"  {name:<14s} pinned {cell(prefix, label, target, False):<22s} rand {cell(prefix, label, target, True)}")


if __name__ == "__main__":
    main()
