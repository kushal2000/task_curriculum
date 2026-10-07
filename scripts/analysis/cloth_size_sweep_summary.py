"""Run 2 (flat pool) on VBD cloths of several sizes: held folds per seed (of 64) and overall.

    python3 scripts/analysis/cloth_size_sweep_summary.py [--md docs/results/large_cloth/size_sweep.md]

10 cm: docs/results/px_matrix/one_flat_s<seed>__on_vbd__<start>__<cut>_s<es>.json (the standard eval).
Others: docs/results/large_cloth/sweep/one_flat_s<seed>__<size>__<start>__<cut>_s<es>.json
(bos14_cloth_size_sweep.sh). Also reports falls and non-finite episodes per size. "?" = missing.
"""

from __future__ import annotations

import argparse
import json
import os

SIZES = ["10cm", "12p5cm", "15cm", "17p5cm", "20cm"]
CELLS = [("pinned", "4cm"), ("rand", "4cm"), ("pinned", "1p5cm"), ("rand", "1p5cm")]


def path(size: str, seed: int, start: str, cut: str, es: int) -> str:
    if size == "10cm":
        return f"docs/results/px_matrix/one_flat_s{seed}__on_vbd__{start}__{cut}_s{es}.json"
    return f"docs/results/large_cloth/sweep/one_flat_s{seed}__{size}__{start}__{cut}_s{es}.json"


def load(p: str):
    return json.load(open(p)) if os.path.isfile(p) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    head = "| size | " + " | ".join(f"{c} {s}" for s, c in CELLS) + " | falls | non-finite |"
    rows = [head, "|---|" + "---|" * (len(CELLS) + 2)]
    for size in SIZES:
        cells, falls, nonfinite, n_all = [], 0, 0, 0
        for start, cut in CELLS:
            per, tot, n = [], 0, 0
            for seed in (1, 2, 3):
                ds = [load(path(size, seed, start, cut, es)) for es in (1, 2)]
                if any(d is None for d in ds):
                    per.append("?")
                    continue
                h = sum(int(d["termination_reasons"].get("fold_held", 0)) for d in ds)
                per.append(str(h))
                tot += h
                n += sum(int(d["num_envs"]) for d in ds)
                falls += sum(int(d["termination_reasons"].get("fall", 0)) for d in ds)
                nonfinite += sum(int(d["termination_reasons"].get("nonfinite", 0)) for d in ds)
                n_all += sum(int(d["num_envs"]) for d in ds)
            share = f" (**{100 * tot / n:.0f}%**)" if n else ""
            cells.append(", ".join(per) + share)
        fall_txt = f"{100 * falls / n_all:.0f}%" if n_all else "?"
        rows.append(f"| {size.replace('p', '.')} | " + " | ".join(cells) + f" | {fall_txt} | {nonfinite} |")
    text = ("held folds per seed s1, s2, s3 (each of 64 = 32 envs x 2 eval seeds), overall share\n\n"
            + "\n".join(rows))
    print(text)
    if a.md:
        open(a.md, "w").write(text + "\n")


if __name__ == "__main__":
    main()
