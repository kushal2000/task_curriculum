"""Latest values of chosen tensorboard scalars for training runs, matched by run-name substring.

    .venv_isaaclab3/bin/python scripts/analysis/tb_tail.py 0_pxpool_flat_s1_158323 [--tags fold done_fold rewards]
"""

from __future__ import annotations

import argparse
import glob
import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

p = argparse.ArgumentParser()
p.add_argument("runs", nargs="+")
p.add_argument("--tags", nargs="*", default=["rewards/step", "episode_final/done_fold", "episode_final/done_fold_held",
                                              "episode_lengths/step", "episode_final/done_start_folded",
                                              "episode_final/done_fold_start_folded"])
p.add_argument("--every", type=int, default=0, help="also print every Nth point of the first tag")
a = p.parse_args()
for run in a.runs:
    for d in sorted(glob.glob(f"outputs/*/*/{run}*/summaries")):
        ea = EventAccumulator(d, size_guidance={"scalars": 0})
        ea.Reload()
        tags = ea.Tags()["scalars"]
        print(f"== {d.split('/')[-2]}")
        for t in a.tags:
            hits = [x for x in tags if x == t or x.endswith("/" + t) or t in x]
            for h in hits[:3]:
                ev = ea.Scalars(h)
                print(f"  {h:<50} step {ev[-1].step:>12,}  {ev[-1].value:.4f}  (n={len(ev)})")
        if a.every:
            h = [x for x in tags if a.tags[0] in x][0]
            ev = ea.Scalars(h)
            print("  " + "  ".join(f"{e.step / 1e6:.0f}M:{e.value:.3f}" for e in ev[:: a.every]))
