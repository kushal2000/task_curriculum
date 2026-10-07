"""Training-time fold rate of rl_games runs, from their tensorboard summaries.

    .venv_isaaclab3/bin/python scripts/analysis/run_fold_rate.py <run-name-substring> ...

Prints, per run, the last value (and the value at a few earlier points) of every scalar whose tag
mentions ``fold`` -- the share of finished episodes the env counted as a held fold.
"""

from __future__ import annotations

import glob
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main() -> None:
    for pattern in sys.argv[1:]:
        for run in sorted(glob.glob(f"outputs/*/*/*{pattern}*")):
            ev = sorted(glob.glob(f"{run}/summaries/events*"))
            if not ev:
                continue
            acc = EventAccumulator(ev[-1], size_guidance={"scalars": 0})
            acc.Reload()
            tags = [t for t in acc.Tags()["scalars"] if "fold" in t.lower() and "/frame" not in t and "shape_" not in t]
            print(run.split("/")[-1])
            for t in sorted(tags):
                vals = acc.Scalars(t)
                if not vals:
                    continue
                pts = [vals[int(i * (len(vals) - 1) / 4)] for i in range(5)]
                print(f"  {t:<48s} " + "  ".join(f"{p.step}:{p.value:.3f}" for p in pts))


if __name__ == "__main__":
    main()
