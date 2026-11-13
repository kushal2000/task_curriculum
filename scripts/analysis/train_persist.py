"""Mean training success of an rl_games run over its last N summary points.

    python scripts/analysis/train_persist.py <run_name> [--last 20] [--tag episode_final/done_persist]

Reads outputs/*/*/<run_name>/summaries/events* (TensorBoard) and prints the mean of the tag's last N
values (``nan`` if the run or tag has none). Used by engaging_fabric_curriculum.sh to decide whether a
contact-layer stage has learned enough to move on.
"""
import argparse
import glob
import math

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--last", type=int, default=20)
    p.add_argument("--tag", default="episode_final/done_persist")
    a = p.parse_args()
    vals = []
    for f in sorted(glob.glob(f"outputs/*/*/{a.run}/summaries/events*")):
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        tags = ea.Tags()["scalars"]
        if a.tag in tags:
            # The first point after a resume has every termination reason at 0 (not a real set of
            # episodes: something must end each one). Skip steps where all episode_final/done_* are 0.
            reasons = [t for t in tags if t.startswith("episode_final/done_")]
            nonzero = set()
            for t in reasons:
                nonzero |= {s.step for s in ea.Scalars(t) if s.value != 0.0}
            vals += [(s.step, s.value) for s in ea.Scalars(a.tag) if not reasons or s.step in nonzero]
    vals = [v for _, v in sorted(vals)][-a.last:]
    print(sum(vals) / len(vals) if vals else math.nan)


if __name__ == "__main__":
    main()
