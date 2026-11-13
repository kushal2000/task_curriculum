"""Upload finished rl_games TensorBoard runs to Weights & Biases (runs trained without --wandb_activate).

    .venv_isaaclab3/bin/python -I scripts/analysis/tb_to_wandb.py --group rel3sRfsC <run_dir> [<run_dir> ...]

Each <run_dir> is an ``outputs/<date>/<time>/<name>`` directory with ``summaries/events.*``. Every scalar
is logged against ``env_steps`` (rl_games' frame counter); the ``*/iter`` and ``*/time`` duplicates of
the reward / length curves are skipped. Config: the ``[ftfab]`` header lines of the matching Slurm log
(fabric, start checkpoint, overrides). The W&B run id is the run name, so re-running replaces nothing --
delete the old run first if a run is uploaded again (e.g. a partial one).
"""

from __future__ import annotations

import argparse
import glob
import re
from collections import defaultdict
from pathlib import Path


def slurm_header(name: str) -> dict:
    """``[ftfab]`` lines of slurm_logs/ftfab*_<job>_0.out, keyed by the job id at the end of ``name``."""
    m = re.search(r"_(\d+)$", name)
    if not m:
        return {}
    logs = glob.glob(f"slurm_logs/ftfab*_{m.group(1)}_*.out")
    out = {"slurm_job": m.group(1)}
    if not logs:
        return out
    for line in open(logs[0], errors="replace"):
        if line.startswith("[ftfab] "):
            body = line[len("[ftfab] "):].strip()
            key = body.split("=", 1)[0].split(":", 1)[0].split()[0]
            out[f"ftfab_{key}"] = body
        if line.startswith("[cloth]") and len(out) > 12:
            break
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--project", default="rigid2deform")
    p.add_argument("--entity", default=None)
    p.add_argument("--group", default=None)
    p.add_argument("--tags", nargs="*", default=[])
    args = p.parse_args()

    import wandb
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    for run_dir in args.runs:
        ev = sorted(glob.glob(str(run_dir / "summaries" / "events*")))
        if not ev:
            print(f"[tb2wb] no events in {run_dir}, skipped")
            continue
        name = run_dir.name
        rows: dict[int, dict] = defaultdict(dict)
        for f in ev:
            ea = EventAccumulator(f, size_guidance={"scalars": 0})
            ea.Reload()
            for tag in ea.Tags()["scalars"]:
                if tag.endswith("/iter") or tag.endswith("/time"):
                    continue
                for s in ea.Scalars(tag):
                    rows[int(s.step)][tag] = float(s.value)
        fabric = re.sub(r"_\d+$", "", name.split("_", 2)[-1]) if name.count("_") >= 2 else name
        cfg = {"run_dir": str(run_dir), "fabric": fabric, **slurm_header(name)}
        run = wandb.init(project=args.project, entity=args.entity, name=name, id=re.sub(r"[^A-Za-z0-9_-]", "-", name),
                         group=args.group, tags=args.tags + [fabric], config=cfg, job_type="train",
                         resume="never", reinit="finish_previous")
        run.define_metric("env_steps")
        run.define_metric("*", step_metric="env_steps")
        for step in sorted(rows):
            run.log({"env_steps": step, **rows[step]})
        print(f"[tb2wb] {name}: {len(rows)} logged points, {len(set(k for r in rows.values() for k in r))} metrics "
              f"-> {run.url}", flush=True)
        run.finish()


if __name__ == "__main__":
    main()
