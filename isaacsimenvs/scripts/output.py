"""Naming run artifacts and collecting Slurm logs under debug_outputs/."""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

from isaacsimenvs import SLURM_STAGING_DIR, TRAIN_LOGS_DIR, VIDEOS_DIR


def job_tag() -> str | None:
    """The Slurm job id (``<array job>_<index>`` for an array task), or None outside Slurm."""
    job = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID")
    if not job:
        return None
    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    return f"{job}_{task}" if task else job


def default_run_dir(run_name: str, now: datetime | None = None) -> Path:
    """``train_logs/<month>/<date>_<run name>_<job id or HHMMSS>/``: one folder per training run.

    ``run_name`` is rl_games' ``config.name`` without its ``<int>_`` prefix (rl_games parses that
    integer; it says nothing about the run). Outside Slurm the time stands in for the job id.
    """
    now = now or datetime.now()
    name = re.sub(r"^\d+_", "", run_name) or "run"
    suffix = job_tag() or now.strftime("%H%M%S")
    return TRAIN_LOGS_DIR / now.strftime("%b").lower() / f"{now:%Y-%m-%d}_{name}_{suffix}"


def default_video_path(name: str, now: datetime | None = None) -> Path:
    """``videos/<month>/<date>_<name>_<HHMMSS>.mp4``: rendered videos dated like the training runs."""
    now = now or datetime.now()
    return VIDEOS_DIR / now.strftime("%b").lower() / f"{now:%Y-%m-%d}_{name}_{now:%H%M%S}.mp4"


def adopt_slurm_log(dest_dir: Path) -> Path | None:
    """Move this Slurm job's log (and GPU-memory log) from SLURM_STAGING_DIR into ``dest_dir``.

    Slurm keeps writing to the moved file -- it holds it open -- and a symlink is left at the
    staging path, so ``tail -f`` on the old name and experiments/_common.sh's end-of-job lines
    still work. No-op outside Slurm or when the log was written somewhere else.
    """
    tag = job_tag()
    if tag is None or not SLURM_STAGING_DIR.is_dir():
        return None
    dest_dir = Path(dest_dir).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    moved = None
    for src in sorted(SLURM_STAGING_DIR.glob(f"*_{tag}.*")):
        if src.is_symlink() or not src.is_file():
            continue
        if src.name.endswith(f"_{tag}.out"):
            target = dest_dir / f"slurm_{tag}.out"
            moved = target
        elif src.name.endswith(f"_{tag}.csv"):
            target = dest_dir / f"gpumem_{tag}.csv"
        else:
            continue
        target = target.resolve()
        os.replace(src, target)
        src.symlink_to(target)
    return moved
