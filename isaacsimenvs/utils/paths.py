"""Where things live, relative to this checkout rather than to one machine.

Every run artifact goes under ``DEBUG_OUTPUTS_DIR`` (``debug_outputs/``, gitignored):

* ``debug_outputs/train_logs/`` (``TRAIN_LOGS_DIR``): Hydra run dirs ``<date>/<time>/0_<name>/``,
  ``wandb/``, ``slurm/`` (scripts/slurm/submit.sh) and ``results/`` (eval JSONs, probe dumps).
* ``debug_outputs/videos/`` (``VIDEOS_DIR``): rendered videos, ``<month>/<date>_<name>_<time>.mp4``.
* ``debug_outputs/deployment/`` (``DEPLOY_DIR``): real-robot run recordings and logs (deployment/).

The Play priors are in ``PRETRAINED_DIR`` (``<repo>/pretrained_policy``, gitignored, or
``$PRETRAINED_POLICY_DIR``), one ``<robot>/{model.pth,config.yaml}`` per robot: ``kuka_sharpa/`` (the generic
SimToolReal prior; on portal symlinks into /share/portal/kk837/simtoolreal/pretrained_policy) and
``flexiv_wuji/`` (the right-Wuji human-like Play prior, run play_scratch_flexiv_right_dr_cont_41213).

Stdlib only: imported by Kit-free code and by deployment/.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEBUG_OUTPUTS_DIR = REPO_ROOT / "debug_outputs"
TRAIN_LOGS_DIR = DEBUG_OUTPUTS_DIR / "train_logs"
VIDEOS_DIR = DEBUG_OUTPUTS_DIR / "videos"
DEPLOY_DIR = DEBUG_OUTPUTS_DIR / "deployment"
#: Where sbatch writes a job's log until the job knows its own folder (scripts/slurm/submit.sh); a
#: training or eval job then moves it there with :func:`adopt_slurm_log`, leaving a symlink.
SLURM_STAGING_DIR = TRAIN_LOGS_DIR / ".slurm"
PRETRAINED_DIR = Path(os.environ.get("PRETRAINED_POLICY_DIR", REPO_ROOT / "pretrained_policy"))
PRETRAINED_KUKA_SHARPA = PRETRAINED_DIR / "kuka_sharpa"
PRETRAINED_FLEXIV_WUJI = PRETRAINED_DIR / "flexiv_wuji"
#: The default prior evals load: Kuka + Sharpa.
PRETRAINED_CHECKPOINT = PRETRAINED_KUKA_SHARPA / "model.pth"
PRETRAINED_CONFIG = PRETRAINED_KUKA_SHARPA / "config.yaml"


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
    staging path, so ``tail -f`` on the old name and scripts/slurm/_common.sh's end-of-job lines
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
