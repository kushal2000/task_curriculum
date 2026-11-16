"""Shared paths and Gym task registration for Isaac Lab environments.

Paths are defined before importing tasks: registration uses CONFIG_DIR.
Simulator classes remain lazy; importing this package does not start a simulator.
"""

from __future__ import annotations

import os as _os
from pathlib import Path as _Path

PACKAGE_ROOT = _Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
SCRIPTS_DIR = PACKAGE_ROOT / "scripts"
CONFIG_DIR = SCRIPTS_DIR / "configs"
DEBUG_OUTPUTS_DIR = REPO_ROOT / "debug_outputs"
TRAIN_LOGS_DIR = DEBUG_OUTPUTS_DIR / "train_logs"
VIDEOS_DIR = DEBUG_OUTPUTS_DIR / "videos"
DEPLOY_DIR = DEBUG_OUTPUTS_DIR / "deployment"
#: Where sbatch writes a job's log until the job knows its own folder (experiments/submit.sh); a
#: training or eval job then moves it there with ``scripts.output.adopt_slurm_log``, leaving a symlink.
SLURM_STAGING_DIR = TRAIN_LOGS_DIR / ".slurm"
PRETRAINED_DIR = _Path(_os.environ.get("PRETRAINED_POLICY_DIR", REPO_ROOT / "pretrained_policy"))
PRETRAINED_KUKA_SHARPA = PRETRAINED_DIR / "kuka_sharpa"
PRETRAINED_FLEXIV_WUJI = PRETRAINED_DIR / "flexiv_wuji"
#: The default prior evals load: Kuka + Sharpa.
PRETRAINED_CHECKPOINT = PRETRAINED_KUKA_SHARPA / "model.pth"
PRETRAINED_CONFIG = PRETRAINED_KUKA_SHARPA / "config.yaml"


# Task registration imports the constants above from this package.
from . import tasks  # noqa: E402,F401
