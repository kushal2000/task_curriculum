"""Where things live, relative to this checkout rather than to one machine.

Every run artifact goes under ``DEBUG_OUTPUTS_DIR`` (``debug_outputs/``, gitignored):

* ``debug_outputs/train_logs/`` (``TRAIN_LOGS_DIR``): Hydra run dirs ``<date>/<time>/0_<name>/``,
  ``wandb/``, ``slurm/`` (scripts/slurm/submit.sh) and ``results/`` (eval JSONs, probe dumps).
* ``debug_outputs/videos/`` (``VIDEOS_DIR``): rendered videos.
* ``debug_outputs/deployment/`` (``DEPLOY_DIR``): real-robot run recordings and logs (deployment/).

The pretrained tool-use prior is ``PRETRAINED_DIR`` (``<repo>/pretrained_policy``, gitignored; on
portal a symlink to /share/portal/kk837/simtoolreal/pretrained_policy), or ``$PRETRAINED_POLICY_DIR``.

Stdlib only: imported by Kit-free code and by deployment/.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEBUG_OUTPUTS_DIR = REPO_ROOT / "debug_outputs"
TRAIN_LOGS_DIR = DEBUG_OUTPUTS_DIR / "train_logs"
VIDEOS_DIR = DEBUG_OUTPUTS_DIR / "videos"
DEPLOY_DIR = DEBUG_OUTPUTS_DIR / "deployment"
SLURM_DIR = TRAIN_LOGS_DIR / "slurm"
RESULTS_DIR = TRAIN_LOGS_DIR / "results"
WANDB_DIR = TRAIN_LOGS_DIR  # wandb.init(dir=...) adds its own "wandb/" level
#: Hydra's run dir; without this Hydra writes ``outputs/<date>/<time>/`` under the working directory.
HYDRA_RUN_DIR = TRAIN_LOGS_DIR / "${now:%Y-%m-%d}" / "${now:%H-%M-%S}"
PRETRAINED_DIR = Path(os.environ.get("PRETRAINED_POLICY_DIR", REPO_ROOT / "pretrained_policy"))
PRETRAINED_CHECKPOINT = PRETRAINED_DIR / "model.pth"
PRETRAINED_CONFIG = PRETRAINED_DIR / "config.yaml"
