"""Where things live, relative to this checkout rather than to one machine.

Every run artifact goes under ``DEBUG_OUTPUTS_DIR`` (``debug_outputs/``, gitignored):

* ``debug_outputs/train_logs/`` (``TRAIN_LOGS_DIR``): Hydra run dirs ``<date>/<time>/0_<name>/``,
  ``wandb/``, ``slurm/`` (scripts/slurm/submit.sh) and ``results/`` (eval JSONs, probe dumps).
* ``debug_outputs/videos/`` (``VIDEOS_DIR``): rendered videos.
* ``debug_outputs/deployment/`` (``DEPLOY_DIR``): real-robot run recordings and logs (deployment/).

The Play priors are in ``PRETRAINED_DIR`` (``<repo>/pretrained_policy``, gitignored, or
``$PRETRAINED_POLICY_DIR``), one ``<robot>/{model.pth,config.yaml}`` per robot: ``kuka_sharpa/`` (the generic
SimToolReal prior; on portal symlinks into /share/portal/kk837/simtoolreal/pretrained_policy) and
``flexiv_wuji/`` (the right-Wuji human-like Play prior, run play_scratch_flexiv_right_dr_cont_41213).

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
PRETRAINED_KUKA_SHARPA = PRETRAINED_DIR / "kuka_sharpa"
PRETRAINED_FLEXIV_WUJI = PRETRAINED_DIR / "flexiv_wuji"
#: The default prior evals load: Kuka + Sharpa.
PRETRAINED_CHECKPOINT = PRETRAINED_KUKA_SHARPA / "model.pth"
PRETRAINED_CONFIG = PRETRAINED_KUKA_SHARPA / "config.yaml"
