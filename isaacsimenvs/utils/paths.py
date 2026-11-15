"""Where things live, relative to this checkout rather than to one machine.

Every run artifact goes under ``OUTPUTS_DIR`` (gitignored): Hydra run dirs ``outputs/<date>/<time>/``,
``outputs/wandb/``, ``outputs/slurm/`` (scripts/slurm/submit.sh), ``outputs/results/`` (eval JSONs),
``outputs/videos/`` and ``outputs/deploy/``.

The pretrained tool-use prior is ``PRETRAINED_DIR`` (``<repo>/pretrained_policy``, gitignored; on
portal a symlink to /share/portal/kk837/simtoolreal/pretrained_policy), or ``$PRETRAINED_POLICY_DIR``.

Stdlib only: imported by Kit-free code and by deploy/.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = REPO_ROOT / "outputs"
SLURM_DIR = OUTPUTS_DIR / "slurm"
RESULTS_DIR = OUTPUTS_DIR / "results"
VIDEOS_DIR = OUTPUTS_DIR / "videos"
WANDB_DIR = OUTPUTS_DIR  # wandb.init(dir=...) adds its own "wandb/" level
PRETRAINED_DIR = Path(os.environ.get("PRETRAINED_POLICY_DIR", REPO_ROOT / "pretrained_policy"))
PRETRAINED_CHECKPOINT = PRETRAINED_DIR / "model.pth"
PRETRAINED_CONFIG = PRETRAINED_DIR / "config.yaml"
