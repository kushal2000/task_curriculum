"""isaacsimenvs: Isaac Sim / Isaac Lab environment package.

Importing ``isaacsimenvs`` (transitively) imports every task subpackage under
``isaacsimenvs.tasks``, firing their ``gym.register`` side effects. After this,
any task id registered by isaacsimenvs is resolvable via ``gym.spec`` /
``gym.make`` / ``isaaclab_tasks.utils.parse_cfg.load_cfg_from_registry``.
"""

import os as _os
from pathlib import Path as _Path

PACKAGE_ROOT = _Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
SCRIPTS_DIR = PACKAGE_ROOT / "scripts"
#: YAML beside its owner: the cloth task's run recipes, DR blocks, fold designs and learner config; the
#: Play task's learner configs.
CLOTH_CONFIG_DIR = PACKAGE_ROOT / "tasks" / "cloth" / "config"
PLAY_CONFIG_DIR = PACKAGE_ROOT / "tasks" / "play" / "config"
#: Play priors and the policy network config (not in git; PRETRAINED_POLICY_DIR overrides the place).
PRETRAINED_DIR = _Path(_os.environ.get("PRETRAINED_POLICY_DIR", REPO_ROOT / "pretrained_policy"))
#: The default prior: Play on the Flexiv + right Wuji (nov8, DR, fingertip friction 0.5).
PRETRAINED_CHECKPOINT = PRETRAINED_DIR / "flexiv_right_wuji" / "nov8_dr_friction_0p5.pth"
PRETRAINED_CONFIG = PRETRAINED_DIR / "config.yaml"

# Task registration reads the config paths, so it comes after them.
from . import tasks  # noqa: E402,F401  side effect: gym.register(...) for each task

__all__ = [
    "CLOTH_CONFIG_DIR", "PACKAGE_ROOT", "PLAY_CONFIG_DIR", "PRETRAINED_CHECKPOINT", "PRETRAINED_CONFIG",
    "PRETRAINED_DIR", "REPO_ROOT", "SCRIPTS_DIR", "tasks",
]
