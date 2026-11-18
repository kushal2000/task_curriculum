"""isaacsimenvs: Isaac Sim / Isaac Lab environment package.

Importing ``isaacsimenvs`` (transitively) imports every task subpackage under
``isaacsimenvs.tasks``, firing their ``gym.register`` side effects. After this,
any task id registered by isaacsimenvs is resolvable via ``gym.spec`` /
``gym.make`` / ``isaaclab_tasks.utils.parse_cfg.load_cfg_from_registry``.
"""

from pathlib import Path as _Path

PACKAGE_ROOT = _Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
SCRIPTS_DIR = PACKAGE_ROOT / "scripts"
CONFIG_DIR = SCRIPTS_DIR / "configs"

# Task registration reads CONFIG_DIR, so it comes after the paths.
from . import tasks  # noqa: E402,F401  side effect: gym.register(...) for each task

__all__ = ["CONFIG_DIR", "PACKAGE_ROOT", "REPO_ROOT", "SCRIPTS_DIR", "tasks"]
