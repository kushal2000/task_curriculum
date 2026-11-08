"""Read config.yaml and apply command-line overrides.

    python policy.py goals.timeout_steps=300 run.duration_s=30

Each override must name a key that already exists in config.yaml, so a typo stops the run
instead of being silently ignored.
"""

import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

HERE = Path(__file__).resolve().parent      # deploy/simtoolreal/utils
REPO = HERE.parents[2]                      # repo root; paths in config.yaml are relative to it
DEFAULT_CONFIG = HERE.parent / "config.yaml"


def load_config(overrides: list[str] | None = None, path: Path = DEFAULT_CONFIG) -> DictConfig:
    """config.yaml + overrides ("a.b=c"). overrides=None reads them from sys.argv."""
    if overrides is None:
        overrides = sys.argv[1:]
    cfg = OmegaConf.load(path)
    OmegaConf.set_struct(cfg, True)         # unknown keys -> error
    return OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))


def repo_path(p: str) -> Path:
    """A path from config.yaml as an absolute path."""
    return REPO / p


if __name__ == "__main__":
    # python deploy/simtoolreal/utils/config.py robot.mode=home  -> prints the final config
    print(OmegaConf.to_yaml(load_config()))
