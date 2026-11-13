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
    """config.yaml + overrides ("a.b=c"). overrides=None reads them from sys.argv.
    task=fold: config.yaml, then fold.overrides, then the command line."""
    if overrides is None:
        overrides = sys.argv[1:]
    cfg = OmegaConf.load(path)
    OmegaConf.set_struct(cfg, True)         # unknown keys -> error
    cli = OmegaConf.from_dotlist(overrides)
    merged = OmegaConf.merge(cfg, cli)
    if merged.task == "fold":
        merged = OmegaConf.merge(cfg, cfg.fold.overrides, cli)
    elif merged.task != "play":
        raise SystemExit(f"task={merged.task}: play | fold")
    return merged


def object_names(cfg: DictConfig) -> list[str]:
    """object.name as a list: one name, or several. The first is the policy's object (object_pose)."""
    name = cfg.object.name
    names = [name] if isinstance(name, str) else list(name)
    unknown = [n for n in names if n not in cfg.objects]
    if not names or unknown:
        raise SystemExit(f"object.name={name}: unknown {unknown}; known objects: {list(cfg.objects)}")
    return names


def object_meshes(cfg: DictConfig) -> list[Path]:
    """The mesh of each object in object.name, in its order."""
    return [repo_path(cfg.objects[n].mesh) for n in object_names(cfg)]


def repo_path(p: str) -> Path:
    """A path from config.yaml as an absolute path."""
    return REPO / p


if __name__ == "__main__":
    # python deploy/simtoolreal/utils/config.py robot.mode=home  -> prints the final config
    print(OmegaConf.to_yaml(load_config()))
