"""Read a task's config.yaml and apply command-line overrides.

Each task folder has one complete config (deploy/simtoolreal/config.yaml, deploy/rigid2deform/config.yaml).
A task's run.py loads its own; the shared nodes (deploy/nodes/) are told which one with config=<path>
(run.py passes it on when it starts them):

    python deploy/nodes/perception.py config=deploy/rigid2deform/config.yaml perception.source=fake

Each other override must name a key that already exists in that config, so a typo stops the run
instead of being silently ignored. The loaded config carries its own path as `config` (repo-relative),
so the recorder stores it with the run.
"""

import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

HERE = Path(__file__).resolve().parent      # deploy/utils
REPO = HERE.parents[1]                      # repo root; paths in a config are relative to it


def load_config(overrides: list[str] | None = None, path: Path | str | None = None) -> DictConfig:
    """The config at `path`, or at the config=<path> override (repo-relative or absolute), + the other
    overrides ("a.b=c"). overrides=None reads them from sys.argv."""
    if overrides is None:
        overrides = sys.argv[1:]
    given = [o.split("=", 1)[1] for o in overrides if o.startswith("config=")]
    overrides = [o for o in overrides if not o.startswith("config=")]
    if path is None:
        if not given:
            raise SystemExit("config=<path> missing, e.g. config=deploy/simtoolreal/config.yaml "
                             "(a task's run.py passes it to the nodes it starts)")
        path = given[-1]
    path = (REPO / path).resolve() if not Path(path).is_absolute() else Path(path)
    if given and (REPO / given[-1]).resolve() != path:
        raise SystemExit(f"config={given[-1]}, but this script reads {path.relative_to(REPO)}")
    if not path.is_file():
        raise SystemExit(f"config {path}: no such file")
    cfg = OmegaConf.load(path)
    cfg.config = str(path.relative_to(REPO))
    OmegaConf.set_struct(cfg, True)         # unknown keys -> error
    return OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))


def object_names(cfg: DictConfig) -> list[str]:
    """object.name as a list: one name, or several. The first is the policy's object (object_pose)."""
    name = cfg.object.name
    names = [name] if isinstance(name, str) else [str(n) for n in name]
    unknown = [n for n in names if n not in cfg.objects]
    if not names or unknown:
        raise SystemExit(f"object.name={name}: unknown {unknown}; known objects: {list(cfg.objects)}")
    return names


def object_meshes(cfg: DictConfig) -> list[Path]:
    """The mesh of each object in object.name, in its order."""
    names = object_names(cfg)
    missing = [n for n in names if cfg.objects[n].get("mesh") is None]
    if missing:
        raise SystemExit(f"objects {missing} have no mesh")
    return [repo_path(cfg.objects[n].mesh) for n in names]


def repo_path(p: str) -> Path:
    """A path from a config as an absolute path."""
    return REPO / p


if __name__ == "__main__":
    # python deploy/utils/config.py config=deploy/simtoolreal/config.yaml robot.mode=home  -> the final config
    print(OmegaConf.to_yaml(load_config()))
