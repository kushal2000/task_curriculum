"""Load repo modules by file path, for the rigid-cloth scripts that must not boot Kit.

`import isaacsimenvs.<anything>` fires `isaacsimenvs/__init__.py`, which imports every task package,
which imports `isaaclab.envs` -- and Isaac Lab's sub-namespaces only resolve after `AppLauncher` has
booted Kit. The geometry module, the MuJoCo probe and the figure scripts deliberately need neither,
so they load what they need straight off disk and the package `__init__` never runs.

`tests/conftest.py` holds the same helper for the same reason and keeps its own copy on purpose: a
conftest has to be importable with nothing but the repo on `sys.path`, and `scripts/` is reached
through implicit namespace packages (`python -m scripts.analysis.X`), which a pytest run does not
set up. Two copies of six lines, rather than a test suite that depends on a scripts directory.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_module(rel_path: str, name: str | None = None):
    """Import `rel_path` (relative to the repo root) as a standalone module."""
    path = REPO_ROOT / rel_path
    name = name or path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - import plumbing
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def rigid_cloth():
    """The chain geometry module -- `rc` in every caller."""
    return load_module("isaacsimenvs/tasks/cloth/utils/rigid_cloth.py", "rigid_cloth")


def rigid_cloth_probe():
    """The MuJoCo probe, which the figure scripts drive for `_build` / `_place` / `_run`."""
    return load_module("scripts/analysis/rigid_cloth_probe.py", "rigid_cloth_probe")
