"""Load repo modules by file path, bypassing package `__init__` chains.

`import isaacsimenvs.<anything>` fires `isaacsimenvs/__init__.py`, which imports every
task package, which imports `isaaclab.envs` — and Isaac Lab's sub-namespaces only resolve
after `AppLauncher` has booted Kit. Booting Kit for a unit test would take minutes and a
GPU.

So the modules under test are deliberately Kit-free (`rigid_cloth`, `fold_shapes`, ...),
and this helper loads them straight off disk so the package `__init__` never runs. If a test here starts
needing Kit, the module it is testing has grown a dependency it should not have.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(rel_path: str, name: str | None = None, package: bool = False):
    """Import `rel_path` (relative to the repo root) as a standalone module; with ``package`` the
    path is a package's ``__init__.py`` and its relative imports resolve inside that directory."""
    path = REPO_ROOT / rel_path
    name = name or path.stem
    spec = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=[str(path.parent)] if package else None
    )
    if spec is None or spec.loader is None:  # pragma: no cover - import plumbing
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def fold_shapes():
    return load_module("isaacsimenvs/tasks/cloth/mdp/geometry/fold_shapes.py", "fold_shapes")


@pytest.fixture(scope="session")
def cloth_geometry():
    return load_module("isaacsimenvs/tasks/cloth/mdp/geometry/grid.py", "cloth_geometry")
