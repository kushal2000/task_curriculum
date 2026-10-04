"""Load repo modules by file path, bypassing package `__init__` chains.

`import isaacsimenvs.<anything>` fires `isaacsimenvs/__init__.py`, which imports every
task package, which imports `isaaclab.envs` — and Isaac Lab's sub-namespaces only resolve
after `AppLauncher` has booted Kit. Booting Kit for a unit test would take minutes and a
GPU.

So the modules under test are deliberately Kit-free (`generate_cartpole`,
`generate_bottle`, `difficulty_math`, `curriculum/schedulers`), and this helper loads
them straight off disk so the package `__init__` never runs. If a test here starts
needing Kit, the module it is testing has grown a dependency it should not have.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(rel_path: str, name: str | None = None):
    """Import `rel_path` (relative to the repo root) as a standalone module."""
    path = REPO_ROOT / rel_path
    name = name or path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - import plumbing
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def gen_cartpole():
    return load_module(
        "isaacsimenvs/tasks/multilink_cartpole/utils/generate_cartpole.py", "gen_cartpole"
    )


@pytest.fixture(scope="session")
def gen_bottle():
    return load_module("isaacsimenvs/tasks/bottle_flip/utils/generate_bottle.py", "gen_bottle")


@pytest.fixture(scope="session")
def difficulty_math():
    return load_module(
        "isaacsimenvs/tasks/multilink_cartpole/utils/difficulty_math.py", "difficulty_math"
    )


@pytest.fixture(scope="session")
def schedulers():
    return load_module("isaacsimenvs/curriculum/schedulers.py", "schedulers")


@pytest.fixture(scope="session")
def rigid_cloth():
    return load_module("isaacsimenvs/tasks/cloth/utils/rigid_cloth.py", "rigid_cloth")


@pytest.fixture(scope="session")
def cloth_geometry():
    return load_module("isaacsimenvs/tasks/cloth/utils/cloth_geometry.py", "cloth_geometry")


@pytest.fixture(scope="session")
def rigid_cloth_adapter():
    """The articulation -> particle-grid bridge.

    Imported as a real package member rather than loaded by path, unlike everything else here,
    because it imports its own dependencies by package path -- which is the right thing for a module
    the env uses, and which a path-load would leave unresolved. That works only because the module
    is deliberately **kit-free**: pure torch, no ``isaaclab`` import, no Isaac Sim. If this fixture
    ever starts needing ``AppLauncher``, the adapter has grown a dependency that puts the grid
    mapping and the reset pose beyond the reach of a CPU test -- which is exactly where they must
    not be, since both fail silently.
    """
    return pytest.importorskip("isaacsimenvs.tasks.cloth.utils.rigid_cloth_adapter")
