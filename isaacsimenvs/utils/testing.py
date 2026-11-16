"""Test helper: load a repo source file by path, without importing its package.

A module like ``isaacsimenvs/tasks/play/utils/point_dr.py`` is Kit-free on its own, but importing
it as ``isaacsimenvs.tasks.play.utils.point_dr`` runs ``isaacsimenvs/tasks/play/__init__.py`` and
the modules it pulls in, which import Isaac Lab -- and Isaac Lab only resolves after
``AppLauncher`` has booted Kit (minutes, and a GPU). Loading the file straight off disk skips those
``__init__`` chains. If a test that uses this starts needing Kit, the module under test has grown a
dependency it should not have.

Stdlib only.
"""

from __future__ import annotations

import importlib.util
import sys

from isaacsimenvs.utils.paths import REPO_ROOT


def load_module(rel_path: str, name: str | None = None):
    """Import ``rel_path`` (relative to the repo root) as a standalone module named ``name``."""
    path = REPO_ROOT / rel_path
    name = name or path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - import plumbing
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


__all__ = ["REPO_ROOT", "load_module"]
