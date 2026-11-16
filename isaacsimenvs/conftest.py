"""Shared pytest setup for every test under isaacsimenvs/ (pytest loads this file automatically).

* Kit guard: in .venv/isaacsim, importing Isaac Lab would boot Kit. ``_NoKitIsaacLab`` makes those
  imports fail fast, and the hooks below turn that failure into a skip, so the Isaac-Lab-backed
  tests skip there and run in .venv/isaaclab3.
* Session fixtures: Kit-free modules that many tests share, loaded once per run with
  ``isaacsimenvs.utils.testing.load_module`` (by file path, bypassing package ``__init__`` chains).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from isaacsimenvs.utils.testing import REPO_ROOT, load_module  # noqa: F401  re-exported


class _NoKitIsaacLab:
    """Make ``import isaaclab*`` raise ModuleNotFoundError where it would boot Kit.

    Under ``.venv/isaacsim`` (Isaac Lab 2.3), importing ``isaaclab`` bootstraps the Omniverse Kit
    kernel: minutes of startup, an interactive EULA prompt, and a ``SystemExit`` when stdin is not a
    terminal -- which ``pytest.importorskip`` does not catch, so one such import took the whole
    session down. The Isaac-Lab-backed tests are written for the kit-less Isaac Lab 3 stack
    (``.venv/isaaclab3``); here they skip instead. Installed only when that stack is absent.
    """

    def find_spec(self, name, path=None, target=None):
        if name == "isaaclab" or name.startswith(("isaaclab.", "isaaclab_")):
            raise ModuleNotFoundError(f"{name}: Isaac Lab would boot Kit here; run this test in .venv/isaaclab3")
        return None


_KITLESS = importlib.util.find_spec("isaaclab_newton") is not None
if not _KITLESS:
    sys.meta_path.insert(0, _NoKitIsaacLab())

_NO_KIT_MARK = "Isaac Lab would boot Kit here"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item):
    yield from _skip_if_kit_blocked()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    yield from _skip_if_kit_blocked()


def _skip_if_kit_blocked():
    """A test (or fixture) that reached a blocked Isaac Lab import skips rather than errors."""
    outcome = yield
    exc = outcome.excinfo
    if exc is not None and issubclass(exc[0], ModuleNotFoundError) and _NO_KIT_MARK in str(exc[1]):
        outcome.force_exception(pytest.skip.Exception(str(exc[1]), _use_item_location=True))
