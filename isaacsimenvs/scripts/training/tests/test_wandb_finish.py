"""Forced Kit shutdown must wait for queued metrics and HTML uploads first."""
import sys
from types import SimpleNamespace
from unittest.mock import Mock

from isaacsimenvs.scripts.training.train import _finish_wandb


def test_active_wandb_run_is_flushed_before_process_exit(monkeypatch):
    module = SimpleNamespace(run=object(), finish=Mock())
    monkeypatch.setitem(sys.modules, 'wandb', module)
    _finish_wandb()
    module.finish.assert_called_once_with(exit_code=0)


def test_nonlogging_rank_does_not_finish_another_run(monkeypatch):
    module = SimpleNamespace(run=None, finish=Mock())
    monkeypatch.setitem(sys.modules, 'wandb', module)
    _finish_wandb()
    module.finish.assert_not_called()


def test_failed_training_flushes_with_failed_status(monkeypatch):
    module = SimpleNamespace(run=object(), finish=Mock())
    monkeypatch.setitem(sys.modules, 'wandb', module)
    _finish_wandb(exit_code=1)
    module.finish.assert_called_once_with(exit_code=1)
