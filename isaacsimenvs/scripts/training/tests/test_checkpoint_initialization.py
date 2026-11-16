"""Recipe initialization loads weights while explicit CLI restores retain their semantics."""
from types import SimpleNamespace as NS

import pytest

from isaacsimenvs.scripts.training.train import _checkpoint_options


def _args(**updates):
    return NS(**({"checkpoint": None, "checkpoint_load_mode": None, "test": False} | updates))


def test_recipe_initialization_is_weights_only_and_paths_are_repo_relative(tmp_path, monkeypatch):
    import isaacsimenvs

    relative = "pretrained_policy/flexiv_wuji_finetuned/flat7RW_fricition.pth"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.touch()
    monkeypatch.setattr(isaacsimenvs, "REPO_ROOT", tmp_path)
    cfg = {"params": {"load_checkpoint": True, "load_path": relative, "checkpoint_load_mode": "weights"}}
    assert _checkpoint_options(_args(), cfg) == {
        "checkpoint": str(path), "checkpoint_load_mode": "weights"}


def test_explicit_cli_checkpoint_and_mode_override_recipe(tmp_path):
    path = tmp_path / "resume.pth"
    path.touch()
    cfg = {"params": {"load_checkpoint": True, "load_path": "missing.pth", "checkpoint_load_mode": "weights"}}
    assert _checkpoint_options(_args(checkpoint=str(path)), cfg)["checkpoint_load_mode"] == "resume"
    assert _checkpoint_options(_args(checkpoint=str(path), checkpoint_load_mode="weights"), cfg) == {
        "checkpoint": str(path), "checkpoint_load_mode": "weights"}


def test_scratch_ignores_checkpoint_path_when_loading_is_disabled():
    cfg = {"params": {"load_checkpoint": False, "load_path": "missing.pth", "checkpoint_load_mode": "weights"}}
    assert _checkpoint_options(_args(), cfg)["checkpoint"] is None


def test_invalid_recipe_checkpoint_fails_before_environment_construction():
    with pytest.raises(ValueError, match="requires a nonempty load_path"):
        _checkpoint_options(_args(), {"params": {"load_checkpoint": True, "load_path": ""}})
    with pytest.raises(FileNotFoundError, match="Checkpoint does not exist"):
        _checkpoint_options(_args(), {"params": {"load_checkpoint": True, "load_path": "missing.pth"}})


def test_recipe_player_uses_network_restore():
    cfg = {"params": {"load_checkpoint": False, "checkpoint_load_mode": "weights"}}
    assert _checkpoint_options(_args(test=True), cfg)["checkpoint_load_mode"] == "resume"


def test_rlgames_weight_initialization_does_not_restore_optimizer_or_counters(monkeypatch):
    import torch
    from rl_games import torch_runner

    class Agent:
        has_central_value = False
        global_rank = 0

        def __init__(self):
            self.model = torch.nn.Linear(2, 1)
            self.optimizer = torch.optim.Adam(self.model.parameters())
            self.epoch_num = self.frame = 0

        def set_weights(self, weights):
            self.model.load_state_dict(weights["model"])

        def restore(self, filename):
            raise AssertionError("A finetune must not call full-state restore")

    agent = Agent()
    target = {"weight": torch.full((1, 2), .75), "bias": torch.full((1,), -.1)}
    historical = {0: {"model": target, "optimizer": {"old": True}, "epoch": 2646, "frame": 2080899072}}
    monkeypatch.setattr(torch_runner.torch_ext, "load_checkpoint", lambda filename: historical)
    torch_runner._restore(agent, {"checkpoint": "yibo.pth", "checkpoint_load_mode": "weights"})
    for name, value in agent.model.state_dict().items():
        torch.testing.assert_close(value, target[name])
    assert agent.optimizer.state == {} and agent.epoch_num == 0 and agent.frame == 0
