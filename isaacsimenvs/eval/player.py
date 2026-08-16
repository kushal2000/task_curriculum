"""Load the released SimToolReal checkpoint and drive it from a torch observation tensor.

A repo-local port of ``deployment/rl_player.py`` from the upstream SimToolReal checkout. It is
ported rather than imported because upstream's version pulls ``isaacgymenvs.utils.reformat``,
which does not exist here.

The checkpoint is an asymmetric-actor-critic LSTM (``rnn.name: lstm``, 1024 units,
``before_mlp: true``, ``seq_length: 16``) trained under the Isaac Gym pipeline with SAPG, with
``normalize_input`` and ``normalize_value`` both on. Its running-mean/std statistics are baked
against the exact observation layout below, so the observation must be assembled by the env
rather than reconstructed here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

# The observation the policy was trained on. `play`'s obs_list produces exactly this.
POLICY_OBS_DIM = 140
POLICY_ACTION_DIM = 29

# SAPG conditions the policy on an exploration coefficient appended as a final observation
# column, so the network takes 141 inputs for a 140-dim environment observation. Upstream's
# player calls this the "SAPG HACK" (deployment/rl_player.py:97-102) and uses 50.0, the value
# the released checkpoint was evaluated at.
#
# Omitting it does not crash: the first Linear would simply reject the shape -- but any code that
# pads or reshapes instead produces finite, plausible, and wrong actions. Asserted below rather
# than trusted.
SAPG_EXPL_COEF = 50.0

# Registered by `isaacgymenvs/__init__.py` upstream. The released config interpolates through all
# of them (`${resolve_default:...}` on the run name, `${eval:"None"}` on several task fields), so
# resolution fails with UnsupportedInterpolationType without them.
_RESOLVERS = {
    "eq": lambda x, y: x.lower() == y.lower(),
    "contains": lambda x, y: x.lower() in y.lower(),
    "if": lambda pred, a, b: a if pred else b,
    "resolve_default": lambda default, arg: default if arg == "" else arg,
    "eval": lambda x: eval(x),  # noqa: S307 - upstream's resolver, used on its own config
}


def _register_resolvers() -> None:
    for name, fn in _RESOLVERS.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, fn)


def _to_dict(cfg: DictConfig) -> dict:
    out: dict = {}
    for key, value in cfg.items():
        out[key] = _to_dict(value) if isinstance(value, DictConfig) else value
    return out


def read_policy_cfg(config_path: str | Path, device: str) -> dict:
    """Read the checkpoint's own ``config.yaml``, resolving upstream's custom interpolations."""
    _register_resolvers()
    raw = yaml.safe_load(Path(config_path).read_text())
    cfg = _to_dict(OmegaConf.create(raw))
    train_cfg = cfg["train"]["params"]["config"]
    train_cfg["device"] = device
    train_cfg["device_name"] = device
    return cfg


class _SpacesOnlyEnv:
    """The minimum rl_games needs to build a player: the spaces, and nothing else.

    ``Runner.create_player`` resolves its env through ``env_configurations``, so a placeholder has
    to be registered even though the real stepping is driven by the caller.
    """

    def __init__(self, num_observations: int, num_actions: int, num_envs: int) -> None:
        from gym import spaces

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_observations,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(num_actions,), dtype=np.float32)
        self.num_envs = num_envs

    def get_env_info(self) -> dict:
        return {"observation_space": self.observation_space, "action_space": self.action_space}

    def set_env_state(self, *args: Any, **kwargs: Any) -> None:
        return None


class PretrainedPlayer:
    """The released checkpoint, ready to map a 140-dim observation batch to 29-dim actions."""

    def __init__(
        self,
        config_path: str | Path,
        checkpoint_path: str | Path,
        num_envs: int,
        device: str = "cuda:0",
        num_observations: int = POLICY_OBS_DIM,
        num_actions: int = POLICY_ACTION_DIM,
    ) -> None:
        from rl_games.common import env_configurations
        from rl_games.torch_runner import Runner

        self.device = device
        self.num_observations = num_observations
        self.num_actions = num_actions

        self.cfg = read_policy_cfg(config_path, device)

        # The space is the *environment's* 140, not the 141 columns handed to `get_action`.
        # rl_games derives both the input normalizer and the network input width from it:
        # `running_mean_std` is built over `obs_shape` (140, matching the checkpoint), while
        # `PpoPlayerContinuous.__init__` sets the network input to
        # `obs_shape + intr_reward_coef_embd.shape[1]` = 140 + 32 = 172, which is the LSTM's
        # `weight_ih_l0` width in the checkpoint. The trailing coefficient column is located by
        # `coef_id_idx = obs_shape[0]` and expanded into that 32-dim embedding internally
        # (`expl_type: mixed_expl_learn_param` selects the 'extra_param' model).
        #
        # Passing 141 here instead makes every one of those 140s a 141 and the restore fails on
        # a shape mismatch -- which is the good outcome. It is recorded because the failure names
        # `running_mean_std`, not the coefficient, and reads like a corrupt checkpoint.
        placeholder = _SpacesOnlyEnv(num_observations, num_actions, num_envs)
        env_configurations.register(
            "rlgpu", {"env_creator": lambda **kw: placeholder, "vecenv_type": "RLGPU"}
        )

        train_cfg = dict(self.cfg["train"])
        train_cfg["load_path"] = str(checkpoint_path)

        runner = Runner()
        runner.load(train_cfg)
        self.player = runner.create_player()
        self.player.init_rnn()
        self.player.has_batch_dimension = True
        self.player.restore(str(checkpoint_path))

    def get_action(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Map a ``(num_envs, 140)`` observation batch to ``(num_envs, 29)`` normalized actions."""
        if obs.shape[-1] != self.num_observations:
            raise ValueError(
                f"expected a {self.num_observations}-dim observation, got {obs.shape[-1]}. "
                "The checkpoint's input normalization is baked against the exact layout in "
                "cfg/task/Play.yaml obs_list -- a different width means a different env, not a "
                "reshape."
            )
        obs = obs.to(self.device)
        expl = torch.full((obs.shape[0], 1), SAPG_EXPL_COEF, device=self.device, dtype=obs.dtype)
        return self.player.get_action(obs=torch.cat([obs, expl], dim=1), is_deterministic=deterministic)

    def reset_rnn(self, env_ids: torch.Tensor | None = None) -> None:
        """Zero the LSTM hidden state, for all envs or a subset.

        rl_games stores states as ``(layers, num_envs, units)`` tensors. Resetting per-env matters
        once episodes end at different steps: carrying a finished episode's hidden state into the
        next one is not what the policy saw in training.
        """
        states = getattr(self.player, "states", None)
        if not states:
            return
        for state in states:
            if state is None:
                continue
            if env_ids is None:
                state.zero_()
            else:
                state[:, env_ids] = 0.0
