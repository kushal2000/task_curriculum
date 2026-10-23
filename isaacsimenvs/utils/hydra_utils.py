"""Hydra decorator variant: configclass defaults + YAML overlay + CLI overrides.

Isaac Lab's stock `hydra_task_config` uses `configclass defaults` as the sole
source; task-level YAML presets are not supported. We want the isaacgymenvs
composition pattern — a task YAML file sits between the configclass defaults
and the Hydra CLI — so this decorator adds that middle layer.

Precedence (lowest → highest):
    1. CartpoleEnvCfg() defaults (Python)
    2. cfg/task/Cartpole.yaml overlay (file — optional, via new gym kwarg
       `env_cfg_yaml_entry_point`)
    3. cfg/experiment/<name>.yaml (optional, `train.py --experiment <name>`): one run recipe's
       `env:` and `agent:` overrides, so the recipe lives in the repo rather than in a launcher
    4. Hydra CLI overrides (e.g. `env.scene.num_envs=4096`)

Implementation mirrors `isaaclab_tasks.utils.hydra.register_task_to_hydra` /
`hydra_task_config`, with one inserted `env_cfg.from_dict(yaml_overlay)` step
before the ConfigStore write. The wrapped function receives `(env_cfg, agent_cfg)`
exactly as Isaac Lab's own decorator does.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from pathlib import Path

import gymnasium as gym
import hydra
import yaml
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from isaaclab.envs.utils.spaces import (
    replace_env_cfg_spaces_with_strings,
    replace_strings_with_env_cfg_spaces,
)
from isaaclab.utils import replace_slices_with_strings, replace_strings_with_slices
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

EXPERIMENT_DIR = Path(__file__).resolve().parents[1] / "cfg" / "experiment"


def _deep_merge(base: dict, over: dict) -> dict:
    """``over`` onto ``base``, recursing into dicts; any other value (lists included) replaces."""
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_experiment(name: str) -> dict:
    """``cfg/experiment/<name>.yaml`` with its ``base:`` chain merged underneath it."""
    with open(EXPERIMENT_DIR / f"{name}.yaml") as f:
        exp = yaml.safe_load(f) or {}
    base = exp.pop("base", None)
    return _deep_merge(load_experiment(base), exp) if base else exp


def hydra_task_config_with_yaml(
    task_name: str,
    agent_cfg_entry_point: str,
    yaml_entry_point: str = "env_cfg_yaml_entry_point",
    experiment: str | None = None,
) -> Callable:
    """Like `isaaclab_tasks.utils.hydra.hydra_task_config`, but layers a task YAML
    overlay on top of the configclass defaults before Hydra CLI merging.

    Args:
        task_name: Gym task id (e.g. ``"Isaacsimenvs-Cartpole-Direct-v0"``). A colon-separated
            prefix is stripped to match Isaac Lab's convention.
        agent_cfg_entry_point: Key in ``gym.register(..., kwargs=...)`` whose value
            is the agent (rl_games) config entry point. Typically
            ``"rl_games_cfg_entry_point"`` or ``"rl_games_sapg_cfg_entry_point"``.
        yaml_entry_point: Key in the task's gym kwargs pointing to the task YAML
            overlay. If absent from the registered kwargs, no overlay is applied
            and behavior matches Isaac Lab's stock decorator.
        experiment: Name of a ``cfg/experiment/`` YAML layered over the task YAML, or ``None``.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            short_name = task_name.split(":")[-1]

            env_cfg = load_cfg_from_registry(short_name, "env_cfg_entry_point")
            agent_cfg = (
                load_cfg_from_registry(short_name, agent_cfg_entry_point)
                if agent_cfg_entry_point
                else None
            )
            # Overlay the task YAML onto the configclass defaults (if registered).
            # `gym.spec(...).kwargs` is where `gym.register(kwargs=...)` puts them.
            yaml_path = gym.spec(short_name).kwargs.get(yaml_entry_point)
            if yaml_path:
                with open(yaml_path) as f:
                    overlay = yaml.safe_load(f) or {}
                # `configclass.from_dict` walks the dict and applies it onto the
                # typed configclass, recursing into nested configclasses — exactly
                # what our hand-rolled `apply_task_overrides` walker does, but
                # upstream-supported.
                env_cfg.from_dict(overlay)
            # A robot overlay (cfg/robot/) on top, for task ids that swap the robot.
            robot_yaml = gym.spec(short_name).kwargs.get("env_cfg_robot_yaml_entry_point")
            if robot_yaml:
                with open(robot_yaml) as f:
                    env_cfg.from_dict(yaml.safe_load(f) or {})

            if experiment:
                exp = load_experiment(experiment)
                if exp.get("task", short_name) != short_name:
                    raise ValueError(f"experiment {experiment!r} is for {exp['task']}, not {short_name}")
                env_cfg.from_dict(exp.get("env") or {})
                if exp.get("agent"):
                    if isinstance(agent_cfg, dict):
                        agent_cfg = _deep_merge(agent_cfg, exp["agent"])
                    else:
                        agent_cfg.from_dict(exp["agent"])
                print(f"[hydra] experiment {experiment}: {EXPERIMENT_DIR / experiment}.yaml", flush=True)

            env_cfg = replace_env_cfg_spaces_with_strings(env_cfg)
            env_cfg_dict = env_cfg.to_dict()
            if isinstance(agent_cfg, dict) or agent_cfg is None:
                agent_cfg_dict = agent_cfg
            else:
                agent_cfg_dict = agent_cfg.to_dict()
            cfg_dict = {"env": env_cfg_dict, "agent": agent_cfg_dict}
            cfg_dict = replace_slices_with_strings(cfg_dict)
            ConfigStore.instance().store(name=short_name, node=cfg_dict)

            @hydra.main(config_path=None, config_name=short_name, version_base="1.3")
            def hydra_main(
                hydra_cfg: DictConfig,
                env_cfg=env_cfg,
                agent_cfg=agent_cfg,
            ):
                resolved = OmegaConf.to_container(hydra_cfg, resolve=True)
                resolved = replace_strings_with_slices(resolved)
                env_cfg.from_dict(resolved["env"])
                env_cfg_restored = replace_strings_with_env_cfg_spaces(env_cfg)
                if isinstance(agent_cfg, dict) or agent_cfg is None:
                    merged_agent = resolved["agent"]
                else:
                    agent_cfg.from_dict(resolved["agent"])
                    merged_agent = agent_cfg
                func(env_cfg_restored, merged_agent, *args, **kwargs)

            hydra_main()

        return wrapper

    return decorator
