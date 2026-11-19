"""Hydra decorator variant: configclass defaults + robot setup + run recipe + CLI overrides.

Isaac Lab's stock `hydra_task_config` takes the configclass defaults and the Hydra CLI. This
decorator puts two YAML layers between them.

Precedence (lowest → highest):
    1. the env configclass defaults (Python): the task itself
    2. robots/setups/<setup>.yaml, the robot setup (`env.robot_setup`), chosen by the command line, else
       the run recipe, else an alias task id (see `resolve_task`)
    3. the run recipe (optional, `train.py --experiment <name or path>`; see recipes.py): its base
       recipe, its DR blocks (tasks/cloth/config/dr/), then its own `env:` and `agent:` overrides, so the recipe
       lives in the repo rather than in a launcher
    4. Hydra CLI overrides (e.g. `env.scene.num_envs=4096`)

Implementation mirrors `isaaclab_tasks.utils.hydra.register_task_to_hydra` /
`hydra_task_config`, with the `env_cfg.from_dict(...)` steps of layers 2 and 3 inserted before the
ConfigStore write. The wrapped function receives `(env_cfg, agent_cfg)` exactly as Isaac Lab's own
decorator does.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

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

from isaacsimenvs import ROBOT_SETUPS_DIR as ROBOT_DIR
from isaacsimenvs.tasks.recipes import deep_merge, load_recipe


def resolve_task(task_id: str) -> tuple[str, str]:
    """``(canonical task id, robot setup or "")``. An alias id (e.g.
    ``Isaacsimenvs-FoldPoolPhysxFlexivWujiRightReal-Direct-v0``) carries both in its gym kwargs."""
    kwargs = gym.spec(task_id).kwargs
    return kwargs.get("canonical_task", task_id), kwargs.get("robot_setup", "")


def _cli_value(key: str) -> str | None:
    """The value of a ``key=value`` hydra override on the command line, if given."""
    import sys

    for arg in sys.argv[1:]:
        if arg.startswith(f"{key}="):
            return arg.split("=", 1)[1]
    return None


def hydra_task_config_with_yaml(
    task_name: str,
    agent_cfg_entry_point: str,
    experiment: str | None = None,
) -> Callable:
    """Like `isaaclab_tasks.utils.hydra.hydra_task_config`, but layers the robot setup and the run
    recipe onto the configclass defaults before Hydra CLI merging.

    Args:
        task_name: Gym task id (e.g. ``"Isaacsimenvs-Cartpole-Direct-v0"``). A colon-separated
            prefix is stripped to match Isaac Lab's convention.
        agent_cfg_entry_point: Key in ``gym.register(..., kwargs=...)`` whose value
            is the agent (rl_games) config entry point. Typically
            ``"rl_games_cfg_entry_point"`` or ``"rl_games_sapg_cfg_entry_point"``.
        experiment: A run recipe (a name in ``tasks/cloth/config/runs/`` or a path; see recipes.py), or ``None``.
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
            canonical, setup = resolve_task(short_name)
            exp = load_recipe(experiment) if experiment else {}
            if exp:
                exp_canonical, exp_setup = resolve_task(exp.get("task", short_name))
                if exp_canonical != canonical:
                    raise ValueError(f"experiment {experiment!r} is for {exp['task']}, not {short_name}")
                exp_setup = (exp.get("env") or {}).get("robot_setup", exp_setup)
                if setup and exp_setup and exp_setup != setup:
                    raise ValueError(
                        f"experiment {experiment!r} is for robot setup {exp_setup!r}, the task id "
                        f"{short_name} for {setup!r}; pick one (env.robot_setup=...)"
                    )
                setup = setup or exp_setup
            setup = _cli_value("env.robot_setup") or setup
            if setup:
                robot_yaml = ROBOT_DIR / f"{setup}.yaml"
                if not robot_yaml.exists():
                    known = sorted(p.stem for p in ROBOT_DIR.glob("*.yaml"))
                    raise ValueError(f"unknown robot setup {setup!r}; known: {known}")
                with open(robot_yaml) as f:
                    env_cfg.from_dict(yaml.safe_load(f) or {})
                env_cfg.robot_setup = setup

            if exp:
                env_cfg.from_dict(exp.get("env") or {})
                if exp.get("agent"):
                    if isinstance(agent_cfg, dict):
                        agent_cfg = deep_merge(agent_cfg, exp["agent"])
                    else:
                        agent_cfg.from_dict(exp["agent"])
                print(f"[hydra] recipe {experiment}: {' <- '.join(exp['files'])}; DR blocks {exp['dr']}", flush=True)
            if setup:
                print(f"[hydra] robot setup {setup}: {ROBOT_DIR / setup}.yaml", flush=True)

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
