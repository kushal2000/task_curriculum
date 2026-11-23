"""Hydra decorator variant: configclass defaults + run recipe + CLI overrides.

Isaac Lab's stock `hydra_task_config` takes the configclass defaults and the Hydra CLI. This
decorator puts the run recipe between them.

Precedence (lowest → highest):
    1. the env configclass defaults (Python): the task itself
    2. the run recipe (optional, `train.py --experiment <name or path>`; see recipes.py): its base
       recipe, its DR blocks (tasks/cloth/config/dr/), then its own `env:` and `agent:` overrides, so the recipe
       lives in the repo rather than in a launcher
    3. Hydra CLI overrides (e.g. `env.scene.num_envs=4096`)

Implementation mirrors `isaaclab_tasks.utils.hydra.register_task_to_hydra` /
`hydra_task_config`, with the `env_cfg.from_dict(...)` step of layer 2 inserted before the
ConfigStore write. The wrapped function receives `(env_cfg, agent_cfg)` exactly as Isaac Lab's own
decorator does.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable, Mapping

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from isaaclab.envs.utils.spaces import (
    replace_env_cfg_spaces_with_strings,
    replace_strings_with_env_cfg_spaces,
)
from isaaclab.utils import replace_slices_with_strings, replace_strings_with_slices
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

from isaacsimenvs.tasks.recipes import deep_merge, load_recipe


def _set_null_fields(cfg, data: Mapping) -> None:
    """Give the fields that are ``None`` in ``cfg`` and set in ``data`` their value.

    Isaac Lab's ``from_dict`` checks a value against the type of the field's current value, so a
    field that is ``None`` (``null`` in task.yaml, e.g. ``cloth.cloth_table_friction``) cannot take a
    float from a recipe, a protocol or the command line. Set those here first, as the annotation
    reads (an int for a float becomes a float, a list for a tuple a tuple); ``from_dict`` then sees
    a value of the right type. Every other field is left to ``from_dict``.
    """
    if not dataclasses.is_dataclass(cfg):
        return
    annotations = {f.name: str(f.type) for f in dataclasses.fields(cfg)}
    for key, value in data.items():
        if key not in annotations:
            continue
        current = getattr(cfg, key)
        if isinstance(value, Mapping):
            _set_null_fields(current, value)
        elif current is None and value is not None:
            kind = annotations[key].split("|")[0].strip()
            if kind == "float" and isinstance(value, int) and not isinstance(value, bool):
                value = float(value)
            elif kind.startswith("tuple") and isinstance(value, list):
                value = tuple(value)
            setattr(cfg, key, value)


def hydra_task_config_with_yaml(
    task_name: str,
    agent_cfg_entry_point: str,
    experiment: str | dict | None = None,
) -> Callable:
    """Like `isaaclab_tasks.utils.hydra.hydra_task_config`, but layers the run recipe onto the
    configclass defaults before Hydra CLI merging.

    Args:
        task_name: Gym task id (e.g. ``"Isaacsimenvs-Cartpole-Direct-v0"``). A colon-separated
            prefix is stripped to match Isaac Lab's convention.
        agent_cfg_entry_point: Key in ``gym.register(..., kwargs=...)`` whose value
            is the agent (rl_games) config entry point. Typically
            ``"rl_games_cfg_entry_point"`` or ``"rl_games_sapg_cfg_entry_point"``.
        experiment: A run recipe (a name in ``tasks/cloth/config/runs/`` or a path; see recipes.py), an
            already composed one (``load_recipe``'s format, e.g. a training's resolved env from its
            run.json, ``training_run.resolved_recipe``), or ``None``.
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
            exp = (experiment if isinstance(experiment, dict) else load_recipe(experiment)) if experiment else {}
            if exp:
                if exp.get("task", short_name) != short_name:
                    raise ValueError(f"experiment {experiment!r} is for {exp['task']}, not {short_name}")
                _set_null_fields(env_cfg, exp.get("env") or {})
                env_cfg.from_dict(exp.get("env") or {})
                if exp.get("agent") and agent_cfg is not None:   # (an evaluation loads no agent config)
                    if isinstance(agent_cfg, dict):
                        agent_cfg = deep_merge(agent_cfg, exp["agent"])
                    else:
                        agent_cfg.from_dict(exp["agent"])
                print(f"[hydra] recipe {experiment}: {' <- '.join(exp['files'])}; DR blocks {exp['dr']}", flush=True)

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
                _set_null_fields(env_cfg, resolved["env"])
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
