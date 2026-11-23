"""Run recipes: one YAML per kind of run, layered over the task's config.

A recipe has up to five keys:

    task:  Isaacsimenvs-RigidCloth-Direct-v0   # the gym id it is written for
    base:  rigid_cloth_right_wuji              # another recipe underneath it (optional)
    dr:    [proprio, keypoints]                # DR blocks from tasks/cloth/config/dr/, in order (optional)
    env:   {...}                               # overrides of the env config
    agent: {...}                               # overrides of the learner config

`load_recipe` composes them as base -> each DR block -> this recipe's own `env` / `agent`, so a
recipe's own values win. A DR block is a plain env overlay (the same keys as a recipe's `env:`).

A recipe's `env.sim` may also hold the step sizes that live elsewhere in the env config
(`SIM_ALIASES`: decimation, the robot / cloth substeps, VBD iterations), so a file can write the
time loop in one block; `load_recipe` moves each to its real place. Only in recipe files: the
command line uses the real names (`env.newton.num_substeps=...`).

A recipe is named by a file stem in `tasks/cloth/config/runs/` or by a path, absolute or relative to the
repo root (e.g. `experiments/nov15_dr_delay/flat7RW_DR_delay.yaml`). A `base:` is looked up next to
the recipe first, then in `tasks/cloth/config/runs/`.

Kit-free (plain YAML), so tests and tools can compose recipes without booting a simulator.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from isaacsimenvs import CLOTH_CONFIG_DIR, REPO_ROOT

RUNS_DIR = CLOTH_CONFIG_DIR / "runs"
DR_DIR = CLOTH_CONFIG_DIR / "dr"

RECIPE_KEYS = {"task", "base", "dr", "env", "agent"}

#: ``env.sim.<alias>`` in a recipe file -> the env config path it sets: policy step = ``decimation`` x
#: ``sim.dt``; robot step = ``sim.dt / robot_substeps``; cloth step = robot step / ``cloth_substeps``,
#: solved with ``vbd_iterations``. None of these names is a field of Isaac Lab's ``SimulationCfg``.
SIM_ALIASES = {
    "decimation": ("decimation",),
    "robot_substeps": ("newton", "num_substeps"),
    "cloth_substeps": ("cloth", "substeps"),
    "vbd_iterations": ("cloth", "vbd_iterations"),
}


def expand_sim_aliases(env: dict, where: str = "recipe") -> dict:
    """``env`` with every ``sim.<alias>`` (``SIM_ALIASES``) moved to its real path. A value given both
    ways in the same overlay is an error, not a silent pick."""
    sim = env.get("sim")
    if not isinstance(sim, dict) or not set(sim) & set(SIM_ALIASES):
        return env
    out = {**env, "sim": {k: v for k, v in sim.items() if k not in SIM_ALIASES}}
    if not out["sim"]:
        del out["sim"]
    for alias, path in SIM_ALIASES.items():
        if alias not in sim:
            continue
        node = out
        for key in path[:-1]:
            node[key] = dict(node.get(key) or {})
            node = node[key]
        if path[-1] in node:
            raise ValueError(f"{where}: env.sim.{alias} and env.{'.'.join(path)} both set; keep one")
        node[path[-1]] = sim[alias]
    return out


def deep_merge(base: dict, over: dict) -> dict:
    """``over`` onto ``base``, recursing into dicts; any other value (lists included) replaces."""
    out = dict(base)
    for k, v in over.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


def recipe_path(name: str, near: Path | None = None) -> Path:
    """The file a recipe name or path refers to. ``near``: the recipe naming it as its ``base``; that
    recipe's directory is searched first (skipping the recipe itself, so ``base: x`` in ``.../x.yaml``
    means ``runs/x.yaml``)."""
    path = Path(name).expanduser()
    if not path.suffix:
        path = path.with_suffix(".yaml")
    if path.is_absolute():
        candidates = [path]
    else:
        candidates = ([near.parent / path] if near else []) + [REPO_ROOT / path, RUNS_DIR / path]
    for candidate in candidates:
        if candidate.is_file() and (near is None or candidate.resolve() != near):
            return candidate.resolve()
    raise FileNotFoundError(f"run recipe {name!r} not found (looked in {[str(c) for c in candidates]})")


def dr_block(name: str) -> dict:
    """``tasks/cloth/config/dr/<name>.yaml``: an env overlay."""
    path = DR_DIR / f"{name}.yaml"
    if not path.is_file():
        known = sorted(p.stem for p in DR_DIR.glob("*.yaml"))
        raise FileNotFoundError(f"unknown DR block {name!r}; known: {known}")
    return _yaml(path)


def load_recipe(name: str, _near: Path | None = None, _seen: frozenset = frozenset()) -> dict:
    """The composed recipe: ``{"task", "env", "agent", "dr", "files"}``. ``dr`` lists every DR block
    applied, base first; ``files`` every recipe file read, base first (for the log)."""
    path = recipe_path(name, _near)
    if path in _seen:
        raise ValueError(f"run recipe inheritance cycle at {path}")
    raw = _yaml(path)
    unknown = set(raw) - RECIPE_KEYS
    if unknown:
        raise ValueError(f"{path}: unknown top-level keys {sorted(unknown)} (allowed: {sorted(RECIPE_KEYS)})")

    base = raw.get("base")
    out = load_recipe(base, path, _seen | {path}) if base else {"env": {}, "agent": {}, "dr": [], "files": []}
    for block in raw.get("dr") or []:
        out["env"] = deep_merge(out["env"], dr_block(block))
        out["dr"] = [*out["dr"], block]
    out["env"] = deep_merge(out["env"], expand_sim_aliases(raw.get("env") or {}, str(path)))
    out["agent"] = deep_merge(out["agent"], raw.get("agent") or {})
    if raw.get("task"):
        out["task"] = raw["task"]
    out["files"] = [*out["files"], str(path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path)]
    return out
