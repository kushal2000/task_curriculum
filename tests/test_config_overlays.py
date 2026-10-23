"""Every key in a task YAML must name a real field on its configclass.

`configclass.from_dict` walks the overlay and applies it onto the typed config, so a
misspelled or stale YAML key fails at env-construction time — after Kit has booted and
the assets have been generated. That is an expensive way to find a typo, and this test
finds it in milliseconds instead.

The check is static (AST over the cfg modules) rather than importing them, because
importing a configclass pulls in `isaaclab.utils`, which needs a booted Kit.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from conftest import REPO_ROOT  # noqa: E402  pytest puts tests/ on sys.path

# Fields DirectRLEnvCfg / SimulationCfg / InteractiveSceneCfg contribute. Listed rather
# than parsed because they live in the installed isaaclab wheel, whose layout is not this
# repo's business.
DIRECT_RL_FIELDS = {
    "decimation", "episode_length_s", "action_space", "observation_space", "state_space",
    "sim", "scene", "viewer", "events", "seed", "rerender_on_reset", "wait_for_textures",
    "num_rerenders_on_reset", "is_finite_horizon", "action_noise_model",
    "observation_noise_model", "xr", "ui_window_class_type",
}
SIM_FIELDS = {"dt", "render_interval", "gravity", "physx", "device", "use_fabric"}
PHYSX_FIELDS = {
    "solver_type", "min_position_iteration_count", "max_position_iteration_count",
    "min_velocity_iteration_count", "max_velocity_iteration_count",
    "bounce_threshold_velocity", "friction_offset_threshold",
    "friction_correlation_distance", "enable_stabilization", "gpu_max_rigid_contact_count",
}
SCENE_FIELDS = {
    "num_envs", "env_spacing", "replicate_physics", "clone_in_fabric",
    "lazy_sensor_update", "filter_collisions",
}

CURRICULUM_CFG = "isaacsimenvs/curriculum/cfg.py"
PLAY_CFG = "isaacsimenvs/tasks/play/play_env_cfg.py"

# (yaml, root configclass, modules). Modules are searched in order and the first
# definition of a class name wins, so a task's own `RewardCfg` shadows play's.
CASES = [
    pytest.param(
        "isaacsimenvs/cfg/task/MultiLinkCartpole.yaml",
        "MultiLinkCartpoleEnvCfg",
        ["isaacsimenvs/tasks/multilink_cartpole/multilink_cartpole_env_cfg.py", CURRICULUM_CFG],
        id="MultiLinkCartpole",
    ),
    pytest.param(
        "isaacsimenvs/cfg/task/BottleFlip.yaml",
        "BottleFlipEnvCfg",
        ["isaacsimenvs/tasks/bottle_flip/bottle_flip_env_cfg.py", PLAY_CFG, CURRICULUM_CFG],
        id="BottleFlip",
    ),
    pytest.param(
        "isaacsimenvs/cfg/task/Play.yaml", "PlayEnvCfg", [PLAY_CFG], id="Play",
    ),
]


def _collect_classes(modules: list[str]) -> dict:
    classes: dict[str, tuple] = {}
    for rel in modules:
        tree = ast.parse((REPO_ROOT / rel).read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef) or node.name in classes:
                continue
            fields, annotations = set(), {}
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    fields.add(stmt.target.id)
                    ann = stmt.annotation
                    annotations[stmt.target.id] = ann.id if isinstance(ann, ast.Name) else None
            bases = [b.id for b in node.bases if isinstance(b, ast.Name)]
            classes[node.name] = (bases, fields, annotations)
    return classes


def _resolve(classes: dict, name: str, seen=None) -> tuple[set, dict] | None:
    """Fields and annotations of `name`, with subclass definitions winning over bases."""
    if name not in classes:
        return None
    seen = seen or set()
    if name in seen:
        return set(), {}
    seen.add(name)

    bases, fields, annotations = classes[name]
    out_fields, out_ann = set(), {}
    for base in bases:  # bases first...
        resolved = _resolve(classes, base, seen)
        if resolved:
            out_fields |= resolved[0]
            out_ann.update(resolved[1])
    out_fields |= fields  # ...so the subclass overrides
    out_ann.update(annotations)
    return out_fields, out_ann


@pytest.mark.parametrize("yaml_rel,root_cls,modules", CASES)
def test_every_yaml_key_maps_to_a_config_field(yaml_rel, root_cls, modules):
    classes = _collect_classes(modules)
    resolved = _resolve(classes, root_cls)
    assert resolved is not None, f"could not find {root_cls} in {modules}"
    fields, annotations = resolved

    overlay = yaml.safe_load((REPO_ROOT / yaml_rel).read_text()) or {}
    problems: list[str] = []

    for key, value in overlay.items():
        if key not in fields and key not in DIRECT_RL_FIELDS:
            problems.append(f"top-level '{key}' is not a field of {root_cls}")
            continue
        if not isinstance(value, dict):
            continue

        if key == "sim":
            problems += [f"sim.{k}" for k in value if k not in SIM_FIELDS]
            problems += [f"sim.physx.{k}" for k in value.get("physx", {}) if k not in PHYSX_FIELDS]
        elif key == "scene":
            problems += [f"scene.{k}" for k in value if k not in SCENE_FIELDS]
        else:
            section_cls = annotations.get(key)
            section = _resolve(classes, section_cls) if section_cls else None
            if not section:
                problems.append(f"cannot resolve section '{key}' (annotated {section_cls})")
                continue
            problems += [
                f"{key}.{k} is not a field of {section_cls}" for k in value if k not in section[0]
            ]

    assert not problems, f"{yaml_rel}:\n  " + "\n  ".join(problems)


def test_cartpole_link_lengths_match_n_max():
    """`n_max` sets the observation width and `link_lengths` sets the geometry; if they
    disagree the env raises at scene setup, so catch it in the config instead."""
    overlay = yaml.safe_load(
        (REPO_ROOT / "isaacsimenvs/cfg/task/MultiLinkCartpole.yaml").read_text()
    )
    geometry = overlay["geometry"]
    assert len(geometry["link_lengths"]) == geometry["n_max"]


def test_curriculum_ranges_are_ordered_and_bounded():
    for rel in ("isaacsimenvs/cfg/task/MultiLinkCartpole.yaml",
                "isaacsimenvs/cfg/task/BottleFlip.yaml"):
        curriculum = yaml.safe_load((REPO_ROOT / rel).read_text())["curriculum"]
        for key in ("init_range", "final_range"):
            lo, hi = curriculum[key]
            assert 0.0 <= lo <= hi <= 1.0, f"{rel}: {key}={curriculum[key]} outside [0, 1]"
        # The adaptive scheduler clamps range_hi into [init_range[1], final_range[1]],
        # so an init ceiling above the final one would freeze the curriculum immediately.
        assert curriculum["init_range"][1] <= curriculum["final_range"][1], rel


# --- cfg/experiment/: run recipes layered over a task YAML (train.py --experiment) ---------------

EXPERIMENT_DIR = REPO_ROOT / "isaacsimenvs/cfg/experiment"
# Per task id: env root configclass, its modules, and the SAPG agent YAML.
EXPERIMENT_TASKS = {
    "Isaacsimenvs-FoldPoolPhysx-Direct-v0": (
        "FoldPoolPhysxEnvCfg",
        [
            "isaacsimenvs/tasks/cloth/physx_env_cfg.py",
            "isaacsimenvs/tasks/cloth/cloth_cfg.py",
            "isaacsimenvs/tasks/cloth/fold_pool_cfg.py",
            "isaacsimenvs/tasks/cloth/rigid_cloth_cfg.py",
            PLAY_CFG,
            CURRICULUM_CFG,
        ],
        "isaacsimenvs/cfg/train/ClothSAPG.yaml",
    ),
}
# The robot-swapped ids (tasks/cloth/__init__.py) share their base task's cfg and agent YAML.
for _robot in ("FlexivWuji", "FlexivWujiHumanLike"):
    EXPERIMENT_TASKS[f"Isaacsimenvs-FoldPoolPhysx{_robot}-Direct-v0"] = EXPERIMENT_TASKS[
        "Isaacsimenvs-FoldPoolPhysx-Direct-v0"
    ]


def _load_experiment(name: str) -> dict:
    """As `hydra_utils.load_experiment`, which needs a booted Kit to import."""
    exp = yaml.safe_load((EXPERIMENT_DIR / f"{name}.yaml").read_text()) or {}
    base = exp.pop("base", None)
    if not base:
        return exp

    def merge(a, b):
        return {**a, **{k: merge(a[k], v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
                        for k, v in b.items()}}

    return merge(_load_experiment(base), exp)


def _env_problems(overlay: dict, classes: dict, cls: str, path: str) -> list[str]:
    fields, annotations = _resolve(classes, cls)
    problems = []
    for key, value in overlay.items():
        where = f"{path}{key}"
        if path == "" and key in DIRECT_RL_FIELDS:
            allowed = {"sim": SIM_FIELDS, "scene": SCENE_FIELDS}.get(key)
            if isinstance(value, dict) and allowed is not None:
                problems += [f"{where}.{k}" for k in value if k not in allowed]
        elif key not in fields:
            problems.append(f"{where} is not a field of {cls}")
        elif isinstance(value, dict):
            sub = annotations.get(key)
            if sub not in classes:
                problems.append(f"cannot resolve {where} (annotated {sub})")
            else:
                problems += _env_problems(value, classes, sub, where + ".")
    return problems


def _agent_problems(overlay: dict, ref: dict, path: str) -> list[str]:
    problems = []
    for key, value in overlay.items():
        if key not in ref:
            problems.append(f"agent.{path}{key} is not in the agent YAML")
        elif isinstance(value, dict):
            problems += _agent_problems(value, ref[key], f"{path}{key}.")
    return problems


@pytest.mark.parametrize("name", sorted(p.stem for p in EXPERIMENT_DIR.glob("*.yaml")))
def test_every_experiment_key_maps_to_a_config_field(name):
    exp = _load_experiment(name)
    assert set(exp) <= {"task", "env", "agent"}, f"{name}: unknown top-level keys {set(exp)}"
    assert exp.get("task") in EXPERIMENT_TASKS, f"{name}: task {exp.get('task')!r} has no checker entry"
    root_cls, modules, agent_yaml = EXPERIMENT_TASKS[exp["task"]]
    problems = _env_problems(exp.get("env") or {}, _collect_classes(modules), root_cls, "")
    problems += _agent_problems(exp.get("agent") or {}, yaml.safe_load((REPO_ROOT / agent_yaml).read_text()), "")
    assert not problems, f"{name}:\n  " + "\n  ".join(problems)
