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

from isaacsimenvs import REPO_ROOT

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

PLAY_CFG = "isaacsimenvs/tasks/play/play_env_cfg.py"

# (yaml, root configclass, modules). Modules are searched in order and the first
# definition of a class name wins, so a task's own `RewardCfg` shadows play's.
CASES = [
    pytest.param(
        "isaacsimenvs/scripts/configs/task/Play.yaml", "PlayEnvCfg", [PLAY_CFG], id="Play",
    ),
    pytest.param(
        "isaacsimenvs/scripts/configs/task/PlayFlexivWujiRightHumanLike.yaml", "PlayFlexivWujiRightHumanLikeEnvCfg",
        [PLAY_CFG], id="PlayFlexivWujiRightHumanLike",
    ),
    pytest.param(
        "isaacsimenvs/scripts/configs/task/RigidBodyCloth.yaml", "RigidClothEnvCfg",
        ["isaacsimenvs/tasks/cloth/backends/rigid/config.py", "isaacsimenvs/tasks/cloth/cloth_cfg.py", "isaacsimenvs/tasks/cloth/mdp/config.py"],
        id="RigidBodyCloth",
    ),
    pytest.param(
        "isaacsimenvs/scripts/configs/task/Cloth.yaml", "DeformableClothEnvCfg",
        ["isaacsimenvs/tasks/cloth/backends/vbd/config.py", "isaacsimenvs/tasks/cloth/cloth_cfg.py", "isaacsimenvs/tasks/cloth/mdp/config.py"],
        id="Cloth",
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


# --- isaacsimenvs/scripts/configs/runs/: run recipes layered over a task YAML (train.py --experiment) ---------------

EXPERIMENT_DIR = REPO_ROOT / "isaacsimenvs/scripts/configs/runs"
# Per task id: env root configclass, its modules, and the SAPG agent YAML.
EXPERIMENT_TASKS = {
    # The deformable (VBD) sheet on Newton.
    "Isaacsimenvs-Cloth-Direct-v0": (
        "DeformableClothEnvCfg",
        [
            "isaacsimenvs/tasks/cloth/backends/vbd/config.py",
            "isaacsimenvs/tasks/cloth/cloth_cfg.py", "isaacsimenvs/tasks/cloth/mdp/config.py",
        ],
        "isaacsimenvs/scripts/configs/train/ClothSAPG.yaml",
    ),
    # Rigid2Deform: the single-hinge rigid sheet on PhysX (tasks/cloth/backends/rigid/backend.py).
    "Isaacsimenvs-RigidBodyCloth-Direct-v0": (
        "RigidClothEnvCfg",
        [
            "isaacsimenvs/tasks/cloth/backends/rigid/config.py",
            "isaacsimenvs/tasks/cloth/cloth_cfg.py", "isaacsimenvs/tasks/cloth/mdp/config.py",
        ],
        "isaacsimenvs/scripts/configs/train/ClothSAPG.yaml",
    ),
    # Play pretraining on the identified right hand (tasks/play/__init__.py).
    "Isaacsimenvs-PlayFlexivWujiRightReal-Direct-v0": (
        "PlayFlexivWujiRightHumanLikeEnvCfg",
        [PLAY_CFG],
        "isaacsimenvs/scripts/configs/train/PlaySAPG.yaml",
    ),
}



def _experiment_loader(directory=EXPERIMENT_DIR):
    """Execute the real pure recipe functions without importing Kit-backed decorators."""
    source = REPO_ROOT / "isaacsimenvs/scripts/configs/loader.py"
    functions = [node for node in ast.parse(source.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name in ("_deep_merge", "load_experiment")]
    namespace = {"EXPERIMENT_DIR": directory, "REPO_ROOT": REPO_ROOT, "Path": Path, "yaml": yaml}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["load_experiment"]


def _load_experiment(name: str) -> dict:
    return _experiment_loader()(name)


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


@pytest.mark.parametrize("name", sorted(p.relative_to(EXPERIMENT_DIR).with_suffix("").as_posix()
                                        for p in EXPERIMENT_DIR.rglob("*.yaml")))
def test_every_experiment_key_maps_to_a_config_field(name):
    exp = _load_experiment(name)
    assert set(exp) <= {"task", "env", "agent"}, f"{name}: unknown top-level keys {set(exp)}"
    assert exp.get("task") in EXPERIMENT_TASKS, f"{name}: task {exp.get('task')!r} has no checker entry"
    root_cls, modules, agent_yaml = EXPERIMENT_TASKS[exp["task"]]
    problems = _env_problems(exp.get("env") or {}, _collect_classes(modules), root_cls, "")
    problems += _agent_problems(exp.get("agent") or {}, yaml.safe_load((REPO_ROOT / agent_yaml).read_text()), "")
    assert not problems, f"{name}:\n  " + "\n  ".join(problems)


def test_nested_yibo_runs_share_environment_and_select_distinct_initialization():
    finetune = _load_experiment("experiments/yibo_our_dr/finetune.yaml")
    scratch = _load_experiment("experiments/yibo_our_dr/scratch.yaml")
    assert finetune == _load_experiment(str(REPO_ROOT / "experiments/yibo_our_dr/finetune.yaml"))
    assert finetune["env"] == scratch["env"] == _load_experiment("yibo_our_dr")["env"]
    assert finetune["env"]["obs"]["keypoint_mode"] == "yibo"
    assert finetune["env"]["robot_preset"] == "yibo"
    init, fresh = finetune["agent"]["params"], scratch["agent"]["params"]
    assert init["load_checkpoint"] is True and init["checkpoint_load_mode"] == "weights"
    assert init["load_path"] == "pretrained_policy/flexiv_wuji_finetuned/flat7RW_fricition.pth"
    assert fresh["load_checkpoint"] is False and fresh["load_path"] == ""
    assert init["config"]["name"] != fresh["config"]["name"]
    assert finetune["env"]["scene"]["num_envs"] // init["config"]["expl_coef_block_size"] == 6


def test_nested_recipe_bases_are_relative_to_their_source_and_reject_cycles(tmp_path):
    (tmp_path / "experiments").mkdir()
    (tmp_path / "base.yaml").write_text("env: {reset: {random_yaw: true}}\n")
    (tmp_path / "experiments" / "child.yaml").write_text("base: ../base.yaml\nenv: {seed: 7}\n")
    load = _experiment_loader(tmp_path)
    assert load("experiments/child") == {"env": {"reset": {"random_yaw": True}, "seed": 7}}
    (tmp_path / "base.yaml").write_text("base: experiments/child\n")
    with pytest.raises(ValueError, match="inheritance cycle"):
        load("experiments/child.yaml")
