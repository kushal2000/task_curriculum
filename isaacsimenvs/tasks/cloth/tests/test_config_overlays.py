"""Every key in a run recipe or a DR block must name a real field on its configclass.

`configclass.from_dict` walks the overlay and applies it onto the typed config, so a
misspelled or stale YAML key fails at env-construction time — after Kit has booted and
the assets have been generated. That is an expensive way to find a typo, and this test
finds it in milliseconds instead.

The check is static (AST over the cfg modules) rather than importing them, because
importing a configclass pulls in `isaaclab.utils`, which needs a booted Kit.
"""

from __future__ import annotations

import ast

import pytest

yaml = pytest.importorskip("yaml")

from isaacsimenvs.conftest import REPO_ROOT  # noqa: E402

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
SCENE_FIELDS = {
    "num_envs", "env_spacing", "replicate_physics", "clone_in_fabric",
    "lazy_sensor_update", "filter_collisions",
}

CLOTH_CFG = [
    "isaacsimenvs/tasks/cloth/env_cfg.py",
    "isaacsimenvs/tasks/cloth/cloth_cfg.py",
    "isaacsimenvs/tasks/cloth/mdp/tracker_cfg.py",
]
RIGID_CLOTH_CFG = ["isaacsimenvs/tasks/cloth/backends/rigid/config.py", *CLOTH_CFG]
VBD_CLOTH_CFG = [
    "isaacsimenvs/tasks/cloth/backends/vbd/config.py", "isaacsimenvs/tasks/cloth/backends/vbd/cloth_cfg.py", *CLOTH_CFG,
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


# --- run recipes (config/runs/, experiments/) and DR blocks (config/dr/) ----------------------

from isaacsimenvs.tasks.recipes import DR_DIR, RUNS_DIR, load_recipe  # noqa: E402

RECIPES = sorted(
    [str(p.relative_to(REPO_ROOT)) for p in RUNS_DIR.glob("*.yaml")]
    + [str(p.relative_to(REPO_ROOT)) for p in (REPO_ROOT / "experiments").rglob("*.yaml")]
)
_CLOTH_SAPG = "isaacsimenvs/tasks/cloth/config/ClothSAPG.yaml"
# Per task id: env root configclass, its modules, and the SAPG agent YAML.
EXPERIMENT_TASKS = {
    "Isaacsimenvs-RigidCloth-Direct-v0": ("RigidClothEnvCfg", RIGID_CLOTH_CFG, _CLOTH_SAPG),
    "Isaacsimenvs-Cloth-Direct-v0": ("VbdClothEnvCfg", VBD_CLOTH_CFG, _CLOTH_SAPG),
}
# The robot-suffixed alias ids (tasks/cloth/__init__.py) share their canonical task's cfg and agent YAML.
for _robot in ("FlexivWujiRightReal", "FlexivWujiRightS2REff", "FlexivWujiRightSharpaGain"):
    EXPERIMENT_TASKS[f"Isaacsimenvs-Cloth{_robot}-Direct-v0"] = EXPERIMENT_TASKS["Isaacsimenvs-Cloth-Direct-v0"]


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


@pytest.mark.parametrize("name", RECIPES)
def test_every_recipe_key_maps_to_a_config_field(name):
    exp = load_recipe(name)
    assert exp.get("task") in EXPERIMENT_TASKS, f"{name}: task {exp.get('task')!r} has no checker entry"
    root_cls, modules, agent_yaml = EXPERIMENT_TASKS[exp["task"]]
    problems = _env_problems(exp["env"], _collect_classes(modules), root_cls, "")
    problems += _agent_problems(exp["agent"], yaml.safe_load((REPO_ROOT / agent_yaml).read_text()), "")
    assert not problems, f"{name}:\n  " + "\n  ".join(problems)


@pytest.mark.parametrize("block", sorted(p.stem for p in DR_DIR.glob("*.yaml")))
def test_every_dr_block_key_maps_to_a_rigid_cloth_field(block):
    overlay = yaml.safe_load((DR_DIR / f"{block}.yaml").read_text()) or {}
    assert overlay, f"{block}: empty DR block"
    problems = _env_problems(overlay, _collect_classes(RIGID_CLOTH_CFG), "RigidClothEnvCfg", "")
    assert not problems, f"{block}:\n  " + "\n  ".join(problems)
