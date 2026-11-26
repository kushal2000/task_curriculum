"""The cloth task's configuration: what the fields are (``schema/``) and what they are set to (YAML).

    schema/            configclasses: every field, its type, what it means, its checks; no task values
    task.yaml          the values both task ids start from (``common`` + ``rigid`` / ``vbd``)
    runs/              run recipes (a training setup on top of task.yaml)
    dr/                DR blocks (ranges only, neutral by default)
    designs/           fold-shape designs
    ClothSAPG.yaml     the learner

The gym entry points are ``rigid_cloth_env_cfg`` and ``vbd_cloth_env_cfg``: the configclass with
task.yaml applied. Run recipes and the command line go on top of that (``tasks/loader.py``).

Kit-free at import; the builders import the schema (Isaac Lab) when called.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path

import yaml

__all__ = ["TASK_YAML", "apply_values", "missing_fields", "rigid_cloth_env_cfg", "rigid_newton_cloth_env_cfg",
           "rigid_ovphysx_cloth_env_cfg", "task_values", "vbd_cloth_env_cfg"]

TASK_YAML = Path(__file__).resolve().parent / "task.yaml"
BACKENDS = ("rigid", "vbd")


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def task_values(backend: str) -> dict:
    """task.yaml's ``common`` with the backend's section merged on top."""
    if backend not in BACKENDS:
        raise ValueError(f"unknown cloth backend {backend!r}; expected one of {BACKENDS}")
    data = yaml.safe_load(TASK_YAML.read_text()) or {}
    unknown = set(data) - {"common", *BACKENDS}
    if unknown:
        raise ValueError(f"{TASK_YAML}: unknown sections {sorted(unknown)}")
    return _merge(data.get("common") or {}, data.get(backend) or {})


def _is_missing(value) -> bool:
    # configclass deep-copies its defaults, so a MISSING default is a copy: compare types, not identity.
    return type(value) is type(dataclasses.MISSING)


def _field_type(obj, key: str) -> str:
    for f in dataclasses.fields(obj):
        if f.name == key:
            return f.type if isinstance(f.type, str) else getattr(f.type, "__name__", str(f.type))
    raise KeyError(key)


def _coerce(annotation: str, value, where: str):
    """``value`` as the annotated type: an int where a float is declared becomes a float, a list
    where a tuple is declared a tuple; anything else that does not match raises."""
    kinds = [a.strip() for a in annotation.split("|")]
    if value is None:
        if "None" in kinds:
            return None
        raise TypeError(f"{where}: null for a field of type {annotation}")
    kind = kinds[0]
    if kind == "float" and isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if kind == "int" and isinstance(value, int) and not isinstance(value, bool):
        return value
    if kind == "bool" and isinstance(value, bool):
        return value
    if kind == "str" and isinstance(value, str):
        return value
    if kind.startswith("tuple") and isinstance(value, (list, tuple)):
        return tuple(value)
    if kind.startswith("list") and isinstance(value, (list, tuple)):
        return list(value)
    raise TypeError(f"{where}: {value!r} ({type(value).__name__}) for a field of type {annotation}")


def apply_values(cfg, values: Mapping, _ns: str = "") -> None:
    """Set ``values`` (nested like the config) on ``cfg``. A field with no value yet (MISSING) is
    set by its annotation (``_coerce``); one that has a value goes through Isaac Lab's own
    ``update_class_from_dict``, as a recipe would."""
    from isaaclab.utils.dict import update_class_from_dict

    for key, value in values.items():
        where = f"{_ns}.{key}" if _ns else key
        if not hasattr(cfg, key):
            raise KeyError(f"task.yaml: {where} is not a field of {type(cfg).__name__}")
        current = getattr(cfg, key)
        if isinstance(value, Mapping):
            if _is_missing(current) or current is None:
                raise TypeError(f"task.yaml: {where} is a block but the field has no config to fill")
            apply_values(current, value, where)
        elif _is_missing(current):
            setattr(cfg, key, _coerce(_field_type(cfg, key), value, f"task.yaml: {where}"))
        else:
            update_class_from_dict(cfg, {key: value}, _ns="/" + _ns.replace(".", "/") if _ns else "")


def missing_fields(cfg, _ns: str = "") -> list[str]:
    """Every field of ``cfg`` (recursively) still MISSING."""
    out = []
    if not dataclasses.is_dataclass(cfg):
        return out
    for f in dataclasses.fields(cfg):
        value = getattr(cfg, f.name)
        where = f"{_ns}.{f.name}" if _ns else f.name
        if _is_missing(value):
            out.append(where)
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            out += missing_fields(value, where)
    return out


def _task_cfg(cfg, backend: str):
    apply_values(cfg, task_values(backend))
    missing = missing_fields(cfg)
    if missing:
        raise ValueError(f"{TASK_YAML} ({backend}) gives no value for: {', '.join(missing)}")
    return cfg


def rigid_cloth_env_cfg():
    """``RigidClothEnvCfg`` with task.yaml (``common`` + ``rigid``): the gym entry point of
    Isaacsimenvs-RigidCloth-Direct-v0."""
    from .schema.rigid_cloth import RigidClothEnvCfg

    return _task_cfg(RigidClothEnvCfg(), "rigid")


def vbd_cloth_env_cfg():
    """``VbdClothEnvCfg`` with task.yaml (``common`` + ``vbd``): the gym entry point of
    Isaacsimenvs-Cloth-Direct-v0."""
    from .schema.vbd import VbdClothEnvCfg

    return _task_cfg(VbdClothEnvCfg(), "vbd")


def rigid_newton_cloth_env_cfg():
    """``RigidNewtonClothEnvCfg``: the PhysX pool's values (``common`` + ``rigid``, without its PhysX
    solver block) plus the VBD world's Newton solver block and gravity compensation. The gym entry
    point of Isaacsimenvs-RigidClothNewton-Direct-v0 (robot and plates on MJWarp)."""
    from .schema.rigid_newton import RigidNewtonClothEnvCfg

    data = yaml.safe_load(TASK_YAML.read_text()) or {}
    rigid = dict(data.get("rigid") or {})
    sim = {k: v for k, v in (rigid.pop("sim", None) or {}).items() if k != "physx"}
    if sim:
        rigid["sim"] = sim
    values = _merge(data.get("common") or {}, rigid)
    vbd = data.get("vbd") or {}
    values["newton"] = vbd["newton"]
    values["use_gravcomp"] = True
    cfg = RigidNewtonClothEnvCfg()
    apply_values(cfg, values)
    missing = missing_fields(cfg)
    if missing:
        raise ValueError(f"{TASK_YAML} (rigid on Newton) gives no value for: {', '.join(missing)}")
    return cfg


def rigid_ovphysx_cloth_env_cfg():
    """``RigidOvPhysxClothEnvCfg``: the PhysX pool's values (``common`` + ``rigid``) on kit-less Isaac
    Lab 3 + ovphysx. Isaac Lab 3's ``SimulationCfg`` has no ``physx`` block, so the rigid section's
    ``sim.physx`` goes to ``physx_scene`` (Isaac Lab 2.3's PhysxCfg fields and defaults), which the
    backend authors on the PhysicsScene prim. The gym entry point of Isaacsimenvs-RigidClothOvPhysx-Direct-v0."""
    from .schema.rigid_ovphysx import RigidOvPhysxClothEnvCfg

    data = yaml.safe_load(TASK_YAML.read_text()) or {}
    rigid = dict(data.get("rigid") or {})
    sim = dict(rigid.pop("sim", None) or {})
    physx = sim.pop("physx", None) or {}
    if sim:
        rigid["sim"] = sim
    values = _merge(data.get("common") or {}, rigid)
    values["physx_scene"] = physx
    cfg = RigidOvPhysxClothEnvCfg()
    apply_values(cfg, values)
    missing = missing_fields(cfg)
    if missing:
        raise ValueError(f"{TASK_YAML} (rigid on ovphysx) gives no value for: {', '.join(missing)}")
    return cfg
