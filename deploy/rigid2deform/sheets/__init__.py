"""Sheet sources: where a fold's particles come from, chosen by the object's perception. Each source is
a module (no classes).

config.yaml: object.name lists what is on the table; every one of those objects' `perception` (all the
same) picks the source here and the perception node to run:

| perception     | object.name                 | source module    | node (topic)                                        |
|----------------|-----------------------------|------------------|-----------------------------------------------------|
| foundationpose | [moving plate, stationary]  | rigid_plates.py  | deploy/nodes/perception.py (object_poses)           |
| cotracker      | [cloth]                     | cloth_points.py  | deploy/nodes/point_tracker.py (object_points): TODO |

Every source module has:
    TOPIC                               what to subscribe
    check(cfg)                          SystemExit if the config does not fit (e.g. object.name's count)
    sheet_size(cfg) -> m                the sheet's side, for fold_geometry (plates: from their meshes)
    init(cfg, spec, geom) -> state      the source's constants (a dict), incl. target_lift (m, the real
                                        success target above training's) and describe (for the log)
    start(state, msg) -> s              the first message; SystemExit if the sheet is not as it must start
    update(state, msg, prev) -> s       every later one; prev = the last s
with s = {"particles" (P, 3) env frame, "stop" (None or why), "status" (a line for the viewer),
"info" (numbers for the recorder), "goal_mesh_pose" ((pos, quat) env frame, or None), + the source's own}.

    src = sheet_source(cfg)
    state = src.init(cfg, spec, fold_geometry(spec, src.sheet_size(cfg)))
"""

from types import ModuleType

from omegaconf import DictConfig

from deploy.utils.config import object_names

from . import cloth_points, rigid_plates

SOURCES = {"foundationpose": rigid_plates, "cotracker": cloth_points}
AVAILABLE = {"foundationpose": True, "cotracker": cloth_points.AVAILABLE}


def sheet_perception(cfg: DictConfig) -> str:
    """The one perception every object in object.name has (objects.<name>.perception)."""
    names = object_names(cfg)
    kinds = {n: cfg.objects[n].get("perception") for n in names}
    if len(set(kinds.values())) != 1:
        raise SystemExit(f"[fold] object.name={names}: their perceptions differ ({kinds}); one sheet, one perception")
    return next(iter(kinds.values()))


def sheet_source(cfg: DictConfig) -> ModuleType:
    """The source module for object.name's perception; SystemExit for an unknown or unfinished one."""
    kind = sheet_perception(cfg)
    if kind not in SOURCES:
        raise SystemExit(f"[fold] object.name={object_names(cfg)}: perception={kind}, not one of {' | '.join(SOURCES)}")
    if not AVAILABLE[kind]:
        raise SystemExit(f"[fold] object.name={object_names(cfg)}: perception={kind} is not implemented yet "
                         f"(rigid2deform/sheets/{SOURCES[kind].__name__.rsplit('.', 1)[-1]}.py)")
    return SOURCES[kind]
