"""Sheet source perception=foundationpose: two rigid plates taped along one edge, each tracked by
FoundationPose as its own mesh (deploy/nodes/perception.py, topic object_poses) -> the sheet's particles.

A plate is a box centred on its origin, so its pose gives its sheet frame only up to the box's
symmetries (identity, 180 deg about each mesh axis). start_frames / update_frames pick the right one:
at the start from the geometry (both lying flat, hinge edges together), then by continuity. Each
plate then carries its half of the grid rigidly (particles).

The tape sits on the plates' top face, hinge_z above their mid-plane, while training's hinge is on
the mid-plane: the observed goal stays training's, the success test expects the folded plate
2 x hinge_z higher (target_lift), where it really lands.

Config: object.name [moving, stationary] (each an objects entry with a mesh and perception:
foundationpose); fold.max_hinge_gap_m. The plates' size comes from their meshes (plate_geometry):
sheet_size = the long side, the hinge along it, hinge_z = half the thickness (tape on the top face);
fold.sheet_size / fold.hinge_z, when set, must agree / override.

The sheet source (sheets/__init__.py has the interface):
    check(cfg)                                 # object.name is two matching plates (run.py, before anything starts)
    size = sheet_size(cfg)                     # m, for fold_geometry
    state = init(cfg, spec, geom)              # constants: plate names, layout, limits
    s = start(state, msg)                      # SystemExit if the plates are not lying flat side by side
    s = update(state, msg, s)                  # every step; s carries the plates' frames to the next
    s["particles"], s["stop"], s["status"], s["info"], s["goal_mesh_pose"]

The plate geometry on its own (the checks use it):
    layout = plate_layout(geom, hinge_z, hinge_axis_mesh)
    poses = plate_poses(layout, origin, r_sheet, fold_rad)          # both plates' mesh poses
    f = start_frames(layout, moving, stationary, max_gap)           # (centre, mesh quat) each -> sheet frames
    f = update_frames(layout, moving, stationary, f)
    parts = particles(layout, f["moving"], f["stationary"])
"""

from __future__ import annotations

import numpy as np

from deploy.rigid2deform.fold import FOLD_FLIP, mat_to_wxyz, quat_to_mat, rotation_angle
from deploy.utils.config import object_names, repo_path
from deploy.utils.spec import base_to_env

TOPIC = "object_poses"
MAX_TILT_DEG = 30.0   # start: each plate's face at most this far from level
MESH_TOL_M = 0.001    # plate meshes: centred, the same size, long side = 2 x short side, to this

# Rotation symmetries of a box centred on its origin: identity and 180 deg about each axis.
BOX_SYMMETRIES = (np.eye(3), np.diag([1.0, -1.0, -1.0]), np.diag([-1.0, 1.0, -1.0]), np.diag([-1.0, -1.0, 1.0]))


def mesh_to_sheet(hinge_axis_mesh: str) -> np.ndarray:
    """M with R_sheet = R_mesh @ M: the sheet axes in the plate mesh's frame. hinge_axis_mesh is the
    mesh axis that runs along the hinge (plate5: x, its 10 cm side); the sheet's x is the mesh's other
    in-plane axis, z the mesh's z."""
    if hinge_axis_mesh == "x":
        return np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])   # sheet x = mesh y, y = -mesh x
    if hinge_axis_mesh == "y":
        return np.eye(3)
    raise ValueError(f"hinge_axis_mesh={hinge_axis_mesh}: x | y")


# --- the plates in the sheet ------------------------------------------------------------------
def plate_layout(geom: dict, hinge_z: float = 0.0, hinge_axis_mesh: str = "x") -> dict:
    """Where the two plates sit in the sheet (fold_geometry's geom): each carries one half of the grid.
    hinge_z: the real hinge line above the plates' mid-plane, m (tape on the top face of 5 mm plates:
    0.0025). hinge_axis_mesh: the plate mesh axis along the hinge."""
    half = 0.25 * geom["size"]                                      # plate centre, |x| in the sheet frame
    return {"geom": geom, "hinge_z": float(hinge_z), "m2s": mesh_to_sheet(hinge_axis_mesh),
            "plate_centre": {"moving": np.array([half, 0.0, 0.0]), "stationary": np.array([-half, 0.0, 0.0])}}


def particles(layout: dict, moving: tuple, stationary: tuple) -> np.ndarray:
    """(P, 3) the grid in the env frame. moving / stationary: (centre, R_sheet) of each plate."""
    g, centre = layout["geom"], layout["plate_centre"]
    out = np.empty_like(g["grid"])
    for (c, r), which, sel in ((moving, "moving", g["on_moving"]), (stationary, "stationary", ~g["on_moving"])):
        out[sel] = c + (g["grid"][sel] - centre[which]) @ r.T
    return out


def hinge_point(layout: dict, which: str, centre: np.ndarray, r_sheet: np.ndarray) -> np.ndarray:
    """Where this plate puts the hinge line's middle (sheet origin), env frame."""
    return centre - r_sheet @ layout["plate_centre"][which]


def goal_plate_pose(layout: dict, stationary: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Where the moving plate's MESH sits when folded (the real fold, hinge_z): for the viewer."""
    c_s, r_s = stationary
    # turned 180 deg about the hinge line (sheet z = hinge_z): (x, z) -> (-x, 2 hinge_z - z), so the
    # moving centre (+half, 0, 0) lands on the stationary centre, 2 hinge_z higher
    centre = c_s + 2.0 * layout["hinge_z"] * r_s[:, 2]
    return centre, mat_to_wxyz(r_s @ FOLD_FLIP @ layout["m2s"].T)


def plate_poses(layout: dict, origin: np.ndarray, r_sheet: np.ndarray, fold_rad: float) -> dict:
    """Mesh poses of both plates for a sheet at origin / r_sheet with the moving plate turned up by
    fold_rad about the (real) hinge line. For the fake perception and the checks.
    Returns {"moving": (centre, quat_wxyz), "stationary": (...)}."""
    m2s, centre = layout["m2s"], layout["plate_centre"]
    out = {}
    out["stationary"] = (origin + r_sheet @ centre["stationary"], mat_to_wxyz(r_sheet @ m2s.T))
    # up = about the sheet's -y, through the hinge line at height hinge_z
    a = -fold_rad
    rot = np.array([[np.cos(a), 0.0, np.sin(a)], [0.0, 1.0, 0.0], [-np.sin(a), 0.0, np.cos(a)]])
    pivot = np.array([0.0, 0.0, layout["hinge_z"]])
    c_local = pivot + rot @ (centre["moving"] - pivot)
    out["moving"] = (origin + r_sheet @ c_local, mat_to_wxyz(r_sheet @ rot @ m2s.T))
    return out


# --- each plate's sheet frame from its tracked mesh pose ---------------------------------------
def _candidates(layout: dict, quat_wxyz) -> list[np.ndarray]:
    """The sheet frames a mesh pose can mean: one per box symmetry."""
    r_mesh = quat_to_mat(quat_wxyz)
    return [r_mesh @ s @ layout["m2s"] for s in BOX_SYMMETRIES]


def _frames(layout: dict, moving: tuple, stationary: tuple, r_m: np.ndarray, r_s: np.ndarray) -> dict:
    """{"moving": (centre, R_sheet), "stationary": (...), "hinge_gap": m}."""
    m = (np.asarray(moving[0], dtype=np.float64), r_m)
    s = (np.asarray(stationary[0], dtype=np.float64), r_s)
    gap = float(np.linalg.norm(hinge_point(layout, "moving", *m) - hinge_point(layout, "stationary", *s)))
    return {"moving": m, "stationary": s, "hinge_gap": gap}


def start_frames(layout: dict, moving: tuple, stationary: tuple, max_gap: float,
                 max_tilt_deg: float = MAX_TILT_DEG) -> dict:
    """moving / stationary: (centre, mesh quat wxyz). Both plates lying flat (sheet z within max_tilt
    of up), hinge edges together: the one symmetry pair that puts both plates' hinge points within
    max_gap and z up. SystemExit if there is none."""
    cos_tilt = np.cos(np.radians(max_tilt_deg))
    best = None
    for r_m in _candidates(layout, moving[1]):
        for r_s in _candidates(layout, stationary[1]):
            if r_m[2, 2] < cos_tilt or r_s[2, 2] < cos_tilt:
                continue
            gap = np.linalg.norm(hinge_point(layout, "moving", moving[0], r_m)
                                 - hinge_point(layout, "stationary", stationary[0], r_s))
            if best is None or gap < best[0]:
                best = (gap, r_m, r_s)
    if best is None:
        raise SystemExit(f"[fold] the plates are not lying flat (each plate's face more than {max_tilt_deg:.0f} "
                         "deg from level)")
    if best[0] > max_gap:
        raise SystemExit(f"[fold] the plates' hinge edges are {100 * best[0]:.1f} cm apart (limit "
                         f"{100 * max_gap:.1f} cm): not side by side, or the wrong plate size / hinge axis")
    return _frames(layout, moving, stationary, best[1], best[2])


def update_frames(layout: dict, moving: tuple, stationary: tuple, prev: dict) -> dict:
    """Per plate, the symmetry closest to its frame in prev (FoundationPose may jump between symmetric poses)."""
    r_m, r_s = (min(_candidates(layout, quat), key=lambda c, r=prev[which][1]: rotation_angle(c, r))
                for which, (_, quat) in (("moving", moving), ("stationary", stationary)))
    return _frames(layout, moving, stationary, r_m, r_s)


# --- the sheet source -------------------------------------------------------------------------
def mesh_box(path) -> tuple[np.ndarray, np.ndarray]:
    """(size (3,), centre (3,)) of an .obj mesh's bounding box, m, in its own frame."""
    with open(path) as f:
        v = np.array([line.split()[1:4] for line in f if line.startswith("v ")], dtype=np.float64)
    lo, hi = v.min(axis=0), v.max(axis=0)
    return hi - lo, 0.5 * (hi + lo)


def plate_geometry(cfg) -> dict:
    """The two plates' sheet, from their meshes: {"sheet_size" (m, the long side), "hinge_axis_mesh"
    (the mesh axis along it: x | y), "thickness" (m, along mesh z), "hinge_z" (m: fold.hinge_z, or
    half the thickness), "describe"}. SystemExit unless object.name is two plates whose meshes are
    boxes centred on their origin, of one size, long side twice the short one, and fold.sheet_size
    (if set) is that long side."""
    names = object_names(cfg)
    if len(names) != 2:
        raise SystemExit(f"[fold] object.name must be [moving plate, stationary plate], got {names}")
    boxes = {}
    for n in names:
        mesh = cfg.objects[n].get("mesh")
        path = repo_path(mesh) if mesh else None
        if path is None or not path.is_file():
            raise SystemExit(f"[fold] objects.{n}.mesh={mesh}: no such file (objects/ is not in git: copy it over)")
        boxes[n] = mesh_box(path)
    (size_a, c_a), (size_b, c_b) = boxes.values()
    if np.abs(size_a - size_b).max() > MESH_TOL_M:
        raise SystemExit(f"[fold] {names[0]} is {np.round(100 * size_a, 2)} cm, {names[1]} {np.round(100 * size_b, 2)} cm: "
                         "the two plates must be the same size")
    for n, (_, c) in boxes.items():
        if np.abs(c).max() > MESH_TOL_M:
            raise SystemExit(f"[fold] {n}'s mesh is not centred on its origin (box centre {np.round(100 * c, 2)} cm)")
    x, y, z = size_a
    long_axis = "x" if x >= y else "y"
    long, short = max(x, y), min(x, y)
    if abs(long - 2.0 * short) > MESH_TOL_M or z > short:
        raise SystemExit(f"[fold] {names[0]} is {100 * x:.1f} x {100 * y:.1f} x {100 * z:.2f} cm: a plate is half the "
                         "sheet, so its long side (along the hinge) must be twice the short one, z the thickness")
    if cfg.fold.sheet_size is not None and abs(cfg.fold.sheet_size - long) > MESH_TOL_M:
        raise SystemExit(f"[fold] fold.sheet_size={cfg.fold.sheet_size}, but the plates make a {100 * long:.1f} cm sheet: "
                         "leave it null for plates")
    hinge_z = float(cfg.fold.hinge_z) if cfg.fold.hinge_z is not None else 0.5 * float(z)
    src = "fold.hinge_z" if cfg.fold.hinge_z is not None else "half the thickness"
    return {"sheet_size": float(long), "hinge_axis_mesh": long_axis, "thickness": float(z), "hinge_z": hinge_z,
            "describe": (f"plates {100 * long:.1f} x {100 * short:.1f} x {100 * z:.2f} cm: moving {names[0]}, "
                         f"stationary {names[1]}; hinge {1000 * hinge_z:.1f} mm above their mid-plane ({src})")}


def check(cfg) -> None:
    """The config, before anything starts: two matching plates (plate_geometry)."""
    plate_geometry(cfg)


def sheet_size(cfg) -> float:
    """m: the sheet the plates make (their long side)."""
    return plate_geometry(cfg)["sheet_size"]


def init(cfg, spec: dict, geom: dict) -> dict:
    """The source's constants: the plates (object.name), their geometry, fold.max_hinge_gap_m."""
    g = plate_geometry(cfg)
    return {"spec": spec, "names": object_names(cfg), "layout": plate_layout(geom, g["hinge_z"], g["hinge_axis_mesh"]),
            "max_gap": float(cfg.fold.max_hinge_gap_m), "target_lift": 2.0 * g["hinge_z"],
            "describe": g["describe"]}


def _plates(state: dict, msg) -> list[tuple]:
    """[moving, stationary] as (centre, mesh quat wxyz), env frame."""
    out, sent = [], [str(n) for n in msg["names"]]   # the bus may deliver a numpy array
    for name in state["names"]:
        if name not in sent:
            raise SystemExit(f"[fold] object_poses has {sent}, no {name}: perception's object.name differs")
        i = sent.index(name)
        out.append(base_to_env(state["spec"], np.asarray(msg["pos"][i]), np.asarray(msg["quat_wxyz"][i])))
    return out


def _result(state: dict, f: dict) -> dict:
    gap, layout = f["hinge_gap"], state["layout"]
    stop = None
    if gap > state["max_gap"]:
        stop = (f"the plates' hinge edges {100 * gap:.1f} cm apart (max_hinge_gap_m "
                f"{100 * state['max_gap']:.1f}): tracking lost or flipped")
    return {"particles": particles(layout, f["moving"], f["stationary"]), "stop": stop,
            "status": f"hinge gap {1000 * gap:.0f} mm", "info": {"hinge_gap": gap},
            "goal_mesh_pose": goal_plate_pose(layout, f["stationary"]), "frames": f}


def start(state: dict, msg) -> dict:
    return _result(state, start_frames(state["layout"], *_plates(state, msg), state["max_gap"]))


def update(state: dict, msg, prev: dict) -> dict:
    return _result(state, update_frames(state["layout"], *_plates(state, msg), prev["frames"]))


def fake_message(state: dict, sheet_pose_base, fold_rad: float) -> tuple[list, np.ndarray, np.ndarray]:
    """(names, pos (2, 3), quat wxyz (2, 4)) of the plates of a sheet at sheet_pose_base [x, y, z,
    qw, qx, qy, qz] (hinge line middle, mid-plane; robot base frame), the moving plate turned up by
    fold_rad: what perception would send. The base frame's axes are the env's (robot root
    quaternion identity), so the sheet pose goes through plate_poses as is."""
    pose = np.asarray(sheet_pose_base, dtype=float)
    p = plate_poses(state["layout"], pose[:3], quat_to_mat(pose[3:]), fold_rad)
    return (state["names"], np.array([p["moving"][0], p["stationary"][0]]),
            np.array([p["moving"][1], p["stationary"][1]]))
