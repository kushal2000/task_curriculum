"""Sheet source perception=cotracker: a real cloth, its grid points tracked by CoTracker. PLACEHOLDER:
nothing here runs yet; selecting a cotracker object stops with a message (sheets/__init__.py).

The plan, with the same interface as rigid_plates.py (TOPIC, check / sheet_size / init / start / update):
  - a shared node deploy/nodes/point_tracker.py (CoTracker on the RealSense, + depth) publishes
    object_points: pos (N, 3) robot base frame, visible (N), the points in the grid's order
    (add the topic to deploy/utils/link.py's table and TOPIC_PORTS, so the recorder keeps it and the
    viewer can draw it);
  - first frame: click the sheet's 4 corners, a homography puts the grid's rest points on the image
    as CoTracker's query points, depth lifts them to 3D: point i is grid point i from the start;
  - here: start(state, msg) checks the sheet lies flat about where training put it; update(state,
    msg, prev) returns the particles (base_to_env), and stops when too many are hidden or lost (the
    corners and the stationary half are what the observation and the success test read);
    target_lift 0 in state; goal_mesh_pose None (the viewer draws no goal mesh, or a point set later).

Config: object.name [cloth] (an objects entry with perception: cotracker); fold.sheet_size; the
node's own settings come with it.
"""

from __future__ import annotations

AVAILABLE = False   # sheets/__init__.py refuses cotracker objects until this works
TOPIC = "object_points"


def check(cfg) -> None:
    raise SystemExit("[fold] perception=cotracker (cloth) is not implemented yet: see sheets/cloth_points.py")


def sheet_size(cfg) -> float:
    """m: fold.sheet_size (a cloth has no mesh to measure)."""
    if cfg.fold.sheet_size is None:
        raise SystemExit("[fold] a cloth needs fold.sheet_size")
    return float(cfg.fold.sheet_size)


def init(cfg, spec: dict, geom: dict) -> dict:
    raise SystemExit("[fold] perception=cotracker (cloth) is not implemented yet: see sheets/cloth_points.py")


def start(state: dict, msg) -> dict:
    raise NotImplementedError


def update(state: dict, msg, prev: dict) -> dict:
    raise NotImplementedError
