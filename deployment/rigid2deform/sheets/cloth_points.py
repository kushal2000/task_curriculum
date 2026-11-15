"""Sheet source perception=cotracker: a real cloth, points on it tracked by CoTracker
(deploy/rigid2deform/cloth_tracker.py, topic object_points) -> the sheet's particles.

object_points (one per tracked camera frame; t = that frame's time):
    mode           perception.cotracker.mode the points were clicked for (the node's config; the policy
                   takes it from here, its own perception.cotracker.mode is not read)
    sigma          perception.cotracker.sigma (fit: the MLS width, x the side), the node's, like mode
    unit (N, 3)    each point's place on the flat sheet, sheet frame in units of the side (x across the
                   hinge from the stationary half to the moving one, y along it, z up, origin on the
                   hinge's middle): from the clicked corners' homography on the first frame
    pos (N, 3)     robot base frame, m (depth at the tracked pixel; meaningless where not visible)
    visible (N)    CoTracker sees the point and the depth there is valid
    size_measured  m: the longest side between the clicked corners (depth on the first frame): the
                   square sheet's side unless fold.sheet_size is set (then start compares the two)

How the points become the grid (fold_geometry's 49 points), by mode:
    grid  the points ARE the grid: a measured one is used as is; the others, each half's rigid fit
          (Kabsch, rest -> now) to its measured points
    kp8   8 points, 4 per half: the moving half's 4 at the observed corners (corner_idx) are used as
          is when measured; the rest of the grid is each half's rigid fit to its 4 points. (The
          stationary frame fold.observe fits to the 21 stationary points is then that 4-point fit.)
    fit   the 4 corners + any points on the cloth: each grid point follows its half's points by a
          weighted rigid fit (moving least squares, Gaussian in the rest distance, perception.cotracker.sigma
          x the side; 0 = one rigid fit per half)
A half with fewer than 3 measured points (occluded) has no fit: the policy keeps the last good
particles (its observation stays the last good one) until a frame fits again; perception.cotracker.max_lost_s
(null = no limit) stops the run after that long. The hinge row (x = 0) belongs to the stationary half,
as in training. No target_lift (a cloth's hinge is its mid-plane, like training's) and no goal mesh.

Config: object.name [cloth] (an objects entry with perception: cotracker); fold.sheet_size (null: the
node's size_measured, the longest side between the clicked corners, as a square's side);
perception.cotracker.max_age_s / max_lost_s. mode and sigma are the node's: they come with object_points,
so only perception.py needs them on its command line. numpy only (the policy's environment); checks/cloth_track_view.py uses the same
functions without the bus.
"""

from __future__ import annotations

import numpy as np

from deploy.utils.config import object_names
from deploy.utils.spec import base_to_env

AVAILABLE = True
TOPIC = "object_points"
MODES = ("grid", "kp8", "fit")
MAX_TILT_DEG = 30.0        # start: each half's face at most this far from level
MAX_START_FOLD_DEG = 25.0  # start: the halves at most this far from flat
IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


# --- rigid fits ---------------------------------------------------------------------------------
def wkabsch(p: np.ndarray, q: np.ndarray, w: np.ndarray | None = None):
    """(R, cp, cq): the weighted rigid fit taking p (K, 3) onto q: q ~ cq + (p - cp) @ R.T. No reflection."""
    w = np.ones(len(p)) if w is None else w
    cp = (w[:, None] * p).sum(0) / w.sum()
    cq = (w[:, None] * q).sum(0) / w.sum()
    u, _, vt = np.linalg.svd((p - cp).T @ (w[:, None] * (q - cq)))
    d = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
    return vt.T @ np.diag([1.0, 1.0, d]) @ u.T, cp, cq


def fit_half(rest_src, cur_src, ok, rest_q, sigma: float = 0.0) -> np.ndarray:
    """rest_q (Q, 3) carried by the measured points (rest_src -> cur_src where ok): one rigid fit, or with
    sigma (m) one per query point, the points weighted by their rest distance (MLS rigid). NaN with < 3."""
    src, dst = rest_src[ok], cur_src[ok]
    if len(src) < 3:
        return np.full((len(rest_q), 3), np.nan)
    if not sigma:
        r, cp, cq = wkabsch(src, dst)
        return cq + (rest_q - cp) @ r.T
    out = np.empty((len(rest_q), 3))
    for i, x in enumerate(rest_q):
        w = np.exp(-((src - x) ** 2).sum(1) / (2.0 * sigma ** 2)) + 1e-6
        r, cp, cq = wkabsch(src, dst, w)
        out[i] = cq + (x - cp) @ r.T
    return out


def particles(geom: dict, mode: str, unit: np.ndarray, xyz: np.ndarray, ok: np.ndarray, sigma: float = 0.0):
    """((P, 3) grid points in xyz's frame, (P,) measured: used as tracked, not fitted). geom: fold_geometry's
    (grid, on_moving, corner_idx, size); unit (N, 3) / xyz (N, 3) / ok (N,) the tracked points; sigma (m) fit only."""
    grid, on_moving = geom["grid"], geom["on_moving"]
    rest = np.asarray(unit, dtype=np.float64) * geom["size"]
    moving_pt = rest[:, 0] > 0                       # the hinge row is the stationary half's
    out = np.full((len(grid), 3), np.nan)
    for half in (False, True):
        src, q = moving_pt == half, on_moving == half
        out[q] = fit_half(rest[src], xyz[src], ok[src], grid[q], sigma if mode == "fit" else 0.0)
    measured = np.zeros(len(grid), dtype=bool)
    if mode == "grid":
        out[ok], measured = xyz[ok], ok.copy()
    elif mode == "kp8":
        for c in geom["corner_idx"]:
            j = int(np.argmin(np.linalg.norm(rest - grid[c], axis=1)))
            if ok[j]:
                out[c], measured[c] = xyz[j], True
    return out, measured


def half_frames(geom: dict, parts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(R_moving, R_stationary): each half's rotation from the flat sheet (sheet frame -> parts' frame)."""
    g, mv = geom["grid"], geom["on_moving"]
    return wkabsch(g[mv], parts[mv])[0], wkabsch(g[~mv], parts[~mv])[0]


def fold_angle_deg(geom: dict, parts: np.ndarray) -> float:
    """The moving half's turn relative to the stationary half: 0 flat, 180 folded."""
    r_m, r_s = half_frames(geom, parts)
    return float(np.degrees(np.arccos(np.clip((np.trace(r_m.T @ r_s) - 1.0) / 2.0, -1.0, 1.0))))


# --- the sheet source ---------------------------------------------------------------------------
def check(cfg) -> None:
    names = object_names(cfg)
    if len(names) != 1:
        raise SystemExit(f"[fold] perception=cotracker: object.name is the one cloth, got {names}")
    if cfg.perception.cotracker.mode not in MODES:
        raise SystemExit(f"[fold] perception.cotracker.mode={cfg.perception.cotracker.mode}: {' | '.join(MODES)}")


def sheet_size(cfg) -> float | None:
    """m: fold.sheet_size; None = measured_size of the first message."""
    check(cfg)
    return None if cfg.fold.sheet_size is None else float(cfg.fold.sheet_size)


def measured_size(msg) -> float:
    """m: the node's size_measured (the longest side between the clicked corners)."""
    size = float(msg["size_measured"])
    if not np.isfinite(size):
        raise SystemExit("[fold] cloth_tracker could not measure the cloth (no depth on it): set fold.sheet_size")
    print(f"[fold] cloth: {100 * size:.1f} cm square (the longest side between the clicked corners)", flush=True)
    return size


def init(cfg, spec: dict, geom: dict) -> dict:
    """mode / sigma: this process's config (fake_message sends them); start() replaces them with the message's."""
    c = cfg.perception.cotracker
    return {"spec": spec, "geom": geom, "mode": str(c.mode), "sigma_unit": float(c.sigma),
            "sigma": float(c.sigma) * geom["size"],
            "max_lost_s": float("inf") if c.max_lost_s is None else float(c.max_lost_s),
            "max_age_s": float(c.max_age_s), "target_lift": 0.0,
            "describe": f"cloth {object_names(cfg)[0]}, CoTracker points"}


def max_age_s(state: dict) -> float:
    """policy.py: object_points older than this = lost (CoTracker online answers in 8-frame batches)."""
    return state["max_age_s"]


def _measure(state: dict, msg) -> dict:
    """The message -> particles (env frame), or NaN particles when a half has < 3 measured points."""
    pos = base_to_env(state["spec"], np.asarray(msg["pos"], dtype=np.float64), IDENTITY)[0]
    ok = np.asarray(msg["visible"], dtype=bool) & np.all(np.isfinite(pos), axis=1)
    parts, measured = particles(state["geom"], state["mode"], np.asarray(msg["unit"]), pos, ok, state["sigma"])
    return {"particles": parts, "n_ok": int(ok.sum()), "n": len(ok), "n_measured": int(measured.sum()),
            "t": float(msg["t"])}


def _result(state: dict, m: dict, stop=None, held: float = 0.0, t_good: float | None = None) -> dict:
    geom, parts = state["geom"], m["particles"]
    angle = fold_angle_deg(geom, parts)
    status = f"points {m['n_ok']}/{m['n']} seen · fold {angle:.0f} deg"
    if held:
        status += f" · **holding the last particles ({1000 * held:.0f} ms)**"
    return {"particles": parts, "stop": stop, "status": status, "goal_mesh_pose": None,
            "info": {"points_seen": m["n_ok"], "grid_measured": m["n_measured"], "fold_angle_deg": angle,
                     "held_s": held},
            "t_good": m["t"] if t_good is None else t_good}


def start(state: dict, msg) -> dict:
    """The first message: its mode / sigma (the node's) from now on; the sheet lies flat."""
    mode = str(msg["mode"])
    if mode not in MODES:
        raise SystemExit(f"[fold] cloth_tracker sent mode {mode}: not one of {' | '.join(MODES)}")
    state["mode"], state["sigma_unit"] = mode, float(msg["sigma"])
    state["sigma"] = state["sigma_unit"] * state["geom"]["size"]
    m = _measure(state, msg)
    if not np.all(np.isfinite(m["particles"])):
        raise SystemExit(f"[fold] the cloth: only {m['n_ok']}/{m['n']} points seen (a half needs 3)")
    r_m, r_s = half_frames(state["geom"], m["particles"])
    tilt = np.degrees(np.arccos(np.clip([r_m[2, 2], r_s[2, 2]], -1.0, 1.0)))
    if tilt.max() > MAX_TILT_DEG:
        raise SystemExit(f"[fold] the cloth is not lying flat face up: halves {tilt.round(0)} deg from level "
                         f"(at most {MAX_TILT_DEG:.0f}; a wrong extrinsics file also does this)")
    angle = fold_angle_deg(state["geom"], m["particles"])
    if angle > MAX_START_FOLD_DEG:
        raise SystemExit(f"[fold] the cloth is not flat: its halves {angle:.0f} deg apart (at most {MAX_START_FOLD_DEG:.0f})")
    print(f"[fold] cloth: mode {mode}" + (f" (sigma {state['sigma_unit']} x the side)" if mode == "fit" else "")
          + f", {m['n_ok']}/{m['n']} points seen, measured {100 * float(msg['size_measured']):.1f} cm, centre "
          f"{np.round(m['particles'].mean(0), 3).tolist()} (env)", flush=True)
    return _result(state, m)


def update(state: dict, msg, prev: dict) -> dict:
    m = _measure(state, msg)
    if np.all(np.isfinite(m["particles"])):
        return _result(state, m)
    held = m["t"] - prev["t_good"]
    stop = None
    if held > state["max_lost_s"]:
        stop = (f"cloth lost: only {m['n_ok']}/{m['n']} points seen for {1000 * held:.0f} ms "
                f"(a half needs 3; max_lost_s {state['max_lost_s']})")
    m["particles"] = prev["particles"]
    return _result(state, m, stop=stop, held=held, t_good=prev["t_good"])


def mode_units(geom: dict, mode: str) -> np.ndarray:
    """(N, 3) where each mode's points lie on the flat sheet (sheet units), in the order the node clicks
    them: grid the 49 grid points; kp8 S far A, S far B, S near A, S near B, M near A, M near B, M far A,
    M far B; fit the 4 corners around the edge (S far A, S far B, M far B, M far A). A = y -0.5."""
    g = geom["grid"] / geom["size"]
    if mode == "grid":
        return g
    if mode == "kp8":
        cx = g[geom["corner_idx"], 0]
        nx, fx = cx.min(), cx.max()
        return np.array([[-fx, -.5, 0], [-fx, .5, 0], [-nx, -.5, 0], [-nx, .5, 0],
                         [nx, -.5, 0], [nx, .5, 0], [fx, -.5, 0], [fx, .5, 0]])
    return np.array([[-.5, -.5, 0], [-.5, .5, 0], [.5, .5, 0], [.5, -.5, 0]])


def fake_message(state: dict, sheet_pose_base, fold_rad: float) -> dict:
    """object_points of a sheet at sheet_pose_base [x, y, z, qw, qx, qy, qz] (hinge line middle, robot base
    frame), the moving half turned up by fold_rad about the hinge, every point seen: grid the 49 grid points,
    kp8 its 8, fit the 4 corners + the 49. The base frame's axes are the env's (robot root quaternion identity)."""
    from deploy.rigid2deform.fold import quat_to_mat

    geom, mode = state["geom"], state["mode"]
    unit = mode_units(geom, mode)
    if mode == "fit":
        unit = np.vstack([unit, mode_units(geom, "grid")])
    p = unit * geom["size"]
    a = -fold_rad                                    # up = about the sheet's -y
    rot = np.array([[np.cos(a), 0.0, np.sin(a)], [0.0, 1.0, 0.0], [-np.sin(a), 0.0, np.cos(a)]])
    p[p[:, 0] > 0] = p[p[:, 0] > 0] @ rot.T
    pose = np.asarray(sheet_pose_base, dtype=float)
    pos = pose[:3] + p @ quat_to_mat(pose[3:]).T
    return {"mode": mode, "sigma": state["sigma_unit"], "unit": unit, "pos": pos,
            "visible": np.ones(len(pos), dtype=bool), "size_measured": geom["size"]}
