"""The point-cloud fold policies' non-live group of points, taken once at the start: pc_node.py runs
`pick` before it streams, and every point_cloud message then carries the same one. Which one the
checkpoint's mode says (pc_obs.mode_of); policy.py feeds it to the network from step ref_step (12) on, as
training did. To take it again, restart the node (and the run: the policy keeps the first message's).

Both come from the FLAT cloth and one click (training's sheet is square, at any heading, folded along its
mid line; which half folds over is the one thing the cloud cannot say):
  label (the reference; rigidcloth_pc_label_*): the cloth's own points, each labelled by the half it is on
        -> ref_xyz (R <= 192, 3), ref_moving (R,) (True = the half that folds over), ref_corners (4, 3)
  goal  (the goal; rigidcloth_pc_goal_*): synthesised as training does (handoff README 4.3): the moving
        half mirrored over the fold line lands on the stationary half at the same height, so the folded
        sheet is the stationary half's rectangle, both layers in one -> N_REF points uniform on it, no
        noise -> goal_xyz (192, 3), goal_corners (4, 3) (that rectangle)

The window (the first camera's live image): put the cloth flat, the robot at home, hands away, the cloth
not touching the tape or anything else at table height (1 cm apart is enough). SPACE freezes both
cameras' frames (the table-height points are drawn grey). Click once on the half that folds over, roughly
the middle of it (not near a corner: there the two fold directions are alike). What it gives is drawn: the
sheet's outline (yellow), the fold line (white), label: the points (orange moving, blue stationary), goal:
the goal points and outline (green). Another click redoes it; SPACE accepts (the window closes, point_cloud
starts). R-click / c clears the click, r back to the live image, q / ESC quit.

From the frozen frames (take):
  1. both cameras' points through the live cloud's steps (box, no flying pixels, no black table, voxels)
  2. table height: the points at most perception.pointcloud.ref_plane_m above training's table top (a hand
     or arm above the cloth is not in it)
  3. the cloth: the connected piece of them (on a voxel_m grid, gaps up to one cell bridged) nearest the
     click's point on the table (the first camera's ray through the pixel, cut at the table top)
  4. the sheet: the minimum-area rectangle around the cloth's points (x, y); its height = their median.
     Fold direction u = the rectangle's side direction nearest the centre -> click direction (pointing to
     the moving half); the fold line = the rectangle's mid line across u
  5. label: moving = on u's side of the fold line; N_REF of the cloth points at random (fewer: all; the
     policy masks the empty slots). goal: N_REF points uniform on the stationary half's rectangle
Warnings (shown, not refused): a side outside training's sheet sizes (spec fold.size_range), not square
(short / long < SQUARE_WARN), the centre more than CENTRE_WARN_M from training's sheet centre, label: fewer
than N_REF cloth points. Refused: fewer than MIN_POINTS cloth points, a click not on the cloth or near a
corner (the fold direction unclear), label: a half with no points.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from deploy.rigid2deform.perception.cotracker_node import BLUE, GREEN, ORANGE, RED, WHITE, YELLOW, panel, pt
from deploy.rigid2deform.pc_obs import MODES, N_REF, ROOT_ENV, TABLE_Z_ENV

WINDOW = "pc_ref_goal"
TABLE_Z_BASE = TABLE_Z_ENV - ROOT_ENV[2]        # 0.33: training's table top above the base
SHEET_CENTRE_BASE = np.array([0.0, -0.51])      # training's sheet centre, robot base frame (x, y)
CENTRE_WARN_M = 0.10
MIN_POINTS = 10
CLICK_MAX_M = 0.03          # the click's point on the table: at most this far from a cloth point
DIAGONAL_MAX = 0.75         # the click direction's smaller / larger part along the sheet's sides: above
                            #   this (within ~8 deg of a diagonal) the fold direction is unclear: refused
SQUARE_WARN = 0.8           # short / long side below this: warned (training's sheet is square)
GOAL_COLOR = (60, 230, 120)
TABLE_COLOR = (150, 150, 150)
UP = np.array([0.0, 0.0, 1.0])


@dataclass
class Taken:
    """What the start took: the reference (mode label) or the goal (mode goal)."""
    mode: str                       # "label" | "goal"
    xyz: np.ndarray                 # (R, 3) base frame, R <= N_REF
    moving: np.ndarray | None       # (R,) bool, True = on the half that folds over (label only)
    sheet: np.ndarray               # (4, 3) the flat sheet: S far A, S far B, M far B, M far A (NaN: none)
    click: np.ndarray               # (3,) the click's point on the table
    n_cloth: int                    # cloth points (before the random N_REF)
    size: tuple[float, float]       # m: across the fold line (along u), along it
    warnings: list[str] = field(default_factory=list)
    error: str | None = None        # set: cannot be used

    @property
    def corners(self) -> np.ndarray:
        """(4, 3) what is sent: label the flat sheet's corners; goal the folded sheet's (the stationary
        half: S far A, S far B, fold line B, fold line A)."""
        if self.mode == "label":
            return self.sheet
        return np.array([self.sheet[0], self.sheet[1], (self.sheet[1] + self.sheet[2]) / 2,
                         (self.sheet[0] + self.sheet[3]) / 2])

    def fields(self) -> dict:
        """The point_cloud header fields that carry it."""
        xyz, corners = np.round(self.xyz, 5).tolist(), np.round(self.corners, 5).tolist()
        if self.mode == "label":
            return {"ref_xyz": xyz, "ref_moving": self.moving.tolist(), "ref_corners": corners}
        return {"goal_xyz": xyz, "goal_corners": corners}

    def lines(self) -> list[str]:
        c = self.sheet.mean(axis=0)
        what = (f"reference {len(self.xyz)} of {self.n_cloth} cloth points ({int(self.moving.sum())} moving, "
                f"{int((~self.moving).sum())} stationary)" if self.mode == "label" else
                f"goal {len(self.xyz)} points on the stationary half (synthesised; from {self.n_cloth} cloth points)")
        return [what, f"sheet {100 * self.size[0]:.1f} (across the fold) x {100 * self.size[1]:.1f} cm, "
                      f"centre ({c[0]:+.3f}, {c[1]:+.3f}, {c[2]:.3f})"]

    def summary(self) -> str:
        return "; ".join(self.lines())


def project(cam, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Base-frame points -> (pixels (N, 2) in cam's colour image, in front of the camera (N,))."""
    p = (xyz - cam.T_base_cam[:3, 3]) @ cam.T_base_cam[:3, :3]
    K = cam.dev.K_color
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.stack([K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2], K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2]], axis=1)
    return uv, p[:, 2] > 0


def on_table(cam, uv, z: float = TABLE_Z_BASE) -> np.ndarray:
    """(3,) where cam's colour-camera ray through pixel uv meets the level plane at height z."""
    ray = cam.T_base_cam[:3, :3] @ (np.linalg.inv(cam.dev.K_color) @ np.array([uv[0], uv[1], 1.0]))
    origin = cam.T_base_cam[:3, 3]
    return origin + ray * (z - origin[2]) / ray[2]


def table_height(cloud: np.ndarray, ref_plane_m: float) -> np.ndarray:
    """Indices of the points at most ref_plane_m above training's table top (step 2)."""
    return np.flatnonzero(cloud[:, 2] <= TABLE_Z_BASE + ref_plane_m)


def cloth_piece(cloud: np.ndarray, low: np.ndarray, click_xy: np.ndarray, cell: float):
    """Step 3: (indices into cloud of the connected piece of the table-height points `low` nearest the
    click, error or None)."""
    if len(low) < MIN_POINTS:
        return low[:0], f"only {len(low)} points at table height: is the cloth in training's box?"
    xy = cloud[low, :2]
    d = np.linalg.norm(xy - click_xy, axis=1)
    k = int(np.argmin(d))
    if d[k] > CLICK_MAX_M:
        return low[:0], f"the click is {100 * d[k]:.0f} cm from the nearest table-height point: click on the cloth"
    ij = np.floor((xy - xy.min(axis=0)) / cell).astype(np.int64) + 1
    occupied = np.zeros((ij[:, 1].max() + 2, ij[:, 0].max() + 2), np.uint8)     # rows y, columns x
    occupied[ij[:, 1], ij[:, 0]] = 1
    occupied = cv2.dilate(occupied, np.ones((3, 3), np.uint8))                    # bridge one-cell gaps
    _, labels = cv2.connectedComponents(occupied, connectivity=8)
    piece = labels[ij[:, 1], ij[:, 0]]
    cloth = low[piece == piece[k]]
    if len(cloth) < MIN_POINTS:
        return cloth, f"only {len(cloth)} cloth points where you clicked"
    return cloth, None


def sheet_frame(cloth_xyz: np.ndarray, click_xy: np.ndarray):
    """Step 4: (centre (3,), u (3,) across the fold toward the moving half, v (3,) along the fold line,
    size (across, along) m, error or None)."""
    (cx, cy), _, _ = rect = cv2.minAreaRect((cloth_xyz[:, :2] * 1000.0).astype(np.float32))   # mm
    box = cv2.boxPoints(rect) / 1000.0                                            # (4, 2) m, in order around
    sides = [box[1] - box[0], box[2] - box[1]]
    lengths = [float(np.linalg.norm(s)) for s in sides]
    centre = np.array([cx / 1000.0, cy / 1000.0, float(np.median(cloth_xyz[:, 2]))])
    d = click_xy - centre[:2]
    along = [abs(d @ s) / n for s, n in zip(sides, lengths)]
    i = int(np.argmax(along))
    if along[1 - i] > DIAGONAL_MAX * along[i]:
        return centre, None, None, (0.0, 0.0), ("the click is near a diagonal of the sheet, so the fold direction "
                                          "is unclear: click in the middle of the half that folds over")
    u2 = sides[i] / lengths[i] * np.sign(d @ sides[i])
    u = np.array([u2[0], u2[1], 0.0])
    return centre, u, np.cross(UP, u), (lengths[i], lengths[1 - i]), None


def take(mode: str, cam, cloud: np.ndarray, click_uv, cfg_pc, size_range, rng: np.random.Generator) -> Taken:
    """Steps 2-5 from cloud (N, 3) (both cameras' frozen frames, the live cloud's steps, all points) and the
    click (pixel in cam's, the first camera's, colour image) on the half that folds over."""
    click = on_table(cam, click_uv)
    nan = np.full((4, 3), np.nan)

    def failed(msg: str, n: int = 0) -> Taken:
        return Taken(mode, np.zeros((0, 3), np.float32), None, nan, click, n, (0.0, 0.0), error=msg)

    cloth, error = cloth_piece(cloud, table_height(cloud, float(cfg_pc.ref_plane_m)), click[:2], float(cfg_pc.voxel_m))
    if error:
        return failed(error, len(cloth))
    pts = cloud[cloth]
    centre, u, v, (along_u, along_v), error = sheet_frame(pts, click[:2])
    if error:
        return failed(error, len(pts))
    hu, hv = along_u / 2, along_v / 2
    sheet = np.array([centre - hu * u - hv * v, centre - hu * u + hv * v,       # S far A, S far B,
                      centre + hu * u + hv * v, centre + hu * u - hv * v])      # M far B, M far A

    if mode == "label":
        moving = (pts - centre) @ u > 0
        xyz = pts
        if len(xyz) > N_REF:
            pick = rng.choice(len(xyz), N_REF, replace=False)
            xyz, moving = xyz[pick], moving[pick]
    else:                                                    # the stationary half's rectangle, both layers
        a, b = rng.uniform(-hu, 0.0, N_REF), rng.uniform(-hv, hv, N_REF)
        xyz, moving = centre + a[:, None] * u + b[:, None] * v, None
    out = Taken(mode, xyz.astype(np.float32), moving, sheet, click, len(pts), (along_u, along_v))

    lo, hi = size_range
    if not (lo <= along_u <= hi and lo <= along_v <= hi):
        out.warnings.append(f"a side outside training's {100 * lo:.0f}-{100 * hi:.0f} cm")
    if min(along_u, along_v) < SQUARE_WARN * max(along_u, along_v):
        out.warnings.append("not square (training's sheet is): something else touching the cloth?")
    off = np.linalg.norm(centre[:2] - SHEET_CENTRE_BASE)
    if off > CENTRE_WARN_M:
        out.warnings.append(f"centre {100 * off:.0f} cm from training's (0, -0.51)")
    if mode == "label":
        if out.moving.all() or not out.moving.any():
            out.error = "all reference points on one half"
        elif len(pts) < N_REF:
            out.warnings.append(f"only {len(pts)} cloth points (training: {N_REF}; the rest masked)")
    return out


def draw_reference(img: np.ndarray, cam, taken: Taken, points: bool = True) -> None:
    """What the start took, on cam's colour image (in place): the sheet's outline and fold line; label the
    points by half (orange moving, blue stationary); goal the goal's outline and points (green)."""
    if not np.isfinite(taken.sheet).all():
        return
    uv, _ = project(cam, taken.sheet)
    cv2.polylines(img, [np.round(uv).astype(np.int32)], True, YELLOW, 1, cv2.LINE_AA)
    a, b = project(cam, np.array([taken.sheet[[0, 3]].mean(axis=0), taken.sheet[[1, 2]].mean(axis=0)]))[0]
    cv2.line(img, pt(a), pt(b), WHITE, 1, cv2.LINE_AA)
    if taken.mode == "goal":
        guv, _ = project(cam, taken.corners)
        cv2.polylines(img, [np.round(guv).astype(np.int32)], True, GOAL_COLOR, 1, cv2.LINE_AA)
    if not points:
        return
    puv, front = project(cam, taken.xyz)
    colours = [GOAL_COLOR] * len(puv) if taken.moving is None else [ORANGE if m else BLUE for m in taken.moving]
    for p, c, f in zip(puv, colours, front):
        if f and np.isfinite(p).all():
            cv2.circle(img, pt(p), 2, c, -1, cv2.LINE_AA)


def pick(cam, cams: list, cloud_of, cfg, mode: str, rng: np.random.Generator) -> Taken:
    """The window (see the docstring) until what the mode (label | goal) needs is accepted. cam: the first
    camera (CloudCamera), clicked in; cams: all of them; cloud_of(frames) -> (N, 3) the live cloud's steps
    on these framesets, all points. KeyboardInterrupt on q / ESC."""
    from deploy.utils.config import repo_path
    from deploy.utils.spec import load_spec

    if mode not in MODES:
        raise SystemExit(f"[pc_ref_goal] mode {mode}: this node takes {' | '.join(MODES)}")
    size_range = load_spec(repo_path(cfg.spec_path))["fold"]["size_range"]
    cfg_pc = cfg.perception.pointcloud
    st = {"frozen": None, "cloud": None, "low_uv": None, "taken": None}

    def on_mouse(event, x, y, flags, param):
        if st["frozen"] is None:
            return
        if event == cv2.EVENT_RBUTTONDOWN:
            st["taken"] = None
        elif event == cv2.EVENT_LBUTTONDOWN:
            t = take(mode, cam, st["cloud"], (float(x), float(y)), cfg_pc, size_range, rng)
            st["taken"] = t
            print(f"[pc_ref_goal] {t.error or t.summary()}"
                  + "".join(f"\n[pc_ref_goal]   WARNING: {w}" for w in t.warnings), flush=True)

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)
    print(f"[pc_ref_goal] mode {mode}. In the window: put the cloth flat (robot at home, hands away, "
          "nothing touching it), SPACE freezes, click once on the half that folds over, SPACE accepts", flush=True)
    try:
        while True:
            if st["frozen"] is None:
                img = cam.dev.latest().color.copy()
                panel(img, ["LIVE  |  the cloth flat, robot at home, hands away, nothing touching the cloth",
                            "SPACE: freeze  |  q: quit"])
            else:
                img = render(st, cam, mode)
            cv2.imshow(WINDOW, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(15) & 0xFF
            if key in (ord("q"), 27):
                raise KeyboardInterrupt
            if key == ord(" "):
                if st["frozen"] is None:
                    st["frozen"] = [c.dev.latest() for c in cams]
                    st["cloud"] = cloud_of(st["frozen"])
                    low = st["cloud"][table_height(st["cloud"], float(cfg_pc.ref_plane_m))]
                    uv, front = project(cam, low)
                    st["low_uv"], st["taken"] = uv[front], None
                    print(f"[pc_ref_goal] frozen: {len(st['cloud'])} points, {len(low)} at table height", flush=True)
                elif st["taken"] is None:
                    print("[pc_ref_goal] click on the half that folds over first", flush=True)
                elif st["taken"].error:
                    print(f"[pc_ref_goal] cannot use it: {st['taken'].error}", flush=True)
                else:
                    return st["taken"]
            elif key == ord("r"):
                st["frozen"], st["cloud"], st["taken"] = None, None, None
            elif key == ord("c"):
                st["taken"] = None
    finally:
        cv2.destroyAllWindows()
        cv2.waitKey(1)


def render(st: dict, cam, mode: str) -> np.ndarray:
    img = st["frozen"][0].color.copy()
    H, W = img.shape[:2]
    uv = np.round(st["low_uv"]).astype(int)
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    img[uv[ok, 1], uv[ok, 0]] = TABLE_COLOR                         # the table-height points
    taken = st["taken"]
    if taken is None:
        panel(img, ["FROZEN  |  click once on the half that folds over (its middle, not a corner)",
                    "r: live  |  q: quit"])
        return img
    draw_reference(img, cam, taken)
    cv2.circle(img, pt(project(cam, taken.click[None])[0][0]), 5, GREEN, 2, cv2.LINE_AA)
    lines = [taken.error] if taken.error else taken.lines()
    lines += [f"WARNING: {w}" for w in taken.warnings]
    lines.append(("click again on the half that folds over" if taken.error else
                  "SPACE: accept and start  |  click again to redo") + "  |  R-click / c clear, r live")
    panel(img, lines, RED if (taken.error or taken.warnings) else WHITE)
    return img
