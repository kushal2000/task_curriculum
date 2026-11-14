"""CoTracker3 on a real cloth, in an OpenCV window: three ways to get the fold's keypoints from tracked
points, to compare before any of it feeds the policy. No robot, no ZMQ.

    ~/miniconda3/envs/cotracker/bin/python deploy/rigid2deform/checks/cloth_track_view.py --mode grid
(cloth_tracker.py is the pipeline's version: the same clicks, then object_points for the policy.)
    ... --mode fit                      # or kp8; --save DIR keeps an overlay mp4 + an npz per run

Camera: the D415 through ~/co-tracker/rs_tracking (CoTracker3 online: 16-frame windows, stride 8, so
results come ~8 frames late; the window shows the delay). Everything stays in the colour camera's frame:
the fold error, footprint and fold angle do not depend on the frame.

Modes (what you click on the frozen frame, and what becomes the sheet's 49 grid points, fold.py's
particles):
  grid  the sheet's 4 corners; a homography puts the training grid (7 x 7, spec fold.grid_rest) on the
        image and CoTracker tracks those 49 points themselves. A point hidden or without depth is
        filled in by its half's rigid fit to the measured ones (hollow on screen).
  fit   the 4 corners, then any number of points on the cloth (snapped to a corner nearby; `a` adds
        --auto_points good features inside the sheet). Each point's sheet coordinates come from the
        corners' homography; every frame, each half's grid follows its tracked points by a moving
        least squares rigid fit (--sigma; 0 = one rigid fit per half).
  kp8   8 points, 4 per half, used as the keypoints directly: the moving half's 4 are the observed
        object keypoints (fold corner_idx: the far edge's corners, and the two 1/3 of the half's width
        from the hinge), the stationary half's 4 their mirror images, which set the stationary frame
        (instead of the 21 stationary grid points). The rest of the grid is each half's rigid fit.

Click order (the sheet frame: x from the stationary half to the moving one across the hinge, y along it).
"A" is the sheet side you start on, "B" the other; which way round does not matter (found from the image):
  corners  S far A, S far B, M far B, M far A          (around the edge, the stationary far edge first)
  kp8      S far A, S far B, S near A, S near B, M near A, M near B, M far A, M far B
           (near = 1/3 of the half's width from the hinge)

Keys: SPACE live -> freeze / frozen -> track; L-click add; R-click undo; c clear; m snap <-> exact;
a (fit) auto-add points; r back to live; q / ESC quit.
On screen: grid blue (stationary) / orange (moving); k0-k3 the observed corners, g0-g3 their fold targets.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from deploy.rigid2deform.fold import fold_geometry, observe, rotation_angle, score  # noqa: E402
from deploy.rigid2deform.sheets.cloth_points import fit_half, wkabsch  # noqa: E402  (the policy's own fits)

SPEC = REPO / "deploy/rigid2deform/spec/spec_flexiv_wuji_right_fold.json"
WINDOW = "cloth tracking"
CORNER_LABELS = ["S far A", "S far B", "M far B", "M far A"]
KP8_LABELS = ["S far A", "S far B", "S near A", "S near B", "M near A", "M near B", "M far A", "M far B"]
INSIDE_MARGIN_PX = 3.0   # fit: an extra point must lie inside the corners' quad by this much
BLUE, ORANGE, YELLOW, GREEN, WHITE, RED = (60, 140, 255), (255, 140, 0), (255, 230, 0), (0, 230, 0), (255, 255, 255), (255, 60, 60)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["grid", "fit", "kp8"], required=True)
    p.add_argument("--cotracker_dir", default=str(Path.home() / "co-tracker"))
    p.add_argument("--checkpoint", default=None, help="default: <cotracker_dir>/checkpoints/scaled_online.pth")
    p.add_argument("--serial", default=None)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=60, choices=[15, 30, 60])
    p.add_argument("--sheet_size", type=float, default=None, help="m; default: the longest side between the corners (depth)")
    p.add_argument("--sigma", type=float, default=0.25, help="fit: MLS width, x sheet size (0 = rigid per half)")
    p.add_argument("--auto_points", type=int, default=30, help="fit: how many points `a` adds")
    p.add_argument("--snap_radius", type=int, default=8)
    p.add_argument("--max_buffer_s", type=float, default=20.0, help="max seconds frozen before going back live")
    p.add_argument("--save", default=None, help="directory: an overlay mp4 + npz per tracking run")
    return p.parse_args()


# --- geometry ---------------------------------------------------------------------------------
def resolve_side(px: np.ndarray, unit: np.ndarray) -> bool:
    """Is the click order mirrored? px / unit: the 4 corners in perimeter order (pixels / sheet units).
    The camera sees the sheet's top: seen from +z, a turn of the sheet's corners and of their image
    (v down) have opposite cross-product signs."""
    cross = lambda a: np.cross(a[1] - a[0], a[2] - a[1])
    return np.sign(cross(px)) == np.sign(cross(unit[:, :2]))


def corner_sides(camera, depth, h, corner_px, n: int = 15) -> np.ndarray | None:
    """(4,) m: the quad's sides, its corners' rays cut by the plane fitted to the depth on an n x n grid
    inside it (a corner's own pixel often has no depth, or the table's). None with too little depth."""
    s = np.linspace(-0.4, 0.4, n)
    unit = np.stack(np.meshgrid(s, s), -1).reshape(-1, 2)
    xyz = camera.pixels_to_xyz(depth, cv2.perspectiveTransform(unit[None], h)[0])
    xyz = xyz[np.all(np.isfinite(xyz), axis=1)]
    if len(xyz) < 0.3 * len(unit):
        return None
    c = xyz.mean(0)
    normal = np.linalg.svd(xyz - c)[2][2]
    intr = camera.intrinsics
    rays = np.c_[(corner_px[:, 0] - intr.ppx) / intr.fx, (corner_px[:, 1] - intr.ppy) / intr.fy, np.ones(4)]
    corners = rays * ((c @ normal) / (rays @ normal))[:, None]
    return np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)


def project(intr, x: np.ndarray) -> np.ndarray:
    """(N, 2) pixels of camera-frame points (N, 3); NaN stays NaN."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.stack([intr.fx * x[:, 0] / x[:, 2] + intr.ppx, intr.fy * x[:, 1] / x[:, 2] + intr.ppy], 1)


class Sheet:
    """The training sheet (spec fold block): the grid in sheet units (side 1), halves, keypoint indices."""

    def __init__(self):
        self.spec = json.loads(SPEC.read_text())
        f = self.spec["fold"]
        self.spec["fold"] = dict(f, size_range=[0.0, 1e9])   # any cloth: the real range is only warned about
        self.size_range = f["size_range"]
        self.unit = np.asarray(f["grid_rest"], dtype=np.float64) / f["sheet_size_nominal"]
        self.n = int(f["resolution"])
        self.on_moving = np.array([p == f["moving_plate"] for p in f["grid_plate"]])
        self.corner_idx = np.asarray(f["corner_idx"])
        cx = self.unit[self.corner_idx, 0]
        self.near_x, self.far_x = float(cx.min()), float(cx.max())
        self.corners_unit = np.array([[-.5, -.5, 0], [-.5, .5, 0], [.5, .5, 0], [.5, -.5, 0]])
        nx, fx = self.near_x, self.far_x
        self.kp8_unit = np.array([[-fx, -.5, 0], [-fx, .5, 0], [-nx, -.5, 0], [-nx, .5, 0],
                                  [nx, -.5, 0], [nx, .5, 0], [fx, -.5, 0], [fx, .5, 0]])

    def grid_index(self, u: np.ndarray) -> int:
        return int(np.argmin(np.linalg.norm(self.unit - u, axis=1)))

    def geometry(self, size: float, stationary_idx=None) -> dict:
        geom = fold_geometry(self.spec, size)
        if stationary_idx is not None:   # kp8: the stationary frame from its 4 points
            st = geom["grid"][np.asarray(stationary_idx)]
            geom.update(stationary_idx=np.asarray(stationary_idx), st_centroid=st.mean(0), st_centred=st - st.mean(0))
        return geom

    def edges(self):
        """Grid neighbour pairs (i over x, then j over y)."""
        n = self.n
        for i in range(n):
            for j in range(n):
                if i + 1 < n:
                    yield i * n + j, (i + 1) * n + j
                if j + 1 < n:
                    yield i * n + j, i * n + j + 1


# --- one tracking run -------------------------------------------------------------------------
class Plan:
    """What is tracked and how its points become the grid: built from the clicks when tracking starts."""

    def __init__(self, mode: str, sheet: Sheet, clicks: np.ndarray, frame, camera, args):
        self.mode, self.sheet = mode, sheet
        clicks = np.asarray(clicks, dtype=np.float64)
        if mode == "kp8":
            unit = sheet.kp8_unit.copy()
            perim = [0, 1, 7, 6]                            # S far A, S far B, M far B, M far A
        else:
            unit = sheet.corners_unit.copy()
            perim = [0, 1, 2, 3]
        self.mirrored = resolve_side(clicks[perim], unit[perim])
        if self.mirrored:
            unit[:, 1] *= -1.0
        h, _ = cv2.findHomography(unit[perim, :2], clicks[perim])
        corner_px = clicks[perim]

        if mode == "grid":
            pts = cv2.perspectiveTransform(sheet.unit[None, :, :2].astype(np.float64), h)[0]
            self.unit = sheet.unit.copy()
        elif mode == "fit":
            extra = clicks[4:]
            u_extra = cv2.perspectiveTransform(extra[None], np.linalg.inv(h))[0] if len(extra) else np.zeros((0, 2))
            pts = np.vstack([clicks[:4], extra])
            self.unit = np.vstack([unit, np.c_[u_extra, np.zeros(len(u_extra))]])
        else:
            pts = clicks
            self.unit = unit
        self.queries = pts.astype(np.float32)

        # sheet size: the corner sides on the depth's plane through the sheet, unless given
        sides = corner_sides(camera, frame.depth, h, corner_px)
        if args.sheet_size:
            self.size = float(args.sheet_size)
        elif sides is not None:
            self.size = float(sides.max())   # a square of the longest side (as cloth_tracker.py)
        else:
            raise ValueError("too little depth on the sheet to measure it: give --sheet_size")
        sides = np.full(4, np.nan) if sides is None else sides
        lo, hi = sheet.size_range
        self.note = ""
        if not lo <= self.size <= hi:
            self.note = f"size {100 * self.size:.1f} cm outside training's {100 * lo:.0f}-{100 * hi:.0f} cm"
        print(f"[view] {mode}: {len(pts)} points; corner sides {np.round(100 * sides, 1)} cm, sheet "
              f"{100 * self.size:.1f} cm{' (given)' if args.sheet_size else ''}; click order "
              f"{'mirrored (A/B swapped)' if self.mirrored else 'as given'}" + (f"; {self.note}" if self.note else ""))

        self.rest = self.unit * self.size                  # tracked points, sheet frame, m
        self.moving_pt = self.unit[:, 0] > 0               # the hinge row is the stationary half's, as in training
        self.sigma = args.sigma * self.size if mode == "fit" else 0.0
        st_idx = None
        if mode == "kp8":
            g = [sheet.grid_index(u) for u in self.unit]
            assert set(g[4:]) == set(sheet.corner_idx.tolist()), (g, sheet.corner_idx)
            st_idx = g[:4]
            self.corner_src = [g.index(c) for c in sheet.corner_idx]   # tracked point of each corner_idx
        self.geom = sheet.geometry(self.size, st_idx)
        self.grid_rest = self.geom["grid"]

    def particles(self, xyz, visible):
        """(49, 3) camera frame, (49,) measured (vs filled in)."""
        ok = visible & np.all(np.isfinite(xyz), axis=1)
        out = np.full((len(self.grid_rest), 3), np.nan)
        for half in (False, True):
            src, q = self.moving_pt == half, self.sheet.on_moving == half
            out[q] = fit_half(self.rest[src], xyz[src], ok[src], self.grid_rest[q], self.sigma)
        measured = np.zeros(len(out), dtype=bool)
        if self.mode == "grid":
            out[ok], measured = xyz[ok], ok
        elif self.mode == "kp8":
            for c, j in zip(self.sheet.corner_idx, self.corner_src):
                if ok[j]:
                    out[c], measured[c] = xyz[j], True
        return out, measured, ok

    def evaluate(self, parts):
        if not np.all(np.isfinite(parts)):
            return None, None, None
        o = observe(self.geom, parts)
        sc = score(self.geom, o)
        g, mv = self.grid_rest, self.sheet.on_moving
        r_m, _, _ = wkabsch(g[mv], parts[mv])
        r_s, _, _ = wkabsch(g[~mv], parts[~mv])
        # fold angle: 0 flat, 180 folded (the moving half's rotation relative to the stationary one)
        return o, sc, float(np.degrees(rotation_angle(r_m, r_s)))


# --- drawing ----------------------------------------------------------------------------------
def pt(uv):
    return int(round(float(uv[0]))), int(round(float(uv[1])))


def finite(uv):
    return np.all(np.isfinite(uv))


def text(img, s, uv, color, scale=0.45):
    cv2.putText(img, s, pt(uv), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, pt(uv), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def panel(img, lines, color=WHITE):
    h = 8 + 18 * len(lines)
    cv2.rectangle(img, (0, 0), (img.shape[1], h), (0, 0, 0), -1)
    for i, s in enumerate(lines):
        cv2.putText(img, s, (6, 17 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.47, color, 1, cv2.LINE_AA)


def draw_grid(img, sheet, uv, measured=None, hollow_unmeasured=False):
    for a, b in sheet.edges():
        if finite(uv[a]) and finite(uv[b]):
            col = ORANGE if sheet.on_moving[a] and sheet.on_moving[b] else BLUE
            cv2.line(img, pt(uv[a]), pt(uv[b]), col, 1, cv2.LINE_AA)
    for i, p in enumerate(uv):
        if finite(p):
            col = ORANGE if sheet.on_moving[i] else BLUE
            filled = measured is None or measured[i] or not hollow_unmeasured
            cv2.circle(img, pt(p), 3, col, -1 if filled else 1, cv2.LINE_AA)


# --- the app ----------------------------------------------------------------------------------
class App:
    def __init__(self, args):
        self.args = args
        sys.path.insert(0, args.cotracker_dir)
        import torch
        from rs_tracking.camera import RealSenseCamera
        from rs_tracking.point_placement import corner_strength_map, sample_points_in_mask, snap_to_corner
        from rs_tracking.tracker import FrameBuffer, OnlineTracker, load_cotracker
        self._snap, self._strength_map, self._sample = snap_to_corner, corner_strength_map, sample_points_in_mask

        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = args.checkpoint or str(Path(args.cotracker_dir) / "checkpoints/scaled_online.pth")
        print(f"[view] loading CoTracker3 online on {device} ...")
        model = load_cotracker(ckpt, device, args.width, args.height)
        self.camera = RealSenseCamera(args.width, args.height, args.fps, args.serial)
        self.intr = self.camera.intrinsics
        print(f"[view] {self.camera.name} {args.width}x{args.height} @ {args.fps} fps")
        self.buffer = FrameBuffer()
        self.tracker = OnlineTracker(model, self.buffer, self.camera, device)
        self.sheet = Sheet()

        self.state = "LIVE"
        self.snap = True
        self.frozen = None
        self.clicks, self.raw_clicks = [], []
        self.plan = None
        self.last_img = None
        self.log = None
        self.writer = None

    # ---- clicks ------------------------------------------------------------------------------
    def need(self) -> int:
        return {"grid": 4, "fit": 4, "kp8": 8}[self.args.mode]

    def label(self, i: int) -> str:
        if self.args.mode == "kp8":
            return KP8_LABELS[i] if i < 8 else ""
        return CORNER_LABELS[i] if i < 4 else f"p{i}"

    def add_click(self, x, y):
        mode = self.args.mode
        if len(self.clicks) >= self.need() and mode != "fit":
            return
        u, v = float(x), float(y)
        if self.snap:
            u, v, _ = self._snap(self.gray, self.strength, u, v, self.args.snap_radius)
        if mode == "fit" and len(self.clicks) >= 4:
            quad = np.array(self.clicks[:4], np.float32)
            if cv2.pointPolygonTest(quad, (u, v), True) < INSIDE_MARGIN_PX:
                print("[view] outside the sheet: not added")
                return
        self.clicks.append((u, v))
        self.raw_clicks.append((float(x), float(y)))

    def auto_points(self):
        if self.args.mode != "fit" or len(self.clicks) < 4:
            return
        mask = np.zeros(self.gray.shape, np.uint8)
        cv2.fillConvexPoly(mask, np.round(np.array(self.clicks[:4])).astype(np.int32), 1)
        pts = self._sample(self.gray, mask, self.args.auto_points, border=6)
        self.clicks += [tuple(map(float, p)) for p in pts]
        self.raw_clicks += [tuple(map(float, p)) for p in pts]
        print(f"[view] {len(pts)} points added ({len(self.clicks) - 4} on the cloth)")

    def on_mouse(self, event, x, y, flags, param):
        if self.state != "SELECT":
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.add_click(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN and self.clicks:
            self.clicks.pop()
            self.raw_clicks.pop()

    # ---- states ------------------------------------------------------------------------------
    def go_live(self):
        self.end_run()
        self.tracker.stop_session()
        self.buffer.stop_recording()
        self.state, self.frozen, self.plan = "LIVE", None, None

    def freeze(self, frame):
        self.buffer.start_recording()
        self.buffer.append(frame)                         # frame 0: the one clicked on
        self.frozen = frame
        self.gray = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY)
        self.strength = self._strength_map(self.gray)
        self.clicks, self.raw_clicks = [], []
        self.state = "SELECT"

    def start_tracking(self):
        if len(self.clicks) < self.need():
            print(f"[view] {self.need() - len(self.clicks)} more click(s) first")
            return
        try:
            self.plan = Plan(self.args.mode, self.sheet, self.clicks, self.frozen, self.camera, self.args)
        except ValueError as e:
            print(f"[view] {e}")
            return
        self.tracker.start_session([tuple(p) for p in self.plan.queries])
        self.trails = collections.deque(maxlen=20)
        self.last_img = None
        self.log = collections.defaultdict(list)
        self.state = "TRACK"

    def end_run(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if self.log and self.args.save and self.plan is not None:
            stamp = self.run_stamp
            out = Path(self.args.save) / f"{stamp}_{self.args.mode}.npz"
            np.savez_compressed(out, mode=self.args.mode, sheet_size=self.plan.size, queries=self.plan.queries,
                                unit=self.plan.unit, mirrored=self.plan.mirrored,
                                **{k: np.asarray(v) for k, v in self.log.items()})
            print(f"[view] saved {out} and its mp4 ({len(self.log['frame_idx'])} frames)")
        self.log = None

    # ---- loop --------------------------------------------------------------------------------
    def run(self):
        cv2.namedWindow(WINDOW)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        self.tracker.start()
        max_buffer = int(self.args.max_buffer_s * self.args.fps)
        try:
            while True:
                frame = self.camera.read()
                self.buffer.append(frame)
                if self.state == "SELECT" and len(self.buffer) >= max_buffer:
                    print(f"[view] frozen over {self.args.max_buffer_s:.0f} s: back to live")
                    self.go_live()
                img = self.render(frame)
                if img is not None:
                    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imshow(WINDOW, bgr)
                    if self.state == "TRACK" and self.args.save and self.last_img is img:
                        self.write(bgr)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    if self.state == "LIVE":
                        self.freeze(frame)
                    elif self.state == "SELECT":
                        self.start_tracking()
                elif key == ord("r"):
                    self.go_live()
                elif key == ord("m"):
                    self.snap = not self.snap
                elif self.state == "SELECT" and key == ord("c"):
                    self.clicks, self.raw_clicks = [], []
                elif self.state == "SELECT" and key == ord("a"):
                    self.auto_points()
        finally:
            self.end_run()
            self.tracker.shutdown()
            self.camera.stop()
            cv2.destroyAllWindows()

    def write(self, bgr):
        if self.writer is None:
            Path(self.args.save).mkdir(parents=True, exist_ok=True)
            self.run_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            out = Path(self.args.save) / f"{self.run_stamp}_{self.args.mode}.mp4"
            self.writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), self.args.fps,
                                          (bgr.shape[1], bgr.shape[0]))
        self.writer.write(bgr)

    # ---- rendering ---------------------------------------------------------------------------
    def render(self, frame):
        if self.state == "LIVE":
            img = frame.rgb.copy()
            panel(img, [f"LIVE  mode {self.args.mode}  |  SPACE: freeze and click  |  q: quit"])
            return img
        if self.state == "SELECT":
            return self.render_select()
        return self.render_track()

    def render_select(self):
        img = self.frozen.rgb.copy()
        mode, n, need = self.args.mode, len(self.clicks), self.need()
        if n >= 4:   # preview: the grid the corners put on the image (as clicked; the side is fixed at SPACE)
            perim = [0, 1, 7, 6] if mode == "kp8" else [0, 1, 2, 3]
            if n >= max(perim) + 1:
                unit = self.sheet.kp8_unit if mode == "kp8" else self.sheet.corners_unit
                h, _ = cv2.findHomography(unit[perim, :2], np.array(self.clicks)[perim])
                if h is not None:
                    uv = cv2.perspectiveTransform(self.sheet.unit[None, :, :2], h)[0]
                    draw_grid(img, self.sheet, uv)
        for i, ((u, v), (x, y)) in enumerate(zip(self.clicks, self.raw_clicks)):
            if (u, v) != (x, y):
                cv2.line(img, pt((x, y)), pt((u, v)), (200, 200, 200), 1, cv2.LINE_AA)
            is_kp = i < need
            cv2.circle(img, pt((u, v)), 5 if is_kp else 3, YELLOW if is_kp else WHITE, -1, cv2.LINE_AA)
            if is_kp:
                text(img, f"{i + 1} {self.label(i)}", (u + 7, v - 7), YELLOW)
        nxt = f"next: {self.label(n)}" if n < need else ("more points on the cloth, or `a`" if mode == "fit" else "done")
        panel(img, [f"CLICK {mode}  {n}/{need}{'+' if mode == 'fit' else ''}  |  {nxt}  |  "
                    f"{'SNAP' if self.snap else 'EXACT'} (m)",
                    "SPACE: track  R-click: undo  c: clear  r: live" + ("  a: auto points" if mode == "fit" else "")],
              YELLOW)
        return img

    def render_track(self):
        behind = self.tracker.num_pending() - self.tracker.step
        results = self.tracker.pop_results(max(1, behind))
        if not results:
            if self.last_img is not None:
                return None
            img = self.frozen.rgb.copy()
            panel(img, [f"TRACK  waiting for the first {self.tracker.window} frames ..."], GREEN)
            return img
        plan = self.plan
        for r in results:   # every result is evaluated (and logged); the last one is shown
            parts, measured, ok = plan.particles(r.xyz.astype(np.float64), r.visible.astype(bool))
            o, sc, angle = plan.evaluate(parts)
            if self.log is not None:
                L = self.log
                L["frame_idx"].append(r.frame_idx)
                L["t_ms"].append(r.frame.timestamp_ms)
                L["uv"].append(r.uv)
                L["visible"].append(r.visible)
                L["xyz"].append(r.xyz)
                L["particles"].append(parts)
                L["fold_error"].append(np.nan if sc is None else sc["fold_error"])
                L["footprint"].append(np.nan if sc is None else sc["footprint"])
                L["fold_angle_deg"].append(np.nan if angle is None else angle)

        img = r.frame.rgb.copy()
        draw_grid(img, self.sheet, project(self.intr, parts), measured, hollow_unmeasured=plan.mode == "grid")
        if plan.mode == "fit":   # the tracked points themselves
            for p, good in zip(r.uv, ok):
                cv2.circle(img, pt(p), 2, WHITE if good else RED, -1 if good else 1, cv2.LINE_AA)
        if plan.mode == "kp8":
            for i, (p, good) in enumerate(zip(r.uv, ok)):
                cv2.circle(img, pt(p), 6, WHITE if good else RED, 1, cv2.LINE_AA)
        if o is not None:
            kp, gl = project(self.intr, o["object_kp"]), project(self.intr, o["targets"])
            for i in range(4):
                if finite(kp[i]) and finite(gl[i]):
                    cv2.line(img, pt(kp[i]), pt(gl[i]), GREEN, 1, cv2.LINE_AA)
                if finite(gl[i]):
                    cv2.drawMarker(img, pt(gl[i]), GREEN, cv2.MARKER_TILTED_CROSS, 12, 2, cv2.LINE_AA)
                    text(img, f"g{i}", gl[i] + (6, 14), GREEN)
                if finite(kp[i]):
                    cv2.circle(img, pt(kp[i]), 6, YELLOW, 2, cv2.LINE_AA)
                    text(img, f"k{i}", kp[i] + (7, -7), YELLOW)
        delay = (len(self.buffer) - 1 - r.frame_idx) * 1000.0 / self.args.fps
        head = (f"TRACK {plan.mode}  frame {r.frame_idx}  delay {delay:.0f} ms  inference "
                f"{self.tracker.inference_ms:.0f} ms  |  r: live  q: quit")
        if sc is None:
            body = f"visible {ok.sum()}/{len(ok)}  |  a half has < 3 measured points: no keypoints"
        else:
            body = (f"visible {ok.sum()}/{len(ok)}  sheet {100 * plan.size:.1f} cm  fold {angle:.0f} deg  "
                    f"error {100 * sc['fold_error']:.1f} cm  footprint {sc['footprint']:.2f}"
                    f"{'  FOLDED' if sc['folded'] else ''}")
        panel(img, [head, body] + ([plan.note] if plan.note else []), GREEN if sc is not None else RED)
        self.last_img = img
        return img


if __name__ == "__main__":
    App(parse_args()).run()
