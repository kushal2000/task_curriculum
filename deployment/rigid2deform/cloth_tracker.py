"""The cloth's perception node (objects with perception: cotracker): RealSense -> CoTracker3 online ->
object_points over ZMQ, in the robot base frame. Usually started by perception.py; on its own:

    ~/miniconda3/envs/cotracker/bin/python deployment/rigid2deform/cloth_tracker.py object.name=[cloth] \
        [perception.cotracker.mode=grid|kp8|fit] [fold.sheet_size=0.15]

Runs in the cotracker environment (~/co-tracker, see realsense_track.py; + pyzmq). Opens the camera the
extrinsics file is for (perception.extrinsics; perception.camera_serial must agree).

1. Like FoundationPose's start: an OpenCV window with the live image. Put the cloth flat, SPACE freezes
   the frame, click the points of perception.cotracker.mode (the banner names the next one), SPACE starts
   tracking and closes the window:
     grid  the 4 corners: S far A, S far B, M far B, M far A (around the edge, the stationary half's far
           edge first). The training grid (7 x 7) goes on the image by their homography: those 49
           points are tracked.
     kp8   8 points: S far A, S far B, S near A, S near B, M near A, M near B, M far A, M far B. near = a
           third of the half's width from the hinge (training's corner_idx), far = the far edge's corners.
     fit   the 4 corners as grid, then any points on the cloth (`a` adds perception.cotracker.auto_points
           good features inside it).
   S = the stationary half, M = the moving one (it turns over onto S); A = the side you start on, B the
   other (either way round: found from the image). A click snaps to a corner within snap_radius (m
   toggles), R-click undoes, c clears, r back to live, q / ESC quits.
2. Tracking, no window: every camera frame goes to CoTracker3 online (16-frame windows, stride 8:
   results come in batches of 8, the newest ~60 ms after its frame); each result -> depth at the point
   (5 x 5 median) -> base frame -> object_points (t = the camera frame's time). camera_image at image_hz:
   the camera image with the grid the policy builds from these points (blue: stationary half, orange:
   moving; hollow = filled in, not seen), k0-k3 the observed corners, g0-g3 their fold targets, the fold
   angle / error / footprint: the viewer's camera panel shows it (the viewer also draws the points in
   3D). perception.frame_stream: every camera frame also goes out raw (camera_frame, from the clicking
   on) for recorder.py record.camera. The terminal prints the rate and the fold state every 5 s. Ctrl-C stops (to click again,
   restart).

object_points: mode, sigma (perception.cotracker's: the policy takes them from here), unit (N, 3) each point's place on the flat sheet (sheet units), pos (N, 3) base
frame, visible (N), uv (N, 2) pixels, size_measured (m: the longest of the corners' sides on the plane
fitted to the depth inside them, at the first frame: the square sheet's side unless fold.sheet_size is
set). perception.cotracker.save_dir: also an overlay mp4 + npz per run.
checks/cloth_track_view.py is the same tracking in a window of its own, without the bus.
"""

from __future__ import annotations

import collections
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))    # repo root, for deploy.*
from deployment.rigid2deform.fold import fold_geometry, observe, score               # noqa: E402
from deployment.rigid2deform.sheets import cloth_points as cp                         # noqa: E402
from deployment.rigid2deform.sheets import sheet_perception                           # noqa: E402
from deployment.utils.config import REPO, load_config, repo_path                      # noqa: E402
from deployment.utils.link import FRAME_PORT, encode_jpeg, open_publisher, send, send_frame  # noqa: E402
from deployment.utils.spec import load_spec                                           # noqa: E402

CONFIG = HERE / "config.yaml"
WINDOW = "cloth_tracker"
CORNER_LABELS = ["S far A", "S far B", "M far B", "M far A"]
KP8_LABELS = ["S far A", "S far B", "S near A", "S near B", "M near A", "M near B", "M far A", "M far B"]
PERIMETER = {"kp8": [0, 1, 7, 6], "grid": [0, 1, 2, 3], "fit": [0, 1, 2, 3]}   # the 4 outer corners, around the edge
NEED = {"grid": 4, "kp8": 8, "fit": 4}
INSIDE_MARGIN_PX = 3.0      # fit: an extra point must lie inside the corners' quad by this much
DEPTH_PATCH = 2             # px: the depth at a point is the median of a (2r+1)^2 patch
BLUE, ORANGE, YELLOW, GREEN, WHITE, RED = (60, 140, 255), (255, 140, 0), (255, 230, 0), (0, 230, 0), (255, 255, 255), (255, 60, 60)


# --- geometry on the image ------------------------------------------------------------------------
def deproject(depth: np.ndarray, uv: np.ndarray, intr) -> np.ndarray:
    """(N, 3) colour-camera frame, m; NaN where the patch has no depth. depth (H, W) m, 0 = none."""
    h, w = depth.shape
    out = np.full((len(uv), 3), np.nan)
    r = DEPTH_PATCH
    for i, (u, v) in enumerate(uv):
        ui, vi = int(round(u)), int(round(v))
        if not (0 <= ui < w and 0 <= vi < h):
            continue
        patch = depth[max(vi - r, 0): vi + r + 1, max(ui - r, 0): ui + r + 1]
        patch = patch[patch > 0]
        if patch.size:
            z = float(np.median(patch))
            out[i] = ((u - intr.cx) / intr.fx * z, (v - intr.cy) / intr.fy * z, z)
    return out


def project(intr, x: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.stack([intr.fx * x[:, 0] / x[:, 2] + intr.cx, intr.fy * x[:, 1] / x[:, 2] + intr.cy], 1)


def mirrored(px: np.ndarray, unit: np.ndarray) -> bool:
    """Is the click order the mirror image of unit's? px / unit: the 4 corners around the edge. The camera
    sees the sheet's top: a turn seen from +z and its image (v down) have opposite cross-product signs."""
    cross = lambda a: np.cross(a[1] - a[0], a[2] - a[1])
    return bool(np.sign(cross(px)) == np.sign(cross(unit[:, :2])))


def corner_sides(depth, intr, h, corner_px, n: int = 15):
    """(4,) m: the quad's sides, the corners' rays cut by the plane fitted to the depth on an n x n grid
    inside it (a corner's own pixel often has no depth, or the table's). None with too little depth."""
    s = np.linspace(-0.4, 0.4, n)
    unit = np.stack(np.meshgrid(s, s), -1).reshape(-1, 2)
    xyz = deproject(depth, cv2.perspectiveTransform(unit[None], h)[0], intr)
    xyz = xyz[np.all(np.isfinite(xyz), axis=1)]
    if len(xyz) < 0.3 * len(unit):
        return None
    c = xyz.mean(0)
    normal = np.linalg.svd(xyz - c)[2][2]
    rays = np.c_[(corner_px[:, 0] - intr.cx) / intr.fx, (corner_px[:, 1] - intr.cy) / intr.fy, np.ones(4)]
    corners = rays * ((c @ normal) / (rays @ normal))[:, None]
    return np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)


def layout(geom: dict, mode: str, clicks: np.ndarray, depth, intr) -> dict:
    """The clicks -> what to track: queries (N, 2) px on the clicked frame, unit (N, 3) their sheet
    coordinates, mirrored, size_measured (m: the longest side, the square's; NaN if the depth is too thin),
    sides (4,)."""
    clicks = np.asarray(clicks, dtype=np.float64)
    perim = PERIMETER[mode]
    unit = cp.mode_units(geom, "fit" if mode == "grid" else mode).copy()   # grid: the 4 clicks are the corners
    flip = mirrored(clicks[perim], unit[perim])
    if flip:
        unit[:, 1] *= -1.0
    h, _ = cv2.findHomography(unit[perim, :2], clicks[perim])
    if mode == "grid":
        unit = cp.mode_units(geom, "grid")
        queries = cv2.perspectiveTransform(unit[None, :, :2], h)[0]
    elif mode == "fit":
        extra = clicks[4:]
        u_extra = cv2.perspectiveTransform(extra[None], np.linalg.inv(h))[0] if len(extra) else np.zeros((0, 2))
        unit = np.vstack([unit, np.c_[u_extra, np.zeros(len(u_extra))]])
        queries = clicks
    else:
        queries = clicks
    sides = corner_sides(depth, intr, h, clicks[perim])
    return {"queries": queries.astype(np.float32), "unit": unit, "mirrored": flip,
            "sides": np.full(4, np.nan) if sides is None else sides,
            "size_measured": float("nan") if sides is None else float(sides.max())}


def preview_grid(geom: dict, mode: str, clicks) -> np.ndarray | None:
    """The grid as the clicks put it on the image (before the side is resolved), or None."""
    perim = PERIMETER[mode]
    if len(clicks) <= max(perim):
        return None
    corners = cp.mode_units(geom, "fit" if mode == "grid" else mode)
    h, _ = cv2.findHomography(corners[perim, :2], np.asarray(clicks, np.float64)[perim])
    return None if h is None else cv2.perspectiveTransform(cp.mode_units(geom, "grid")[None, :, :2], h)[0]


# --- drawing --------------------------------------------------------------------------------------
def pt(uv):
    return int(round(float(uv[0]))), int(round(float(uv[1])))


def finite(uv):
    return bool(np.all(np.isfinite(uv)))


def text(img, s, uv, color, scale=0.45):
    cv2.putText(img, s, pt(uv), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, pt(uv), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def panel(img, lines, color=WHITE):
    cv2.rectangle(img, (0, 0), (img.shape[1], 8 + 18 * len(lines)), (0, 0, 0), -1)
    for i, s in enumerate(lines):
        cv2.putText(img, s, (6, 17 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.47, color, 1, cv2.LINE_AA)


def grid_edges(n: int):
    for i in range(n):
        for j in range(n):
            if i + 1 < n:
                yield i * n + j, (i + 1) * n + j
            if j + 1 < n:
                yield i * n + j, i * n + j + 1


def draw_grid(img, geom, uv, hollow=None):
    on_moving, n = geom["on_moving"], int(round(np.sqrt(len(uv))))
    for a, b in grid_edges(n):
        if finite(uv[a]) and finite(uv[b]):
            cv2.line(img, pt(uv[a]), pt(uv[b]), ORANGE if on_moving[a] and on_moving[b] else BLUE, 1, cv2.LINE_AA)
    for i, p in enumerate(uv):
        if finite(p):
            cv2.circle(img, pt(p), 3, ORANGE if on_moving[i] else BLUE,
                       1 if hollow is not None and hollow[i] else -1, cv2.LINE_AA)


# --- camera and tracker ---------------------------------------------------------------------------
class Capture(threading.Thread):
    """Every new camera frame (rgb, depth m, t) into a queue while recording; the newest always. With
    frame_port: also every frame raw as camera_frame (recorder.py record.camera), from the clicking on,
    on its own socket (one per thread)."""

    def __init__(self, cam, frame_port: int | None = None):
        super().__init__(daemon=True)
        self.cam, self.queue, self.recording, self.running = cam, collections.deque(), False, True
        self.frame_port = frame_port
        self.newest = cam.latest_rgbd()

    def run(self):
        pub = open_publisher(self.frame_port) if self.frame_port else None
        last_t, seq = None, 0
        try:
            while self.running:
                f = self.cam.latest_rgbd()
                if f[2] == last_t:
                    time.sleep(0.002)
                    continue
                last_t = f[2]
                self.newest = f
                if self.recording:
                    self.queue.append(f)
                if pub is not None:
                    send_frame(pub, seq, f[2], f[0])
                    seq += 1
        finally:
            if pub is not None:
                pub.close(linger=0)

    def drain(self) -> list:
        out = []
        while self.queue:
            out.append(self.queue.popleft())
        return out


class Tracker:
    """CoTracker3 online: start(first frame, queries), then add(frame) -> [(frame, uv (N, 2), visible (N))]
    for each frame a 16-frame window (stride 8) finished, in order."""

    def __init__(self, checkpoint: str, width: int, height: int):
        import torch
        from cotracker.predictor import CoTrackerOnlinePredictor

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CoTrackerOnlinePredictor(checkpoint=checkpoint).to(self.device)
        self.step = self.model.step
        self.window = 2 * self.step
        self.inference_ms = 0.0
        video = torch.zeros(1, self.window, 3, height, width, device=self.device)   # warm up
        self.model(video[:, :1], is_first_step=True, queries=torch.tensor([[[0.0, width / 2, height / 2]]],
                                                                          device=self.device), add_support_grid=True)
        self.model(video, add_support_grid=True)

    def _video(self, frames):
        v = np.stack([f[0] for f in frames])
        return self.torch.from_numpy(v).to(self.device).permute(0, 3, 1, 2)[None].float()

    def start(self, frame, queries: np.ndarray):
        q = self.torch.tensor([[[0.0, float(u), float(v)] for u, v in queries]], device=self.device)
        self.model(self._video([frame]), is_first_step=True, queries=q, add_support_grid=True)
        self.frames, self.start_idx = [frame], 0

    def add(self, frame) -> list:
        self.frames.append(frame)
        out = []
        while len(self.frames) >= self.window:
            chunk = self.frames[:self.window]
            t0 = time.perf_counter()
            with self.torch.no_grad():
                tracks, vis = self.model(self._video(chunk), add_support_grid=True)
            tracks, vis = tracks[0].cpu().numpy(), vis[0].cpu().numpy()   # (frames so far, N, ...)
            self.inference_ms = 1000.0 * (time.perf_counter() - t0)
            for k in range(0 if self.start_idx == 0 else self.step, self.window):
                out.append((chunk[k], tracks[self.start_idx + k], vis[self.start_idx + k]))
            self.start_idx += self.step
            self.frames = self.frames[self.step:]
        return out


# --- the node -------------------------------------------------------------------------------------
class Node:
    def __init__(self, cfg):
        from deployment.utils.realsense import RealSense

        self.cfg, p, c = cfg, cfg.perception, cfg.perception.cotracker
        self.mode = str(c.mode)
        spec = load_spec(repo_path(cfg.spec_path))
        self.size_range = spec["fold"]["size_range"]
        spec["fold"] = dict(spec["fold"], size_range=[0.0, 1e9])   # this node only draws; the policy checks the size
        self.spec = spec
        self.size_cfg = None if cfg.fold.sheet_size is None else float(cfg.fold.sheet_size)
        # until the corners are measured: the training sheet (the click layout only needs its proportions)
        self.set_size(self.size_cfg or spec["fold"]["sheet_size_nominal"])

        calib_path = repo_path(p.extrinsics)
        calib = json.loads(calib_path.read_text())
        self.T_base_cam = np.asarray(calib["T_base_cam"], dtype=float)
        serial = p.camera_serial or calib.get("camera_serial")
        if calib.get("camera_serial") is not None and serial != calib["camera_serial"]:
            raise SystemExit(f"[cloth_tracker] perception.camera_serial={serial}, but {calib_path.name} is for "
                             f"camera {calib['camera_serial']}")
        cdir = Path(str(c.cotracker_dir)).expanduser()
        sys.path.insert(0, str(cdir))
        from rs_tracking.point_placement import corner_strength_map, sample_points_in_mask, snap_to_corner
        self._snap, self._strength_map, self._sample = snap_to_corner, corner_strength_map, sample_points_in_mask
        ckpt = str(Path(str(c.checkpoint)).expanduser()) if c.checkpoint else str(cdir / "checkpoints/scaled_online.pth")
        print(f"[cloth_tracker] loading CoTracker3 online ({ckpt}) ...", flush=True)
        self.tracker = Tracker(ckpt, p.camera_width, p.camera_height)
        self.cam = RealSense(serial=serial, width=p.camera_width, height=p.camera_height, fps=p.camera_fps)
        self.cam.start()
        self.intr = self.cam.intrinsics
        self.capture = Capture(self.cam, cfg.bus[FRAME_PORT] if p.frame_stream else None)
        self.capture.start()
        self.pub = open_publisher(cfg.bus.object_pose_port)
        print(f"[cloth_tracker] {self.cam.name} {self.cam.serial}, {p.camera_width}x{p.camera_height} at "
              f"{p.camera_fps} fps, extrinsics {calib_path.name}, mode {self.mode}, sheet "
              + (f"{100 * self.size_cfg:.1f} cm (fold.sheet_size)" if self.size_cfg else "measured from the clicked corners"),
              flush=True)

        self.snap, self.frozen = True, None
        self.clicks, self.raw = [], []
        self.seq, self.t_image = 0, 0.0
        self.writer, self.log = None, None

    def set_size(self, size: float):
        lo, hi = self.size_range
        self.size_note = ("" if lo <= size <= hi else
                          f"sheet {100 * size:.1f} cm outside training's {100 * lo:.0f}-{100 * hi:.0f} cm: the policy refuses it")
        self.geom = fold_geometry(self.spec, size)
        if self.cfg.fold.success_tolerance_m is not None:   # as policy.py: the FOLDED shown here is its test
            self.geom["tolerance"] = float(self.cfg.fold.success_tolerance_m)
        self.sigma = float(self.cfg.perception.cotracker.sigma) * size

    # ---- 1. clicking, in an OpenCV window (closes when tracking starts) ----------------------------
    def label(self, i: int) -> str:
        if self.mode == "kp8":
            return KP8_LABELS[i] if i < 8 else ""
        return CORNER_LABELS[i] if i < 4 else f"p{i}"

    def on_mouse(self, event, x, y, flags, param):
        if self.frozen is None:
            return
        if event == cv2.EVENT_RBUTTONDOWN and self.clicks:
            self.clicks.pop()
            self.raw.pop()
        if event != cv2.EVENT_LBUTTONDOWN or (len(self.clicks) >= NEED[self.mode] and self.mode != "fit"):
            return
        u, v = float(x), float(y)
        if self.snap:
            u, v, _ = self._snap(self.gray, self.strength, u, v, int(self.cfg.perception.cotracker.snap_radius))
        if self.mode == "fit" and len(self.clicks) >= 4:
            if cv2.pointPolygonTest(np.array(self.clicks[:4], np.float32), (u, v), True) < INSIDE_MARGIN_PX:
                print("[cloth_tracker] outside the cloth: not added", flush=True)
                return
        self.clicks.append((u, v))
        self.raw.append((float(x), float(y)))

    def auto_points(self):
        if self.mode != "fit" or len(self.clicks) < 4:
            return
        mask = np.zeros(self.gray.shape, np.uint8)
        cv2.fillConvexPoly(mask, np.round(np.array(self.clicks[:4])).astype(np.int32), 1)
        pts = [tuple(map(float, p)) for p in
               self._sample(self.gray, mask, int(self.cfg.perception.cotracker.auto_points), border=6)]
        self.clicks += pts
        self.raw += pts
        print(f"[cloth_tracker] {len(pts)} points added ({len(self.clicks) - 4} on the cloth)", flush=True)

    def freeze(self):
        self.frozen = self.capture.newest
        self.gray = cv2.cvtColor(self.frozen[0], cv2.COLOR_RGB2GRAY)
        self.strength = self._strength_map(self.gray)
        self.clicks, self.raw = [], []

    def select(self) -> None:
        """The window: live view, SPACE freezes, click, SPACE starts tracking (window closed). q quits."""
        cv2.namedWindow(WINDOW)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        print(f"[cloth_tracker] in the window: SPACE freezes the frame, click the {self.mode} points "
              "(the banner says which), SPACE starts tracking", flush=True)
        try:
            while True:
                if self.frozen is None:
                    img = self.capture.newest[0].copy()
                    panel(img, [f"LIVE  mode {self.mode}  |  SPACE: freeze and click  |  q: quit"])
                else:
                    img = self.render_select()
                cv2.imshow(WINDOW, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                key = cv2.waitKey(15) & 0xFF
                if key in (ord("q"), 27):
                    raise KeyboardInterrupt
                if key == ord(" "):
                    if self.frozen is None:
                        self.freeze()
                    elif len(self.clicks) < NEED[self.mode]:
                        print(f"[cloth_tracker] {NEED[self.mode] - len(self.clicks)} more click(s) first", flush=True)
                    else:
                        return
                elif key == ord("r"):
                    self.frozen = None
                elif key == ord("m"):
                    self.snap = not self.snap
                elif key == ord("c"):
                    self.clicks, self.raw = [], []
                elif key == ord("a"):
                    self.auto_points()
        finally:
            cv2.destroyAllWindows()
            cv2.waitKey(1)

    # ---- 2. tracking, no window: object_points + camera_image (the viewer shows it) ----------------
    def start_tracking(self) -> bool:
        """The clicks -> tracking. False (back to clicking) when the size cannot be measured."""
        lay = layout(self.geom, self.mode, self.clicks, self.frozen[1], self.intr)
        measured = lay["size_measured"]
        if self.size_cfg is None and not np.isfinite(measured):
            print("[cloth_tracker] no depth on the cloth to measure it: click again, or set fold.sheet_size", flush=True)
            return False
        self.set_size(self.size_cfg or measured)
        self.lay = lay
        sides = lay["sides"]
        uneven = np.isfinite(sides).all() and sides.min() < 0.85 * sides.max()
        print(f"[cloth_tracker] tracking {len(lay['queries'])} points ({self.mode}); corner sides "
              f"{np.round(100 * sides, 1).tolist()} cm -> sheet {100 * self.geom['size']:.1f} cm square "
              f"({'fold.sheet_size' if self.size_cfg else 'the longest side'})"
              + ("; the sides differ by more than 15%: a corner off, or the cloth not square" if uneven else "")
              + (f"; {self.size_note}" if self.size_note else "")
              + f"; click order {'mirrored (A/B swapped)' if lay['mirrored'] else 'as given'}. "
              "The viewer's camera panel shows it; Ctrl-C stops", flush=True)
        self.capture.queue.clear()
        self.tracker.start(self.frozen, lay["queries"])
        self.capture.recording = True
        self.log = collections.defaultdict(list) if self.cfg.perception.cotracker.save_dir else None
        return True

    def end_run(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if self.log:
            out = Path(str(self.cfg.perception.cotracker.save_dir)).expanduser() / f"{self.stamp}_{self.mode}.npz"
            np.savez_compressed(out, mode=self.mode, unit=self.lay["unit"], queries=self.lay["queries"],
                                size_measured=self.lay["size_measured"], sheet_size=self.geom["size"],
                                T_base_cam=self.T_base_cam, **{k: np.asarray(v) for k, v in self.log.items()})
            print(f"[cloth_tracker] saved {out} and its mp4", flush=True)
        self.log = None

    # ---- one tracked frame -------------------------------------------------------------------------
    def handle(self, frame, uv, vis) -> dict:
        rgb, depth, t = frame
        xyz = deproject(depth, uv, self.intr)
        ok = vis.astype(bool) & np.all(np.isfinite(xyz), axis=1)
        pos = xyz @ self.T_base_cam[:3, :3].T + self.T_base_cam[:3, 3]
        send(self.pub, "object_points", {"seq": self.seq, "t": t, "mode": self.mode,
                                         "sigma": float(self.cfg.perception.cotracker.sigma), "unit": self.lay["unit"],
                                         "pos": np.where(np.isfinite(pos), pos, 0.0), "visible": ok, "uv": uv,
                                         "size_measured": self.lay["size_measured"]})
        self.seq += 1
        parts, measured = cp.particles(self.geom, self.mode, self.lay["unit"], xyz, ok, self.sigma)
        r = {"frame": frame, "uv": uv, "ok": ok, "parts": parts, "measured": measured, "o": None, "sc": None}
        if np.all(np.isfinite(parts)):
            r["o"] = observe(self.geom, parts)
            r["sc"] = score(self.geom, r["o"])
            r["angle"] = cp.fold_angle_deg(self.geom, parts)
        if self.log is not None:
            for k, v in (("t", t), ("uv", uv), ("visible", ok), ("pos_base", pos), ("particles_cam", parts),
                         ("fold_error", r["sc"]["fold_error"] if r["sc"] else np.nan),
                         ("footprint", r["sc"]["footprint"] if r["sc"] else np.nan),
                         ("fold_angle_deg", r.get("angle", np.nan))):
                self.log[k].append(v)
        return r

    def run(self):
        self.select()
        while not self.start_tracking():
            self.select()
        n, t_print, last = 0, time.time(), None
        while True:
            frames = self.capture.drain()
            if not frames:
                time.sleep(0.002)
                continue
            results = []
            for f in frames:
                results += [self.handle(*x) for x in self.tracker.add(f)]
            if not results:
                continue
            last = results[-1]
            n += len(results)
            t = last["frame"][2]
            hz = self.cfg.perception.image_hz
            if (hz > 0 and t - self.t_image >= 1.0 / hz) or self.log is not None:
                img = self.render_track(last)
                if hz > 0 and t - self.t_image >= 1.0 / hz:
                    send(self.pub, "camera_image", {"seq": self.seq, "t": t, "jpeg": encode_jpeg(
                        img, self.cfg.perception.image_jpeg_quality)})
                    self.t_image = t
                self.save_frame(img)
            if time.time() - t_print > 5.0:
                sc = last["sc"]
                state = ("a half has < 3 points seen" if sc is None else
                         f"fold {last['angle']:.0f} deg, error {100 * sc['fold_error']:.1f} cm, footprint {sc['footprint']:.2f}")
                print(f"[cloth_tracker] {n / (time.time() - t_print):.0f} Hz, seen {last['ok'].sum()}/{len(last['ok'])}, "
                      f"inference {self.tracker.inference_ms:.0f} ms, latency {1000 * (time.time() - t):.0f} ms; {state}",
                      flush=True)
                n, t_print = 0, time.time()

    def close(self):
        self.end_run()
        self.capture.running = False
        self.cam.stop()
        self.pub.close(linger=0)

    def save_frame(self, img):
        save = self.cfg.perception.cotracker.save_dir
        if not save or self.log is None:
            return
        if self.writer is None:
            Path(str(save)).expanduser().mkdir(parents=True, exist_ok=True)
            self.stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            out = Path(str(save)).expanduser() / f"{self.stamp}_{self.mode}.mp4"
            self.writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), self.cfg.perception.camera_fps,
                                          (img.shape[1], img.shape[0]))
        self.writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # ---- rendering -------------------------------------------------------------------------------
    def render_select(self):
        img = self.frozen[0].copy()
        need, n = NEED[self.mode], len(self.clicks)
        uv = preview_grid(self.geom, self.mode, self.clicks)
        if uv is not None:
            draw_grid(img, self.geom, uv)
        for i, ((u, v), (x, y)) in enumerate(zip(self.clicks, self.raw)):
            if (u, v) != (x, y):
                cv2.line(img, pt((x, y)), pt((u, v)), (200, 200, 200), 1, cv2.LINE_AA)
            kp = i < need
            cv2.circle(img, pt((u, v)), 5 if kp else 3, YELLOW if kp else WHITE, -1, cv2.LINE_AA)
            if kp:
                text(img, f"{i + 1} {self.label(i)}", (u + 7, v - 7), YELLOW)
        nxt = (f"next: {self.label(n)}" if n < need else
               ("more points on the cloth, or a" if self.mode == "fit" else "done: SPACE"))
        panel(img, [f"CLICK {self.mode}  {n}/{need}{'+' if self.mode == 'fit' else ''}  |  {nxt}  |  "
                    f"{'SNAP' if self.snap else 'EXACT'} (m)",
                    "SPACE: track  R-click: undo  c: clear  r: live" + ("  a: auto points" if self.mode == "fit" else "")],
              YELLOW)
        return img

    def render_track(self, r):
        rgb, _, t = r["frame"]
        img = rgb.copy()
        draw_grid(img, self.geom, project(self.intr, r["parts"]), hollow=~r["measured"] if self.mode == "grid" else None)
        for p, good in zip(r["uv"], r["ok"]):
            if self.mode == "fit":
                cv2.circle(img, pt(p), 2, WHITE if good else RED, -1 if good else 1, cv2.LINE_AA)
            elif self.mode == "kp8":
                cv2.circle(img, pt(p), 6, WHITE if good else RED, 1, cv2.LINE_AA)
        o, sc = r["o"], r["sc"]
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
        head = (f"TRACK {self.mode}  latency {1000 * (time.time() - t):.0f} ms  inference "
                f"{self.tracker.inference_ms:.0f} ms")
        if sc is None:
            body = f"seen {r['ok'].sum()}/{len(r['ok'])}  |  a half has < 3 points seen: no grid"
        else:
            body = (f"seen {r['ok'].sum()}/{len(r['ok'])}  sheet {100 * self.geom['size']:.1f} cm  fold {r['angle']:.0f} deg"
                    f"  error {100 * sc['fold_error']:.1f} cm  footprint {sc['footprint']:.2f}{'  FOLDED' if sc['folded'] else ''}")
        panel(img, [head, body] + ([self.size_note] if self.size_note else []), GREEN if sc is not None else RED)
        return img


def main() -> None:
    if any(a.startswith("config=") for a in sys.argv[1:]):
        raise SystemExit(f"[cloth_tracker] this script reads {CONFIG.relative_to(REPO)}; drop config=")
    cfg = load_config(path=CONFIG)
    if sheet_perception(cfg) != "cotracker":
        raise SystemExit(f"[cloth_tracker] object.name={cfg.object.name}: not a cotracker object (object.name=cloth)")
    cp.check(cfg)
    node = Node(cfg)
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.close()


if __name__ == "__main__":
    main()
