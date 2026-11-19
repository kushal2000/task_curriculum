"""A RealSense D415 as a depth source for point clouds, and the depth -> points steps every cloud needs.

Two ways to get the depth (`DepthCamera(depth=...)`):
    raw               the camera's own depth (z16), through librealsense's post-processing filters
                      (raw_filters: spatial, temporal, hole_filling; run in the disparity domain)
    foundationstereo  the camera's two IR images (the emitter's dots on) -> FoundationStereo (TensorRT
                      engine, `FoundationStereo` below) -> disparity -> depth
Either way the depth is in the depth sensor's frame (= IR camera 1: rectified, no distortion) and the
colour image comes from the same frameset. The extrinsics files calibrate the COLOUR camera, so
T_color_depth (from the device) carries the points over:

    cam = DepthCamera(serial, depth="raw", width=640, height=480, fps=60, raw_filters=["spatial"])
    cam.start()
    f = cam.latest()                  # DepthFrame: color (H, W, 3), depth (H, W) m or None, ir1, ir2, t
    depth, K = f.depth, cam.K_depth   # foundationstereo: depth, K = fs(f.ir1, f.ir2, cam.K_depth, cam.baseline)
    keep = valid_depth(depth) & ~depth_edges(depth, 0.01)
    p_depth = backproject(depth, K, keep)                   # (N, 3) in the depth sensor's frame
    p_base = transform(T_base_cam @ cam.T_color_depth, p_depth)
    rgb, seen = color_at(f.color, cam.K_color, transform(cam.T_color_depth, p_depth))
    # every pixel at once (rigid2deform/perception/cloud_camera.py): rays = pixel_rays(K, depth.shape, R_base_depth) once, then
    # p_all = t_base_depth + rays * depth[..., None]

Needs numpy + pyrealsense2 (+ cv2, torch and tensorrt for FoundationStereo, in its environment).
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np

DEPTHS = ("raw", "foundationstereo")
RAW_FILTERS = ("spatial", "temporal", "hole_filling")


class DepthFrame:
    """color (H, W, 3) uint8 RGB; depth (H, W) float32 m, 0 = none (raw; converted from z16 when first
    read, not in the camera thread); ir1, ir2 (H, W) uint8 left / right IR (foundationstereo); t:
    time.time() when the frameset arrived."""

    def __init__(self, color, z16, depth_scale, ir1, ir2, t) -> None:
        self.color, self.ir1, self.ir2, self.t = color, ir1, ir2, t
        self._z16, self._scale, self._depth = z16, depth_scale, None

    @property
    def depth(self) -> np.ndarray | None:
        if self._depth is None and self._z16 is not None:
            self._depth = self._z16.astype(np.float32) * np.float32(self._scale)
        return self._depth


class DepthCamera:
    """One D415: colour + (depth | the IR pair), the newest frameset kept by a background thread."""

    def __init__(self, serial: str, depth: str, width: int, height: int, fps: int, raw_filters=()) -> None:
        import pyrealsense2 as rs

        if depth not in DEPTHS:
            raise ValueError(f"depth={depth}: one of {DEPTHS}")
        bad = [f for f in raw_filters if f not in RAW_FILTERS]
        if bad:
            raise ValueError(f"raw_filters {bad}: each one of {RAW_FILTERS}")
        serials = [d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()]
        if serial not in serials:
            raise RuntimeError(f"RealSense {serial} not found, connected: {serials}")
        self.rs, self.serial, self.depth_kind = rs, serial, depth
        self.width, self.height, self.fps = width, height, fps
        self.raw_filters = list(raw_filters) if depth == "raw" else []
        self._pipe = rs.pipeline()
        self._lock = threading.Lock()
        self._latest: DepthFrame | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.n_frames = 0

    def start(self) -> None:
        rs = self.rs
        cfg = rs.config()
        cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        if self.depth_kind == "raw":
            cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        else:
            cfg.enable_stream(rs.stream.infrared, 1, self.width, self.height, rs.format.y8, self.fps)
            cfg.enable_stream(rs.stream.infrared, 2, self.width, self.height, rs.format.y8, self.fps)
        profile = self._pipe.start(cfg)
        sensor = profile.get_device().first_depth_sensor()
        sensor.set_option(rs.option.emitter_enabled, 1.0)        # the dots: texture for both kinds of depth
        self._depth_scale = sensor.get_depth_scale()
        self.name = profile.get_device().get_info(rs.camera_info.name)

        def video(stream, index=0):
            return profile.get_stream(stream, index).as_video_stream_profile()

        color = video(rs.stream.color)
        dprof = video(rs.stream.depth) if self.depth_kind == "raw" else video(rs.stream.infrared, 1)
        for p in (color, dprof):
            i = p.get_intrinsics()
            if any(abs(c) > 1e-9 for c in i.coeffs):
                raise RuntimeError(f"{self.serial}: {p.stream_name()} reports distortion {list(i.coeffs)}; "
                                   "this code assumes none")
        self.K_color = _K(color.get_intrinsics())
        self.K_depth = _K(dprof.get_intrinsics())
        self.T_color_depth = _T(dprof.get_extrinsics_to(color))       # depth-frame point -> colour frame
        if self.depth_kind == "foundationstereo":
            self.baseline = abs(video(rs.stream.infrared, 1).get_extrinsics_to(video(rs.stream.infrared, 2)).translation[0])
        self._filters = []
        if self.raw_filters:
            self._filters = ([rs.disparity_transform(True)]
                             + [{"spatial": rs.spatial_filter, "temporal": rs.temporal_filter,
                                 "hole_filling": rs.hole_filling_filter}[f]() for f in self.raw_filters]
                             + [rs.disparity_transform(False)])
        for _ in range(30):                                   # auto exposure settles
            self._pipe.wait_for_frames()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        while self._latest is None:
            time.sleep(0.01)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._pipe.stop()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                fs = self._pipe.wait_for_frames(1000)
            except RuntimeError:
                continue
            t = time.time()
            c = fs.get_color_frame()
            if not c:
                continue
            z16 = ir1 = ir2 = None
            if self.depth_kind == "raw":
                d = fs.get_depth_frame()
                if not d:
                    continue
                for f in self._filters:              # in this thread: temporal keeps state between frames
                    d = f.process(d)
                z16 = np.asanyarray(d.get_data()).copy()
            else:
                a, b = fs.get_infrared_frame(1), fs.get_infrared_frame(2)
                if not a or not b:
                    continue
                ir1, ir2 = np.asanyarray(a.get_data()).copy(), np.asanyarray(b.get_data()).copy()
            frame = DepthFrame(np.asanyarray(c.get_data()).copy(), z16, self._depth_scale, ir1, ir2, t)
            with self._lock:
                self._latest = frame
                self.n_frames += 1

    def latest(self) -> DepthFrame:
        """The newest frameset (the same object until a new one arrives: compare .t)."""
        with self._lock:
            return self._latest


class FoundationStereo:
    """FoundationStereo's TensorRT engine (the checkout's trt/fs_trt.py) on a rectified IR pair.

    The pair is resized to the engine's size (INTER_AREA) and the depth comes out at that size, with K
    scaled to match; a pixel whose match would lie left of the right image (u - disparity < 0) or with
    disparity <= 0 gets depth 0."""

    def __init__(self, fs_dir: str | Path, engine: str | Path) -> None:
        fs_dir = Path(fs_dir).expanduser()
        engine = Path(engine).expanduser()
        engine = engine if engine.is_absolute() else fs_dir / engine
        sys.path.insert(0, str(fs_dir / "trt"))
        import torch
        from fs_trt import FoundationStereoTRT

        self.torch = torch
        self.model = FoundationStereoTRT(str(engine))
        self.height, self.width = int(self.model.height), int(self.model.width)
        self.engine = engine

    def __call__(self, ir1: np.ndarray, ir2: np.ndarray, K: np.ndarray, baseline: float) -> tuple[np.ndarray, np.ndarray]:
        """(depth (h, w) float32 m at the engine's size, its 3 x 3 K)."""
        import cv2

        torch = self.torch
        H, W = ir1.shape
        sx, sy = self.width / W, self.height / H
        imgs = [ir if (W, H) == (self.width, self.height) else
                cv2.resize(ir, (self.width, self.height), interpolation=cv2.INTER_AREA) for ir in (ir1, ir2)]
        left, right = (torch.from_numpy(i).cuda().float()[None, None].expand(1, 3, -1, -1) for i in imgs)
        disp = self.model(left, right)                                     # (h, w) px, cuda
        Ks = K.copy()
        Ks[0, 0], Ks[1, 1] = K[0, 0] * sx, K[1, 1] * sy
        Ks[0, 2], Ks[1, 2] = (K[0, 2] + 0.5) * sx - 0.5, (K[1, 2] + 0.5) * sy - 0.5
        u = torch.arange(self.width, device=disp.device, dtype=disp.dtype)[None]
        ok = (disp > 0) & (u - disp >= 0)
        depth = torch.where(ok, Ks[0, 0] * baseline / disp.clamp(min=1e-3), torch.zeros_like(disp))
        return depth.cpu().numpy().astype(np.float32), Ks


# ---------------------------------------------------------------------------------------------------
# depth -> points
# ---------------------------------------------------------------------------------------------------
def valid_depth(depth: np.ndarray, min_m: float = 0.2, max_m: float = 3.0) -> np.ndarray:
    return np.isfinite(depth) & (depth > min_m) & (depth < max_m)


def depth_edges(depth: np.ndarray, jump_m: float) -> np.ndarray:
    """(H, W) bool: pixels with an 8-neighbour more than jump_m nearer or further (flying pixels sit
    between a near and a far surface, so both sides flag them). Missing neighbours (0) do not count."""
    import cv2

    kernel = np.ones((3, 3), np.uint8)
    far = cv2.dilate(depth, kernel)                                    # 0 = missing: never the max
    near = cv2.erode(np.where(depth > 0, depth, np.inf).astype(np.float32), kernel)
    return (far - depth > jump_m) | (depth - near > jump_m)


def pixel_rays(K: np.ndarray, shape: tuple[int, int], R: np.ndarray | None = None) -> np.ndarray:
    """(H, W, 3) float32: each pixel's point at depth 1 (z along the optical axis), rotated by R (e.g. the
    camera's rotation in the base frame). point = t + rays * depth[..., None]: every pixel at once."""
    v, u = np.mgrid[0:shape[0], 0:shape[1]]
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones(shape)], axis=-1)
    return (rays @ (np.eye(3) if R is None else R).T).astype(np.float32)


def backproject(depth: np.ndarray, K: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """(N, 3) float32 points in the camera's frame (x right, y down, z forward) of the pixels in keep."""
    v, u = np.nonzero(keep)
    z = depth[v, u]
    return np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], axis=1).astype(np.float32)


def transform(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    return (p @ T[:3, :3].T + T[:3, 3]).astype(np.float32)


def color_at(color: np.ndarray, K_color: np.ndarray, p_color: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The colour image's pixel under each point (colour-camera frame): ((N, 3) uint8 RGB, (N,) bool
    inside the image and in front of the camera; RGB 0 where not)."""
    H, W = color.shape[:2]
    z = p_color[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.round(K_color[0, 0] * p_color[:, 0] / z + K_color[0, 2])
        v = np.round(K_color[1, 1] * p_color[:, 1] / z + K_color[1, 2])
    ok = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    rgb = np.zeros((len(p_color), 3), np.uint8)
    rgb[ok] = color[v[ok].astype(int), u[ok].astype(int)]
    return rgb, ok


def voxel_downsample(p: np.ndarray, voxel_m: float) -> np.ndarray:
    """One point per occupied voxel: the mean of the points in it."""
    if len(p) == 0 or voxel_m <= 0:
        return p
    keys = np.floor(p / voxel_m).astype(np.int64)
    keys -= keys.min(axis=0)
    span = keys.max(axis=0) + 1
    flat = (keys[:, 0] * span[1] + keys[:, 1]) * span[2] + keys[:, 2]
    if span.prod() <= 4 * len(p) + 1_000_000:          # a dense count over the voxels: no sort
        counts = np.bincount(flat, minlength=int(span.prod()))
        occupied = np.flatnonzero(counts)
        inverse = (np.cumsum(counts > 0) - 1)[flat]
        counts = counts[occupied]
    else:
        _, inverse, counts = np.unique(flat, return_inverse=True, return_counts=True)
        inverse = inverse.reshape(-1)
    out = np.stack([np.bincount(inverse, weights=p[:, i], minlength=len(counts)) for i in range(3)], axis=1)
    return (out / counts[:, None]).astype(np.float32)


def _K(i) -> np.ndarray:
    return np.array([[i.fx, 0.0, i.ppx], [0.0, i.fy, i.ppy], [0.0, 0.0, 1.0]])


def _T(e) -> np.ndarray:
    """librealsense extrinsics (rotation column-major) -> 4 x 4: a point in the source frame -> target."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(e.rotation).reshape(3, 3).T
    T[:3, 3] = e.translation
    return T
