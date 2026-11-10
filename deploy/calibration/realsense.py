"""RealSense depth camera -> colored point cloud in the color camera's optical frame (x right, y down,
z forward), which is the frame the extrinsics are calibrated for.

A background thread keeps the latest frame, so the viser loop never blocks on the camera.

    cam = RealSense()                     # the only one connected, or RealSense(serial="1431...")
    cam.start()
    frame = cam.latest()                  # Frame(points (N,3) m, colors (N,3) uint8, image, t)
    cam.stop()
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class Frame:
    points: np.ndarray  # (N, 3) float32, color optical frame, m
    colors: np.ndarray  # (N, 3) uint8
    image: np.ndarray   # (H, W, 3) uint8 color image
    t: float


@dataclass
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    model: str
    coeffs: list[float]

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class RealSense:
    def __init__(self, serial: str | None = None, width: int = 1280, height: int = 720, fps: int = 30,
                 min_depth: float = 0.15, max_depth: float = 2.5) -> None:
        import pyrealsense2 as rs

        self.rs = rs
        self.width, self.height, self.fps = width, height, fps
        self.min_depth, self.max_depth = min_depth, max_depth
        devices = rs.context().query_devices()
        serials = [d.get_info(rs.camera_info.serial_number) for d in devices]
        if not serials:
            raise RuntimeError("no RealSense found")
        if serial is None:
            if len(serials) > 1:
                raise RuntimeError(f"several RealSense found {serials}, pick one with --serial")
            serial = serials[0]
        elif serial not in serials:
            raise RuntimeError(f"RealSense {serial} not found, connected: {serials}")
        self.serial = serial
        self.name = next(d.get_info(rs.camera_info.name) for d in devices
                         if d.get_info(rs.camera_info.serial_number) == serial)
        self._pipe = rs.pipeline()
        self._align = rs.align(rs.stream.color)
        self._lock = threading.Lock()
        self._latest: tuple[np.ndarray, np.ndarray, float] | None = None  # depth (m), color, t
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.intrinsics: Intrinsics | None = None

    def start(self) -> None:
        rs = self.rs
        cfg = rs.config()
        cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        profile = self._pipe.start(cfg)
        self._depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        i = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.intrinsics = Intrinsics(i.width, i.height, i.fx, i.fy, i.ppx, i.ppy, str(i.model), list(i.coeffs))
        u, v = np.meshgrid(np.arange(i.width), np.arange(i.height))
        self._ray_x = ((u - i.ppx) / i.fx).astype(np.float32)
        self._ray_y = ((v - i.ppy) / i.fy).astype(np.float32)
        for _ in range(30):  # auto exposure settles
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
                fs = self._align.process(self._pipe.wait_for_frames(1000))
            except RuntimeError:
                continue
            d, c = fs.get_depth_frame(), fs.get_color_frame()
            if not d or not c:
                continue
            depth = np.asanyarray(d.get_data()).astype(np.float32) * self._depth_scale
            with self._lock:
                self._latest = (depth, np.asanyarray(c.get_data()).copy(), time.time())

    def _frame(self, depth: np.ndarray, color: np.ndarray, t: float) -> Frame:
        ok = np.isfinite(depth) & (depth > self.min_depth) & (depth < self.max_depth)
        z = depth[ok]
        pts = np.stack([self._ray_x[ok] * z, self._ray_y[ok] * z, z], axis=1)
        return Frame(pts.astype(np.float32), color[ok], color, t)

    def latest(self) -> Frame:
        with self._lock:
            depth, color, t = self._latest
        return self._frame(depth, color, t)

    def latest_rgbd(self) -> tuple[np.ndarray, np.ndarray, float]:
        """(rgb (H, W, 3) uint8, depth (H, W) float32 m with 0 where invalid, t) of the newest frame,
        depth aligned to the color image."""
        with self._lock:
            depth, color, t = self._latest
        depth = depth.copy()
        depth[~np.isfinite(depth) | (depth < self.min_depth) | (depth > self.max_depth)] = 0.0
        return color, depth, t

    def latest_color(self) -> tuple[np.ndarray, float]:
        """(rgb (H, W, 3) uint8, t) of the newest frame, without the depth work of latest_rgbd. Each new
        frame is a new array, so it can be kept or sent without copying."""
        with self._lock:
            _, color, t = self._latest
        return color, t

    def K(self) -> np.ndarray:
        """3x3 color camera matrix (after start())."""
        i = self.intrinsics
        return np.array([[i.fx, 0.0, i.cx], [0.0, i.fy, i.cy], [0.0, 0.0, 1.0]])
