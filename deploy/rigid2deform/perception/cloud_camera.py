"""The point-cloud policies' perception node: both D415s -> the live cloud the policy sees -> point_cloud.

Usually started by perception/main.py (policy.checkpoint a point-cloud policy); on its own:

    ~/miniconda3/envs/foundationstereo/bin/python deploy/rigid2deform/perception/cloud_camera.py \
        [perception.pointcloud.depth=raw|foundationstereo]

Runs in the foundationstereo environment (pyrealsense2, TensorRT, torch, cv2 with a window; + pyzmq);
.venv_deploy's cv2 cannot open the window. Every setting: config.yaml perception.pointcloud.

First the non-live group of points, which policy.checkpoint's mode says (cloud_obs.mode_of; cloud_reference.py), in a
window with the first camera's image; nothing is sent before it is accepted, then every point_cloud
message carries the same one. To take it again, restart. Either mode: put the cloth flat, SPACE
freezes, click once on the half that folds over, SPACE accepts.
  label  the reference: the cloth's points -> ref_xyz (R <= 192, 3) base frame, ref_moving (R,) (True = the
         half that folds over), ref_corners (4, 3)
  goal   the goal, synthesised from the flat cloth as training does (the stationary half's rectangle, both
         layers in one) -> goal_xyz (192, 3), goal_corners (4, 3)

Then each step (at most perception.pointcloud.rate_hz, waiting for a new frame of the first camera), for each
camera's newest frameset:
  1. depth: raw = the camera's own (through raw_filters) | foundationstereo = FoundationStereo on its IR
     pair, at the engine's size
  2. drop invalid pixels and flying ones (a neighbour more than edge_jump_m nearer / further)
  3. pixels -> points in the depth sensor's frame -> the robot base frame (T_base_cam of the extrinsics
     file, which calibrates the colour camera, x the device's depth -> colour transform)
  4. keep the points in training's box (in_box: the table top, 2 cm below it, no ceiling)
  5. the table: a point within table_band_m of the table top (or below it) whose colour (the colour
     image's pixel under it) is black is dropped. Black: chroma (brightest - darkest RGB channel) <=
     table_chroma_max, and either dark (brightest channel <= table_dark_max) or a cool grey up to
     table_grey_max (red - blue <= table_warm_max: the glossy tape's highlights; the brown wood is warm,
     R - B ~ +24). Chroma, not HSV saturation: the darkest pixels' S jumps to 130-250. Measured Nov 18
     on both cameras: 0.3-0.6 % of the black tape kept; wood 0-5 %, the purple / cyan cloth 0 % removed
then both cameras' points together -> one per voxel_m voxel -> at most cloud_obs.N_POINTS (1024) at
random -> point_cloud (deploy/utils/link.py send_cloud; t = the older camera frame's time, so the policy
sees how old it is). The hand and the arm stay in the cloud: training's cloud has them.

For the viewer: camera_image at perception.image_hz = the first camera's image with the sent points drawn
on it (green) and the reference (its outline, fold line, points: orange moving, blue stationary); with perception.frame_stream every frame of the first camera goes out as camera_frame too
(recorder.py record.camera). The terminal prints the rate, the time of each stage and the point counts
every 5 s. Ctrl-C stops.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))    # repo root, for deploy.*
from deploy.rigid2deform.perception import cloud_reference                                       # noqa: E402
from deploy.rigid2deform.cloud_obs import BOX_HI_BASE, BOX_LO_BASE, N_POINTS, ROOT_ENV, TABLE_Z_ENV, mode_of   # noqa: E402
from deploy.utils.config import load_config, repo_path                                  # noqa: E402
from deploy.utils.depth_camera import (DepthCamera, FoundationStereo, color_at, depth_edges,  # noqa: E402
                                       pixel_rays, transform, valid_depth, voxel_downsample)
from deploy.utils.link import (CLOUD_PORT, FRAME_PORT, encode_jpeg, open_publisher, send,     # noqa: E402
                               send_cloud, send_frame)

CONFIG = HERE.parent / "config.yaml"
TABLE_Z_BASE = TABLE_Z_ENV - ROOT_ENV[2]   # 0.33: training's table top above the base
REPORT_EVERY_S = 5.0
STAGES = ("depth", "points", "table", "merge")


class CloudCamera:
    """One camera of the cloud: the device, where it is, and its frame -> base-frame points."""

    def __init__(self, extrinsics: Path, depth: str, cfg) -> None:
        p, pc = cfg.perception, cfg.perception.pointcloud
        calib = json.loads(extrinsics.read_text())
        if calib.get("camera_serial") is None:
            raise SystemExit(f"[cloud_camera] {extrinsics.name} has no camera_serial: which camera is it?")
        self.T_base_cam = np.asarray(calib["T_base_cam"], dtype=float)
        self.name = extrinsics.name
        self.dev = DepthCamera(str(calib["camera_serial"]), depth, int(p.camera_width), int(p.camera_height),
                               int(p.camera_fps), list(pc.raw_filters))
        self.dev.start()
        self.T_base_depth = self.T_base_cam @ self.dev.T_color_depth
        self.T_color_base = self.dev.T_color_depth @ np.linalg.inv(self.T_base_depth)
        self.edge_jump = float(pc.edge_jump_m)
        self.table_top = TABLE_Z_BASE + float(pc.table_band_m)
        self.dark_max, self.grey_max = int(pc.table_dark_max), int(pc.table_grey_max)
        self.chroma_max, self.warm_max = int(pc.table_chroma_max), int(pc.table_warm_max)
        self._rays = {}                                # depth image size -> (H, W, 3) base-frame rays

    def points(self, frame, fs: FoundationStereo | None, times: dict) -> tuple[np.ndarray, dict]:
        """(N, 3) float32 base-frame points of this frameset, after steps 1-5; counts of each step."""
        t0 = time.perf_counter()
        if fs is None:
            depth, K = frame.depth, self.dev.K_depth
        else:
            depth, K = fs(frame.ir1, frame.ir2, self.dev.K_depth, self.dev.baseline)
        t1 = time.perf_counter()
        if depth.shape not in self._rays:              # K is fixed for a size: rays once
            self._rays[depth.shape] = pixel_rays(K, depth.shape, self.T_base_depth[:3, :3])
        valid = valid_depth(depth)
        keep = valid & ~depth_edges(depth, self.edge_jump)
        p_all = self.T_base_depth[:3, 3].astype(np.float32) + self._rays[depth.shape] * depth[..., None]
        inside = keep & in_box(p_all)
        p_base = p_all[inside]
        t2 = time.perf_counter()
        low = np.flatnonzero(p_base[:, 2] < self.table_top)
        rgb, seen = color_at(frame.color, self.dev.K_color, transform(self.T_color_base, p_base[low]))
        rgb = rgb.astype(np.int16)
        hi = rgb.max(axis=1)
        grey = (hi - rgb.min(axis=1)) <= self.chroma_max
        cool = (rgb[:, 0] - rgb[:, 2]) <= self.warm_max
        black = ~seen | (grey & ((hi <= self.dark_max) | ((hi <= self.grey_max) & cool)))
        drop = np.zeros(len(p_base), bool)
        drop[low[black]] = True
        t3 = time.perf_counter()
        for k, dt in zip(STAGES, (t1 - t0, t2 - t1, t3 - t2)):
            times[k] = times.get(k, 0.0) + dt
        counts = {"valid": int(valid.sum()), "edges": int(valid.sum() - keep.sum()), "box": len(p_base),
                  "table": int(drop.sum())}
        return p_base[~drop], counts

    def overlay(self, image: np.ndarray, xyz: np.ndarray, ref=None) -> np.ndarray:
        """The colour image with base-frame points drawn on it (green, 3 x 3 px), then the reference."""
        out = image.copy()
        p_cam = transform(np.linalg.inv(self.T_base_cam), xyz)
        _, seen = color_at(image, self.dev.K_color, p_cam)
        K, z = self.dev.K_color, p_cam[seen, 2]
        u = np.round(K[0, 0] * p_cam[seen, 0] / z + K[0, 2]).astype(int)
        v = np.round(K[1, 1] * p_cam[seen, 1] / z + K[1, 2]).astype(int)
        H, W = out.shape[:2]
        for dv in (-1, 0, 1):
            for du in (-1, 0, 1):
                out[np.clip(v + dv, 0, H - 1), np.clip(u + du, 0, W - 1)] = (40, 230, 60)
        if ref is not None:
            cloud_reference.draw_reference(out, self, ref)
        return out


def merged_cloud(cams: list, frames: list, fs, voxel_m: float, times: dict, counts: dict) -> np.ndarray:
    """Every camera's points of these framesets (steps 1-5), together, one per voxel: (N, 3) float32."""
    parts = []
    for cam, frame in zip(cams, frames):
        pts, c = cam.points(frame, fs, times)
        parts.append(pts)
        for k, v in c.items():
            counts[f"{cam.name}:{k}"] = counts.get(f"{cam.name}:{k}", 0) + v
    t0 = time.perf_counter()
    vox = voxel_downsample(np.concatenate(parts), voxel_m)
    times["merge"] = times.get("merge", 0.0) + time.perf_counter() - t0
    return vox


def in_box(p: np.ndarray) -> np.ndarray:
    """(...,) inside training's box (cloud_obs.BOX_LO_BASE .. BOX_HI_BASE, no top) for (..., 3) points known
    to be finite, a component at a time (an (H, W, 3) image of points: ~1 ms instead of ~10)."""
    out = np.ones(p.shape[:-1], bool)
    for i in range(3):
        c = p[..., i]
        if np.isfinite(BOX_LO_BASE[i]):
            out &= c >= BOX_LO_BASE[i]
        if np.isfinite(BOX_HI_BASE[i]):
            out &= c <= BOX_HI_BASE[i]
    return out


class FrameStream(threading.Thread):
    """Every new frame of one camera out as camera_frame (recorder.py record.camera), on its own socket."""

    def __init__(self, dev: DepthCamera, port: int) -> None:
        super().__init__(daemon=True)
        self.dev, self.port, self.running = dev, port, True

    def run(self) -> None:
        pub = open_publisher(self.port)
        last_t, seq = None, 0
        try:
            while self.running:
                f = self.dev.latest()
                if f.t == last_t:
                    time.sleep(0.002)
                    continue
                last_t = f.t
                send_frame(pub, seq, f.t, f.color)
                seq += 1
        finally:
            pub.close(linger=0)


def checkpoint_mode(cfg) -> str:
    """policy.checkpoint's mode (label | goal): which non-live group the policy needs (cloud_reference.py)."""
    import torch

    path = repo_path(cfg.policy.checkpoint)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "pc_args" not in ckpt:
        raise SystemExit(f"[cloud_camera] {path.name} has no pc_args (not a distilled point-cloud student): "
                         "which non-live group (reference or goal) it needs is unknown")
    return mode_of(ckpt["pc_args"])


def main() -> None:
    cfg = load_config(path=CONFIG)
    p, pc = cfg.perception, cfg.perception.pointcloud
    mode = checkpoint_mode(cfg)
    depth = str(pc.depth)
    fs = None
    if depth == "foundationstereo":
        print(f"[cloud_camera] loading FoundationStereo ({pc.fs_engine}) ...", flush=True)
        fs = FoundationStereo(pc.foundationstereo_dir, pc.fs_engine)
    cams = [CloudCamera(repo_path(e), depth, cfg) for e in pc.extrinsics]
    for c in cams:
        print(f"[cloud_camera] {c.dev.name} {c.dev.serial} ({c.name}): {p.camera_width}x{p.camera_height} at "
              f"{p.camera_fps} fps, depth {depth}"
              + (f" (filters {list(pc.raw_filters) or 'none'})" if fs is None else
                 f" (FoundationStereo {fs.width}x{fs.height}, baseline {1000 * c.dev.baseline:.1f} mm)"),
              flush=True)

    pub = open_publisher(cfg.bus[CLOUD_PORT])
    image_pub = open_publisher(cfg.bus.object_pose_port) if p.image_hz > 0 else None
    stream = FrameStream(cams[0].dev, cfg.bus[FRAME_PORT]) if p.frame_stream else None
    if stream is not None:
        stream.start()
    rng = np.random.default_rng()
    period = 1.0 / float(pc.rate_hz)
    voxel_m = float(pc.voxel_m)

    try:
        ref = cloud_reference.pick(cams[0], cams, lambda frames: merged_cloud(cams, frames, fs, voxel_m, {}, {}),
                                   cfg, mode, rng)
    except KeyboardInterrupt:
        if stream is not None:
            stream.running = False
        for c in cams:
            c.dev.stop()
        return
    ref_fields = ref.fields()
    print(f"[cloud_camera] {ref.summary()}", flush=True)

    seq, last_t, t_image = 0, None, 0.0
    report = {"t": time.time(), "steps": 0, "times": {}, "counts": {}, "voxels": 0, "sent": 0, "age": 0.0}
    print(f"[cloud_camera] point_cloud on port {cfg.bus[CLOUD_PORT]}, at most {pc.rate_hz:g} Hz, "
          f"{N_POINTS} points, voxel {1000 * pc.voxel_m:g} mm. Ctrl-C stops.", flush=True)
    try:
        while True:
            t_start = time.perf_counter()
            while (first := cams[0].dev.latest()).t == last_t:          # a new frame of the first camera
                time.sleep(0.001)
            last_t = first.t
            frames = [first] + [c.dev.latest() for c in cams[1:]]

            vox = merged_cloud(cams, frames, fs, voxel_m, report["times"], report["counts"])
            xyz = vox if len(vox) <= N_POINTS else vox[rng.choice(len(vox), N_POINTS, replace=False)]

            t_frames = [f.t for f in frames]
            send_cloud(pub, seq, min(t_frames), xyz, depth=depth, t_cams=t_frames, **ref_fields)
            seq += 1
            report["steps"] += 1
            report["voxels"] += len(vox)
            report["sent"] += len(xyz)
            report["age"] += time.time() - min(t_frames)

            if image_pub is not None and first.t - t_image >= 1.0 / p.image_hz:
                t_image = first.t
                send(image_pub, "camera_image", {"seq": seq, "t": first.t, "image_hz": p.image_hz,
                                                 "jpeg": encode_jpeg(cams[0].overlay(first.color, xyz, ref),
                                                                     p.image_jpeg_quality)})

            now = time.time()
            if now - report["t"] >= REPORT_EVERY_S:
                print_report(report, cams, now)
                report = {"t": now, "steps": 0, "times": {}, "counts": {}, "voxels": 0, "sent": 0, "age": 0.0}
            time.sleep(max(0.0, period - (time.perf_counter() - t_start)))
    except KeyboardInterrupt:
        pass
    finally:
        if stream is not None:
            stream.running = False
        for c in cams:
            c.dev.stop()


def print_report(report: dict, cams: list, now: float) -> None:
    n = max(report["steps"], 1)
    times = " ".join(f"{k} {1000 * report['times'].get(k, 0.0) / n:.1f}" for k in STAGES)
    per_cam = "; ".join(
        f"{c.name.split('_')[0]} " + " ".join(f"{k} {report['counts'].get(f'{c.name}:{k}', 0) // n}"
                                               for k in ("valid", "edges", "box", "table"))
        for c in cams)
    stale = [f"{c.name} {now - c.dev.latest().t:.1f} s" for c in cams if now - c.dev.latest().t > 0.5]
    print(f"[cloud_camera] {report['steps'] / (now - report['t']):.1f} Hz, ms/step: {times}; "
          f"age {1000 * report['age'] / n:.0f} ms; per camera (px / points): {per_cam}; "
          f"voxels {report['voxels'] // n}, sent {report['sent'] // n}"
          + (f"; NO NEW FRAMES: {', '.join(stale)}" if stale else ""), flush=True)


if __name__ == "__main__":
    main()
