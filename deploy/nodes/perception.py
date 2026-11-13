"""The perception node: the object pose in the robot base frame, published over ZMQ (object_pose).

perception.source=camera (default): RealSense -> FoundationPose. Runs in the FoundationPose
environment on the robot PC (needs its GPU), from the repo root:

    ~/miniconda3/envs/foundationpose/bin/python deploy/nodes/perception.py config=deploy/<task>/config.yaml

  It opens the camera the extrinsics file was calibrated for (its camera_serial), and refuses another.
  1. First frame: an OpenCV window opens; click 4 corners of a box around the object, SAM turns it
     into a mask, FoundationPose registers the pose. The window then closes. With a list in
     object.name, once per object, in the list's order (the terminal says which).
  2. Every later frame: FoundationPose tracks from the last pose; the pose goes out over ZMQ, and
     the color image (camera_image) at image_hz for the viewer. object_pose is the first object's (the
     policy's); with a list in object.name, object_poses also carries every tracked object. Each object
     costs one FoundationPose step per frame; perception.parallel_track runs them in threads, one CUDA
     stream each, so they overlap on the GPU. The status line prints the tracking time per frame.
  3. With perception.frame_stream (default on), a thread sends every camera frame raw (camera_frame,
     camera_frame_port) at the camera's fps, whatever FoundationPose's speed; recorder.py
     record.camera=true encodes them to an mp4. Nothing listening: the frames are dropped.

  Follows upstream's live_tracking.py (github.com/kushal2000/FoundationPose, RealSense, UDP), with
  ZMQ instead of UDP. Frames: FoundationPose gives T_cam_obj (cam =
  the color optical frame, as calibrate_compare.py calibrates it); T_base_obj = T_base_cam @ T_cam_obj.

perception.source=fake: a fixed pose (perception.fake_pose_base), no camera, to test without one:

    .venv_deploy/bin/python deploy/nodes/perception.py config=deploy/simtoolreal/config.yaml perception.source=fake

  Objects after the first: 15 cm further along base x each. An object whose perception
  (objects.<name>.perception) is not foundationpose is refused. A task that needs its objects placed
  relative to each other has its own fake sender (deploy/rigid2deform/fake_perception.py).
"""

import json
import logging
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root, for deploy.*
from deploy.utils.config import load_config, object_meshes, object_names, repo_path                      # noqa: E402
from deploy.utils.link import FRAME_PORT, encode_jpeg, open_publisher, send, send_frame, send_object_poses   # noqa: E402


def publish_fake(cfg, pub) -> None:
    """A fixed pose at fake_hz, until Ctrl-C. Objects after the first: 15 cm further along base x each."""
    p = cfg.perception
    names = object_names(cfg)
    pose = np.asarray(p.fake_pose_base, dtype=float)
    pos = np.array([pose[:3] + [0.15 * i, 0.0, 0.0] for i in range(len(names))])
    quat = np.tile(pose[3:] / np.linalg.norm(pose[3:]), (len(names), 1))
    print(f"[perception] fake: {names}, pos {pos.tolist()}, quat_wxyz {quat[0]} at {p.fake_hz} Hz. "
          "Ctrl-C stops.", flush=True)
    seq = 0
    while True:
        send_object_poses(pub, seq, time.time(), names, pos, quat)
        seq += 1
        time.sleep(1.0 / p.fake_hz)


def track_camera(cfg, pub) -> None:
    """RealSense -> FoundationPose, until Ctrl-C."""
    from deploy.calibration.realsense import RealSense

    p = cfg.perception
    if p.foundationpose_dir is None or p.extrinsics is None:
        raise SystemExit("[perception] set perception.foundationpose_dir and perception.extrinsics "
                         "(config.yaml or the command line)")
    fp_dir = Path(p.foundationpose_dir).expanduser()
    if not (fp_dir / "estimater.py").is_file():
        raise SystemExit(f"[perception] perception.foundationpose_dir={fp_dir}: no estimater.py there")
    calib_path = repo_path(p.extrinsics)
    calib = json.loads(calib_path.read_text())
    T_base_cam = np.asarray(calib["T_base_cam"], dtype=float)  # extrinsics
    calib_serial = calib.get("camera_serial")
    names, mesh_paths = object_names(cfg), object_meshes(cfg)

    # the extrinsics belong to one camera: open that one, and refuse another
    serial = p.camera_serial or calib_serial
    if calib_serial is not None and serial != calib_serial:
        raise SystemExit(f"[perception] perception.camera_serial={serial}, but {calib_path.name} is for "
                         f"camera {calib_serial}")
    cam = RealSense(serial=serial, width=p.camera_width, height=p.camera_height, fps=p.camera_fps)
    cam.start()
    K = cam.K()  # intrinsics
    print(f"[perception] {cam.name} {cam.serial}, {p.camera_width}x{p.camera_height} at {p.camera_fps} fps, "
          f"extrinsics {calib_path.name}, meshes {[m.name for m in mesh_paths]}", flush=True)
    parallel = p.parallel_track and len(names) > 1
    ests = load_foundationpose(fp_dir, mesh_paths, own_rasterizer=parallel)
    tracker = ParallelTracker(ests) if parallel else None
    stop_frames = threading.Event()
    if p.frame_stream:
        threading.Thread(target=stream_frames, args=(cam, cfg.bus[FRAME_PORT], stop_frames), daemon=True).start()
    try:
        # 1. register on the first frame
        T_cam_last = [None] * len(names)   # each object's newest T_cam_obj
        for i, (name, est) in enumerate(zip(names, ests)):
            print(f"[perception] object {i + 1}/{len(names)}: {name}. Click 4 corners of a box around it.",
                  flush=True)
            rgb, depth, t = cam.latest_rgbd()
            mask = select_mask(rgb)
            t0 = time.time()
            T_cam_last[i] = est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask, iteration=p.register_iters)
            print(f"[perception] {name} registered in {time.time() - t0:.2f} s.", flush=True)
        print(f"[perception] tracking{' in parallel' if parallel else ''}. Ctrl-C stops.", flush=True)

        # 2. track every new frame
        seq, last_t, n, t_print, t_image, track_s = 0, t, 0, time.time(), 0.0, 0.0
        while True:
            rgb, depth, t = cam.latest_rgbd()
            if t == last_t:            # no new camera frame yet
                time.sleep(0.002)
                continue
            last_t = t
            t0 = time.perf_counter()
            if tracker is not None:
                T_cam_last = tracker.track(rgb, depth, K, p.track_iters)
            else:
                T_cam_last = [est.track_one(rgb=rgb, depth=depth, K=K, iteration=p.track_iters) for est in ests]
            track_s += time.perf_counter() - t0
            T_base_objs = [T_base_cam @ T for T in T_cam_last]
            pos = np.array([T[:3, 3] for T in T_base_objs])
            send_object_poses(pub, seq, t, names, pos, np.array([matrix_to_quat_wxyz(T[:3, :3]) for T in T_base_objs]))
            seq += 1
            if p.image_hz > 0 and t - t_image >= 1.0 / p.image_hz:   # JPEG costs a few ms per frame
                send(pub, "camera_image", {"seq": seq, "t": t, "jpeg": encode_jpeg(rgb, p.image_jpeg_quality)})
                t_image = t

            n += 1
            if time.time() - t_print > 5.0:
                print(f"[perception] {n / (time.time() - t_print):.1f} Hz, track {1000 * track_s / n:.1f} ms/frame, "
                      f"pos {np.round(pos, 3).tolist()}, latency {1000 * (time.time() - t):.0f} ms", flush=True)
                n, t_print, track_s = 0, time.time(), 0.0
    finally:
        if tracker is not None:
            tracker.close()
        stop_frames.set()
        cam.stop()


class ParallelTracker:
    """track_one for every object at once: one thread and one CUDA stream per object, so their GPU work
    overlaps (the Python between kernels still takes turns). The objects share the refiner network,
    which keeps no state between calls that matters; each has its own rasterizer (own_rasterizer)."""

    def __init__(self, ests: list) -> None:
        import torch
        from concurrent.futures import ThreadPoolExecutor

        self.ests = ests
        self.streams = [torch.cuda.Stream() for _ in ests]
        self.pool = ThreadPoolExecutor(max_workers=len(ests), thread_name_prefix="track")
        self._torch = torch

    def _one(self, i: int, rgb, depth, K, iters: int) -> np.ndarray:
        with self._torch.cuda.stream(self.streams[i]):
            return self.ests[i].track_one(rgb=rgb, depth=depth, K=K, iteration=iters)   # ends in .cpu(): synced

    def track(self, rgb, depth, K, iters: int) -> list:
        """-> T_cam_obj per object, all tracked on the same frame."""
        futures = [self.pool.submit(self._one, i, rgb, depth, K, iters) for i in range(len(self.ests))]
        return [f.result() for f in futures]

    def close(self) -> None:
        self.pool.shutdown(wait=True)


def stream_frames(cam, port: int, stop: threading.Event) -> None:
    """Every new camera frame -> camera_frame (raw RGB), until stop. Its own socket (one per thread)."""
    pub = open_publisher(port)
    seq, last_t = 0, None
    while not stop.is_set():
        rgb, t = cam.latest_color()
        if t == last_t:
            time.sleep(0.002)
            continue
        last_t = t
        send_frame(pub, seq, t, rgb)
        seq += 1


def load_foundationpose(fp_dir: Path, mesh_paths: list[Path], own_rasterizer: bool = False) -> list:
    """One FoundationPose per mesh (each keeps its own last pose), sharing the networks, and the
    rasterizer unless own_rasterizer (needed to track in parallel threads). Imports from the
    FoundationPose checkout."""
    sys.path.insert(0, str(fp_dir))
    import nvdiffrast.torch as dr
    import trimesh
    from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor
    from Utils import set_logging_format

    set_logging_format(level=logging.WARNING)   # its INFO lines: dozens per frame

    scorer, refiner, glctx = ScorePredictor(), PoseRefinePredictor(), dr.RasterizeCudaContext()
    ests = []
    for mesh_path in mesh_paths:
        mesh = trimesh.load(str(mesh_path), process=False)   # process=False: keep the mesh frame as is
        ests.append(FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
                                   scorer=scorer, refiner=refiner,
                                   glctx=dr.RasterizeCudaContext() if own_rasterizer else glctx, debug=0,
                                   debug_dir="/tmp/foundationpose"))
    return ests


def select_mask(rgb: np.ndarray) -> np.ndarray:
    """OpenCV window: click 4 corners of a box around the object -> SAM mask (H, W) bool. The window
    closes itself.

    FoundationPose's refiner makes CUDA the default tensor type, which stops transformers from loading
    SAM. CPU defaults while SAM runs, then back (as upstream live_tracking.py).
    """
    import cv2
    import torch
    from generate_mask import generate_binary_mask_box   # from the FoundationPose checkout

    cuda_default = torch.tensor(0.0).is_cuda
    if cuda_default:
        torch.set_default_tensor_type(torch.FloatTensor)
    try:
        mask = generate_binary_mask_box(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), polygon_refinement=True)
    finally:
        if cuda_default:
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
    if mask is None or not np.any(mask):
        raise SystemExit("[perception] empty mask, nothing selected")
    return mask > 0


def matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w, x, y, z), w >= 0."""
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(R).as_quat()   # scipy is xyzw
    q = np.array([w, x, y, z])
    return q if w >= 0 else -q


def check_perception(cfg) -> None:
    """Every object in object.name must be one this node tracks: objects.<name>.perception
    foundationpose, or unset (a plain mesh)."""
    for n in object_names(cfg):
        kind = cfg.objects[n].get("perception", "foundationpose")
        if kind != "foundationpose":
            raise SystemExit(f"[perception] objects.{n}.perception={kind}: this node is FoundationPose; "
                             f"that object needs its own perception node")


def main() -> None:
    cfg = load_config()
    check_perception(cfg)
    if cfg.perception.source not in ("camera", "fake"):
        raise SystemExit(f"[perception] perception.source={cfg.perception.source}: camera | fake")
    pub = open_publisher(cfg.bus.object_pose_port)
    try:
        (publish_fake if cfg.perception.source == "fake" else track_camera)(cfg, pub)
    except KeyboardInterrupt:
        pass
    finally:
        pub.close(linger=0)


if __name__ == "__main__":
    main()
