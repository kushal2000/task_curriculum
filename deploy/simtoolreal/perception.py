"""The perception node: the object pose in the robot base frame, published over ZMQ (object_pose).

perception.source=camera (default): RealSense -> FoundationPose. Runs in the FoundationPose
environment on the robot PC (needs its GPU), from the repo root:

    ~/miniconda3/envs/foundationpose/bin/python deploy/simtoolreal/perception.py

  It opens the camera the extrinsics file was calibrated for (its camera_serial), and refuses another.
  1. First frame: an OpenCV window opens; click 4 corners of a box around the object, SAM turns it
     into a mask, FoundationPose registers the pose. The window then closes.
  2. Every later frame: FoundationPose tracks from the last pose; the pose goes out over ZMQ, and
     the color image (camera_image) at image_hz for the viewer.

  Follows upstream's live_tracking.py (github.com/kushal2000/FoundationPose, RealSense, UDP), with
  ZMQ instead of UDP. Frames: FoundationPose gives T_cam_obj (cam =
  the color optical frame, as calibrate_compare.py calibrates it); T_base_obj = T_base_cam @ T_cam_obj.

perception.source=fake: a fixed pose (perception.fake_pose_base), no camera, to test without one:

    .venv_deploy/bin/python deploy/simtoolreal/perception.py perception.source=fake
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))               # deploy/simtoolreal
sys.path.insert(0, str(HERE.parents[1]))    # repo root, for deploy.calibration
from utils.config import load_config, repo_path                      # noqa: E402
from utils.link import encode_jpeg, open_publisher, send             # noqa: E402


def publish_fake(cfg, pub) -> None:
    """A fixed pose at fake_hz, until Ctrl-C."""
    p = cfg.perception
    pose = np.asarray(p.fake_pose_base, dtype=float)
    pos, quat = pose[:3], pose[3:] / np.linalg.norm(pose[3:])
    print(f"[perception] fake: pos {pos}, quat_wxyz {quat} at {p.fake_hz} Hz. Ctrl-C stops.", flush=True)
    seq = 0
    while True:
        send(pub, "object_pose", {"seq": seq, "t": time.time(), "pos": pos, "quat_wxyz": quat})
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
    mesh_path = repo_path(cfg.objects[cfg.object.name].mesh)

    # the extrinsics belong to one camera: open that one, and refuse another
    serial = p.camera_serial or calib_serial
    if calib_serial is not None and serial != calib_serial:
        raise SystemExit(f"[perception] perception.camera_serial={serial}, but {calib_path.name} is for "
                         f"camera {calib_serial}")
    cam = RealSense(serial=serial, width=p.camera_width, height=p.camera_height, fps=p.camera_fps)
    cam.start()
    K = cam.K()  # intrinsics
    print(f"[perception] {cam.name} {cam.serial}, {p.camera_width}x{p.camera_height} at {p.camera_fps} fps, "
          f"extrinsics {calib_path.name}, mesh {mesh_path.name}", flush=True)
    est = load_foundationpose(fp_dir, mesh_path)
    try:
        # 1. register on the first frame
        rgb, depth, t = cam.latest_rgbd()
        mask = select_mask(rgb)
        t0 = time.time()
        est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask, iteration=p.register_iters)
        print(f"[perception] registered in {time.time() - t0:.2f} s. Ctrl-C stops.", flush=True)

        # 2. track every new frame
        seq, last_t, n, t_print, t_image = 0, t, 0, time.time(), 0.0
        while True:
            rgb, depth, t = cam.latest_rgbd()
            if t == last_t:            # no new camera frame yet
                time.sleep(0.002)
                continue
            last_t = t
            T_base_obj = T_base_cam @ est.track_one(rgb=rgb, depth=depth, K=K, iteration=p.track_iters)
            send(pub, "object_pose", {"seq": seq, "t": t, "pos": T_base_obj[:3, 3],
                                      "quat_wxyz": matrix_to_quat_wxyz(T_base_obj[:3, :3])})
            seq += 1
            if p.image_hz > 0 and t - t_image >= 1.0 / p.image_hz:   # JPEG costs a few ms per frame
                send(pub, "camera_image", {"seq": seq, "t": t, "jpeg": encode_jpeg(rgb, p.image_jpeg_quality)})
                t_image = t

            n += 1
            if time.time() - t_print > 5.0:
                print(f"[perception] {n / (time.time() - t_print):.1f} Hz, pos {np.round(T_base_obj[:3, 3], 3)}, "
                      f"latency {1000 * (time.time() - t):.0f} ms", flush=True)
                n, t_print = 0, time.time()
    finally:
        cam.stop()


def load_foundationpose(fp_dir: Path, mesh_path: Path):
    """Build FoundationPose for one mesh. Imports from the FoundationPose checkout."""
    sys.path.insert(0, str(fp_dir))
    import nvdiffrast.torch as dr
    import trimesh
    from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor
    from Utils import set_logging_format

    set_logging_format(level=logging.WARNING)   # its INFO lines: dozens per frame

    mesh = trimesh.load(str(mesh_path), process=False)   # process=False: keep the mesh frame as is
    return FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
                          scorer=ScorePredictor(), refiner=PoseRefinePredictor(),
                          glctx=dr.RasterizeCudaContext(), debug=0, debug_dir="/tmp/foundationpose")


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


def main() -> None:
    cfg = load_config()
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
