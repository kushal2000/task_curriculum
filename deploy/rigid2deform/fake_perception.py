"""perception.source=fake for rigid2deform: a still sheet, no camera, to test without one.

    .venv_deploy/bin/python deploy/rigid2deform/fake_perception.py [overrides]

Sends what object.name's perception node would, at perception.fake_hz until Ctrl-C.
perception.fake_pose_base is the SHEET's pose (hinge line middle, mid-plane, robot base frame); the
moving half is turned up about the hinge by fold.fake_fold_deg.
  foundationpose: object_pose and object_poses, the two plates (object.name [moving, stationary]) where they
                  lie in that sheet (sheets/rigid_plates.py fake_message), as
                  deploy/nodes/perception.py does.
  cotracker:      not yet (sheets/cloth_points.py).
"""

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))    # repo root, for deploy.*
from deploy.rigid2deform.fold import fold_geometry                               # noqa: E402
from deploy.rigid2deform.sheets import rigid_plates, sheet_source                # noqa: E402
from deploy.utils.config import load_config, repo_path                           # noqa: E402
from deploy.utils.link import open_publisher, send_object_poses                  # noqa: E402
from deploy.utils.spec import load_spec                                          # noqa: E402


def main() -> None:
    cfg = load_config(path=HERE / "config.yaml")
    if cfg.task != "fold":
        raise SystemExit(f"[fake_perception] task={cfg.task}: only fold so far")
    src = sheet_source(cfg)
    if src is not rigid_plates:
        raise SystemExit(f"[fake_perception] {src.__name__}: no fake sender yet")
    spec = load_spec(repo_path(cfg.spec_path))
    sheet = src.init(cfg, spec, fold_geometry(spec, src.sheet_size(cfg)))
    names, pos, quat = src.fake_message(sheet, cfg.perception.fake_pose_base, np.radians(cfg.fold.fake_fold_deg))
    pub = open_publisher(cfg.bus.object_pose_port)
    hz = cfg.perception.fake_hz
    print(f"[fake_perception] {names}, sheet at {cfg.perception.fake_pose_base}, moving half "
          f"turned {cfg.fold.fake_fold_deg} deg, at {hz} Hz. Ctrl-C stops.", flush=True)
    seq = 0
    try:
        while True:
            send_object_poses(pub, seq, time.time(), names, pos, quat)
            seq += 1
            time.sleep(1.0 / hz)
    except KeyboardInterrupt:
        pass
    finally:
        pub.close(linger=0)


if __name__ == "__main__":
    main()
