"""perception.source=fake for rigid2deform: a still sheet, no camera, to test without one.

    .venv_deploy/bin/python deploy/rigid2deform/perception/fake_sheet.py [overrides]

Sends what object.name's perception node would, at perception.fake_hz until Ctrl-C.
perception.fake_pose_base is the SHEET's pose (hinge line middle, mid-plane, robot base frame); the
moving half is turned up about the hinge by fold.fake_fold_deg.
  foundationpose: object_pose and object_poses, the two plates (object.name [moving, stationary]) where they
                  lie in that sheet (sheets/rigid_plates.py fake_message), as
                  deploy/nodes/perception.py does.
  cotracker:      object_points, the points perception.cotracker.mode tracks, all seen
                  (sheets/cloth_points.py fake_message), as cloth_tracker.py does.
"""

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))    # repo root, for deploy.*
from deploy.rigid2deform.fold import fold_geometry                               # noqa: E402
from deploy.rigid2deform.sheets import cloth_points, rigid_plates, sheet_source  # noqa: E402
from deploy.utils.config import load_config, repo_path                           # noqa: E402
from deploy.utils.link import open_publisher, send, send_object_poses            # noqa: E402
from deploy.utils.spec import load_spec                                          # noqa: E402


def main() -> None:
    cfg = load_config(path=HERE.parent / "config.yaml")
    if cfg.task != "fold":
        raise SystemExit(f"[fake_sheet] task={cfg.task}: only fold so far")
    src = sheet_source(cfg)
    spec = load_spec(repo_path(cfg.spec_path))
    size = src.sheet_size(cfg) or spec["fold"]["size_range"][0]   # a cloth without fold.sheet_size: the smallest trained
    sheet = src.init(cfg, spec, fold_geometry(spec, size))
    fold_rad = np.radians(cfg.fold.fake_fold_deg)
    if src is rigid_plates:
        names, pos, quat = src.fake_message(sheet, cfg.perception.fake_pose_base, fold_rad)
        what = names
    else:
        points = cloth_points.fake_message(sheet, cfg.perception.fake_pose_base, fold_rad)
        what = f"cloth, {len(points['pos'])} points ({points['mode']})"
    pub = open_publisher(cfg.bus.object_pose_port)
    hz = cfg.perception.fake_hz
    print(f"[fake_sheet] {what}, sheet at {cfg.perception.fake_pose_base}, moving half "
          f"turned {cfg.fold.fake_fold_deg} deg, at {hz} Hz. Ctrl-C stops.", flush=True)
    seq = 0
    try:
        while True:
            if src is rigid_plates:
                send_object_poses(pub, seq, time.time(), names, pos, quat)
            else:
                send(pub, "object_points", {"seq": seq, "t": time.time(), **points})
            seq += 1
            time.sleep(1.0 / hz)
    except KeyboardInterrupt:
        pass
    finally:
        pub.close(linger=0)


if __name__ == "__main__":
    main()
