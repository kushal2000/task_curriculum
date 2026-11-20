"""A stand-in for the point-cloud perception node (pc_node.py), to test policy.py's point-cloud loop
without cameras: a still, flat, square sheet, as point_cloud messages (deploy/utils/link.py send_cloud).

    .venv_deploy/bin/python deploy/rigid2deform/perception/pc_fake_node.py [overrides]
    .venv_deploy/bin/python deploy/rigid2deform/run.py robot.arm=mujoco robot.hand=fake [mode=pc_goal]

The sheet: side fold.sheet_size (null: 0.15 m), at perception.fake_pose_base (the sheet's pose: hinge
line middle, mid-plane, robot base frame; sheet x across the hinge, the moving half at x > 0), at
perception.fake_hz. Each message:
  xyz         the live cloud: the sheet's surface on a 5 mm grid (the voxel size), at most 1024 points
              at random, 1 mm depth noise (no robot points: the real cloud has the hand and the arm)
  and, the same in every message, what config.yaml's mode (pc_label | pc_goal) needs:
    label: ref_xyz (192, 3) random points of the sheet, ref_moving (192,) True on the half that folds
           over (x > 0), ref_corners (4, 3) the sheet's: S far A, S far B, M far B, M far A
    goal:  goal_xyz (192, 3) the sheet folded: the moving half turned over onto the stationary one
           (x -> -x), 192 random of both layers, at the same height (as training's goal), goal_corners
           (4, 3) the folded sheet's (the stationary half: S far A, S far B, fold line B, fold line A)
  (as pc_ref_goal.py)
  depth       "fake"; t_cams [t]
"""

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))    # repo root, for deploy.*
from deploy.rigid2deform.pc_obs import BOX_HI_BASE, BOX_LO_BASE, N_POINTS, N_REF   # noqa: E402
from deploy.rigid2deform.mode import cloud_mode, resolve                     # noqa: E402
from deploy.rigid2deform.fold import quat_to_mat                             # noqa: E402
from deploy.utils.config import load_config                                  # noqa: E402
from deploy.utils.link import CLOUD_PORT, open_publisher, send_cloud         # noqa: E402

CONFIG = HERE.parent / "config.yaml"
VOXEL_M = 0.005
NOISE_M = 0.001


def crop_mask(p_base: np.ndarray) -> np.ndarray:
    """(N,) inside training's box (pc_obs.BOX_LO_BASE .. BOX_HI_BASE, no top) and finite. The real
    node's version: pc_node.py in_box."""
    return np.isfinite(p_base).all(-1) & (p_base >= BOX_LO_BASE).all(-1) & (p_base <= BOX_HI_BASE).all(-1)


def to_base(local: np.ndarray, pose_base) -> np.ndarray:
    pose = np.asarray(pose_base, dtype=float)
    return pose[:3] + local @ quat_to_mat(pose[3:]).T


def sheet_points(size: float, pose_base) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(N, 3) the flat sheet on a VOXEL_M grid, base frame; (N,) on the moving half (sheet x > 0);
    (N, 3) the same points folded (the moving half turned over onto the stationary one), base frame."""
    u = np.arange(-size / 2 + VOXEL_M / 2, size / 2, VOXEL_M)
    x, y = np.meshgrid(u, u, indexing="ij")
    local = np.stack([x.ravel(), y.ravel(), np.zeros(x.size)], axis=1)
    folded = local * [-1.0, 1.0, 1.0]
    folded[local[:, 0] < 0] = local[local[:, 0] < 0]
    return to_base(local, pose_base), local[:, 0] > 0, to_base(folded, pose_base)


def corners(size: float, pose_base) -> np.ndarray:
    """(4, 3) S far A, S far B, M far B, M far A (stationary half at sheet x < 0), base frame."""
    h = size / 2
    return to_base(np.array([[-h, -h, 0.0], [-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0]]), pose_base)


def main() -> None:
    cfg = load_config(path=CONFIG)
    size = float(cfg.fold.sheet_size or 0.15)
    mode = cloud_mode(resolve(cfg))
    if mode is None:
        raise SystemExit(f"[pc_fake_node] mode={cfg.mode} is not a point-cloud mode (pc_label | pc_goal)")
    sheet, moving, folded = sheet_points(size, cfg.perception.fake_pose_base)
    inside = crop_mask(sheet)
    if not inside.all():
        print(f"[pc_fake_node] {int((~inside).sum())} of {len(sheet)} sheet points are outside training's box", flush=True)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(sheet), N_REF, replace=False)
    sheet_corners = corners(size, cfg.perception.fake_pose_base)
    if mode == "label":
        fields = {"ref_xyz": sheet[pick], "ref_moving": moving[pick], "ref_corners": sheet_corners}
        what = f"reference {N_REF} ({int(moving[pick].sum())} moving)"
    else:
        s = sheet_corners                    # the folded sheet: the stationary half (as pc_ref_goal.py)
        fields = {"goal_xyz": folded[pick], "goal_corners": np.array([s[0], s[1], (s[1] + s[2]) / 2, (s[0] + s[3]) / 2])}
        what = f"goal {N_REF} (the folded sheet)"

    pub = open_publisher(cfg.bus[CLOUD_PORT])
    hz = float(cfg.perception.fake_hz)
    print(f"[pc_fake_node] a {100 * size:.0f} cm sheet at {list(cfg.perception.fake_pose_base)}: {len(sheet)} points "
          f"(sending at most {N_POINTS}), {what} for mode {mode}, on port "
          f"{cfg.bus[CLOUD_PORT]} at {hz:g} Hz. Ctrl-C stops.", flush=True)
    seq = 0
    try:
        while True:
            t = time.time()
            live = sheet[inside]
            if len(live) > N_POINTS:
                live = live[rng.choice(len(live), N_POINTS, replace=False)]
            live = live + rng.normal(0.0, NOISE_M, live.shape)
            send_cloud(pub, seq, t, live, depth="fake", t_cams=[t], **fields)
            seq += 1
            time.sleep(1.0 / hz)
    except KeyboardInterrupt:
        pass
    finally:
        pub.close(linger=0)


if __name__ == "__main__":
    main()
