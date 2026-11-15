"""Measure where the real eraser lies, for checks/record_lift.py's --x --y --yaw --table_top.

Run perception.py (camera) first, with the eraser lying flat where the lift will start, then:

    .venv_deploy/bin/python deploy/simtoolreal/checks/measure_eraser.py [--seconds 3] [--out <dir>]

It averages FoundationPose's object_pose (the mesh's own frame, robot base frame) over --seconds and
converts it to what record_lift.py places in the env frame:
  - x, y:      the box centre (the mesh's bounding-box centre, 35 mm from the mesh origin);
  - table_top: the box centre's z minus half the box's vertical side, i.e. the top of whatever the
               eraser lies on (table, pad);
  - yaw:       the mesh's long (x) side about env z, in (-180, 180], as FoundationPose sees the mesh
               frame. The box looks the same turned 180 deg, but the mesh frame (and so the policy's
               keypoint order) does not: keep the raw value (Nov 8's rollouts: about 179 deg).
It also says how far the eraser is from lying flat (record_lift.py assumes the 2.8 cm side vertical)
and how much the pose jittered. With --out, T_base_obj.npy (for view_lifts.py --measured) and
measured.json are written there.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))   # repo root, for deploy.*
from deploy.utils.config import REPO, load_config                    # noqa: E402
from deploy.utils.link import open_subscriber, recv_wait              # noqa: E402
from deploy.utils.quat import quat_apply                              # noqa: E402
from deploy.utils.spec import base_to_env, load_spec                  # noqa: E402

MESH = HERE.parent / "objects" / "flat_eraser" / "flat_eraser.obj"


def mesh_box(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(extents, bounding-box centre in the mesh frame) of an .obj, metres (as record_lift.py)."""
    v = np.array([[float(x) for x in l.split()[1:4]] for l in open(path) if l.startswith("v ")])
    lo, hi = v.min(axis=0), v.max(axis=0)
    return hi - lo, (hi + lo) / 2


def mean_quat(qs: np.ndarray) -> np.ndarray:
    """Average of unit quaternions (wxyz): the main eigenvector of sum q q^T."""
    w, v = np.linalg.eigh(qs.T @ qs)
    q = v[:, -1]
    return q if q[0] >= 0 else -q


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seconds", type=float, default=3.0)
    p.add_argument("--out", type=Path)
    args, overrides = p.parse_known_args()
    cfg = load_config(overrides, path=HERE.parent / "config.yaml")
    spec = load_spec(REPO / cfg.spec_path)
    size, centre_in_mesh = mesh_box(MESH)

    sub = open_subscriber(cfg.bus.host, cfg.bus.object_pose_port, "object_pose", latest_only=False)
    print(f"[measure] waiting for object_pose on {cfg.bus.host}:{cfg.bus.object_pose_port} ...", flush=True)
    pos, quat, t_end = [], [], None
    while t_end is None or time.time() < t_end:
        m = recv_wait(sub, timeout_s=60.0 if t_end is None else 5.0)   # the first: FoundationPose loads
        if m is None:
            raise SystemExit("[measure] no object_pose: is perception.py running and tracking?")
        if t_end is None:
            t_end = time.time() + args.seconds
        pos.append(m["pos"])
        quat.append(m["quat_wxyz"])
    pos, quat = np.array(pos), np.array(quat)
    quat *= np.sign(quat @ quat[0])[:, None]          # one hemisphere before averaging
    p_base, q_base = pos.mean(axis=0), mean_quat(quat)

    p_env, q_env = base_to_env(spec, p_base, q_base)
    centre = p_env + quat_apply(q_env, centre_in_mesh)
    ax = [quat_apply(q_env, e) for e in np.eye(3)]   # the mesh axes in the env frame
    tilt = np.degrees(np.arccos(min(1.0, abs(ax[1][2]))))   # flat: the mesh y (2.8 cm) side vertical
    yaw = np.degrees(np.arctan2(ax[0][1], ax[0][0]))
    table_top = centre[2] - size[1] / 2

    # jitter: the box centre of every sample, base frame
    cs = np.array([pp + quat_apply(qq, centre_in_mesh) for pp, qq in zip(pos, quat)])
    jit_mm = 1e3 * cs.std(axis=0)

    print(f"[measure] {len(pos)} poses over {args.seconds:.1f} s; box centre jitter (std) "
          f"{np.round(jit_mm, 1).tolist()} mm")
    print(f"[measure] base frame: mesh origin {np.round(p_base, 4).tolist()}, box centre "
          f"{np.round(cs.mean(axis=0), 4).tolist()}")
    print(f"[measure] env frame: box centre {np.round(centre, 4).tolist()}, yaw {yaw:+.1f} deg, "
          f"tilt from flat {tilt:.1f} deg")
    print(f"[measure] -> table_top {table_top:.4f} (spec's bare table {spec['table_top_z_env']:.4f}; "
          f"+{100 * (table_top - spec['table_top_z_env']):.1f} cm)")
    if tilt > 5.0:
        print(f"[measure] WARNING: {tilt:.1f} deg from flat: record_lift.py starts the box flat "
              "(the 2.8 cm side vertical); check the eraser or the tracking")
    print(f"\n  --x {centre[0]:.4f} --y {centre[1]:.4f} --yaw {yaw:.1f} --table_top {table_top:.4f}\n")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        T = np.eye(4)
        from scipy.spatial.transform import Rotation
        w, x, y, z = q_base
        T[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
        T[:3, 3] = p_base
        np.save(args.out / "T_base_obj.npy", T)
        (args.out / "measured.json").write_text(json.dumps({
            "t": time.time(), "n": len(pos), "seconds": args.seconds,
            "extrinsics": cfg.perception.extrinsics,
            "mesh_pos_base": p_base.tolist(), "mesh_quat_wxyz_base": q_base.tolist(),
            "box_centre_env": centre.tolist(), "yaw_deg": yaw, "tilt_deg": tilt,
            "table_top_env": table_top, "jitter_mm": jit_mm.tolist(),
            "record_lift_args": f"--x {centre[0]:.4f} --y {centre[1]:.4f} --yaw {yaw:.1f} --table_top {table_top:.4f}",
        }, indent=2))
        print(f"[measure] saved {args.out / 'T_base_obj.npy'} and measured.json")


if __name__ == "__main__":
    main()
