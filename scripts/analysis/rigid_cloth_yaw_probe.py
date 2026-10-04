"""Which root-pose write actually turns the chain? Measured from slat POSITIONS, not quaternions.

Written when the chain reset axis-aligned whatever yaw the task drew, and `RigidClothEnv` only
warned about it. Every earlier diagnosis read orientations back through the same API whose quaternion
convention was in question, so this one judges by where the slats' centres end up instead: the vector
from the first slat to the last one IS the chain's heading, with no convention involved. It found the
adapter writing a wxyz quaternion into an xyzw API; the fix is `cloth_adapter.task_yaw` plus the
xyzw write, and `RigidClothEnv._calibrate_and_verify_placement` now RAISES rather than warns if a
yawed reset comes back wrong. This still runs, and is the thing to run if it ever does.

Each mode writes a requested yaw, steps once, and reports the heading the slats actually took:

    adapter        ``ChainAsRigidObject.write_root_pose_to_sim`` -- what every reset calls today
    xyzw           ``write_root_pose_to_sim_index`` with an (x, y, z, w) quaternion, as documented
    xyzw_nojoint   the same, without the flat-joint write the adapter does afterwards
    wxyz           ``write_root_pose_to_sim_index`` with (w, x, y, z), as the adapter sends

    scripts/newton_py -m scripts.analysis.rigid_cloth_yaw_probe --variant box3-mid
"""

from __future__ import annotations

import argparse
import math
import sys


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="Isaacsimenvs-RigidCloth-Direct-v0")
    parser.add_argument("--variant", default="box3-mid")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--sim_device", default="cuda:0")
    AppLauncher.add_app_launcher_args(parser)
    args, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args + [f"env.rigid_cloth.variant={args.variant}"]
    try:
        AppLauncher(args)
    except ImportError as exc:
        print(f"[yaw] kit-less: {exc}", flush=True)

    import gymnasium as gym
    import torch

    import isaacsimenvs  # noqa: F401
    from isaacsimenvs.eval.protocol import disable_randomization, use_single_object_variant
    from isaacsimenvs.tasks.cloth.utils import rigid_cloth as rc
    from isaacsimenvs.tasks.cloth.utils.rigid_cloth_adapter import _to_torch
    from isaacsimenvs.utils.hydra_utils import hydra_task_config_with_yaml

    @hydra_task_config_with_yaml(args.task, "")
    def run(env_cfg, agent_cfg) -> None:
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.sim.device = args.sim_device
        disable_randomization(env_cfg)
        use_single_object_variant()
        env = gym.make(args.task, cfg=env_cfg)
        inner = env.unwrapped
        env.reset()

        obj = inner.object
        art = inner._chain
        spec = inner.cfg.rigid_cloth.spec
        names = list(art.body_names)
        first = names.index(rc.slat_link_name(0))
        last = names.index(rc.slat_link_name(spec.num_slats - 1))
        n = inner.num_envs
        origins = inner.scene.env_origins
        dt = float(inner.cfg.sim.dt)
        # Chain-frame x of the first and last slat: the heading of (last - first) at zero yaw.
        frames = rc.chain_frames(spec, None)
        base = math.degrees(math.atan2(0.0, frames[-1][0] - frames[0][0]))

        def heading() -> tuple[float, float, float]:
            st = _to_torch(art.data.body_state_w)
            d = st[0, last, :2] - st[0, first, :2]
            h = math.degrees(math.atan2(float(d[1]), float(d[0]))) - base
            c = (st[0, :, :3].mean(dim=0) - origins[0]).tolist()
            return h, c[0] * 1e3, c[1] * 1e3

        def settle() -> None:
            inner.sim.step(render=False)
            inner.scene.update(dt)

        def write_direct(yaw: float, layout: str, joints: bool) -> None:
            ids = torch.arange(n, device=inner.device)
            pose = torch.zeros((n, 7), device=inner.device)
            off = obj.root_offset
            c, s = math.cos(yaw), math.sin(yaw)
            pose[:, 0] = origins[:, 0] + c * float(off[0]) - s * float(off[1])
            pose[:, 1] = origins[:, 1] + s * float(off[0]) + c * float(off[1])
            pose[:, 2] = obj._spawn_z
            h = 0.5 * yaw
            if layout == "xyzw":
                pose[:, 5], pose[:, 6] = math.sin(h), math.cos(h)
            else:
                pose[:, 3], pose[:, 6] = math.cos(h), math.sin(h)
            art.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
            art.write_root_velocity_to_sim(torch.zeros((n, 6), device=inner.device), ids)
            if joints:
                obj._write_flat_joints(ids)

        for yaw_deg in (0.0, 90.0, 45.0, -120.0):
            yaw = math.radians(yaw_deg)
            for mode in ("adapter", "xyzw", "xyzw_nojoint", "wxyz"):
                if mode == "adapter":
                    pose = torch.zeros((n, 7), device=inner.device)
                    pose[:, :3] = origins
                    pose[:, 3] = math.cos(0.5 * yaw)
                    pose[:, 6] = math.sin(0.5 * yaw)
                    obj.write_root_pose_to_sim(pose)
                    obj.write_root_velocity_to_sim(torch.zeros((n, 6), device=inner.device))
                elif mode == "xyzw":
                    write_direct(yaw, "xyzw", True)
                elif mode == "xyzw_nojoint":
                    write_direct(yaw, "xyzw", False)
                else:
                    write_direct(yaw, "wxyz", True)
                before = [round(float(v), 3) for v in _to_torch(art.data.root_link_pose_w)[0, 3:7]]
                settle()
                after = [round(float(v), 3) for v in _to_torch(art.data.root_link_pose_w)[0, 3:7]]
                h, cx, cy = heading()
                print(
                    f"[yaw] req {yaw_deg:+7.1f}  {mode:<13} got {h:+8.2f} deg  "
                    f"centroid ({cx:+7.2f}, {cy:+7.2f}) mm  root quat written {before} after {after}",
                    flush=True,
                )
                if mode in ("adapter", "xyzw") and yaw_deg in (0.0, 90.0):
                    st = _to_torch(art.data.body_state_w)
                    lp = _to_torch(art.data.body_link_pose_w)
                
                
                    state = inner.sim.physics_manager.get_state_0() if hasattr(
                        inner.sim.physics_manager, "get_state_0") else None
                    bq = state.body_q.numpy() if state is not None else None
                    model = inner.sim.physics_manager.get_model()
                    labels = [s.rsplit("/", 1)[-1] for s in model.body_label]
                    for i in range(spec.num_slats):
                        b = names.index(rc.slat_link_name(i))
                        nb = [k for k, s in enumerate(labels) if s == rc.slat_link_name(i)][0]
                        print(
                            f"[yaw]    slat_{i} pos_env_mm "
                            f"{[round(float(v) * 1e3, 1) for v in (st[0, b, :3] - origins[0])]} "
                            f"body_state_w q {[round(float(v), 3) for v in st[0, b, 3:7]]} "
                            f"body_link_pose_w q {[round(float(v), 3) for v in lp[0, b, 3:7]]} "
                            f"newton body_q q {[round(float(v), 3) for v in bq[nb][3:7]] if bq is not None else None}",
                            flush=True,
                        )
                    if yaw_deg == 0.0 and mode == "adapter":
                        jxc = model.joint_X_c.numpy()
                        jxp = model.joint_X_p.numpy()
                        jl = [s.rsplit("/", 1)[-1] for s in model.joint_label]
                        jt = model.joint_type.numpy()
                        for j, lab in enumerate(jl):
                            if "slat" in lab or (j < len(jl) and jt[j] == 4 and "RigidCloth" in model.joint_label[j]):
                                print(f"[yaw]    joint {lab} type {int(jt[j])} X_p {[round(float(v), 4) for v in jxp[j]]} X_c {[round(float(v), 4) for v in jxc[j]]}", flush=True)
                            if j > 400:
                                break
        env.close()

    run()


if __name__ == "__main__":
    main()
    import os

    os._exit(0)
