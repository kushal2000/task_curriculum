"""Replay a G1 + Wuji Hand 2 joint trajectory in Isaac Sim (Isaac Lab).

Kinematic: every frame writes the joint positions (and root pose, for a floating-base
trajectory) into PhysX and renders, without stepping physics, so the robot shows exactly the
trajectory. Physics-level tracking is ``replay_mujoco.py --physics``, which carries the
upstream servo gains; this side has none to be faithful to.

The URDF is converted to USD on first use and cached under ``assets/g1_wuji2_description/usd/``;
Isaac Lab reconverts by itself when the URDF or the conversion options change.

    # GUI (needs a display)
    .venv_isaacsim/bin/python scripts/g1_wuji/replay_isaacsim.py traj.npz
    # headless mp4 (GPU node)
    .venv_isaacsim/bin/python scripts/g1_wuji/replay_isaacsim.py traj.npz --headless --video out.mp4

A trajectory is a ``.npz`` as documented in ``g1_wuji.py``, or a wuji-hand-teleop clip dir.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

import g1_wuji
import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument(
    "trajectory", help=".npz trajectory or wuji-hand-teleop clip directory"
)
parser.add_argument(
    "--variant", type=int, choices=g1_wuji.VARIANTS, default=29, help="G1 DoF variant"
)
parser.add_argument(
    "--speed", type=float, default=1.0, help="playback speed multiplier"
)
parser.add_argument(
    "--loop", action="store_true", help="repeat until the window is closed (GUI only)"
)
parser.add_argument("--video", help="render to this .mp4 through a camera sensor")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video:
    args.enable_cameras = True
simulation_app = AppLauncher(args).app

# Everything below needs the running app.
import isaaclab.sim as sim_utils
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

# Camera pose relative to the pelvis: in front and to the side, looking at hand height.
CAMERA_EYE = np.array([1.6, 1.0, 0.45])
CAMERA_TARGET = np.array([0.0, 0.0, 0.25])


def convert_urdf(variant: int, floating: bool) -> str:
    urdf = g1_wuji.model_path(variant, "urdf")
    cfg = UrdfConverterCfg(
        asset_path=str(urdf),
        usd_dir=str(
            g1_wuji.ASSET_DIR
            / "usd"
            / f"{urdf.stem}_{'floating' if floating else 'fixed'}"
        ),
        fix_base=not floating,
        # Keep every link, fingertip sensor frames included, so link names match the URDF.
        merge_fixed_joints=False,
        make_instanceable=False,
        self_collision=False,
        # Replay never steps physics; the drives only need to exist and hold still.
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=100.0, damping=10.0
            ),
        ),
    )
    return UrdfConverter(cfg).usd_path


def build_scene(
    usd_path: str, variant: int, fps: float
) -> tuple[sim_utils.SimulationContext, Articulation]:
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / fps, device=args.device)
    )
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9))
    light.func("/World/light", light)
    robot = Articulation(
        ArticulationCfg(
            prim_path="/World/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=usd_path,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, g1_wuji.FIXED_PELVIS_HEIGHT),
                joint_pos=g1_wuji.stand_pose(variant),
            ),
            actuators={
                "all": ImplicitActuatorCfg(
                    joint_names_expr=[".*"], stiffness=None, damping=None
                )
            },
        )
    )
    return sim, robot


def main() -> None:
    traj = g1_wuji.load_trajectory(args.trajectory)
    print(
        f"{len(traj)} frames at {traj.fps:g} fps, {len(traj.joint_names)} joints driven, "
        f"{'floating' if traj.floating else 'fixed'} base"
    )

    sim, robot = build_scene(
        convert_urdf(args.variant, traj.floating), args.variant, traj.fps
    )
    camera = None
    if args.video:
        camera = Camera(
            CameraCfg(
                prim_path="/World/Camera",
                update_period=0.0,
                height=args.height,
                width=args.width,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0, clipping_range=(0.05, 50.0)
                ),
            )
        )
    sim.reset()

    # PhysX orders joints its own way; place the trajectory by name.
    dev = robot.device
    q = torch.tensor(
        traj.full(robot.joint_names, g1_wuji.stand_pose(args.variant)),
        dtype=torch.float32,
        device=dev,
    )
    zeros = torch.zeros_like(q[0:1])
    pelvis = (
        traj.root_pos
        if traj.floating
        else np.tile([0.0, 0.0, g1_wuji.FIXED_PELVIS_HEIGHT], (len(traj), 1))
    )
    if traj.floating:
        root = torch.tensor(
            np.concatenate([traj.root_pos, traj.root_quat], axis=1),
            dtype=torch.float32,
            device=dev,
        )

    def show(t: int) -> None:
        if traj.floating:
            robot.write_root_pose_to_sim(root[t : t + 1])
            robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev))
        robot.write_joint_state_to_sim(q[t : t + 1], zeros)
        robot.set_joint_position_target(q[t : t + 1])
        robot.write_data_to_sim()
        sim.forward()
        if camera is not None:
            eye, target = (
                torch.tensor(pelvis[t] + off, dtype=torch.float32, device=dev)[None]
                for off in (CAMERA_EYE, CAMERA_TARGET)
            )
            camera.set_world_poses_from_view(eye, target)
        sim.render()

    frame_dt = 1.0 / (traj.fps * args.speed)
    if camera is not None:
        import imageio.v2 as imageio

        # The RTX renderer needs a few frames before its output settles.
        for _ in range(10):
            show(0)
        with imageio.get_writer(
            args.video, fps=1.0 / frame_dt, macro_block_size=1
        ) as writer:
            for t in range(len(traj)):
                show(t)
                camera.update(frame_dt, force_recompute=True)
                writer.append_data(camera.data.output["rgb"][0, ..., :3].cpu().numpy())
        print(f"wrote {len(traj)} frames to {args.video}")
        return

    sim.set_camera_view(
        eye=(pelvis[0] + CAMERA_EYE).tolist(),
        target=(pelvis[0] + CAMERA_TARGET).tolist(),
    )
    while simulation_app.is_running():
        for t in range(len(traj)):
            start = time.perf_counter()
            show(t)
            if not simulation_app.is_running():
                break
            time.sleep(max(0.0, frame_dt - (time.perf_counter() - start)))
        if not args.loop:
            break


if __name__ == "__main__":
    # Kit shutdown can hang (it does headless on bos14, even after a clean run), so force-exit
    # instead of waiting for a teardown, as isaacsimenvs/train.py does.
    exit_code = 0
    try:
        main()
    except BaseException:  # noqa: BLE001 -- anything, Ctrl-C included, must still reach os._exit
        traceback.print_exc()
        exit_code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
