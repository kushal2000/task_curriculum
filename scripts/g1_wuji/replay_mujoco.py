"""Replay a G1 + Wuji Hand 2 joint trajectory in MuJoCo.

Kinematic by default: every frame writes the positions straight into ``qpos``, so what you see
is exactly the trajectory, feasible or not. ``--physics`` instead feeds each frame to the
model's position servos (upstream gains) and simulates, which shows whether the robot can
actually track it, and prints the tracking error.

    # interactive viewer (needs a display)
    .venv_isaaclab3/bin/python scripts/g1_wuji/replay_mujoco.py traj.npz
    # headless mp4 (MUJOCO_GL=egl on a GPU node, osmesa elsewhere)
    MUJOCO_GL=egl .venv_isaaclab3/bin/python scripts/g1_wuji/replay_mujoco.py traj.npz \
        --video out.mp4 --physics

A trajectory is a ``.npz`` as documented in ``g1_wuji.py``, or a wuji-hand-teleop clip dir.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import g1_wuji
import mujoco
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "trajectory", help=".npz trajectory or wuji-hand-teleop clip directory"
    )
    p.add_argument(
        "--variant",
        type=int,
        choices=g1_wuji.VARIANTS,
        default=29,
        help="G1 DoF variant",
    )
    p.add_argument(
        "--physics",
        action="store_true",
        help="track with position servos instead of setting qpos",
    )
    p.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    p.add_argument(
        "--loop", action="store_true", help="repeat until the viewer is closed"
    )
    p.add_argument("--video", help="render to this .mp4 instead of opening the viewer")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    return p.parse_args()


@dataclass
class ContactStat:
    frames: int = 0  # frames with at least one contact of this class
    depth: float = 0.0  # deepest penetration, m
    pair: tuple[str, str] = ("", "")  # bodies at the deepest penetration
    force: float = 0.0  # peak normal force, N (--physics only)


def _body_part(model: mujoco.MjModel, geom: int) -> str | None:
    """Which part of the robot a geom belongs to; None for the floor."""
    name = model.body(model.geom_bodyid[geom]).name
    if name.startswith("left_wuji"):
        return "left hand"
    if name.startswith("right_wuji"):
        return "right hand"
    return None if name == "world" else "body"


def _contact_class(a: str | None, b: str | None) -> str | None:
    if a is None or b is None:
        return None  # feet on the floor: expected, not reported
    if a == b:
        return "within-hand" if a != "body" else "body-body"
    if "body" in (a, b):
        return "hand-body"
    return "hand-hand"


class Replayer:
    def __init__(
        self, traj: g1_wuji.Trajectory, variant: int, physics: bool, speed: float
    ):
        if physics and traj.floating:
            raise SystemExit(
                "--physics needs a fixed-base trajectory: the G1 balance controller is not modelled"
            )
        self.traj, self.physics = traj, physics
        kind = "scene_floating" if traj.floating else "scene_fixed"
        self.model = mujoco.MjModel.from_xml_path(
            str(g1_wuji.model_path(variant, kind))
        )
        self.data = mujoco.MjData(self.model)

        m = self.model
        hinges = [
            i for i in range(m.njnt) if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
        ]
        names = [m.joint(i).name for i in hinges]
        self.q = traj.full(names, g1_wuji.stand_pose(variant))  # (T, n_hinge)
        self.qpos_adr = m.jnt_qposadr[hinges]
        # Actuators are not in joint order (hands are THJ0.. by actuator name), so map through
        # each actuator's transmission target.
        col = {j: c for c, j in enumerate(hinges)}
        self.ctrl_cols = np.array([col[m.actuator_trnid[a, 0]] for a in range(m.nu)])

        self.frame_dt = 1.0 / (traj.fps * speed)
        self.substeps = max(1, round(self.frame_dt / m.opt.timestep))
        self.track_err: list[np.ndarray] = []
        self.geom_part = [_body_part(m, g) for g in range(m.ngeom)]
        self.contacts: dict[str, ContactStat] = {}

    def reset(self) -> None:
        self.track_err.clear()
        self.contacts.clear()
        mujoco.mj_resetData(self.model, self.data)
        self._set_qpos(0)
        mujoco.mj_forward(self.model, self.data)

    def _set_qpos(self, t: int) -> None:
        self.data.qpos[self.qpos_adr] = self.q[t]
        if self.traj.floating:
            self.data.qpos[0:3] = self.traj.root_pos[t]
            self.data.qpos[3:7] = self.traj.root_quat[t]

    def frame(self, t: int) -> None:
        if self.physics:
            self.data.ctrl[:] = self.q[t, self.ctrl_cols]
            mujoco.mj_step(self.model, self.data, nstep=self.substeps)
            self.track_err.append(self.data.qpos[self.qpos_adr] - self.q[t])
        else:
            self._set_qpos(t)
            mujoco.mj_forward(self.model, self.data)
        self._record_contacts()

    def _record_contacts(self) -> None:
        """Kinematic replay does not resolve contacts, so this is penetration: how deep the
        trajectory drives geometry into itself. Under --physics it is what the solver allowed,
        plus the force it took."""
        m, d = self.model, self.data
        seen = set()
        force = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            cls = _contact_class(self.geom_part[c.geom1], self.geom_part[c.geom2])
            if cls is None:
                continue
            stat = self.contacts.setdefault(cls, ContactStat())
            seen.add(cls)
            if -c.dist > stat.depth:
                stat.depth = -c.dist
                stat.pair = (
                    m.body(m.geom_bodyid[c.geom1]).name,
                    m.body(m.geom_bodyid[c.geom2]).name,
                )
            if self.physics:
                mujoco.mj_contactForce(m, d, i, force)
                stat.force = max(stat.force, force[0])
        for cls in seen:
            self.contacts[cls].frames += 1

    def report(self) -> None:
        n = len(self.traj)
        for cls in ("hand-hand", "hand-body", "body-body", "within-hand"):
            s = self.contacts.get(cls)
            if s is None:
                print(f"contact {cls}: none")
                continue
            force = f", peak force {s.force:.1f} N" if self.physics else ""
            print(
                f"contact {cls}: {s.frames}/{n} frames, deepest {s.depth * 1000:.1f} mm "
                f"({s.pair[0]} / {s.pair[1]}){force}"
            )
        if not self.track_err:
            return
        err = np.abs(np.array(self.track_err))
        names = [self.model.joint(int(j)).name for j in self.model.actuator_trnid[:, 0]]
        per_joint = err.max(axis=0)[self.ctrl_cols]
        hand = np.array(["_wuji_" in n for n in names])
        rmse = lambda e: float(np.sqrt((e**2).mean()))
        print(
            f"tracking RMSE: body {rmse(err[:, self.ctrl_cols][:, ~hand]):.4f} rad, "
            f"hands {rmse(err[:, self.ctrl_cols][:, hand]):.4f} rad"
        )
        worst = np.argsort(per_joint)[::-1][:5]
        print(
            "worst joints (max |error|, rad): "
            + ", ".join(f"{names[i]} {per_joint[i]:.3f}" for i in worst)
        )


def make_camera(model: mujoco.MjModel) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = model.body("pelvis").id
    cam.distance, cam.azimuth, cam.elevation = 2.2, 150.0, -15.0
    return cam


def render_video(r: Replayer, path: str, width: int, height: int) -> None:
    import imageio.v2 as imageio

    r.model.vis.global_.offwidth = max(r.model.vis.global_.offwidth, width)
    r.model.vis.global_.offheight = max(r.model.vis.global_.offheight, height)
    renderer = mujoco.Renderer(r.model, height=height, width=width)
    cam = make_camera(r.model)
    r.reset()
    with imageio.get_writer(path, fps=1.0 / r.frame_dt, macro_block_size=1) as writer:
        for t in range(len(r.traj)):
            r.frame(t)
            renderer.update_scene(r.data, camera=cam)
            writer.append_data(renderer.render())
    print(f"wrote {len(r.traj)} frames to {path}")


def run_viewer(r: Replayer, loop: bool) -> None:
    import mujoco.viewer

    with mujoco.viewer.launch_passive(r.model, r.data) as viewer:
        cam = make_camera(r.model)
        viewer.cam.type, viewer.cam.trackbodyid = cam.type, cam.trackbodyid
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = (
            cam.distance,
            cam.azimuth,
            cam.elevation,
        )
        while viewer.is_running():
            r.reset()
            for t in range(len(r.traj)):
                start = time.perf_counter()
                with viewer.lock():
                    r.frame(t)
                viewer.sync()
                if not viewer.is_running():
                    break
                time.sleep(max(0.0, r.frame_dt - (time.perf_counter() - start)))
            if not loop:
                break


def main() -> None:
    args = parse_args()
    traj = g1_wuji.load_trajectory(args.trajectory)
    print(
        f"{len(traj)} frames at {traj.fps:g} fps, {len(traj.joint_names)} joints driven, "
        f"{'floating' if traj.floating else 'fixed'} base"
    )
    r = Replayer(traj, args.variant, args.physics, args.speed)
    if args.video:
        render_video(r, args.video, args.width, args.height)
    else:
        run_viewer(r, args.loop)
    r.report()


if __name__ == "__main__":
    main()
