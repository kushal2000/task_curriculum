"""Check the camera extrinsics with the arm: aim the flange axis at the eraser FoundationPose tracks.

FoundationPose's live_tracking.py places the eraser in the robot base frame using the extrinsics and
streams it here over UDP. The arm runs aioflexiv's operational space control (OSC) on the flange: this
script slides the flange in x and y only, keeping the height and orientation it starts with, until the
flange axis (its z axis, the tool direction) hits the eraser's top center. With the orientation fixed,
that is the flange center plus a constant x/y offset: height x tan(tilt) (--aim center puts the flange
center straight above the eraser instead). When the arm settles, where the axis hits and the eraser are
printed in the base frame. Look along the axis: if the extrinsics are right, it goes through the eraser's
center (the center of its bounding box, the box the tracker draws; the mesh origin can be elsewhere:
flat_eraser.obj's is 35 mm from the center).

    # FoundationPose env, terminal 1
    python live_tracking.py --mesh_path <eraser.obj> --publish_udp 127.0.0.1:5005
    # deploy env, terminal 2 (check in sim first)
    python deploy/calibration/follow_eraser.py --arm mujoco --viewer
    python deploy/calibration/follow_eraser.py --arm <ROBOT_SN>

Before you start, move the arm (free drive in calibrate_compare.py, or move_home.py) to a pose with the
flange pointing roughly down, 5-10 cm above the eraser: the height and orientation it keeps. Ctrl+C stops.
The arm holds when the tracking stops (lost, re-registering) or the target would leave --max_travel from
the start.

Not bumping the table: the floor is the eraser's top (tracked center + --eraser_half_height; the table is
below it). The flange target's z never changes. The lowest point of the arm over the table (links 2-7 and
the flange rim, from the MuJoCo model at the measured joints, further than --guard_radius from the base
axis: the base sits below the table top, so the shoulder may be lower than the floor without being over the
table) must be --min_clearance above the floor to start, and is checked every tick: below half of it, the
arm stops following and holds the last target until Ctrl+C.

aioflexiv's Rizon 4s MuJoCo model is used for OSC: its attachment_site is the URDF's flange exactly, the
frame the extrinsics are calibrated in. No tool is attached, so the model has no payload.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from deploy.joint_reach.core import DEFAULT_SPEC, Spec  # noqa: E402


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, help="robot serial or 'mujoco'")
    p.add_argument("--udp", default="127.0.0.1:5005", help="host:port live_tracking.py --publish_udp sends to")
    p.add_argument("--aim", choices=("axis", "center"), default="axis",
                   help="axis: the flange axis hits the eraser; center: the flange center is straight above it")
    p.add_argument("--max_tilt", type=float, default=45.0,
                   help="--aim axis: refuse to start if the flange is further than this from pointing down [deg]")
    p.add_argument("--offset_xy", type=float, nargs=2, default=[0.0, 0.0],
                   help="aim at the eraser + this, base frame [m] (e.g. beside it instead of at it)")
    p.add_argument("--speed", type=float, default=0.04, help="flange target speed [m/s]")
    p.add_argument("--max_travel", type=float, default=0.30, help="xy radius from the start it may go [m]")
    p.add_argument("--deadband", type=float, default=0.003, help="ignore eraser moves smaller than this [m]")
    p.add_argument("--smooth", type=float, default=0.3, help="eraser xy low-pass, weight of a new pose")
    p.add_argument("--stale_s", type=float, default=0.3, help="hold when no pose came for this long")
    p.add_argument("--min_clearance", type=float, default=0.05,
                   help="the arm's lowest point must be this far above the eraser top to start [m]; "
                        "below half of it, it stops")
    p.add_argument("--eraser_half_height", type=float, default=0.015,
                   help="eraser top = tracked center z + this [m] (flat_eraser.obj is 28 mm thick)")
    p.add_argument("--flange_radius", type=float, default=0.04, help="for the flange rim's lowest point [m]")
    p.add_argument("--guard_radius", type=float, default=0.20,
                   help="the table guard checks the arm only further than this from the base axis [m]")
    p.add_argument("--settle_s", type=float, default=1.0, help="report the error after the arm is still this long")
    p.add_argument("--rate", type=float, default=50.0, help="target update rate [Hz]")
    # OSC gains are accelerations (the controller scales them by the task-space mass): kd = 2 sqrt(kp) is
    # critically damped
    p.add_argument("--kp_pos", type=float, default=600.0, help="OSC position stiffness [1/s^2]")
    p.add_argument("--kd_pos", type=float, default=50.0, help="OSC position damping [1/s]")
    p.add_argument("--kp_rot", type=float, default=300.0, help="OSC orientation stiffness [1/s^2]")
    p.add_argument("--kd_rot", type=float, default=35.0, help="OSC orientation damping [1/s]")
    p.add_argument("--spec", default=str(DEFAULT_SPEC), help="its default_q is the MuJoCo start pose")
    p.add_argument("--sim_start_deg", type=float, nargs=7, default=None,
                   help="MuJoCo start joints [deg] instead of the spec's default_q")
    p.add_argument("--viewer", action="store_true", help="MuJoCo viewer for --arm mujoco")
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    return p.parse_args()


def fmt(v) -> str:
    return "[" + " ".join(f"{x:+.3f}" for x in v) + "]"


def tilt_deg(R: np.ndarray) -> float:
    """Angle of the flange z axis (the tool direction) from straight down."""
    return float(np.degrees(np.arccos(np.clip(-R[2, 2], -1.0, 1.0))))


def aim_point(ee: np.ndarray, aim: str, z: float) -> np.ndarray:
    """The xy the flange points at: where its axis crosses height z (aim axis), or the flange center."""
    if aim == "center":
        return ee[:2, 3].copy()
    a = ee[:3, 2]  # flange z axis, pointing down (--max_tilt keeps a[2] well below 0)
    return ee[:2, 3] + (z - ee[2, 3]) / a[2] * a[:2]


# ------------------------------------------------------------------------------------------- tracking


class EraserFeed:
    """The latest eraser center pose (4x4, base frame) from live_tracking.py --publish_udp."""

    def __init__(self, addr: str) -> None:
        host, port = addr.rsplit(":", 1)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, int(port)))
        self.sock.setblocking(False)
        self.T: np.ndarray | None = None
        self.t_rx = -np.inf
        self.n = 0  # poses received

    def poll(self) -> None:
        """Read everything that arrived, keep the newest."""
        while True:
            try:
                data = self.sock.recv(65536)
            except BlockingIOError:
                return
            try:
                self.T = np.asarray(json.loads(data)["T_base_center"], float).reshape(4, 4)
            except (ValueError, KeyError):
                continue
            self.t_rx, self.n = time.monotonic(), self.n + 1

    def age(self) -> float:
        return time.monotonic() - self.t_rx

    async def wait(self, addr: str) -> None:
        print(f"[follow] waiting for eraser poses on udp {addr} (live_tracking.py --publish_udp {addr})")
        while self.T is None:
            self.poll()
            await asyncio.sleep(0.02)
        print(f"[follow] eraser at {fmt(self.T[:3, 3])} m (base)")


# ------------------------------------------------------------------------------------------ follower


class XYFollower:
    """The flange target: the start pose slid in x and y at --speed until the flange aims at the eraser.
    Its z and orientation never change, so where it aims is the flange center + a constant offset.

    update_eraser filters the tracked eraser, step moves the target one tick toward it.
    """

    def __init__(self, args: argparse.Namespace, ee_start: np.ndarray, eraser_xy: np.ndarray,
                 aim_z: float) -> None:
        self.args = args
        self.T0 = ee_start.copy()
        self.xy0 = ee_start[:2, 3].copy()
        self.xy_cmd = self.xy0.copy()        # the target's xy now
        self.aim_offset = self.xy0 - aim_point(ee_start, args.aim, aim_z)  # flange center - where it aims
        self.eraser_xy = eraser_xy.copy()    # filtered eraser position

    def update_eraser(self, xy: np.ndarray) -> None:
        """A new tracked eraser xy: low-pass it, ignoring jitter below --deadband."""
        if np.linalg.norm(xy - self.eraser_xy) > self.args.deadband:
            self.eraser_xy += self.args.smooth * (xy - self.eraser_xy)

    def aim_goal(self) -> np.ndarray:
        """Where the flange should aim."""
        return self.eraser_xy + np.asarray(self.args.offset_xy)

    def goal(self) -> tuple[np.ndarray, bool]:
        """Where the flange center should end up, kept within --max_travel of the start; and if clipped."""
        goal = self.aim_goal() + self.aim_offset
        r = np.linalg.norm(goal - self.xy0)
        if r <= self.args.max_travel:
            return goal, False
        return self.xy0 + (goal - self.xy0) * self.args.max_travel / r, True

    def step(self, dt: float) -> str:
        """Move the target xy one tick toward the goal; returns a status line."""
        goal, clipped = self.goal()
        delta = goal - self.xy_cmd
        max_step = self.args.speed * dt
        if np.linalg.norm(delta) > max_step:
            delta *= max_step / np.linalg.norm(delta)
        self.xy_cmd = self.xy_cmd + delta
        return "clipped to --max_travel" if clipped else "tracking"

    def target(self) -> np.ndarray:
        """4x4 flange target: the start pose at the commanded xy."""
        T = self.T0.copy()
        T[:2, 3] = self.xy_cmd
        return T

    def settled(self, ee: np.ndarray) -> bool:
        """The target reached the goal and the flange reached the target."""
        return (np.linalg.norm(self.goal()[0] - self.xy_cmd) < 0.002
                and np.linalg.norm(ee[:3, 3] - self.target()[:3, 3]) < 0.002)


# ------------------------------------------------------------------------------------------------ arm


class TableGuard:
    """The lowest point of the arm over the table at measured joints: the corners of each link's geometry
    bounding box (links 2-7; the base and link 1 sit at the mount) and the flange rim, from aioflexiv's
    Rizon 4s MuJoCo model, further than `radius` from the base axis (closer is the robot's mount, not the
    table). Bounding boxes overestimate how low the arm reaches, which errs on the safe side."""

    SKIP_BODIES = ("world", "base", "link1")

    def __init__(self, floor_z: float, flange_radius: float, radius: float) -> None:
        import mujoco
        from aioflexiv.mujoco_model import default_mujoco_model_path

        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(default_mujoco_model_path("rizon4s")))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.geoms = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name not in self.SKIP_BODIES]
        signs = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float)
        # (n_geom, 8, 3) bounding box corners in each geom's frame
        self.corners = np.stack([m.geom_aabb[g, :3] + signs * m.geom_aabb[g, 3:] for g in self.geoms])
        self.site = m.site("attachment_site").id
        self.floor_z, self.r, self.radius = floor_z, flange_radius, radius

    def clearance(self, qpos: np.ndarray) -> tuple[float, str]:
        """Height of the arm's lowest point above the floor [m], and which part it is."""
        d = self.data
        d.qpos[:7] = qpos
        self.mj.mj_kinematics(self.model, d)
        pts = (np.einsum("gij,gkj->gki", d.geom_xmat[self.geoms].reshape(-1, 3, 3), self.corners)
               + d.geom_xpos[self.geoms][:, None])  # (n_geom, 8, 3) corners in the base frame
        z = np.where(np.hypot(pts[..., 0], pts[..., 1]) > self.radius, pts[..., 2], np.inf).min(1)
        i = int(np.argmin(z))
        lowest, part = z[i], self.model.body(self.model.geom_bodyid[self.geoms[i]]).name
        n = d.site_xmat[self.site].reshape(3, 3)[:, 2]  # flange disk normal
        rim = d.site_xpos[self.site, 2] - self.r * np.sqrt(max(0.0, 1.0 - n[2] ** 2))
        if rim < lowest and np.hypot(*d.site_xpos[self.site, :2]) > self.radius - self.r:
            lowest, part = rim, "flange rim"
        return float(lowest - self.floor_z), part


def make_controller(args: argparse.Namespace):
    from aioflexiv import FlexivController
    from aioflexiv.mujoco_model import default_mujoco_model_path

    kwargs = dict(mujoco_model_path=default_mujoco_model_path("rizon4s"))  # flange = the URDF's flange
    if args.arm == "mujoco":
        q0 = np.radians(args.sim_start_deg) if args.sim_start_deg else Spec.load(args.spec).default_q[:7]
        kwargs.update(mujoco_viewer=args.viewer, mujoco_initial_qpos=q0)
    return FlexivController(args.arm, **kwargs)


def start_osc(c, args: argparse.Namespace) -> None:
    """Switch to OSC, holding the flange where it is (the null space holds the joints where they are)."""
    with c.state_lock:
        c.ee_kp = np.array([args.kp_pos] * 3 + [args.kp_rot] * 3)
        c.ee_kd = np.array([args.kd_pos] * 3 + [args.kd_rot] * 3)
    c.switch("osc")


async def confirm(args: argparse.Namespace, ee: np.ndarray, q: np.ndarray, guard: TableGuard) -> bool:
    """Print the start; False if it is not safe to start or the user does not confirm."""
    height = ee[2, 3] - guard.floor_z
    clearance, part = guard.clearance(q)
    tilt = tilt_deg(ee[:3, :3])
    print(f"[follow] flange at {fmt(ee[:3, 3])} m, {height * 100:.1f} cm above the eraser top "
          f"(z {guard.floor_z:+.3f}), {tilt:.1f} deg from pointing down; height and orientation stay")
    print(f"[follow] lowest point of the arm: {part}, {clearance * 100:.1f} cm above the eraser top")
    if clearance < args.min_clearance:
        print(f"[follow] less than --min_clearance {args.min_clearance * 100:.0f} cm: raise the arm first")
        return False
    if args.aim == "axis":
        if tilt > args.max_tilt:
            print(f"[follow] more than --max_tilt {args.max_tilt:.0f} deg: the axis would hit the table far "
                  "away. Point the flange more down, or use --aim center")
            return False
        offset = ee[:2, 3] - aim_point(ee, "axis", guard.floor_z)
        print(f"[follow] aiming the axis: the flange center stays {np.linalg.norm(offset) * 100:.1f} cm "
              f"from the eraser center ({fmt(offset)} m)")
    if args.yes or args.arm == "mujoco":
        return True
    # input() in a thread: the torque loop runs on this event loop and must keep running
    answer = await asyncio.to_thread(input, "[follow] the arm will move in x/y only. Type 'go' to start: ")
    return answer.strip() == "go"


# ------------------------------------------------------------------------------------------------ main


async def follow(args: argparse.Namespace, c, f: XYFollower, feed: EraserFeed, guard: TableGuard) -> None:
    dt = 1.0 / args.rate
    last_n, t_print = feed.n, 0.0
    settled_since, reported = None, False  # report once per stop, after the arm stayed still --settle_s
    tripped = None  # why the table guard stopped the following
    while c.running:
        t_tick = time.monotonic()

        # 1. table guard: on the measured joints; when tripped, the last target stays (z = the start's)
        clearance, part = guard.clearance(c.state["qpos"])
        if tripped is None and clearance < args.min_clearance / 2:
            tripped = f"STOPPED: {part} {clearance * 100:.1f} cm above the eraser top, holding (Ctrl+C)"
            print(f"\n[follow] {tripped}")

        # 2. tracking -> flange target
        feed.poll()
        if feed.n != last_n:
            last_n = feed.n
            f.update_eraser(feed.T[:2, 3])
        fresh = feed.age() < args.stale_s
        if tripped:
            status = tripped
        elif fresh:
            status = f.step(dt)
        else:
            status = f"no pose for {feed.age():.1f} s, holding"
        c.set_now("ee_desired", f.target())

        # 3. report: where the flange really aims (measured pose) vs the eraser
        ee = c.state["ee"]
        aimed = aim_point(ee, args.aim, guard.floor_z)
        err = aimed - f.aim_goal()
        dz = ee[2, 3] - f.T0[2, 3]
        if fresh and not tripped and f.settled(ee):
            settled_since = settled_since or time.monotonic()
        else:
            settled_since, reported = None, False
        if settled_since is not None and not reported and time.monotonic() - settled_since > args.settle_s:
            reported = True
            print(f"\n[follow] SETTLED  eraser {fmt(feed.T[:3, 3])}  flange {fmt(ee[:3, 3])}  aims at "
                  f"{fmt(aimed)}  xy error {fmt(err * 1000)} mm, |{np.linalg.norm(err) * 1000:.1f}| mm\n"
                  f"[follow] check by eye that the flange {'axis' if args.aim == 'axis' else 'center'} "
                  "is on the eraser center")
        if time.monotonic() - t_print > 0.5:
            t_print = time.monotonic()
            print(f"\r[follow] {status:<36} eraser xy {fmt(f.eraser_xy)}  aims at {fmt(aimed)}  "
                  f"xy err {np.linalg.norm(err) * 1000:5.1f} mm  dz {dz * 1000:+5.1f} mm  "
                  f"clear {clearance * 100:4.1f} cm", end="", flush=True)

        await asyncio.sleep(max(0.0, dt - (time.monotonic() - t_tick)))
    raise RuntimeError("arm torque loop stopped")


async def run(args: argparse.Namespace) -> int:
    feed = EraserFeed(args.udp)
    await feed.wait(args.udp)

    c = make_controller(args)
    await c.start()  # joint impedance, holding where it is
    try:
        start_osc(c, args)
        await asyncio.sleep(0.2)
        ee, q = c.state["ee"].copy(), c.state["qpos"].copy()
        floor_z = feed.T[2, 3] + args.eraser_half_height  # the eraser's top
        guard = TableGuard(floor_z, args.flange_radius, args.guard_radius)
        if not await confirm(args, ee, q, guard):
            print("[follow] not moved")
            return 1
        await follow(args, c, XYFollower(args, ee, feed.T[:2, 3], floor_z), feed, guard)
    finally:
        print("\n[follow] stopping")
        await c.stop()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(run(_parse())))
    except KeyboardInterrupt:
        pass
