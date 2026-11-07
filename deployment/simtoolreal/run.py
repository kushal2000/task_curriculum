"""Run the Play policy (pretrained_policy/nov6_dr.pth) on the Flexiv Rizon 4s + right Wuji Hand 2, at 60 Hz.

The deploy loop. Every policy step: read the joints and the object pose; test the goal
(core.SuccessTracker) and, when it is reached, move to the next one (core.GoalSampler: training's
random walk of goals); build the 134-dim observation (core.build_obs), run the policy, turn the action
into joint targets (core.ActionPipeline), check them, send them. A goal gets at most 600 steps (10 s),
as in training.

Robot (--robot):
  isaac         sim_server.py, the training env as a plant (lock-step: one request = one policy step).
                Object pose from the sim.
  real          aioflexiv (--arm: robot serial, "mujoco" = aioflexiv's MuJoCo Rizon without a hand, or
                "fake") + aiowuji (--hand: serial, "auto", "dry_run" = connected but never enabled, or
                "fake"). Object pose from --object.
Object (--object, real only): "fake" = a fixed pose (--object_pose_base, default: lying flat at the
table centre); FoundationPose comes later.
Mode (--mode): "policy", or "replay" = send the joint targets a sim rollout recorded
(record_sim_rollout.py) open loop, to tell a driver / plant mismatch from a policy problem.

Safety (abort, or only count with --on_safety count): an arm target more than --arm_lag_hard_deg
(25) from the measured angle; more than --arm_lag_deg (10) for --arm_lag_s (0.5 s) in a row -- in
training's own env the arm lags its target by up to ~19 deg when the hand presses on something, so
SimToolReal's plain 10 deg check would fire within seconds; the palm centre below --min_palm_height
above the table.

The real loop's imperfections can be put back in for sim tests, within what training randomized:
--obs_delay / --action_delay (policy steps, training: up to 3), --object_delay (up to 10),
--object_noise (1 cm, 5 deg).

    # sim2sim (sim_server.py running):
    .venv_deploy/bin/python deployment/simtoolreal/run.py --robot isaac --address <node>:6060
    # no hardware: MuJoCo arm + fake hand, fake object
    .venv_deploy/bin/python deployment/simtoolreal/run.py --robot real --arm mujoco --hand fake --object fake
    # hardware, hand connected but not enabled
    .venv_deploy/bin/python deployment/simtoolreal/run.py --robot real --arm <SN> --hand dry_run --hand_sn <SN> ...
    # open-loop replay of a sim rollout
    ... --mode replay --replay docs/results/simtoolreal/rec1/rollout.npz --replay_env 0
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import contextlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from deployment.simtoolreal.core import (  # noqa: E402
    DEFAULT_CKPT, DEFAULT_SPEC, ActionPipeline, GoalSampler, ObsInputs, RobotFK, Spec, SuccessTracker,
    build_obs, load_policy, quat_apply, quat_from_angle_axis, quat_mul,
)

OBJECTS_JSON = HERE / "objects.json"


class Abort(RuntimeError):
    pass


@dataclass
class State:
    q: np.ndarray
    qd: np.ndarray
    object_pos: np.ndarray   # env frame
    object_quat: np.ndarray  # wxyz, env frame
    reset_by_sim: bool = False


# ------------------------------------------------------------------------------------------ backends


class IsaacBackend:
    """sim_server.py over a socket, lock-step: command() steps the sim one policy step."""

    lockstep = True

    def __init__(self, address: str) -> None:
        from multiprocessing.connection import Client

        host, port = address.rsplit(":", 1)
        for attempt in range(600):  # the server may still be starting Isaac Sim
            try:
                self.conn = Client((host, int(port)), authkey=b"simtoolreal")
                break
            except (ConnectionRefusedError, OSError):
                if attempt == 0:
                    print(f"[run] waiting for sim_server at {address} ...", flush=True)
                time.sleep(1.0)
        else:
            raise SystemExit(f"no sim_server at {address}")
        self.info = self._call({"cmd": "info"})
        self.goal = None
        self.state: State | None = None

    # Only plain Python values cross the socket: the two venvs pickle numpy differently (2.x vs 1.x).
    def _call(self, req: dict):
        self.conn.send(req)
        return self.conn.recv()

    def _set(self, d: dict) -> State:
        a = lambda k: np.asarray(d[k], dtype=np.float64)  # noqa: E731
        self.state = State(a("q"), a("qd"), a("object_pos"), a("object_quat"), bool(d["reset_by_sim"]))
        return self.state

    def _goal(self):
        if self.goal is None:
            return None
        return [np.asarray(self.goal[0], float).tolist(), np.asarray(self.goal[1], float).tolist()]

    async def start(self) -> None:
        self._set(self._call({"cmd": "reset", "goal": self._goal()}))

    async def stop(self) -> None:
        with contextlib.suppress(Exception):
            self._call({"cmd": "close"})
        self.conn.close()

    def read(self) -> State:
        return self.state

    def command(self, targets: np.ndarray) -> None:
        self._set(self._call({"cmd": "step", "targets": np.asarray(targets, float).tolist(), "goal": self._goal()}))

    async def move_to(self, q: np.ndarray) -> None:
        """The sim starts every episode at training's reset pose; a new episode is a sim reset."""
        self._set(self._call({"cmd": "reset", "goal": self._goal()}))

    async def new_episode(self) -> None:
        await self.move_to(None)


class FixedObject:
    """A fixed object pose (env frame): the pipeline test before the tracker is connected."""

    def __init__(self, pos_env: np.ndarray, quat_env: np.ndarray) -> None:
        self.pos, self.quat = np.asarray(pos_env, float), np.asarray(quat_env, float)

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        return self.pos.copy(), self.quat.copy()


class RealBackend:
    """aioflexiv arm + aiowuji hand (deploy/joint_reach/run_real.py's ArmIO / HandIO) + an object source."""

    lockstep = False

    def __init__(self, args, spec: Spec, obj: FixedObject) -> None:
        from deploy.joint_reach.run_real import ArmIO, FakeJoints, HandIO

        na = spec.n_arm
        self.na, self.obj = na, obj
        q0 = spec.default_q
        self.arm_kp = np.asarray(spec["stiffness"][:na]) * args.arm_gain_scale
        self.arm_kd = np.asarray(spec["damping"][:na]) * args.arm_gain_scale
        self.hand_kp = np.asarray(spec["stiffness"][na:]) * args.hand_kp_per_nm
        self.hand_kd = np.asarray(spec["damping"][na:]) * args.hand_kd_per_nm
        self.move_speed = args.move_speed
        self.arm = FakeJoints(q0[:na], 0.05) if args.arm == "fake" else ArmIO(
            args.arm, self.arm_kp, self.arm_kd, q0[:na], args.viewer)
        if args.hand == "fake":
            self.hand = FakeJoints(q0[na:], 0.05)
        else:
            sn = args.hand_sn if args.hand == "dry_run" else (None if args.hand == "auto" else args.hand)
            self.hand = HandIO(sn, self.hand_kp, self.hand_kd, args.hand_effort_limit, args.hand == "dry_run",
                               "diff", ())
        self._fake = (args.arm == "fake", args.hand == "fake")

    async def start(self) -> None:
        await self.arm.start()
        await self.hand.start()

    async def stop(self) -> None:
        with contextlib.suppress(Exception):
            await self.hand.stop()
        with contextlib.suppress(Exception):
            await self.arm.stop()

    def read(self) -> State:
        qa, qda = self.arm.read()
        qh, qdh = self.hand.read()
        pos, quat = self.obj.read()
        return State(np.concatenate([qa, qh]), np.concatenate([qda, qdh]), pos, quat)

    def command(self, targets: np.ndarray) -> None:
        self.arm.command(targets[:self.na])
        self.hand.command(targets[self.na:])

    def hold(self) -> None:
        """Targets = the measured pose: stop where we are."""
        s = self.read()
        self.command(s.q)

    async def move_to(self, q: np.ndarray) -> None:
        na = self.na
        if self._fake[0]:
            await self.arm.move_to(q[:na])
        else:
            await self.arm.move_to(q[:na], self.move_speed)
        if self._fake[1]:
            await self.hand.move_to(q[na:])
        else:
            await self.hand.move_to(q[na:], 0.5)
        self.read()  # prime the hand's velocity difference

    async def go_home(self, q: np.ndarray) -> None:
        """Stop, open the hand to q's hand part (lets go of the object), then move the arm to q's arm part.

        The hand goes first so the fingers do not drag the object along the arm's path. The arm move is a
        joint-space Ruckig move at --move_speed: it does not plan around the table.
        """
        na = self.na
        self.hold()
        if self._fake[1]:
            await self.hand.move_to(q[na:])
        else:
            await self.hand.move_to(q[na:], 0.5)
        self.hand.command(q[na:])  # keep the hand open while the arm moves
        if self._fake[0]:
            await self.arm.move_to(q[:na])
        else:
            await self.arm.move_to(q[:na], self.move_speed)
        self.command(q)
        self.read()  # prime the hand's velocity difference


# ------------------------------------------------------------------------------------- imperfections


class Delay:
    """A value seen 0..n steps late: fixed n, or uniform in 0..n each step (training's DR)."""

    def __init__(self, n: int, mode: str, rng: np.random.Generator) -> None:
        self.n, self.mode, self.rng = n, mode, rng
        self.buf: collections.deque = collections.deque(maxlen=n + 1)

    def reset(self) -> None:
        self.buf.clear()

    def __call__(self, x):
        if self.n == 0:
            return x
        if not self.buf:  # training flushes the queue with the first value of an episode
            self.buf.extend([x] * (self.n + 1))
        self.buf.appendleft(x)
        k = self.n if self.mode == "fixed" else int(self.rng.integers(0, self.n + 1))
        return self.buf[k]


def perturb(pos, quat, rng, xyz_std: float, max_deg: float):
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis)
    angle = math.radians(rng.uniform(-max_deg, max_deg))
    return pos + rng.standard_normal(3) * xyz_std, quat_mul(quat_from_angle_axis(angle, axis), quat)


# ------------------------------------------------------------------------------------------- safety


class Safety:
    def __init__(self, args, spec: Spec, fk: RobotFK, table_top: float, guard=None) -> None:
        self.args, self.spec, self.fk, self.table_top, self.guard = args, spec, fk, table_top, guard
        self.lag_steps_limit = max(1, int(round(args.arm_lag_s / spec["policy_dt"])))
        self.lag_run = 0
        self.counts = collections.Counter()

    def reset(self) -> None:
        self.lag_run = 0

    def _trip(self, name: str, msg: str) -> None:
        self.counts[name] += 1
        if self.args.on_safety == "abort":
            raise Abort(msg)

    def check(self, q: np.ndarray, targets: np.ndarray) -> float:
        na = self.spec.n_arm
        lag = float(np.degrees(np.abs(targets[:na] - q[:na]).max()))
        if lag > self.args.arm_lag_hard_deg:
            self._trip("arm_lag_hard", f"arm target {lag:.1f} deg from the measured angle "
                                       f"(limit {self.args.arm_lag_hard_deg})")
        self.lag_run = self.lag_run + 1 if lag > self.args.arm_lag_deg else 0
        if self.lag_run >= self.lag_steps_limit:
            self.lag_run = 0
            self._trip("arm_lag_sustained", f"arm lagged its target by > {self.args.arm_lag_deg} deg "
                                            f"for {self.args.arm_lag_s} s")
        p, r = self.fk.link_pose(self.spec["palm_body"], q)
        palm_z = (p + quat_apply(r, np.asarray(self.spec["palm_center_offset"])))[2] - self.table_top
        if palm_z < self.args.min_palm_height:
            self._trip("palm_low", f"palm centre {palm_z * 100:.1f} cm above the table")
        return lag

    def hand_clearance(self, q: np.ndarray, targets: np.ndarray) -> tuple[float, str, np.ndarray]:
        """Whole-hand check against the real table (TableGuard). Touching the table is allowed (the
        policy grasps a flat object off it); pressing into it is not:
          measured pose: every wrist/hand point >= --min_hand_height (default -0.5 cm: FK/calibration slack)
          arm targets, with the hand at its measured angles: >= -(--max_press_depth) (default 1 cm). This
            is what the stiff arm is told to push; the hand's own push is capped by --hand_effort_limit.
        Returns the measured (height above the table top, link, point)."""
        if self.guard is None:
            return np.inf, "", np.full(3, np.nan)
        na = self.spec.n_arm
        dz, link, point = self.guard.lowest(q)
        if dz < self.args.min_hand_height:
            self._trip("hand_low", f"measured: {link} {dz * 100:.1f} cm above the table top (limit "
                                   f"{self.args.min_hand_height * 100:.1f}) at env xy ({point[0]:.3f}, {point[1]:.3f})")
        x = targets.copy()
        x[na:] = q[na:]
        for region, (dzt, linkt, pt) in self.guard.lowest_by_region(x).items():
            limit = -self.args.max_press_depth if region == "train" else self.args.min_hand_height
            if dzt < limit:
                where = "training's table" if region == "train" else "the real table outside training's table"
                self._trip(f"arm_press_{region}", f"arm target puts {linkt} {-dzt * 100:.1f} cm into {where} "
                                                  f"(limit {-limit * 100:.1f}) at env xy ({pt[0]:.3f}, {pt[1]:.3f})")
        return dz, link, point


def move_clear(guard, args, na: int, q_from: np.ndarray, q_to: np.ndarray, hand_first: bool) -> str | None:
    """None if a scripted move keeps the whole hand out of the real table, else why not.

    The move: the hand goes to q_to's hand angles first (hand_first, go_home) or last (move_to), and the arm
    moves in joint space. Ruckig synchronizes the joints in time, so the path is close to the straight
    line in joint space; it is checked at 2 deg steps of the largest arm joint change."""
    if guard is None:
        return None
    a, b = q_from.copy(), q_to.copy()
    hand = b[na:] if hand_first else a[na:]
    poses = [a]
    if hand_first:
        poses.append(np.concatenate([a[:na], b[na:]]))
    n = max(2, int(np.degrees(np.abs(b[:na] - a[:na]).max()) / 2.0) + 1)
    for s in np.linspace(0.0, 1.0, n):
        poses.append(np.concatenate([a[:na] + s * (b[:na] - a[:na]), hand]))
    poses.append(b)
    for i, x in enumerate(poses):
        dz, link, p = guard.lowest(x)
        if dz < args.min_hand_height:
            return (f"{link} would be {dz * 100:.1f} cm above the table top at env xy ({p[0]:.3f}, {p[1]:.3f}), "
                    f"{i}/{len(poses) - 1} of the way")
    return None


def preflight_table(guard, args, spec: Spec, start_q: np.ndarray, replay, dt: float) -> bool:
    """Before anything moves: the start pose, and in replay mode every recorded step, must keep the whole
    hand out of the real table. The recorded sim joint angles are where the hand will go (the sim table
    is smaller, so past its edge the sim hand went down where the real table is); the arm targets with
    the recorded hand are what the arm will push. With --hand dry_run the hand does not follow its
    targets, so the replay is also checked with the hand at the start pose's (open) angles."""
    lim, press = args.min_hand_height, -args.max_press_depth
    dz, link, _ = guard.lowest(start_q)
    if dz < lim:
        print(f"[run] REFUSED: the start pose puts {link} {dz * 100:.1f} cm above the table top")
        return False
    if replay is None:
        return True
    na = spec.n_arm
    variants = [("hand as recorded", None)]
    if args.hand == "dry_run":
        variants.append(("hand open (dry_run)", start_q[na:]))
    ok = True
    for label, hand in variants:
        for what, src in (("hand", replay["q_sim"]), ("arm target", replay["targets"])):
            lows = []
            for t, x in enumerate(src):
                x = x.copy()
                x[na:] = replay["q_sim"][min(t, len(replay["q_sim"]) - 1), na:] if hand is None else hand
                if what == "hand":
                    dz, link, _ = guard.lowest(x)
                    lows.append((dz, link, dz - lim))
                else:  # pressing allowed on training's table, not on the rest of the real table
                    rg = guard.lowest_by_region(x)
                    m = [(v[0], v[1], v[0] - (press if k == "train" else lim)) for k, v in rg.items()]
                    lows.append(min(m, key=lambda v: v[2]))
            z = np.array([v[0] for v in lows])
            slack = np.array([v[2] for v in lows])
            bad = np.nonzero(slack < 0)[0]
            lim_txt = (f"limit {lim * 100:.1f}" if what == "hand" else
                       f"limit {press * 100:.1f} on training's table, {lim * 100:.1f} elsewhere")
            print(f"[run] replay preflight ({label}, {what}): lowest point {z.min() * 100:.1f} cm above the "
                  f"table top ({lim_txt})")
            if len(bad):
                t0 = bad[0]
                print(f"[run] REFUSED: at t = {t0 * dt:.2f} s {lows[t0][1]} would be {z[t0] * 100:.1f} cm above "
                      f"the table top ({len(bad) * dt:.1f} s in all past the limit). "
                      f"Use --replay_max_s {max(0.0, t0 * dt - 0.5):.1f} or another --replay_env.")
                ok = False
    return ok


# ---------------------------------------------------------------------------------------------- args


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot", choices=("isaac", "real"), required=True)
    p.add_argument("--mode", choices=("policy", "replay"), default="policy")
    p.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    p.add_argument("--spec", default=str(DEFAULT_SPEC))
    p.add_argument("--expl_coef", type=float, default=0.0, help="SAPG block (0 = last, task reward only)")
    p.add_argument("--duration_s", type=float, default=300.0, help="policy time to run")
    p.add_argument("--max_goals", type=int, default=0, help="stop after this many goals (0: no limit)")
    p.add_argument("--goal_timeout_steps", type=int, default=0, help="0: training's episode length (600)")
    p.add_argument("--on_episode_end", choices=("stop", "next_goal", "home"), default=None,
                   help="real robot, when a goal times out or the object is lost: stop the run, go on with "
                        "the next goal, or go home and wait for Enter (default: stop; isaac always resets)")
    p.add_argument("--success_tolerance", type=float, default=None, help="m before keypoint_scale; default 0.01")
    p.add_argument("--seed", type=int, default=0)
    # isaac
    p.add_argument("--address", default="127.0.0.1:6060", help="sim_server host:port")
    # real
    p.add_argument("--arm", default="mujoco", help="robot serial, 'mujoco', or 'fake'")
    p.add_argument("--hand", default="fake", help="hand serial, 'auto', 'dry_run', or 'fake'")
    p.add_argument("--hand_sn", default=None, help="hand serial for --hand dry_run")
    p.add_argument("--object", choices=("fake",), default="fake")
    p.add_argument("--object_name", default="flat_eraser", help=f"an entry of {OBJECTS_JSON.name}")
    p.add_argument("--object_pose_base", type=float, nargs=7, default=None, metavar=("X", "Y", "Z", "QW", "QX", "QY", "QZ"),
                   help="fixed object pose in the robot base frame (default: flat at the table centre)")
    p.add_argument("--arm_gain_scale", type=float, default=1.0, help="x the sim's arm kp and kd")
    p.add_argument("--hand_kp_per_nm", type=float, default=2.0,
                   help="hand kp [A/rad] = this x sim kp [Nm/rad] (Wuji's sim2real convention; UNVERIFIED -- "
                        "aiowuji sysid gives the torque constant: kp_A = kp_Nm / (1.95 Kt))")
    p.add_argument("--hand_kd_per_nm", type=float, default=math.sqrt(2.0), help="hand kd [A s/rad] = this x sim kd")
    p.add_argument("--hand_effort_limit", type=float, default=0.5, help="A (aiowuji default 0.5, max 1.5)")
    p.add_argument("--move_speed", type=float, default=0.3, help="rad/s for the move to the start pose")
    p.add_argument("--settle_s", type=float, default=1.0, help="hold at the start pose before the policy")
    p.add_argument("--home_on_exit", choices=("ask", "yes", "no"), default="ask",
                   help="real robot, when the run ends: open the hand and move back to the start pose "
                        "(ask: wait for Enter first)")
    p.add_argument("--viewer", action="store_true", help="MuJoCo viewer for --arm mujoco")
    p.add_argument("--viser", action="store_true", help="live viser view: robot, targets, table, object, goal")
    p.add_argument("--viser_port", type=int, default=8091)
    p.add_argument("--viser_hold", action="store_true", help="keep the viser page up after the run until Enter")
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    # replay
    p.add_argument("--replay", default=None, help="rollout.npz from record_sim_rollout.py")
    p.add_argument("--replay_env", type=int, default=0)
    # sim-test imperfections
    p.add_argument("--obs_delay", type=int, default=0)
    p.add_argument("--action_delay", type=int, default=0)
    p.add_argument("--object_delay", type=int, default=0)
    p.add_argument("--delay_mode", choices=("fixed", "random"), default="random")
    p.add_argument("--object_noise", action="store_true", help="training's 1 cm / 5 deg object-pose noise")
    # safety
    p.add_argument("--arm_lag_hard_deg", type=float, default=25.0)
    p.add_argument("--arm_lag_deg", type=float, default=10.0)
    p.add_argument("--arm_lag_s", type=float, default=0.5)
    p.add_argument("--min_palm_height", type=float, default=0.02, help="m above the table top")
    p.add_argument("--min_hand_height", type=float, default=-0.01,
                   help="m: wrist/hand mesh points over the real table stay above top + this: measured always, "
                        "arm targets outside training's table (negative: contact allowed; training's own contact "
                        "reaches -0.7 cm)")
    p.add_argument("--max_press_depth", type=float, default=0.02,
                   help="m: how far the arm targets (hand at its measured angles) may put the hand into training's "
                        "table (training presses up to ~6 cm: 3.4%% of steps > 1 cm, 2.5%% > 2 cm, 0.8%% > 4 cm)")
    p.add_argument("--table_x_half", type=float, default=1.0,
                   help="real table half-width in x (env frame, m); its top and near edge are training's")
    p.add_argument("--table_far_y", type=float, default=-1.0, help="real table far edge y (env frame, m)")
    p.add_argument("--table_edge_margin", type=float, default=0.03,
                   help="m past the near edge (towards the base) still checked")
    p.add_argument("--replay_max_s", type=float, default=0.0, help="replay only the first this many s (0: all)")
    p.add_argument("--on_safety", choices=("abort", "count"), default="abort")
    # pacing / logs
    p.add_argument("--realtime", action="store_true", help="isaac: pace the loop at 60 Hz wall time (real: always)")
    p.add_argument("--log_dir", default=str(HERE / "logs"))
    p.add_argument("--tag", default="")
    return p.parse_args()


def load_object(name: str) -> tuple[np.ndarray, np.ndarray]:
    """(scales, collision-box corners in the object frame) of a known object."""
    objects = json.loads(OBJECTS_JSON.read_text())
    if name not in objects:
        raise SystemExit(f"unknown object {name}; known: {sorted(objects)}")
    scales = np.asarray(objects[name]["scale"], dtype=np.float64)
    half = scales * 0.04 / 2  # the grasp box (object_base_size 0.04 per unit of scale)
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half
    return scales, corners


# ---------------------------------------------------------------------------------------------- run


async def run(args: argparse.Namespace) -> int:
    spec = Spec.load(args.spec)
    fk = RobotFK(spec)
    rng = np.random.default_rng(args.seed)
    na = spec.n_arm
    dt = float(spec["policy_dt"])
    timeout = args.goal_timeout_steps or int(spec["episode_length_steps"])
    max_steps = int(round(args.duration_s / dt))
    on_end = "reset" if args.robot == "isaac" else (args.on_episode_end or "stop")

    # ---- backend and object
    if args.robot == "isaac":
        robot = IsaacBackend(args.address)
        info = robot.info
        if info["joint_names"] != spec.joint_names:
            raise SystemExit("sim joint order differs from the spec")
        scales = np.asarray(info["object_scales"], dtype=np.float64)
        corners = np.asarray(info["object_corners"], dtype=np.float64)
        table_top = float(info["table_top_z"])
        obj_label = Path(info["object_urdf"]).name
    else:
        scales, corners = load_object(args.object_name)
        table_top = float(spec["table_top_z_env"])
        if args.object_pose_base is None:  # lying flat (object frame = base axes) at the table centre
            tp = np.asarray(spec["table_pos_env"], dtype=np.float64)
            pos_env = np.array([tp[0], tp[1], table_top + scales[2] * 0.04 / 2])
            quat_env = np.array([1.0, 0, 0, 0])
        else:
            v = np.asarray(args.object_pose_base, dtype=np.float64)
            pos_env, quat_env = spec.base_to_env(v[:3], v[3:] / np.linalg.norm(v[3:]))
        robot = RealBackend(args, spec, FixedObject(pos_env, quat_env))
        obj_label = f"{args.object_name} (fixed pose, env {np.round(pos_env, 3).tolist()})"
    print(f"[run] robot {args.robot}, mode {args.mode}, object {obj_label}, scales {np.round(scales, 3).tolist()}, "
          f"table top {table_top:.3f} m (env frame)")

    # ---- replay source
    replay = None
    if args.mode == "replay":
        if not args.replay:
            raise SystemExit("--mode replay needs --replay <rollout.npz>")
        r = np.load(args.replay)
        e = args.replay_env
        ends = np.nonzero(r["done"][:, e])[0]
        n_rep = int(ends[0]) if len(ends) else len(r["targets"])
        replay = {"start": r["prev_targets"][0, e].astype(np.float64), "targets": r["targets"][:n_rep, e].astype(np.float64),
                  "q_sim": r["q"][1:n_rep + 1, e].astype(np.float64)}
        if args.replay_max_s > 0:
            n_rep = min(n_rep, int(round(args.replay_max_s / dt)))
            replay = {k: (v[:n_rep] if k != "start" else v) for k, v in replay.items()}
        max_steps = min(max_steps, n_rep)
        print(f"[run] replaying {n_rep} steps ({n_rep * dt:.1f} s) of env {e} from {args.replay}")

    start_q = replay["start"] if replay is not None else spec.default_q
    guard = None
    if args.robot == "real":  # the sim has its own (training's) table; the guard is for the real one
        from deployment.simtoolreal.table_guard import TableGuard
        guard = TableGuard(spec, args.table_x_half, args.table_far_y, args.table_edge_margin)
        print(f"[run] table guard: every wrist/hand point >= {args.min_hand_height * 100:.1f} cm above the real "
              f"table top (z {guard.top:.3f}, x +-{guard.x_half:.2f}, y {guard.far_y:.2f}..{guard.near_y:.2f} "
              f"+ {guard.edge_margin:.2f} m, env frame)")
        if not preflight_table(guard, args, spec, start_q, replay, dt):
            return 1
    if args.robot == "real":
        print(f"[run] start pose arm [deg] {np.round(np.degrees(start_q[:na]), 1).tolist()}")
        print(f"[run] arm kp {np.round(robot.arm_kp, 1).tolist()} Nm/rad, kd {np.round(robot.arm_kd, 2).tolist()}")
        print(f"[run] hand kp {np.round(robot.hand_kp, 3).tolist()} A/rad (UNVERIFIED conversion), "
              f"effort limit {args.hand_effort_limit} A")
        hardware = args.arm not in ("mujoco", "fake") or args.hand not in ("fake", "dry_run")
        if hardware and not args.yes:
            if input("[run] hardware will move. Type 'go' to start: ").strip() != "go":
                print("[run] not started")
                return 1

    policy = load_policy(args.checkpoint, expl_coef=args.expl_coef) if args.mode == "policy" else None
    sampler = GoalSampler(spec, corners, seed=args.seed, table_top_z=table_top)
    tracker = SuccessTracker(spec, tolerance=args.success_tolerance)
    safety = Safety(args, spec, fk, table_top, guard)
    obs_delay, act_delay, obj_delay = (Delay(n, args.delay_mode, rng)
                                       for n in (args.obs_delay, args.action_delay, args.object_delay))

    log = collections.defaultdict(list)
    counts = collections.Counter()
    goals_per_episode: list[int] = []
    pace = args.robot == "real" or args.realtime

    def compute(st: State, pipe: ActionPipeline, goal):
        """obs -> policy -> targets (runs in a worker thread: the arm's torque loop keeps stepping)."""
        obj_pos, obj_quat = obj_delay((st.object_pos, st.object_quat))
        if args.object_noise:
            obj_pos, obj_quat = perturb(obj_pos, obj_quat, rng, 0.01, 5.0)
        x = ObsInputs(q=st.q, qd=st.qd, prev_targets=pipe.prev, object_pos=obj_pos, object_quat=obj_quat,
                      goal_pos=goal[0], goal_quat=goal[1], object_scales=scales)
        obs = obs_delay(build_obs(spec, fk, x))
        action = act_delay(policy.act(obs))
        return obs, action, pipe.step(action)

    viewer = None
    if args.viser:
        from deployment.simtoolreal.live_viser import LiveViewer
        viewer = LiveViewer(spec, scales, port=args.viser_port, guard=guard, min_hand_height=args.min_hand_height)

    def show(st: State, targets, t_s: float, status: int, clear=(np.inf, "", np.full(3, np.nan))) -> None:
        if viewer is not None:
            viewer.publish(st.q, targets, st.object_pos, st.object_quat, goal[0], goal[1], t_s, counts["goals"], status,
                           clear[0], clear[2])

    async with contextlib.AsyncExitStack() as stack:
        if viewer is not None:
            stack.callback(viewer.close)
        goal = sampler.first()
        if isinstance(robot, IsaacBackend):
            robot.goal = goal
        await robot.start()
        stack.push_async_callback(robot.stop)

        async def begin_episode(first: bool) -> tuple[State, ActionPipeline]:
            nonlocal goal
            if not first:
                goal = sampler.first()
                if isinstance(robot, IsaacBackend):
                    robot.goal = goal
            if first or isinstance(robot, IsaacBackend):
                if guard is not None:
                    why = move_clear(guard, args, na, robot.read().q, start_q, hand_first=False)
                    if why:
                        raise Abort(f"not moving to the start pose: {why}. Move the arm up by hand first")
                print("[run] moving to the start pose", flush=True)
                await robot.move_to(start_q)
            if args.robot == "real" and first:
                await asyncio.sleep(args.settle_s)
            if policy is not None:
                policy.reset()
            tracker.near_steps = 0
            safety.reset()
            for d in (obs_delay, act_delay, obj_delay):
                d.reset()
            st = robot.read()
            show(st, st.q, 0.0, 0)
            return st, ActionPipeline(spec, st.q)

        st, pipe = await begin_episode(first=True)
        if policy is not None:  # warm the policy up off the clock, then clear its memory
            await asyncio.to_thread(compute, st, ActionPipeline(spec, st.q), goal)
            policy.reset()
            for d in (obs_delay, act_delay, obj_delay):
                d.reset()

        ep_goals, on_goal = 0, 0
        t_start = time.perf_counter()
        t_next = t_start
        stopped = None
        print("[run] running", flush=True)
        try:
            for t in range(max_steps):
                t_read = time.perf_counter()
                st = robot.read()

                # -- the goal test, on the state the last targets produced
                if t > 0 and replay is None:
                    reached = tracker.update(st.object_pos, st.object_quat, goal[0], goal[1])
                    log["reached"].append(reached)
                    on_goal += 1
                    if reached:
                        counts["goals"] += 1
                        ep_goals += 1
                        on_goal = 0
                        goal = sampler.next()
                        if isinstance(robot, IsaacBackend):
                            robot.goal = goal
                        print(f"[run] t={t * dt:6.1f} s  goal {counts['goals']} reached", flush=True)
                    ended = "object_lost" if st.reset_by_sim or st.object_pos[2] < table_top - 0.05 else (
                        "goal_timeout" if on_goal >= timeout else None)
                    if ended:
                        counts[ended] += 1
                        goals_per_episode.append(ep_goals)
                        print(f"[run] t={t * dt:6.1f} s  episode {len(goals_per_episode)} ended ({ended}) "
                              f"after {ep_goals} goals", flush=True)
                        ep_goals, on_goal = 0, 0
                        if on_end == "stop":
                            stopped = ended
                            break
                        if on_end == "next_goal":
                            goal = sampler.next()
                        else:
                            if on_end == "home":
                                why = move_clear(guard, args, na, robot.read().q, start_q, hand_first=True)
                                if why:
                                    raise Abort(f"not going home: {why}")
                                await robot.go_home(start_q)
                                await asyncio.to_thread(input, "[run] put the object back, then press Enter ")
                            st, pipe = await begin_episode(first=False)
                            t_next = time.perf_counter()

                # -- targets
                if replay is not None:
                    obs, action, targets = None, None, replay["targets"][t].copy()
                    pipe.prev = targets.copy()
                else:
                    obs, action, targets = await asyncio.to_thread(compute, st, pipe, goal)
                lag = safety.check(st.q, targets)
                clear = safety.hand_clearance(st.q, targets)
                robot.command(targets)
                t_sent = time.perf_counter()
                show(st, targets, t * dt, 1, clear)

                for k, v in (("q", st.q), ("qd", st.qd), ("targets", targets), ("object_pos", st.object_pos),
                             ("object_quat", st.object_quat), ("goal_pos", goal[0]), ("goal_quat", goal[1]),
                             ("arm_lag_deg", lag), ("hand_clearance_m", clear[0]), ("t_read", t_read - t_start), ("t_sent", t_sent - t_start)):
                    log[k].append(v)
                if obs is not None:
                    log["obs"].append(obs)
                    log["action"].append(action)

                if pace:
                    t_next += dt
                    wait = t_next - time.perf_counter()
                    if wait > 0:
                        await asyncio.sleep(wait)
                    else:
                        counts["overruns"] += 1
                        if -wait > dt:  # a whole step behind: drop it, don't burst
                            t_next = time.perf_counter()
                if args.max_goals and counts["goals"] >= args.max_goals:
                    stopped = "max_goals"
                    break
        except Abort as e:
            print(f"[run] ABORT: {e}", flush=True)
            stopped = f"abort: {e}"
        except (KeyboardInterrupt, asyncio.CancelledError):
            print("[run] interrupted", flush=True)
            stopped = "interrupted"
        finally:
            if isinstance(robot, RealBackend):
                with contextlib.suppress(Exception):
                    robot.hold()
                home = args.home_on_exit
                if home == "ask":
                    with contextlib.suppress(EOFError, KeyboardInterrupt):
                        ans = await asyncio.to_thread(input, "[run] holding. Enter: go home, 'n' + Enter: stay ")
                        home = "no" if ans.strip().lower() == "n" else "yes"
                if home == "yes":
                    with contextlib.suppress(Exception):
                        why = move_clear(guard, args, na, robot.read().q, start_q, hand_first=True)
                        if why:
                            print(f"[run] NOT going home, holding: {why}. Move the arm up by hand "
                                  "(free drive), then run move_home.py", flush=True)
                            home = "no"
                if home == "yes":
                    print("[run] going home", flush=True)
                    try:
                        await robot.go_home(start_q)
                        print("[run] home", flush=True)
                    except Exception as e:  # e.g. the torque loop already stopped
                        print(f"[run] go home failed: {e!r}", flush=True)
            if viewer is not None and log["q"]:
                with contextlib.suppress(Exception):
                    show(robot.read(), log["targets"][-1], len(log["q"]) * dt, 2)
                if args.viser_hold:
                    await asyncio.to_thread(input, "[run] viser still up; press Enter to exit ")

    steps = len(log["q"])
    minutes = steps * dt / 60.0
    goals_per_episode.append(ep_goals)
    summary = {
        "steps": steps, "policy_minutes": round(minutes, 2), "stopped": stopped,
        "goals": counts["goals"], "goals_per_minute": round(counts["goals"] / max(minutes, 1e-9), 2),
        "goals_per_episode": goals_per_episode, "object_lost": counts["object_lost"],
        "goal_timeout": counts["goal_timeout"], "safety": dict(safety.counts), "overruns": counts["overruns"],
    }
    if steps:
        loop_ms = 1e3 * (np.asarray(log["t_sent"]) - np.asarray(log["t_read"]))
        summary.update(read_to_send_ms_median=round(float(np.median(loop_ms)), 2),
                       read_to_send_ms_max=round(float(loop_ms.max()), 2),
                       arm_lag_deg_max=round(float(np.max(log["arm_lag_deg"])), 1))
        if pace and steps > 1:
            period = 1e3 * np.diff(np.asarray(log["t_read"]))
            summary.update(period_ms_median=round(float(np.median(period)), 2), period_ms_max=round(float(period.max()), 2))
    if replay is not None and steps > 1:
        q = np.asarray(log["q"])[1:]
        err = np.degrees(np.abs(q - replay["q_sim"][:len(q)]))
        summary.update(replay_err_deg_arm_max=round(float(err[:, :na].max()), 2),
                       replay_err_deg_arm_median=round(float(np.median(err[:, :na].max(1))), 2),
                       replay_err_deg_hand_max=round(float(err[:, na:].max()), 2),
                       replay_err_deg_hand_median=round(float(np.median(err[:, na:].max(1))), 2))
    print("[run] summary " + json.dumps(summary))
    out = Path(args.log_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"run_{args.robot}_{args.mode}_{args.tag + '_' if args.tag else ''}{time.strftime('%Y%m%d_%H%M%S')}"
    np.savez_compressed(out / f"{stem}.npz", **{k: np.asarray(v) for k, v in log.items()})
    (out / f"{stem}.json").write_text(json.dumps({**summary, "args": vars(args), "object": obj_label,
                                                  "object_scales": scales.tolist()}, indent=1) + "\n")
    print(f"[run] log {out / stem}.npz")
    return 0


def main() -> int:
    return asyncio.run(run(parse()))


if __name__ == "__main__":
    raise SystemExit(main())
