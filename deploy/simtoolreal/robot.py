"""The robot node: Flexiv Rizon 4s arm (aioflexiv) + right Wuji Hand 2 (aiowuji). Usually started by
run.py; the README has the details.

robot.arm   serial number | mujoco (aioflexiv's MuJoCo Rizon, same code path)
robot.hand  serial number | auto | dry_run (connected, never enabled) | fake (utils/fake_hand.py)
robot.mode  home        hand, then arm, to home (spec default_q, the training start pose), then exit
            free_drive  arm movable by hand (kp 0, Flexiv compensates gravity); hand never enabled
            policy      the arm must be at home; the hand is brought home; then forward policy.py's
                        joint_target, with two safety checks (check_tracking, check_target_step)
            openloop    as policy, but the targets come from a sim rollout (openloop.path, made by
                        checks/record_lift.py), sent one per policy step with no feedback; this node
                        then also publishes joint_target, goal_pose and sim_object_pose itself

Every mode publishes joint_state at rate_hz, and ends (Ctrl-C, or the arm's torque loop stopped:
E-stop / enabling button / fault) by holding the measured pose and then stopping both: the hand goes
limp. The Flexiv E-stop stops only the arm; this node then stops the hand.

Gains: the arm runs aioflexiv's joint impedance, tau = kp (q* - q) - kd dq (Flexiv adds gravity),
with the sim's kp, kd (Nm/rad). The Wuji firmware takes A/rad: kp, kd = sim / (hand_kp/kd_scale x Kt),
Kt per joint from the aiowuji sysid (robot.hand_kt).
"""

import asyncio
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))    # deploy/simtoolreal
from utils.config import load_config, repo_path                                  # noqa: E402
from utils.fake_hand import FakeHand                                             # noqa: E402
from utils.link import age, open_publisher, open_subscriber, recv_latest, send   # noqa: E402
from utils.openloop import load_rollout                                          # noqa: E402
from utils.spec import load_spec                                                 # noqa: E402

MODES = ("policy", "home", "free_drive", "openloop")
AT_HOME_MODES = ("policy", "openloop")   # refuse to start with the arm away from home
ARM, HAND = slice(0, 7), slice(7, None)   # spec joint order: 7 arm joints, then 20 hand joints
ARM_STOPPED = "arm torque loop stopped (E-stop, enabling button or fault)"


# ---------------------------------------------------------------------------------- arm and hand

def open_arm(cfg, spec):
    from aioflexiv import FlexivController
    kwargs = {}
    if cfg.arm == "mujoco":
        kwargs = dict(mujoco_viewer=cfg.mujoco_viewer, mujoco_initial_qpos=spec["default_q"][ARM])
    return FlexivController(str(cfg.arm), **kwargs)


def set_arm_gains(arm, cfg, spec, free: bool = False) -> None:
    """The sim's gains, or free drive (kp 0, light damping)."""
    with arm.state_lock:
        arm.kp = np.zeros(7) if free else spec["stiffness"][ARM] * cfg.arm_gain_scale
        arm.kd = spec["damping"][ARM] * (cfg.free_drive_kd_scale if free else cfg.arm_gain_scale)


def open_hand(cfg, spec, never_enable: bool = False):
    """aiowuji WujiController, or a FakeHand. never_enable: connect but never enable (as dry_run)."""
    if cfg.hand == "fake":
        return FakeHand(spec["default_q"][HAND])
    from aiowuji import JOINT_NAMES, WujiController
    serial = cfg.hand_sn if cfg.hand == "dry_run" else (None if cfg.hand == "auto" else str(cfg.hand))
    effort = np.full(len(JOINT_NAMES), float(cfg.hand_effort_limit))
    for name, amps in (cfg.hand_effort_limit_joints or {}).items():
        if name not in JOINT_NAMES:
            raise SystemExit(f"[robot] robot.hand_effort_limit_joints: unknown joint {name!r}; one of {JOINT_NAMES}")
        effort[JOINT_NAMES.index(name)] = float(amps)
    hand_kt = dict(cfg.hand_kt or {})
    if set(hand_kt) != set(JOINT_NAMES):
        raise SystemExit(f"[robot] robot.hand_kt needs exactly the aiowuji joints {JOINT_NAMES}; got {sorted(hand_kt)}")
    kt = np.array([float(hand_kt[n]) for n in JOINT_NAMES])     # N m/A
    return WujiController(serial, kp=spec["stiffness"][HAND] / (cfg.hand_kp_scale * kt),
                          kd=spec["damping"][HAND] / (cfg.hand_kd_scale * kt), effort_limit=effort,
                          dry_run=cfg.hand == "dry_run" or never_enable)


def read_q(arm, hand) -> np.ndarray | None:
    """All 27 joint angles, or None once the arm's torque loop has stopped. Raises if the hand's stopped."""
    if not arm.running:
        return None
    if hand.error is not None:
        raise RuntimeError(f"hand stopped: {hand.error}")
    return np.r_[arm.state["qpos"], hand.state["qpos"]].astype(float)


def command(arm, hand, q: np.ndarray) -> None:
    """Joint targets (27) to the arm and the hand."""
    arm.set_now("q_desired", q[ARM])
    hand.set_now("q_desired", clip_hand(q[HAND]))


def hold(arm, hand) -> np.ndarray | None:
    """Targets = the measured pose: stay where it is. Returns that pose (None if the arm stopped)."""
    q = read_q(arm, hand)
    if q is not None:
        command(arm, hand, q)
    return q


def clip_hand(q: np.ndarray) -> np.ndarray:
    """The sim's thumb_cmc_abd lower limit is 0.03 deg past aiowuji's documented range, which it rejects."""
    from aiowuji import JOINT_LIMITS
    return np.clip(q, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])


def hand_velocity(q: np.ndarray, t: float, memo: dict, window_s: float) -> np.ndarray:
    """PLACEHOLDER until measured on the hand. aiowuji: the SDK's qvel can read > 0.5 rad/s on a still
    joint, so (as the old deploy) use the backward difference of the position over >= window_s
    (one policy step), held between updates. memo keeps the reference point."""
    if "q" not in memo:
        memo.update(q=q, t=t, dq=np.zeros_like(q))
    elif t - memo["t"] >= window_s:
        memo.update(dq=(q - memo["q"]) / (t - memo["t"]), q=q, t=t)
    return memo["dq"]


async def publish_joint_state(cfg, spec, arm, hand, status: dict) -> None:
    """joint_state at rate_hz until cancelled or the arm stops. status["tripped"]: a safety trip's reason;
    status["ready"]: policy mode has brought the hand home and now forwards joint_target."""
    pub = open_publisher(cfg.bus.joint_state_port)
    memo, seq = {}, 0
    try:
        while (q := read_q(arm, hand)) is not None:
            now = time.time()
            dq = np.r_[arm.state["qvel"], hand_velocity(q[HAND], now, memo, spec["policy_dt"])]
            send(pub, "joint_state", {"seq": seq, "t": now, "q": q, "dq": dq, "tripped": status.get("tripped"),
                                     "ready": bool(status.get("ready"))})
            seq += 1
            await asyncio.sleep(1.0 / cfg.robot.rate_hz)
    finally:
        pub.close(linger=0)


# ------------------------------------------------------------------------------- home and safety

def home_errors(cfg, spec, q: np.ndarray, check_hand: bool) -> list[str]:
    """Joints further from home (spec default_q) than the tolerance, as "name 12.3 deg"."""
    err = np.degrees(np.abs(q - spec["default_q"]))
    tol = np.r_[np.full(7, cfg.home_tolerance_arm_deg), np.full(20, cfg.home_tolerance_hand_deg)]
    joints = range(27) if check_hand else range(7)
    return [f"{spec['joint_names'][i]} {err[i]:.1f} deg" for i in joints if err[i] > tol[i]]


def check_tracking(cfg, spec, commanded: np.ndarray, q: np.ndarray) -> str | None:
    """An arm joint too far from the target it is being sent (stuck, hit something, wrong gains)."""
    err = np.degrees(np.abs(commanded[ARM] - q[ARM]))
    i = int(np.argmax(err))
    if err[i] > cfg.max_arm_tracking_deg:
        return f"arm tracking: {spec['joint_names'][i]} {err[i]:.1f} deg from its target"
    return None


def check_target_step(cfg, spec, commanded: np.ndarray, new: np.ndarray) -> str | None:
    """A new target too far from the last one sent: more than training's action pipeline can move
    in one step, so a bug (policy restarted elsewhere, wrong joint order, ...). Checked before sending."""
    step = np.degrees(np.abs(new - commanded))
    limit = np.r_[np.full(7, cfg.max_target_step_arm_deg), np.full(20, cfg.max_target_step_hand_deg)]
    i = int(np.argmax(step / limit))
    if step[i] > limit[i]:
        return f"target jump: {spec['joint_names'][i]} {step[i]:.1f} deg in one step (limit {limit[i]:.0f})"
    return None


# ----------------------------------------------------------------------------------------- modes

async def go_home(cfg, spec, arm, hand) -> str:
    """Hand first (opens, lets go), then the arm, jerk-limited; the arm in a straight joint-space line."""
    rc, home = cfg.robot, spec["default_q"]
    if not hand.dry_run:
        print("[robot] home: hand ...", flush=True)
        await hand.move(clip_hand(home[HAND]), max_velocity=rc.hand_move_speed)
    print("[robot] home: arm ...", flush=True)
    v = rc.move_speed
    await arm.move(home[ARM], max_velocity=v, max_acceleration=2.0 * v, max_jerk=10.0 * v)
    set_arm_gains(arm, rc, spec)          # move() keeps the gains, but make sure
    q = read_q(arm, hand)
    if q is None:
        return ARM_STOPPED
    far = home_errors(rc, spec, q, check_hand=not hand.dry_run)
    return f"home NOT reached: {', '.join(far)}" if far else "home reached"


async def free_drive(cfg, spec, arm, hand) -> str:
    """Arm movable by hand until Ctrl-C; prints how far each arm joint is from home every second."""
    set_arm_gains(arm, cfg.robot, spec, free=True)
    try:
        while (q := read_q(arm, hand)) is not None:
            err = np.degrees(q[ARM] - spec["default_q"][ARM])
            i = int(np.argmax(np.abs(err)))
            print(f"[robot] free drive: arm to home, worst {spec['joint_names'][i]} {err[i]:+.1f} deg; "
                  f"all {np.round(err, 1).tolist()}", flush=True)
            await asyncio.sleep(1.0)
        return ARM_STOPPED
    finally:
        hold(arm, hand)                   # targets = here, before the stiffness comes back
        set_arm_gains(arm, cfg.robot, spec)


async def hand_to_home(rc, spec, arm, hand, mode: str) -> str | None:
    """The hand went limp when the last run stopped: bring it home. Returns why not, or None."""
    if hand.dry_run:
        return None
    print(f"[robot] {mode}: hand to home ...", flush=True)
    await hand.move(clip_hand(spec["default_q"][HAND]), max_velocity=rc.hand_move_speed)
    q = read_q(arm, hand)
    if q is None:
        return ARM_STOPPED
    if far := home_errors(rc, spec, q, check_hand=True):
        return f"hand did not reach home: {', '.join(far)}"
    return None


async def follow_policy(cfg, spec, arm, hand, status: dict) -> str:
    """Forward joint_target to the robot until Ctrl-C, with the two safety checks.
    No new target for target_timeout_s: hold. A safety trip: hold, ignore every later target."""
    rc = cfg.robot
    if (why := await hand_to_home(rc, spec, arm, hand, "policy")) is not None:
        return why

    sub = open_subscriber(cfg.bus.host, cfg.bus.policy_port, "joint_target")
    commanded = read_q(arm, hand)         # what the robot is being sent (at start: where it is)
    status["ready"] = True                # policy.py starts only now, from this pose (not mid hand-move)
    target, last_seq, following = None, None, False
    try:
        while (q := read_q(arm, hand)) is not None:
            target = recv_latest(sub) or target
            fresh = target is not None and age(target) < rc.target_timeout_s
            if not status.get("tripped"):
                trip = check_tracking(rc, spec, commanded, q)
                if trip is None and fresh and target["seq"] != last_seq:
                    trip = check_target_step(rc, spec, commanded, target["targets"])
                    last_seq = target["seq"]
                if trip:
                    status["tripped"] = trip
                    print(f"[robot] SAFETY TRIP: {trip}. Holding; targets ignored. Ctrl-C to end.", flush=True)
                    commanded = hold(arm, hand)
                elif fresh:
                    if not following:
                        print("[robot] following joint_target", flush=True)
                        following = True
                    commanded = target["targets"]
                    command(arm, hand, commanded)
                elif following:           # targets stopped
                    print(f"[robot] no joint_target for {rc.target_timeout_s} s: holding", flush=True)
                    following = False
                    commanded = hold(arm, hand)
            await asyncio.sleep(1.0 / rc.rate_hz)
        return ARM_STOPPED
    finally:
        sub.close(linger=0)


def send_openloop(pub, roll: dict, k: int, seq: int, targets: bool = True) -> None:
    """Step k of the rollout on the bus: its joint targets, the sim's goal, where the sim object was."""
    now = time.time()
    if targets:
        send(pub, "joint_target", {"seq": seq, "t": now, "targets": roll["targets"][k]})
    if roll.get("goals") is not None:   # a goal sequence (record_lift.py --goals train)
        pos, quat = roll["goals"][0][k], roll["goals"][1][k]
        index = int(np.count_nonzero(np.any(np.diff(roll["goals"][0][:k + 1], axis=0) != 0, axis=1)))
    else:
        pos, quat, index = roll["goal"][0], roll["goal"][1], 0
    send(pub, "goal_pose", {"seq": seq, "t": now, "pos": pos, "quat_wxyz": quat, "goal_index": index})
    send(pub, "sim_object_pose", {"seq": seq, "t": now, "pos": roll["object_pos"][k],
                                  "quat_wxyz": roll["object_quat"][k], "step": k})


async def play_openloop(cfg, spec, arm, hand, status: dict) -> str:
    """Send the rollout's joint targets, target k from k * dt after the start, with the two safety
    checks; after the last one, keep holding it (the object stays lifted) until Ctrl-C.
    A safety trip: hold the measured pose, send nothing more."""
    rc = cfg.robot
    roll = load_rollout(repo_path(cfg.openloop.path), spec)
    targets, dt, n = roll["targets"], roll["dt"], len(roll["targets"])
    pub = open_publisher(cfg.bus.policy_port)

    async def announce() -> None:   # the viewer shows where the object goes while the hand moves home
        seq = 0
        while True:
            send_openloop(pub, roll, 0, seq, targets=False)
            seq += 1
            await asyncio.sleep(0.1)

    announcer = asyncio.create_task(announce())
    try:
        why = await hand_to_home(rc, spec, arm, hand, "openloop")
        announcer.cancel()
        if why is not None:
            return why
        print(f"[robot] openloop: sending {n} targets ({n * dt:.1f} s) from {roll['path']}", flush=True)
        commanded = read_q(arm, hand)
        t0, sent, ended = time.perf_counter(), -1, False
        while (q := read_q(arm, hand)) is not None:
            tick = int((time.perf_counter() - t0) / dt)
            k = min(tick, n - 1)
            if not status.get("tripped"):
                trip = check_tracking(rc, spec, commanded, q)
                if trip is None and tick != sent:
                    trip = check_target_step(rc, spec, commanded, targets[k])
                if trip:
                    status["tripped"] = trip
                    print(f"[robot] SAFETY TRIP at {tick * dt:.2f} s: {trip}. Holding. Ctrl-C to end.", flush=True)
                    commanded = hold(arm, hand)
                elif tick != sent:
                    commanded = targets[k]
                    command(arm, hand, commanded)
                    send_openloop(pub, roll, k, tick)
                    sent = tick
                    if k == n - 1 and not ended:
                        ended = True
                        print("[robot] openloop: last target sent; holding it. Ctrl-C to end.", flush=True)
            await asyncio.sleep(1.0 / rc.rate_hz)
        return ARM_STOPPED
    finally:
        announcer.cancel()
        pub.close(linger=0)


# ------------------------------------------------------------------------------------------ main

async def main() -> None:
    cfg = load_config()
    rc = cfg.robot
    if rc.mode not in MODES:
        raise SystemExit(f"[robot] robot.mode={rc.mode}: {' | '.join(MODES)}")
    spec = load_spec(repo_path(cfg.spec_path))

    arm = open_arm(rc, spec)
    hand = open_hand(rc, spec, never_enable=rc.mode == "free_drive")
    await arm.start()                     # holds where it is
    set_arm_gains(arm, rc, spec)
    await hand.start()
    status = {}
    publisher = asyncio.create_task(publish_joint_state(cfg, spec, arm, hand, status))
    reason = "?"
    try:
        # policy and openloop check only the arm here: they bring the hand home themselves
        far = home_errors(rc, spec, read_q(arm, hand), check_hand=not hand.dry_run and rc.mode not in AT_HOME_MODES)
        print(f"[robot] mode {rc.mode}; arm {rc.arm}, hand {rc.hand}; "
              f"{'at home' if not far else 'away from home: ' + ', '.join(far)}. Ctrl-C stops.", flush=True)
        if rc.mode in AT_HOME_MODES and far:
            raise SystemExit("[robot] not at home: run robot.mode=home (or free_drive) first")
        if rc.mode == "home":
            reason = await go_home(cfg, spec, arm, hand)
        elif rc.mode == "free_drive":
            reason = await free_drive(cfg, spec, arm, hand)
        elif rc.mode == "openloop":
            reason = await play_openloop(cfg, spec, arm, hand, status)
        else:
            reason = await follow_policy(cfg, spec, arm, hand, status)
    except (KeyboardInterrupt, asyncio.CancelledError):
        reason = "Ctrl-C"
    finally:
        publisher.cancel()
        try:
            hold(arm, hand)
        except RuntimeError:              # the hand already stopped
            pass
        await hand.stop()                 # disables it: the fingers go limp
        await arm.stop()
        print(f"[robot] stopped: {reason}", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
