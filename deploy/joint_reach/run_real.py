"""Run the joint-reach policy on the real Flexiv Rizon 4s + Wuji Hand 2 (left), at 60 Hz.

One asyncio process. aioflexiv runs the arm's joint impedance law in a Python torque loop (Flexiv
compensates gravity, as the sim has none on the robot); aiowuji streams joint targets to the hand's
firmware PD from a 1 kHz thread. Every policy step: read both, build the observation, run the policy
(in a worker thread, so the arm's torque loop keeps stepping), turn the action into joint targets
through the sim's pipeline, send them.

Goals come from real_test_sequence.pt (isaacsimenvs/eval/joint_reach_eval.py): a start pose and a
chain of goals with hold times. The robot first moves to the start pose with Ruckig, then the policy
runs the chain. Each goal is scored as in sim (reached: every arm joint within 1 deg and every hand
joint within 3 deg at the end of its hold) and printed next to the sim's result.

    # no hardware: MuJoCo arm (aioflexiv's bundled Rizon, no hand on it) + a first-order fake hand
    python deploy/joint_reach/run_real.py --arm mujoco --hand fake --ckpt <ckpt>.pth --sequence <seq>.pt

    # hardware, hand connected but never enabled (aiowuji dry_run), arm live
    python deploy/joint_reach/run_real.py --arm <ROBOT_SN> --hand dry_run ...

    # hardware, both live; --num_goals 3 runs only the first three goals
    python deploy/joint_reach/run_real.py --arm <ROBOT_SN> --hand <HAND_SN> --num_goals 3 ...

Install (the robot PC; Python >= 3.12 for aioflexiv, no Isaac Sim):
    git submodule update --init third_party/aioflexiv third_party/aiowuji
    pip install torch numpy pyyaml omegaconf gym==0.23.1 -e rl_games -e third_party/aioflexiv -e third_party/aiowuji
and copy the checkpoint and real_test_sequence.pt over (outputs/ and docs/results/ are not in git).
Check the install with check_offline.py first.

--mode replay sends the targets the sim recorded instead of running the policy (open loop), to tell a
driver/plant mismatch from a policy problem. Each run saves every step to --log_dir.

Hand gains: the sim's are the vendor MJCF servo's, in Nm/rad, while the firmware takes A/rad and its
torque constant is not identified (see the --hand_kp_per_nm help). The policy was trained with the
hand's stiffness randomized x0.5-5 and damping x0.5-4 around the sim's.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from deploy.joint_reach.core import DEFAULT_SPEC, ActionPipeline, GoalSequence, Policy, Spec, build_obs  # noqa: E402


class Abort(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------------- IO


class FakeJoints:
    """Joints that follow their target as a first-order lag: a no-hardware stand-in."""

    def __init__(self, q0: np.ndarray, tau_s: float) -> None:
        self.q, self.target, self.tau = q0.copy(), q0.copy(), tau_s
        self._t = time.perf_counter()

    async def start(self) -> None:
        self._t = time.perf_counter()

    async def stop(self) -> None:
        pass

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        now = time.perf_counter()
        dt, self._t = max(now - self._t, 1e-6), now
        step = (self.target - self.q) * (1.0 - math.exp(-dt / self.tau))
        self.q = self.q + step
        return self.q.copy(), step / dt

    def command(self, q: np.ndarray) -> None:
        self.target = q.copy()

    async def move_to(self, q: np.ndarray) -> None:
        self.q, self.target = q.copy(), q.copy()


class ArmIO:
    """aioflexiv FlexivController in joint impedance mode."""

    def __init__(self, robot: str, kp: np.ndarray, kd: np.ndarray, q_start: np.ndarray, viewer: bool) -> None:
        from aioflexiv import FlexivController

        kwargs = {}
        if robot == "mujoco":
            kwargs = dict(mujoco_viewer=viewer, mujoco_initial_qpos=q_start)
        self.c = FlexivController(robot, **kwargs)
        self.kp, self.kd = kp, kd

    async def start(self) -> None:
        await self.c.start()
        with self.c.state_lock:
            self.c.kp, self.c.kd = self.kp.copy(), self.kd.copy()

    async def stop(self) -> None:
        await self.c.stop()

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        s = self.c.state
        if not self.c.running:
            raise Abort("arm torque loop stopped")
        return np.asarray(s["qpos"], float).copy(), np.asarray(s["qvel"], float).copy()

    def command(self, q: np.ndarray) -> None:
        self.c.set_now("q_desired", q)

    async def move_to(self, q: np.ndarray, max_velocity: float) -> None:
        await self.c.move(q, max_velocity=max_velocity, max_acceleration=2.0 * max_velocity,
                          max_jerk=10.0 * max_velocity)
        with self.c.state_lock:  # move() keeps the gains, but make sure
            self.c.kp, self.c.kd = self.kp.copy(), self.kd.copy()


class HandIO:
    """aiowuji WujiController in impedance (firmware PD) mode.

    The SDK's joint velocity is not trusted (aiowuji: it can read >0.5 rad/s on a still joint), so
    by default the velocity is the backward difference of the position between policy steps.
    """

    def __init__(self, sn: str | None, kp, kd, effort_limit: float, dry_run: bool, vel: str) -> None:
        from aiowuji import WujiController

        self.c = WujiController(sn, kp=kp, kd=kd, effort_limit=effort_limit, dry_run=dry_run)
        self.vel = vel
        self._last = None

    async def start(self) -> None:
        await self.c.start()

    async def stop(self) -> None:
        await self.c.stop()

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        if self.c.error is not None:
            raise Abort(f"hand stopped: {self.c.error}")
        s = self.c.state
        q, now = np.asarray(s["qpos"], float).copy(), time.perf_counter()
        if self.vel == "sdk":
            qd = np.asarray(s["qvel"], float).copy()
        elif self._last is None:
            qd = np.zeros_like(q)
        else:
            qd = (q - self._last[0]) / max(now - self._last[1], 1e-4)
        self._last = (q, now)
        return q, qd

    def command(self, q: np.ndarray) -> None:
        self.c.set_now("q_desired", q)

    async def move_to(self, q: np.ndarray, max_velocity: float) -> None:
        await self.c.move(q, max_velocity=max_velocity)
        self._last = None


# ---------------------------------------------------------------------------------------------- main


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="joint-reach policy checkpoint (.pth)")
    p.add_argument("--sequence", required=True, help="real_test_sequence.pt from joint_reach_eval")
    p.add_argument("--spec", default=str(DEFAULT_SPEC))
    p.add_argument("--policy_config", default=None, help="default: pretrained_policy/config.yaml")
    p.add_argument("--arm", required=True, help="robot serial, 'mujoco', or 'fake'")
    p.add_argument("--hand", required=True, help="hand serial ('auto': the only one found), 'dry_run', or 'fake'")
    p.add_argument("--hand_sn", default=None, help="hand serial for --hand dry_run")
    p.add_argument("--mode", choices=("policy", "replay"), default="policy")
    p.add_argument("--num_goals", type=int, default=None, help="run only the first N goals")
    p.add_argument("--device", default="cpu", help="policy device")
    p.add_argument("--arm_gain_scale", type=float, default=1.0, help="x the sim's arm kp and kd")
    p.add_argument("--hand_kp_per_nm", type=float, default=2.0,
                   help="hand kp [A/rad] = this x sim kp [Nm/rad]. 2.0 is Wuji's own sim2real deploy "
                        "convention (wuji-isaaclab: kp_A = 2 x sim kp, kd x sqrt 2); aiowuji's default "
                        "torque constant (MJCF force range / 2 A, firmware 1.95x) would give ~0.5-3.5")
    p.add_argument("--hand_kd_per_nm", type=float, default=math.sqrt(2.0),
                   help="hand kd [A s/rad] = this x sim kd [Nm s/rad]")
    p.add_argument("--hand_effort_limit", type=float, default=0.5, help="A (aiowuji default 0.5, max 1.5)")
    p.add_argument("--hand_vel", choices=("diff", "sdk"), default="diff")
    p.add_argument("--move_speed", type=float, default=0.3, help="rad/s for the move to the start pose")
    p.add_argument("--settle_s", type=float, default=1.0, help="hold at the start pose before the policy")
    p.add_argument("--max_arm_track_err", type=float, default=0.15,
                   help="rad; abort if an arm joint lags its target by more (sim: < 0.02)")
    p.add_argument("--viewer", action="store_true", help="MuJoCo viewer for --arm mujoco")
    p.add_argument("--log_dir", default="deploy/logs")
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    return p.parse_args()


async def run(args: argparse.Namespace) -> int:
    spec = Spec.load(args.spec)
    seq = GoalSequence.load(args.sequence, args.num_goals)
    seq.check(spec)
    n, na = spec.num_joints, spec.n_arm
    goal_at = seq.goal_at()
    n_steps = len(goal_at)
    rec = None
    import torch

    rec = torch.load(args.sequence, map_location="cpu", weights_only=False)
    if args.mode == "replay":
        replay = rec["sim_trajectory"]["target"].double().numpy()[:n_steps]

    arm_kp = spec.stiffness[:na] * args.arm_gain_scale
    arm_kd = spec.damping[:na] * args.arm_gain_scale
    hand_kp = spec.stiffness[na:] * args.hand_kp_per_nm
    hand_kd = spec.damping[na:] * args.hand_kd_per_nm

    print(f"[jr] {len(seq.goals)} goals, {n_steps} steps = {n_steps * spec.dt:.0f} s at {1 / spec.dt:.0f} Hz, mode {args.mode}")
    print(f"[jr] start pose arm [deg] {np.round(np.degrees(seq.start[:na]), 1).tolist()}")
    far = np.abs(seq.goals[:, :na] - seq.start[:na]).max()
    print(f"[jr] farthest goal from the start pose: {np.degrees(far):.1f} deg on one arm joint")
    print(f"[jr] arm kp {np.round(arm_kp, 1).tolist()} Nm/rad, kd {np.round(arm_kd, 2).tolist()} Nm s/rad")
    print(f"[jr] hand kp {np.round(hand_kp, 3).tolist()} A/rad")
    print(f"[jr] hand kd {np.round(hand_kd, 4).tolist()} A s/rad, effort limit {args.hand_effort_limit} A, "
          f"velocity from {args.hand_vel}")
    if not args.yes and (args.arm not in ("mujoco", "fake") or args.hand not in ("fake", "dry_run")):
        if input("[jr] hardware will move. Type 'go' to start: ").strip() != "go":
            print("[jr] not started")
            return 1

    policy = None
    if args.mode == "policy":
        kw = {} if args.policy_config is None else {"policy_config": args.policy_config}
        policy = Policy(args.ckpt, num_obs=4 * n, num_actions=n, device=args.device, **kw)
        policy.act(build_obs(spec, seq.start, np.zeros(n), seq.start, seq.start))  # warm up
        policy.reset()

    if args.arm == "fake":
        arm = FakeJoints(seq.start[:na], 0.05)
    else:
        arm = ArmIO(args.arm, arm_kp, arm_kd, seq.start[:na], args.viewer)
    if args.hand == "fake":
        hand = FakeJoints(seq.start[na:], 0.05)
    else:
        sn = args.hand_sn if args.hand == "dry_run" else (None if args.hand == "auto" else args.hand)
        hand = HandIO(sn, hand_kp, hand_kd, args.hand_effort_limit, args.hand == "dry_run", args.hand_vel)

    log = {k: np.zeros((n_steps, n)) for k in ("obs_q", "obs_qd", "goal", "action", "target")}
    log["obs"] = np.zeros((n_steps, 4 * n))
    log["t_read"], log["t_sent"] = np.zeros(n_steps), np.zeros(n_steps)
    seg_end = np.cumsum(seq.hold_steps) - 1
    results = []
    done_steps = 0

    async with contextlib.AsyncExitStack() as stack:
        await arm.start()
        stack.push_async_callback(arm.stop)
        await hand.start()
        stack.push_async_callback(hand.stop)

        print("[jr] moving to the start pose")
        if isinstance(arm, FakeJoints):
            await arm.move_to(seq.start[:na])
        else:
            await arm.move_to(seq.start[:na], args.move_speed)
        if isinstance(hand, FakeJoints):
            await hand.move_to(seq.start[na:])
        else:
            await hand.move_to(seq.start[na:], 0.5)
        await asyncio.sleep(args.settle_s)
        arm.read(), hand.read()  # prime the hand's velocity difference

        pipe = ActionPipeline(spec, seq.start)
        period, t_next, overruns = spec.dt, time.perf_counter(), 0
        print("[jr] running" + ("" if args.mode == "policy" else " (replaying sim targets)"))
        t_start = time.perf_counter()
        try:
            for t in range(n_steps):
                t_read = time.perf_counter()
                q_arm, qd_arm = arm.read()
                q_hand, qd_hand = hand.read()
                q, qd = np.concatenate([q_arm, q_hand]), np.concatenate([qd_arm, qd_hand])
                obs = build_obs(spec, q, qd, pipe.prev, goal_at[t])
                if policy is not None:
                    action = await asyncio.to_thread(policy.act, obs)
                    target = pipe.step(action)
                else:
                    action = np.zeros(n)
                    target = replay[t].copy()
                    pipe.prev = target.copy()
                lag = np.abs(target[:na] - q_arm).max()
                if lag > args.max_arm_track_err:
                    raise Abort(f"arm lags its target by {np.degrees(lag):.1f} deg (limit "
                                f"{np.degrees(args.max_arm_track_err):.1f})")
                arm.command(target[:na])
                hand.command(target[na:])
                t_sent = time.perf_counter()
                for k, v in (("obs_q", q), ("obs_qd", qd), ("goal", goal_at[t]), ("action", action),
                             ("target", target), ("obs", obs)):
                    log[k][t] = v
                log["t_read"][t], log["t_sent"][t] = t_read - t_start, t_sent - t_start
                done_steps = t + 1

                k = int(np.searchsorted(seg_end, t))
                if t == seg_end[k]:
                    # Score the goal with the state at the end of its hold (the next read is ~now).
                    qa, _ = arm.read()
                    qh, _ = hand.read()
                    err = np.abs(seq.goals[k] - np.concatenate([qa, qh]))
                    ea, eh = np.degrees(err[:na].max()), np.degrees(err[na:].max())
                    ok = ea < np.degrees(spec.arm_tol) and eh < np.degrees(spec.hand_tol)
                    sim = rec.get("sim_segments", {}) if rec is not None else {}
                    sim_txt = ""
                    if "arm_err_deg" in sim and k < len(sim["arm_err_deg"]):
                        sim_txt = (f"   sim: arm {float(sim['arm_err_deg'][k]):.2f} hand "
                                   f"{float(sim['hand_err_deg'][k]):.2f} {'ok' if float(sim['reached'][k]) else 'MISS'}")
                    worst = spec.joint_names[int(np.argmax(err))]
                    print(f"[jr] goal {k + 1:2d}/{len(seq.goals)}: arm {ea:5.2f} deg, hand {eh:5.2f} deg "
                          f"(worst {worst}) {'ok' if ok else 'MISS'}{sim_txt}", flush=True)
                    results.append({"goal": k, "arm_err_deg": ea, "hand_err_deg": eh, "reached": bool(ok),
                                    "joint_err_deg": np.degrees(err).tolist()})

                t_next += period
                wait = t_next - time.perf_counter()
                if wait > 0:
                    await asyncio.sleep(wait)
                else:
                    overruns += 1
                    if -wait > period:  # more than a whole step behind: drop it, don't burst
                        t_next = time.perf_counter()
        except Abort as e:
            print(f"[jr] ABORT: {e}", flush=True)
        except (KeyboardInterrupt, asyncio.CancelledError):
            print("[jr] interrupted", flush=True)
        finally:
            with contextlib.suppress(Exception):
                qa, _ = arm.read()
                qh, _ = hand.read()
                arm.command(qa)
                hand.command(qh)
            dts = np.diff(log["t_read"][:done_steps]) * 1e3
            timing = {"steps": done_steps, "overruns": overruns}
            if len(dts):
                timing.update(period_ms_median=float(np.median(dts)), period_ms_max=float(dts.max()),
                              read_to_send_ms_max=float(((log["t_sent"] - log["t_read"])[:done_steps] * 1e3).max()))
            print(f"[jr] timing {json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in timing.items()})}")
            if results:
                print(f"[jr] reached {sum(r['reached'] for r in results)}/{len(results)} goals")
            out = Path(args.log_dir)
            out.mkdir(parents=True, exist_ok=True)
            path = out / f"jr_{args.mode}_{time.strftime('%Y%m%d_%H%M%S')}.npz"
            meta = {"args": vars(args), "timing": timing, "results": results, "joint_names": spec.joint_names,
                    "arm_kp": arm_kp.tolist(), "arm_kd": arm_kd.tolist(), "hand_kp_A": hand_kp.tolist(),
                    "hand_kd_A": hand_kd.tolist()}
            np.savez_compressed(path, **{k: v[:done_steps] for k, v in log.items()}, meta=json.dumps(meta))
            print(f"[jr] log {path}")
    return 0


def main() -> int:
    return asyncio.run(run(_parse()))


if __name__ == "__main__":
    raise SystemExit(main())
