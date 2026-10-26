"""Move the Flexiv arm to the joint-reach training default pose (the spec's default_q), then hold it.

aioflexiv's own home (DEFAULT_HOME_QPOS, [0, -40, 0, 90, 0, 40, 0] deg) is the Flexiv RDK example
posture, not the pose the policy was trained around. This moves to default_q from spec_*.json with
the same joint impedance gains and Ruckig move run_real.py uses for its start pose. The hand is not
touched.

    python deploy/joint_reach/move_home.py --arm mujoco --viewer   # check the pose in MuJoCo first
    python deploy/joint_reach/move_home.py --arm <ROBOT_SN>
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from deploy.joint_reach.core import DEFAULT_SPEC, Spec  # noqa: E402
from deploy.joint_reach.run_real import ArmIO  # noqa: E402


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, help="robot serial or 'mujoco'")
    p.add_argument("--spec", default=str(DEFAULT_SPEC))
    p.add_argument("--move_speed", type=float, default=0.2, help="rad/s, the fastest joint")
    p.add_argument("--hold_s", type=float, default=2.0, help="hold at the pose before stopping")
    p.add_argument("--arm_gain_scale", type=float, default=1.0, help="x the sim's arm kp and kd")
    p.add_argument("--viewer", action="store_true", help="MuJoCo viewer for --arm mujoco")
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    return p.parse_args()


async def run(args: argparse.Namespace) -> int:
    spec = Spec.load(args.spec)
    na = spec.n_arm
    home = spec.default_q[:na].copy()
    if np.any(home < spec.lower[:na]) or np.any(home > spec.upper[:na]):
        raise ValueError("default_q is outside the spec's arm joint limits")
    arm = ArmIO(args.arm, spec.stiffness[:na] * args.arm_gain_scale, spec.damping[:na] * args.arm_gain_scale,
                home, args.viewer)

    await arm.start()
    try:
        q, _ = arm.read()
        print(f"[home] now    [deg] {np.round(np.degrees(q), 1).tolist()}")
        print(f"[home] target [deg] {np.round(np.degrees(home), 1).tolist()}")
        print(f"[home] largest move {np.degrees(np.abs(home - q).max()):.1f} deg at {args.move_speed} rad/s")
        if not args.yes and args.arm != "mujoco":
            if input("[home] the arm will move. Type 'go' to start: ").strip() != "go":
                print("[home] not moved")
                return 1
        await arm.move_to(home, args.move_speed)
        arm.command(home)
        await asyncio.sleep(args.hold_s)
        q, _ = arm.read()
        print(f"[home] reached, max error {np.degrees(np.abs(home - q).max()):.2f} deg")
    finally:
        await arm.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(_parse())))
