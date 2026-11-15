#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio

import numpy as np

from aioflexiv import FlexivController


async def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Stream zero user torque on hardware, or pass robot_sn=mujoco to run MuJoCo."
        )
    )
    parser.add_argument("robot_sn", nargs="?")
    args = parser.parse_args()

    controller = FlexivController(args.robot_sn, friction_comp_scale=99)
    await controller.start()
    try:
        base = np.ones(controller.dof)
        with controller.state_lock:
            controller.kp = base * 80.0
            controller.kd = base * 4.0
            print("Moving to initial position...")
        await controller.move()

        await asyncio.sleep(3.0)
        print("starting torque hold with zero torque command. Keep clear of the robot.")

        controller.switch("torque")
        controller.set_freq(10)
        zero_torque = np.zeros(controller.dof)
        while True:
            await controller.set("torque", zero_torque)
    finally:
        await controller.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
