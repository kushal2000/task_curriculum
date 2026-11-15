#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio

import numpy as np

from aioflexiv import FlexivController


async def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run joint impedance on hardware, or pass robot_sn=mujoco to run MuJoCo."
        )
    )
    parser.add_argument("robot_sn", nargs="?")
    args = parser.parse_args()

    controller = FlexivController(args.robot_sn)
    await controller.start()
    try:
        base = np.ones(controller.dof)
        with controller.state_lock:
            controller.kp = base * 80.0
            controller.kd = base * 4.0
            print("Moving to initial position...")
        await controller.move()

        controller.switch("impedance")
        controller.kp = np.ones(controller.dof) * 1024.0
        controller.kd = np.ones(controller.dof) * 24.0
        controller.set_freq(50)

        q0 = controller.initial_qpos.copy()
        while True:
            await controller.set("q_desired", q0)
    finally:
        await controller.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
