#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio

import numpy as np

from aioflexiv import FlexivController


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run OSC hold on hardware, or pass robot_sn=mujoco to run MuJoCo."
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

        controller.switch("osc")
        controller.ee_kp = np.ones(6) * 1024.0
        controller.ee_kd = np.ones(6) * 32.0
        controller.null_kp = np.zeros(controller.dof)
        controller.null_kd = np.zeros(controller.dof)
        controller.set_freq(50)

        target = controller.initial_ee.copy()
        while True:
            await controller.set("ee_desired", target)
    finally:
        await controller.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
