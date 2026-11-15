from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import deque

from .config import config_path, resolve_robot_sn, save_last_robot_sn
from ._loader import load_rt


def _timestamp_seconds(timestamp) -> float:
    return float(timestamp[0]) + float(timestamp[1]) * 1e-9


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Flexiv RT joint-torque mode with zero user-commanded torque."
    )
    parser.add_argument(
        "robot_sn",
        nargs="?",
        help=(
            "Robot serial number. If omitted, use the latest serial saved in "
            f"{config_path()}."
        ),
    )
    parser.add_argument("--friction-comp-scale", type=float, default=100.0)
    args = parser.parse_args()
    try:
        robot_sn = resolve_robot_sn(args.robot_sn)
    except Exception as exc:
        parser.error(str(exc))

    rt = load_rt()
    ctrl = None
    iterations = 0
    max_abs_drift = 0.0
    wall_dts = deque(maxlen=5000)
    robot_dts = deque(maxlen=5000)

    print(f"Connecting to {robot_sn}")
    print("Switching to RT_JOINT_TORQUE in 3 seconds. Keep clear of the robot.")
    print("After it starts, press Ctrl+C to stop.")
    time.sleep(3.0)

    wall_start = time.perf_counter()
    try:
        ctrl = rt.ActiveTorqueControl(robot_sn, [], False)
        save_last_robot_sn(robot_sn)
        initial = ctrl.states()
        q0 = list(initial.q)
        zero_torque = [0.0] * len(q0)

        last_wall = time.perf_counter()
        last_ts = tuple(initial.timestamp)

        while True:
            state = ctrl.read_once(1000)
            now = time.perf_counter()
            ts = tuple(state.timestamp)

            wall_dts.append(now - last_wall)
            robot_dts.append(_timestamp_seconds(ts) - _timestamp_seconds(last_ts))
            last_wall = now
            last_ts = ts

            drift = max(abs(state.q[i] - q0[i]) for i in range(len(q0)))
            max_abs_drift = max(max_abs_drift, drift)

            ctrl.write_once(zero_torque, True, True, args.friction_comp_scale)
            iterations += 1

    except KeyboardInterrupt:
        print("\nCtrl+C received, stopping robot...")
    finally:
        if ctrl is not None:
            ctrl.stop()

    elapsed = time.perf_counter() - wall_start
    result = {
        "elapsed_s": elapsed,
        "iterations": iterations,
        "max_abs_joint_drift_rad": max_abs_drift,
        "rate_hz_est": iterations / elapsed if elapsed > 0 else None,
        "robot_dt_median_ms": statistics.median(robot_dts) * 1000 if robot_dts else None,
        "robot_dt_max_ms": max(robot_dts) * 1000 if robot_dts else None,
        "wall_dt_median_ms": statistics.median(wall_dts) * 1000 if wall_dts else None,
        "wall_dt_max_ms": max(wall_dts) * 1000 if wall_dts else None,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
