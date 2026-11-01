"""The Flexiv arm in its own process, so the calibration UI's numpy and viser work never takes the GIL from
aioflexiv's Python torque loop (same reason as live_viser.py).

The arm holds its pose in joint impedance with the sim's gains (ArmIO, as run_real.py), or is in free
drive: no stiffness, light damping, the robot's gravity compensation on, so it can be moved by hand.
The hand is not connected: its joints stay at the spec's default_q.

    robot = RobotProcess(spec, arm="<ROBOT_SN>" | "mujoco" | "none")
    robot.q()               # (n,) canonical order: measured arm, hand at default_q
    robot.set_free(True)    # free drive
    robot.set_free(False)   # hold where it is now
    robot.close()
"""

from __future__ import annotations

import multiprocessing as mp
import time

import numpy as np

# shared buffer: [status, free requested, free applied, q (n)]
_ST_STARTING, _ST_READY, _ST_FAILED = 0.0, 1.0, -1.0


class RobotProcess:
    def __init__(self, spec, arm: str, arm_gain_scale: float = 1.0, viewer: bool = False,
                 free_kd_scale: float = 0.2) -> None:
        na = spec.n_arm
        self.can_move = arm != "none"
        ctx = mp.get_context("spawn")
        self._buf = ctx.Array("d", 3 + spec.num_joints, lock=False)
        self._buf[3:] = spec.default_q.tolist()
        self._err = ctx.Queue()
        self._stop = ctx.Event()
        self._proc = ctx.Process(
            target=_serve, daemon=True,
            args=(self._buf, self._err, self._stop, arm, spec.default_q[:na].tolist(),
                  (spec.stiffness[:na] * arm_gain_scale).tolist(), (spec.damping[:na] * arm_gain_scale).tolist(),
                  (spec.damping[:na] * free_kd_scale).tolist(), viewer))
        self._proc.start()
        while self._buf[0] == _ST_STARTING:
            self._check()
            time.sleep(0.05)
        self._check()

    def _check(self) -> None:
        if self._buf[0] == _ST_FAILED or not self._proc.is_alive():
            msg = self._err.get(timeout=1.0) if not self._err.empty() else "robot process died"
            raise RuntimeError(msg)

    def q(self) -> np.ndarray:
        self._check()
        return np.asarray(self._buf[3:], float)

    @property
    def free(self) -> bool:
        return self._buf[2] > 0

    def set_free(self, on: bool) -> None:
        """Free drive on or off (hold the current pose); blocks until applied."""
        if not self.can_move:
            raise RuntimeError("no arm connected (--arm none)")
        self._buf[1] = float(on)
        while self._buf[2] != self._buf[1]:
            self._check()
            time.sleep(0.01)

    def close(self) -> None:
        self._stop.set()
        self._proc.join(timeout=5.0)
        if self._proc.is_alive():
            self._proc.terminate()


def _serve(buf, err, stop, arm: str, q_start, kp, kd, free_kd, viewer: bool) -> None:
    import asyncio
    import sys
    import traceback
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from deploy.joint_reach.run_real import ArmIO

    na = len(kp)

    async def main() -> None:
        if arm == "none":
            buf[0] = _ST_READY
            while not stop.is_set():
                await asyncio.sleep(0.1)
            return
        arm_io = ArmIO(arm, np.asarray(kp), np.asarray(kd), np.asarray(q_start), viewer)
        try:
            await arm_io.start()
            arm_io.command(arm_io.read()[0])  # hold where it is
            buf[0] = _ST_READY
            while not stop.is_set():
                q = arm_io.read()[0]
                buf[3:3 + na] = q.tolist()
                if buf[1] != buf[2]:
                    c = arm_io.c
                    with c.state_lock:  # the target first: back to holding, the arm stays where it is
                        c.q_desired = q.copy()
                        c.kp, c.kd = ((np.zeros(na), np.asarray(free_kd)) if buf[1]
                                      else (arm_io.kp.copy(), arm_io.kd.copy()))
                    buf[2] = buf[1]
                if buf[2]:
                    arm_io.command(q)  # follow the hand, so turning free drive off holds here
                await asyncio.sleep(1 / 60)
        finally:
            await arm_io.stop()

    try:
        asyncio.run(main())
    except BaseException:
        err.put(traceback.format_exc())
        buf[0] = _ST_FAILED
