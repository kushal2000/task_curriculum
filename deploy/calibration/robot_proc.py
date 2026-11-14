"""The Flexiv arm in its own process, so the calibration UI's numpy and viser work never takes the GIL from
aioflexiv's Python torque loop (same reason as live_viser.py).

The arm holds its pose in joint impedance with the sim's gains (ArmIO, as run_real.py), or is in free
drive: no stiffness, light damping, the robot's gravity compensation on, so it can be moved by hand; a
soft wall before each joint limit keeps it out of Flexiv's abrupt soft limits (deploy/utils/free_drive.py).
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
                 free_kd_scale: float = 0.2, free_limit_margin_deg: float = 10.0,
                 free_ramp_s: float = 0.3) -> None:
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
                  (spec.damping[:na] * free_kd_scale).tolist(), viewer,
                  spec.lower[:na].tolist(), spec.upper[:na].tolist(), float(np.radians(free_limit_margin_deg)),
                  free_ramp_s))
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


def _serve(buf, err, stop, arm: str, q_start, kp, kd, free_kd, viewer: bool, lower, upper, margin, ramp_s) -> None:
    import asyncio
    import sys
    import traceback
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from deploy.joint_reach.run_real import ArmIO
    from deploy.utils.free_drive import blend_gains, wall_gains

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
            c = arm_io.c
            lo = np.maximum(lower, c.robot.info["q_min"])  # the tighter of the spec's and the robot's limits
            hi = np.minimum(upper, c.robot.info["q_max"])
            buf[0] = _ST_READY
            # a: 1 holding, 0 free; it ramps over ramp_s, so the stiffness never comes or goes in one step
            a, q_hold, t_last = 1.0, arm_io.read()[0], time.perf_counter()
            while not stop.is_set():
                q = arm_io.read()[0]
                buf[3:3 + na] = q.tolist()
                now = time.perf_counter()
                da, t_last = (now - t_last) / max(ramp_s, 1e-3), now
                want = 0.0 if buf[1] else 1.0
                if a != want or not want:
                    if want:  # back to holding: the target follows the arm until the stiffness is back
                        a, q_hold = min(a + da, 1.0), q.copy()
                    else:     # to free: the holding torque fades; then the target follows the hand, up to the walls
                        a = max(a - da, 0.0)
                    free = wall_gains(q, lo, hi, margin, arm_io.kp, np.asarray(free_kd), arm_io.kd)[:3]
                    with c.state_lock:
                        c.q_desired, c.kp, c.kd = blend_gains(a, (q_hold, arm_io.kp, arm_io.kd), free)
                if a == want:
                    buf[2] = buf[1]
                await asyncio.sleep(1 / 60)
        finally:
            await arm_io.stop()

    try:
        asyncio.run(main())
    except BaseException:
        err.put(traceback.format_exc())
        buf[0] = _ST_FAILED
