from __future__ import annotations

import asyncio
from pathlib import Path
import threading
import time

import numpy as np

from .robot import (
    DEFAULT_MUJOCO_TIMESTEP,
    DEFAULT_MUJOCO_SITE_NAME,
    AUTO_MUJOCO_TOOL_PAYLOAD,
    FlexivRobotInterface,
    MujocoRobotInterface,
    is_mujoco_robot_sn,
)


# Flexiv's platform-defined Home target is only exposed through PLAN-Home /
# primitive Home. This numeric fallback mirrors the repeated RDK example
# refJntPos posture used for Cartesian motions.
DEFAULT_HOME_QPOS = np.deg2rad(np.array([0.0, -40.0, 0.0, 90.0, 0.0, 40.0, 0.0]))


def _as_vector(value, size: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape == ():
        return np.ones(size) * float(arr)
    if arr.shape != (size,):
        raise ValueError(f"{name} must have shape {(size,)}, got {arr.shape}")
    return arr


def _rotation_error_vector(R_goal: np.ndarray, R_current: np.ndarray) -> np.ndarray:
    R_err = R_goal @ R_current.T
    cos_angle = np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0)
    angle = float(np.arccos(cos_angle))
    skew_vec = np.array(
        [
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1],
        ]
    )
    if angle < 1e-6:
        return 0.5 * skew_vec
    return skew_vec * (angle / (2.0 * np.sin(angle)))


def _damped_pinv(matrix: np.ndarray, *, damping: float, rcond: float) -> np.ndarray:
    damping = float(damping)
    rcond = float(rcond)
    if damping < 0.0:
        raise ValueError("damping must be non-negative")
    if rcond < 0.0:
        raise ValueError("rcond must be non-negative")

    u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
    if singular_values.size == 0:
        return np.zeros(matrix.T.shape)

    cutoff = rcond * singular_values[0]
    damping_sq = damping**2
    inv_singular_values = np.zeros_like(singular_values)
    keep = singular_values > cutoff
    if damping_sq == 0.0:
        inv_singular_values[keep] = 1.0 / singular_values[keep]
    else:
        kept = singular_values[keep]
        inv_singular_values[keep] = kept / (kept * kept + damping_sq)
    return (vh.T * inv_singular_values) @ u.T


class FlexivController:
    """Async Python-owned torque controller for Flexiv robots.

    Supported modes:
    - `impedance`: joint-space spring damper
    - `osc`: operational-space control using MuJoCo model data by default
    - `torque`: direct user torque
    """

    def __init__(
        self,
        robot: str | FlexivRobotInterface | MujocoRobotInterface | None = None,
        *,
        link_name: str = "flange",
        tool: str | None = None,
        enable_gravity_comp: bool = True,
        enable_soft_limits: bool = True,
        friction_comp_scale: float = 100.0,
        ext_offset: bool = True,
        clip: bool = True,
        torque_diff_limit: float | None = None,
        auto_clear_fault: bool = True,
        fault_clear_timeout_sec: int = 30,
        operational_timeout_sec: int = 30,
        model_backend: str = "mujoco",
        mujoco_model_path: str | Path | None = None,
        mujoco_site_name: str | None = DEFAULT_MUJOCO_SITE_NAME,
        mujoco_body_name: str = "link7",
        mujoco_velocity_source: str = "dtheta",
        mujoco_viewer: bool | None = None,
        mujoco_realtime: bool = True,
        mujoco_timestep: float = DEFAULT_MUJOCO_TIMESTEP,
        mujoco_initial_qpos: np.ndarray | list[float] | None = None,
        mujoco_tool_payload=AUTO_MUJOCO_TOOL_PAYLOAD,
    ) -> None:
        mujoco_tool_payload_auto = (
            isinstance(mujoco_tool_payload, str)
            and mujoco_tool_payload == AUTO_MUJOCO_TOOL_PAYLOAD
        )
        if isinstance(robot, (FlexivRobotInterface, MujocoRobotInterface)):
            self.robot = robot
            if tool is not None:
                self.robot.tool = tool
        elif is_mujoco_robot_sn(robot):
            self.robot = MujocoRobotInterface(
                robot,
                link_name=link_name,
                tool=tool,
                enable_gravity_comp=enable_gravity_comp,
                enable_soft_limits=enable_soft_limits,
                friction_comp_scale=friction_comp_scale,
                verbose=False,
                model_backend=model_backend,
                mujoco_model_path=mujoco_model_path,
                mujoco_site_name=mujoco_site_name,
                mujoco_body_name=mujoco_body_name,
                mujoco_velocity_source=mujoco_velocity_source,
                mujoco_viewer=True if mujoco_viewer is None else mujoco_viewer,
                mujoco_realtime=mujoco_realtime,
                mujoco_timestep=mujoco_timestep,
                mujoco_initial_qpos=mujoco_initial_qpos,
                mujoco_tool_payload=None
                if mujoco_tool_payload_auto
                else mujoco_tool_payload,
            )
        else:
            self.robot = FlexivRobotInterface(
                robot,
                link_name=link_name,
                tool=tool,
                enable_gravity_comp=enable_gravity_comp,
                enable_soft_limits=enable_soft_limits,
                friction_comp_scale=friction_comp_scale,
                auto_clear_fault=auto_clear_fault,
                fault_clear_timeout_sec=fault_clear_timeout_sec,
                operational_timeout_sec=operational_timeout_sec,
                model_backend=model_backend,
                mujoco_model_path=mujoco_model_path,
                mujoco_site_name=mujoco_site_name,
                mujoco_body_name=mujoco_body_name,
                mujoco_velocity_source=mujoco_velocity_source,
                mujoco_tool_payload=mujoco_tool_payload,
            )

        self.state_lock = threading.Lock()
        self.type = "impedance"
        self.ext_offset = bool(ext_offset)
        self.running = False
        self.task: asyncio.Task | None = None
        self.state: dict[str, np.ndarray] | None = None

        self.kp = np.ones(7) * 80.0
        self.kd = np.ones(7) * 4.0
        self.ee_kp = np.array([150.0, 150.0, 150.0, 20.0, 20.0, 20.0])
        self.ee_kd = np.array([20.0, 20.0, 20.0, 3.0, 3.0, 3.0])
        self.null_kp = np.ones(7) * 5.0
        self.null_kd = np.ones(7) * 1.0
        self.osc_pinv_damping = 1e-3
        self.osc_pinv_rcond = 1e-6
        self.torque = np.zeros(7)

        self.clip = bool(clip)
        self.torque_diff_limit = torque_diff_limit
        self.torque_limit: np.ndarray | None = None

        self.initial_qpos: np.ndarray | None = None
        self.initial_ee: np.ndarray | None = None
        self.q_desired: np.ndarray | None = None
        self.ee_desired: np.ndarray | None = None

        self._update_freq = 50.0
        self._last_update_time: dict[str, float] = {}
        self._last_loop_time: float | None = None
        self._last_commanded_torque: np.ndarray | None = None
        self.tau_ext_offset: np.ndarray | None = None

    @property
    def dof(self) -> int:
        return self.robot.dof

    def _sync_shapes(self) -> None:
        dof = self.dof
        self.kp = _as_vector(self.kp, dof, "kp")
        self.kd = _as_vector(self.kd, dof, "kd")
        self.null_kp = _as_vector(self.null_kp, dof, "null_kp")
        self.null_kd = _as_vector(self.null_kd, dof, "null_kd")
        self.torque = _as_vector(self.torque, dof, "torque")
        self.torque_limit = _as_vector(self.robot.torque_limit, dof, "torque_limit")
        self._last_commanded_torque = np.zeros(dof)
        if self.tau_ext_offset is None:
            self.tau_ext_offset = np.zeros(dof)

    def initialize(self) -> None:
        state = self.robot.state
        self.state = state
        self.initial_qpos = state["qpos"].copy()
        self.initial_ee = state["ee"].copy()
        self.q_desired = self.initial_qpos.copy()
        self.ee_desired = self.initial_ee.copy()

    def _capture_tau_ext_offset(self) -> None:
        if self.state is None:
            raise RuntimeError("Controller state has not been initialized")
        tau_ext = _as_vector(self.state["tau_ext"], self.dof, "tau_ext")
        with self.state_lock:
            self.tau_ext_offset = tau_ext.copy()

    async def start(self) -> asyncio.Task:
        self.robot.start()
        self._sync_shapes()
        self.initialize()
        self._capture_tau_ext_offset()
        self.running = True
        self._last_loop_time = time.perf_counter()
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self._run())
        await asyncio.sleep(0.1)
        return self.task

    async def stop(self) -> None:
        self.running = False
        if self.task is not None:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        self.robot.stop()

    async def _run(self) -> None:
        try:
            while self.running:
                self.step()
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            raise
        except Exception:
            self.running = False
            try:
                self.robot.stop()
            finally:
                raise

    def switch(self, controller_type: str) -> None:
        if controller_type not in {"impedance", "osc", "torque"}:
            raise ValueError(f"Unknown controller type: {controller_type}")
        self.type = controller_type
        if self.robot.started:
            self.initialize()
        self._last_update_time.clear()

    def set_freq(self, freq: float) -> None:
        self._update_freq = float(freq)

    async def set(self, attr: str, value) -> None:
        current_time = time.perf_counter()
        dt = 1.0 / self._update_freq

        if attr not in self._last_update_time:
            with self.state_lock:
                setattr(self, attr, np.asarray(value, dtype=float))
            self._last_update_time[attr] = current_time + dt
            await asyncio.sleep(dt)
            return

        target_time = self._last_update_time[attr] + dt
        with self.state_lock:
            setattr(self, attr, np.asarray(value, dtype=float))

        sleep_time = target_time - current_time
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)
        self._last_update_time[attr] = target_time

    def set_now(self, attr: str, value) -> None:
        with self.state_lock:
            setattr(self, attr, np.asarray(value, dtype=float))

    def step(self) -> None:
        now = time.perf_counter()
        dt = 1e-3 if self._last_loop_time is None else max(now - self._last_loop_time, 1e-4)
        self._last_loop_time = now

        if self.type == "impedance":
            state = self.robot.state_minimal()
            tau = self._impedance_torque(state)
        elif self.type == "osc":
            state = self.robot.state
            tau = self._osc_torque(state)
        elif self.type == "torque":
            state = self.robot.state_minimal()
            with self.state_lock:
                tau = np.asarray(self.torque, dtype=float).copy()
        else:
            raise ValueError(f"Unknown controller type: {self.type}")

        controller_tau = tau.copy()
        tau = self._apply_ext_offset(tau)
        tau = self._clip_torque(tau, dt)
        self.robot.step(tau)
        state["controller_torque"] = controller_tau
        state["last_torque"] = tau.copy()
        self.state = state

    def _apply_ext_offset(self, tau: np.ndarray) -> np.ndarray:
        with self.state_lock:
            enabled = self.ext_offset
            offset = (
                np.zeros_like(tau)
                if self.tau_ext_offset is None
                else np.asarray(self.tau_ext_offset, dtype=float).copy()
            )
        if not enabled:
            return np.asarray(tau, dtype=float)
        return np.asarray(tau, dtype=float) - offset

    def _clip_torque(self, tau: np.ndarray, dt: float) -> np.ndarray:
        tau = np.asarray(tau, dtype=float)
        if not self.clip:
            self._last_commanded_torque = tau.copy()
            return tau

        last = (
            np.zeros_like(tau)
            if self._last_commanded_torque is None
            else self._last_commanded_torque
        )
        if self.torque_diff_limit is not None:
            torque_diff_limit = float(self.torque_diff_limit)
            if torque_diff_limit <= 0:
                raise ValueError("torque_diff_limit must be positive or None")
            max_delta = torque_diff_limit * dt
            tau = last + np.clip(tau - last, -max_delta, max_delta)
        if self.torque_limit is not None:
            tau = np.clip(tau, -self.torque_limit, self.torque_limit)
        self._last_commanded_torque = tau.copy()
        return tau

    def _impedance_torque(self, state: dict[str, np.ndarray]) -> np.ndarray:
        q = state["qpos"]
        dq = state["qvel"]
        with self.state_lock:
            q_goal = np.asarray(self.q_desired, dtype=float).copy()
            kp = np.asarray(self.kp, dtype=float).copy()
            kd = np.asarray(self.kd, dtype=float).copy()
        return kp * (q_goal - q) - kd * dq

    def _osc_torque(self, state: dict[str, np.ndarray]) -> np.ndarray:
        q = state["qpos"]
        dq = state["qvel"]
        ee = state["ee"]
        jac = state["jac"]
        mm = state["mm"]
        ee_vel = jac @ dq

        with self.state_lock:
            ee_goal = np.asarray(self.ee_desired, dtype=float).copy()
            ee_kp = np.asarray(self.ee_kp, dtype=float).copy()
            ee_kd = np.asarray(self.ee_kd, dtype=float).copy()
            null_kp = np.asarray(self.null_kp, dtype=float).copy()
            null_kd = np.asarray(self.null_kd, dtype=float).copy()
            q0 = np.asarray(self.initial_qpos, dtype=float).copy()
            osc_pinv_damping = float(self.osc_pinv_damping)
            osc_pinv_rcond = float(self.osc_pinv_rcond)

        twist_error = np.zeros(6)
        twist_error[:3] = ee_goal[:3, 3] - ee[:3, 3]
        twist_error[3:] = _rotation_error_vector(ee_goal[:3, :3], ee[:3, :3])

        minv = np.linalg.pinv(mm)
        mx_inv = jac @ minv @ jac.T
        mx = _damped_pinv(
            mx_inv,
            damping=osc_pinv_damping,
            rcond=osc_pinv_rcond,
        )

        task_cmd = ee_kp * twist_error - ee_kd * ee_vel
        wrench = mx @ task_cmd
        feedback = jac.T @ wrench

        jbar = minv @ jac.T @ mx
        null_projector = np.eye(self.dof) - jac.T @ jbar.T
        null_cmd = null_kp * (q0 - q) - null_kd * dq
        null = null_projector @ null_cmd

        return feedback + null

    async def move(
        self,
        qpos=None,
        *,
        freq: float = 50.0,
        max_velocity=1.0,
        max_acceleration=2.0,
        max_jerk=5.0,
        minimum_duration: float | None = None,
    ) -> None:
        """Move to a joint target using Ruckig and the Python torque loop.

        This stays inside the active `FlexivController`: it switches to the
        Python joint-impedance controller, asks Ruckig for a smooth joint-space
        trajectory, and updates `q_desired` at `freq` Hz while the background
        1 kHz-ish torque loop tracks it.
        """
        if not self.running:
            raise RuntimeError("Call await controller.start() before move()")
        if freq <= 0:
            raise ValueError("freq must be positive")
        from ruckig import InputParameter, Result, Ruckig, Trajectory

        self.switch("impedance")
        self.set_freq(freq)
        state = self.state if self.state is not None else self.robot.state_minimal()
        start = np.asarray(state["qpos"], dtype=float)
        start_vel = np.asarray(state["qvel"], dtype=float)
        goal = (
            DEFAULT_HOME_QPOS.copy()
            if qpos is None
            else np.asarray(qpos, dtype=float)
        )

        if goal.shape != (self.dof,):
            raise ValueError(f"qpos must have shape {(self.dof,)}, got {goal.shape}")

        inp = InputParameter(self.dof)
        inp.current_position = start.tolist()
        inp.current_velocity = start_vel.tolist()
        inp.current_acceleration = np.zeros(self.dof).tolist()
        inp.target_position = goal.tolist()
        inp.target_velocity = np.zeros(self.dof).tolist()
        inp.target_acceleration = np.zeros(self.dof).tolist()
        inp.max_velocity = _as_vector(max_velocity, self.dof, "max_velocity").tolist()
        inp.max_acceleration = _as_vector(
            max_acceleration, self.dof, "max_acceleration"
        ).tolist()
        inp.max_jerk = _as_vector(max_jerk, self.dof, "max_jerk").tolist()
        if minimum_duration is not None:
            inp.minimum_duration = float(minimum_duration)

        otg = Ruckig(self.dof)
        trajectory = Trajectory(self.dof)
        result = otg.calculate(inp, trajectory)
        if result not in (Result.Working, Result.Finished):
            raise RuntimeError(f"Ruckig failed to generate trajectory: {result}")

        steps = max(int(np.ceil(trajectory.duration * freq)), 1)
        for i in range(steps):
            t = min((i + 1) / freq, trajectory.duration)
            q_desired, _, _ = trajectory.at_time(t)
            await self.set("q_desired", np.asarray(q_desired, dtype=float))

        await self.set("q_desired", goal)
