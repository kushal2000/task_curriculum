from __future__ import annotations

from pathlib import Path
from typing import Any
import time
import warnings

import numpy as np

from ._loader import load_rt
from .config import resolve_robot_sn, save_last_robot_sn
from .mujoco_model import (
    DEFAULT_MUJOCO_SITE_NAME,
    MujocoModelBackend,
    default_mujoco_model_path,
    default_mujoco_scene_path,
)
from .tools import coerce_tool_payload, read_active_tool_payload, switch_active_tool


MUJOCO_ROBOT_SN = "mujoco"
DEFAULT_MUJOCO_TIMESTEP = 0.001
AUTO_MUJOCO_TOOL_PAYLOAD = "auto"


def is_mujoco_robot_sn(robot_sn: str | None) -> bool:
    return isinstance(robot_sn, str) and robot_sn.strip().lower() == MUJOCO_ROBOT_SN


class FlexivRobotInterface:
    """Low-level bridge for Flexiv RT joint-torque control.

    This class owns the C++ `ActiveTorqueControl` shim. It does not use
    `flexiv::rdk::Scheduler`; timing is driven by waiting for fresh robot state
    timestamps and immediately writing the next torque command from Python.
    """

    def __init__(
        self,
        robot_sn: str | None = None,
        *,
        link_name: str = "flange",
        tool: str | None = None,
        enable_gravity_comp: bool = True,
        enable_soft_limits: bool = True,
        friction_comp_scale: float = 100.0,
        network_interface_whitelist: list[str] | None = None,
        verbose: bool = False,
        auto_clear_fault: bool = True,
        fault_clear_timeout_sec: int = 30,
        operational_timeout_sec: int = 30,
        model_backend: str = "mujoco",
        mujoco_model_path: str | Path | None = None,
        mujoco_site_name: str | None = DEFAULT_MUJOCO_SITE_NAME,
        mujoco_body_name: str = "link7",
        mujoco_velocity_source: str = "dtheta",
        mujoco_tool_payload: Any = AUTO_MUJOCO_TOOL_PAYLOAD,
    ) -> None:
        self.robot_sn = resolve_robot_sn(robot_sn)
        self.link_name = link_name
        self.tool = tool
        self.enable_gravity_comp = enable_gravity_comp
        self.enable_soft_limits = enable_soft_limits
        self.friction_comp_scale = friction_comp_scale
        self.network_interface_whitelist = network_interface_whitelist or []
        self.verbose = verbose
        self.auto_clear_fault = bool(auto_clear_fault)
        self.fault_clear_timeout_sec = int(fault_clear_timeout_sec)
        self.operational_timeout_sec = int(operational_timeout_sec)
        self.model_backend = str(model_backend).lower()
        if self.model_backend not in {"rdk", "mujoco"}:
            raise ValueError("model_backend must be 'rdk' or 'mujoco'")
        self.mujoco_model_path = (
            None
            if mujoco_model_path is None
            else Path(mujoco_model_path).expanduser()
        )
        self.mujoco_site_name = mujoco_site_name
        self.mujoco_body_name = mujoco_body_name
        self.mujoco_velocity_source = str(mujoco_velocity_source).lower()
        if self.mujoco_velocity_source not in {"dq", "dtheta"}:
            raise ValueError("mujoco_velocity_source must be 'dq' or 'dtheta'")
        self.mujoco_tool_payload = mujoco_tool_payload

        self._rt = None
        self._ctrl = None
        self._info: dict[str, Any] | None = None
        self._last_torque: np.ndarray | None = None
        self._mujoco_backend: MujocoModelBackend | None = None
        self._active_tool_payload = None

    @property
    def started(self) -> bool:
        return self._ctrl is not None

    @property
    def info(self) -> dict[str, Any]:
        if self._info is None:
            raise RuntimeError("Robot interface has not been started")
        return self._info

    @property
    def dof(self) -> int:
        return int(self.info["dof"])

    @property
    def torque_limit(self) -> np.ndarray:
        tau_max = np.asarray(self.info.get("tau_max", []), dtype=float)
        if (
            tau_max.shape == (self.dof,)
            and np.all(np.isfinite(tau_max))
            and np.all(tau_max > 0)
        ):
            return tau_max.copy()
        raise RuntimeError(
            "Flexiv RDK did not report valid RobotInfo.tau_max torque limits; "
            "refusing to use guessed fallback limits."
        )

    def start(self) -> None:
        if self._ctrl is not None:
            return
        if self.tool:
            switch_active_tool(
                self.robot_sn,
                self.tool,
                network_interface_whitelist=self.network_interface_whitelist,
                verbose=self.verbose,
            )
        self._active_tool_payload = self._resolve_mujoco_tool_payload()
        self._rt = load_rt()
        self._ctrl = self._rt.ActiveTorqueControl(
            self.robot_sn,
            self.network_interface_whitelist,
            self.verbose,
            self.auto_clear_fault,
            self.fault_clear_timeout_sec,
            self.operational_timeout_sec,
        )
        self._info = dict(self._ctrl.info())
        self._initialize_model_backend()
        try:
            save_last_robot_sn(self.robot_sn)
        except Exception as exc:
            warnings.warn(
                f"Could not save last robot serial to aioflexiv config: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
        self._last_torque = np.zeros(self.dof)

    def stop(self) -> None:
        if self._ctrl is not None:
            self._ctrl.stop()
            self._ctrl = None
            self._mujoco_backend = None

    def _require_started(self):
        if self._ctrl is None:
            raise RuntimeError("Robot interface has not been started")
        return self._ctrl

    def _initialize_model_backend(self) -> None:
        self._mujoco_backend = None
        if self.model_backend != "mujoco":
            return
        if self._info is None:
            raise RuntimeError("Robot info must be available before model backend setup")
        model_name = self._info.get("model_name")
        if self.mujoco_model_path is None and not model_name:
            warnings.warn(
                "Flexiv RDK RobotInfo.model_name is unavailable; defaulting to the "
                "Rizon4 MuJoCo model. Rebuild the RT extension or pass "
                "mujoco_model_path for explicit model selection.",
                RuntimeWarning,
                stacklevel=2,
            )
        model_path = (
            self.mujoco_model_path
            if self.mujoco_model_path is not None
            else default_mujoco_model_path(str(model_name or ""))
        )
        self._mujoco_backend = MujocoModelBackend(
            model_path,
            site_name=self.mujoco_site_name,
            body_name=self.mujoco_body_name,
            tool_payload=self._active_tool_payload,
        )

    def _resolve_mujoco_tool_payload(self):
        if self.model_backend != "mujoco":
            return None
        if not (
            isinstance(self.mujoco_tool_payload, str)
            and self.mujoco_tool_payload == AUTO_MUJOCO_TOOL_PAYLOAD
        ):
            return coerce_tool_payload(self.mujoco_tool_payload)
        try:
            payload = read_active_tool_payload(
                self.robot_sn,
                network_interface_whitelist=self.network_interface_whitelist,
                verbose=self.verbose,
            )
        except Exception as exc:
            warnings.warn(
                "Could not read active Flexiv tool payload; MuJoCo OSC model "
                f"will omit the payload: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
        return None if payload.is_empty else payload

    def _last_or_raw_torque(self, raw) -> np.ndarray:
        return (
            self._last_torque.copy()
            if self._last_torque is not None
            else np.asarray(raw.tau_des, dtype=float)
        )

    @staticmethod
    def _raw_array(raw, name: str, size: int) -> np.ndarray:
        value = getattr(raw, name, None)
        if value is None:
            return np.zeros(size)
        return np.asarray(value, dtype=float)

    def state_minimal(self, timeout_ms: int = 1000) -> dict[str, np.ndarray]:
        ctrl = self._require_started()
        raw = ctrl.read_once(timeout_ms)
        state = {
            "timestamp": tuple(raw.timestamp),
            "qpos": np.asarray(raw.q, dtype=float),
            "qvel": np.asarray(raw.dtheta, dtype=float),
            "dq": np.asarray(raw.dq, dtype=float),
            "theta": np.asarray(raw.theta, dtype=float),
            "dtheta": np.asarray(raw.dtheta, dtype=float),
            "tau": np.asarray(raw.tau, dtype=float),
            "tau_des": np.asarray(raw.tau_des, dtype=float),
            "tau_ext": np.asarray(raw.tau_ext, dtype=float),
            "last_torque": self._last_or_raw_torque(raw),
        }
        if getattr(self, "model_backend", "rdk") == "mujoco":
            state["model_backend"] = np.asarray(["mujoco"], dtype=object)
            state["mujoco_velocity_source"] = np.asarray(
                [self.mujoco_velocity_source], dtype=object
            )
            if self._mujoco_backend is not None:
                state["mujoco_model_path"] = np.asarray(
                    [str(self._mujoco_backend.path)], dtype=object
                )
                state["mujoco_frame_type"] = np.asarray(
                    [self._mujoco_backend.frame_type], dtype=object
                )
                state["mujoco_frame_name"] = np.asarray(
                    [self._mujoco_backend.frame_name], dtype=object
                )
                state["mujoco_tool_name"] = np.asarray(
                    [self._mujoco_backend.tool_name], dtype=object
                )
                state["mujoco_tool_mass"] = np.asarray(
                    [self._mujoco_backend.tool_mass], dtype=float
                )
        return state

    @property
    def state(self) -> dict[str, np.ndarray]:
        ctrl = self._require_started()
        if getattr(self, "model_backend", "rdk") == "mujoco":
            raw = ctrl.read_once(1000)
            return self._mujoco_state_from_raw(raw)

        raw = dict(ctrl.read_once_full(self.link_name, 1000))
        state = {
            key: np.asarray(value, dtype=float)
            for key, value in raw.items()
            if key != "timestamp"
        }
        state["timestamp"] = tuple(raw["timestamp"])
        state["last_torque"] = (
            self._last_torque.copy()
            if self._last_torque is not None
            else np.asarray(state["tau_des"], dtype=float)
        )
        return state

    def _mujoco_state_from_raw(self, raw) -> dict[str, np.ndarray]:
        backend = self._mujoco_backend
        if backend is None:
            raise RuntimeError("MuJoCo model backend has not been initialized")

        qpos = np.asarray(raw.q, dtype=float)
        dq = np.asarray(raw.dq, dtype=float)
        dtheta = np.asarray(raw.dtheta, dtype=float)
        qvel = dtheta if self.mujoco_velocity_source == "dtheta" else dq
        model_state = backend.state(qpos, qvel)
        state = {
            "timestamp": tuple(raw.timestamp),
            "qpos": qpos,
            "qvel": qvel.copy(),
            "dq": dq,
            "theta": np.asarray(raw.theta, dtype=float),
            "dtheta": dtheta,
            "tau": np.asarray(raw.tau, dtype=float),
            "tau_des": np.asarray(raw.tau_des, dtype=float),
            "tau_ext": np.asarray(raw.tau_ext, dtype=float),
            "tcp_pose": self._raw_array(raw, "tcp_pose", 7),
            "tcp_vel": self._raw_array(raw, "tcp_vel", 6),
            "flange_pose": self._raw_array(raw, "flange_pose", 7),
            "ft_sensor_raw": self._raw_array(raw, "ft_sensor_raw", 6),
            "ext_wrench_in_tcp": self._raw_array(raw, "ext_wrench_in_tcp", 6),
            "ext_wrench_in_world": self._raw_array(raw, "ext_wrench_in_world", 6),
            "ext_wrench_in_tcp_raw": self._raw_array(
                raw, "ext_wrench_in_tcp_raw", 6
            ),
            "ext_wrench_in_world_raw": self._raw_array(
                raw, "ext_wrench_in_world_raw", 6
            ),
            "last_torque": self._last_or_raw_torque(raw),
            "model_backend": np.asarray(["mujoco"], dtype=object),
            "mujoco_velocity_source": np.asarray(
                [self.mujoco_velocity_source], dtype=object
            ),
        }
        state.update(model_state)
        return state

    def step(self, torque: np.ndarray | list[float]) -> None:
        ctrl = self._require_started()
        tau = np.asarray(torque, dtype=float)
        if tau.shape != (self.dof,):
            raise ValueError(f"Expected torque shape {(self.dof,)}, got {tau.shape}")
        ctrl.write_once(
            tau.tolist(),
            self.enable_gravity_comp,
            self.enable_soft_limits,
            self.friction_comp_scale,
        )
        self._last_torque = tau.copy()


class MujocoRobotInterface:
    """MuJoCo-backed robot interface with the same surface as the RT bridge."""

    def __init__(
        self,
        robot_sn: str | None = MUJOCO_ROBOT_SN,
        *,
        link_name: str = "flange",
        tool: str | None = None,
        enable_gravity_comp: bool = True,
        enable_soft_limits: bool = True,
        friction_comp_scale: float = 100.0,
        network_interface_whitelist: list[str] | None = None,
        verbose: bool = False,
        model_backend: str = "mujoco",
        mujoco_model_path: str | Path | None = None,
        mujoco_site_name: str | None = DEFAULT_MUJOCO_SITE_NAME,
        mujoco_body_name: str = "link7",
        mujoco_velocity_source: str = "dtheta",
        mujoco_viewer: bool = True,
        mujoco_realtime: bool = True,
        mujoco_timestep: float = DEFAULT_MUJOCO_TIMESTEP,
        mujoco_initial_qpos: np.ndarray | list[float] | None = None,
        mujoco_tool_payload: Any = None,
    ) -> None:
        if robot_sn is not None and not is_mujoco_robot_sn(robot_sn):
            raise ValueError("MujocoRobotInterface robot_sn must be 'mujoco'")
        self.robot_sn = MUJOCO_ROBOT_SN
        self.link_name = link_name
        self.tool = tool
        self.enable_gravity_comp = bool(enable_gravity_comp)
        self.enable_soft_limits = bool(enable_soft_limits)
        self.friction_comp_scale = float(friction_comp_scale)
        self.network_interface_whitelist = network_interface_whitelist or []
        self.verbose = verbose
        self.model_backend = str(model_backend).lower()
        if self.model_backend != "mujoco":
            raise ValueError("MuJoCo simulation requires model_backend='mujoco'")
        self.mujoco_model_path = (
            None
            if mujoco_model_path is None
            else Path(mujoco_model_path).expanduser()
        )
        self.mujoco_site_name = mujoco_site_name
        self.mujoco_body_name = mujoco_body_name
        self.mujoco_velocity_source = str(mujoco_velocity_source).lower()
        if self.mujoco_velocity_source not in {"dq", "dtheta"}:
            raise ValueError("mujoco_velocity_source must be 'dq' or 'dtheta'")
        self.mujoco_viewer = bool(mujoco_viewer)
        self.mujoco_realtime = bool(mujoco_realtime)
        self.mujoco_timestep = float(mujoco_timestep)
        if self.mujoco_timestep <= 0.0:
            raise ValueError("mujoco_timestep must be positive")
        self.mujoco_initial_qpos = (
            None
            if mujoco_initial_qpos is None
            else np.asarray(mujoco_initial_qpos, dtype=float)
        )
        self.mujoco_tool_payload = coerce_tool_payload(mujoco_tool_payload)

        self._mujoco = None
        self._mujoco_backend: MujocoModelBackend | None = None
        self._viewer = None
        self._info: dict[str, Any] | None = None
        self._last_torque: np.ndarray | None = None
        self._last_total_torque: np.ndarray | None = None
        self._joint_limited: np.ndarray | None = None
        self._last_wall_time: float | None = None
        self._started = False

    @property
    def started(self) -> bool:
        return self._started

    @property
    def info(self) -> dict[str, Any]:
        if self._info is None:
            raise RuntimeError("MuJoCo robot interface has not been started")
        return self._info

    @property
    def dof(self) -> int:
        return int(self.info["dof"])

    @property
    def model(self):
        backend = self._require_started()
        return backend.model

    @property
    def data(self):
        backend = self._require_started()
        return backend.data

    @property
    def torque_limit(self) -> np.ndarray:
        return np.asarray(self.info["tau_max"], dtype=float).copy()

    def start(self) -> None:
        if self._started:
            return

        model_path = (
            self.mujoco_model_path
            if self.mujoco_model_path is not None
            else default_mujoco_scene_path("Rizon4")
        )
        self._mujoco_backend = MujocoModelBackend(
            model_path,
            site_name=self.mujoco_site_name,
            body_name=self.mujoco_body_name,
            tool_payload=self.mujoco_tool_payload,
        )
        self._mujoco = self._mujoco_backend._mujoco
        self._mujoco_backend.model.opt.timestep = self.mujoco_timestep
        self._disable_builtin_actuators()
        # The bundled models set gravcomp="1" on every body. step() compensates gravity
        # itself when enable_gravity_comp is set, as Flexiv does, so turn MuJoCo's off.
        self._mujoco_backend.model.body_gravcomp[:] = 0.0
        self._joint_limited = np.asarray(
            self._mujoco_backend.model.jnt_limited, dtype=np.uint8
        ).copy()
        self._reset_initial_state()

        dof = int(self._mujoco_backend.model.nv)
        tau_max = self._torque_limits()
        q_range = np.asarray(self._mujoco_backend.model.jnt_range[:dof], dtype=float)
        self._info = {
            "serial_num": MUJOCO_ROBOT_SN,
            "software_ver": "mujoco",
            "model_name": self._sim_model_name(),
            "license_type": "simulation",
            "dof": dof,
            "manipulator_dof": dof,
            "q_min": q_range[:, 0].tolist(),
            "q_max": q_range[:, 1].tolist(),
            "dq_max": [float("inf")] * dof,
            "tau_max": tau_max.tolist(),
            "K_q_nom": [0.0] * dof,
            "K_x_nom": [0.0] * 6,
            "has_FT_sensor": False,
            "simulated": True,
            "mujoco_model_path": str(self._mujoco_backend.path),
            "mujoco_timestep": self.mujoco_timestep,
            "mujoco_tool_name": self._mujoco_backend.tool_name,
            "mujoco_tool_mass": self._mujoco_backend.tool_mass,
        }
        self._last_torque = np.zeros(dof)
        self._last_total_torque = np.zeros(dof)
        self._last_wall_time = time.perf_counter()
        self._started = True
        if self.mujoco_viewer:
            self._open_viewer()

    def stop(self) -> None:
        self._close_viewer()
        self._started = False
        self._mujoco = None
        self._mujoco_backend = None
        self._info = None
        self._last_wall_time = None

    def _require_started(self) -> MujocoModelBackend:
        if self._mujoco_backend is None or self._mujoco is None or not self._started:
            raise RuntimeError("MuJoCo robot interface has not been started")
        return self._mujoco_backend

    def _sim_model_name(self) -> str:
        backend = self._require_loaded()
        path_text = str(backend.path).lower()
        if "rizon4s" in path_text:
            return "Rizon4S"
        if "rizon4" in path_text:
            return "Rizon4"
        return backend.path.stem

    def _disable_builtin_actuators(self) -> None:
        backend = self._require_loaded()
        model = backend.model
        data = backend.data
        if model.nu:
            model.actuator_gainprm[:, :] = 0.0
            model.actuator_biasprm[:, :] = 0.0
            data.ctrl[:] = 0.0

    def _require_loaded(self) -> MujocoModelBackend:
        if self._mujoco_backend is None or self._mujoco is None:
            raise RuntimeError("MuJoCo model has not been loaded")
        return self._mujoco_backend

    def _reset_initial_state(self) -> None:
        backend = self._require_loaded()
        model = backend.model
        data = backend.data
        if self.mujoco_initial_qpos is not None:
            qpos = self.mujoco_initial_qpos
        elif model.nkey:
            qpos = np.asarray(model.key_qpos[0], dtype=float)
        else:
            qpos = np.zeros(model.nq)
        if qpos.shape != (model.nq,):
            raise ValueError(f"mujoco_initial_qpos must have shape {(model.nq,)}")
        data.qpos[:] = qpos
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        data.qfrc_applied[:] = 0.0
        self._mujoco.mj_forward(model, data)

    def _torque_limits(self) -> np.ndarray:
        backend = self._require_loaded()
        model = backend.model
        dof = int(model.nv)
        ranges = np.asarray(model.jnt_actfrcrange[:dof], dtype=float)
        limited = np.asarray(model.jnt_actfrclimited[:dof], dtype=bool)
        limits = np.maximum(np.abs(ranges[:, 0]), np.abs(ranges[:, 1]))
        if ranges.shape != (dof, 2) or not np.all(limited) or np.any(limits <= 0):
            return np.ones(dof) * np.inf
        return limits

    def _open_viewer(self) -> None:
        backend = self._require_started()
        try:
            import mujoco.viewer

            self._viewer = mujoco.viewer.launch_passive(backend.model, backend.data)
        except Exception as exc:
            warnings.warn(
                f"Could not open MuJoCo viewer; continuing headless: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            self._viewer = None

    def _close_viewer(self) -> None:
        viewer = self._viewer
        self._viewer = None
        if viewer is None:
            return
        try:
            viewer.close()
        except Exception:
            pass

    def _sync_viewer(self) -> None:
        viewer = self._viewer
        if viewer is None:
            return
        try:
            if hasattr(viewer, "is_running") and not viewer.is_running():
                self._close_viewer()
                return
            viewer.sync()
        except Exception as exc:
            warnings.warn(
                f"MuJoCo viewer sync failed; closing viewer: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            self._close_viewer()

    def _timestamp(self) -> tuple[int, int]:
        t = float(self.data.time)
        seconds = int(t)
        nanoseconds = int(round((t - seconds) * 1e9))
        if nanoseconds >= 1_000_000_000:
            seconds += 1
            nanoseconds -= 1_000_000_000
        return seconds, nanoseconds

    def state_minimal(self, timeout_ms: int = 1000) -> dict[str, np.ndarray]:
        del timeout_ms
        backend = self._require_started()
        self._mujoco.mj_forward(backend.model, backend.data)
        qpos = np.asarray(backend.data.qpos, dtype=float).copy()
        qvel = np.asarray(backend.data.qvel, dtype=float).copy()
        tau = (
            np.zeros(self.dof)
            if self._last_total_torque is None
            else self._last_total_torque.copy()
        )
        tau_des = (
            np.zeros(self.dof)
            if self._last_torque is None
            else self._last_torque.copy()
        )
        return {
            "timestamp": self._timestamp(),
            "qpos": qpos,
            "qvel": qvel,
            "dq": qvel.copy(),
            "theta": qpos.copy(),
            "dtheta": qvel.copy(),
            "tau": tau,
            "tau_des": tau_des,
            "tau_ext": np.zeros(self.dof),
            "last_torque": tau_des.copy(),
            "model_backend": np.asarray(["mujoco"], dtype=object),
            "mujoco_velocity_source": np.asarray(
                [self.mujoco_velocity_source], dtype=object
            ),
            "mujoco_model_path": np.asarray([str(backend.path)], dtype=object),
            "mujoco_frame_type": np.asarray([backend.frame_type], dtype=object),
            "mujoco_frame_name": np.asarray([backend.frame_name], dtype=object),
            "mujoco_tool_name": np.asarray([backend.tool_name], dtype=object),
            "mujoco_tool_mass": np.asarray([backend.tool_mass], dtype=float),
            "mujoco_timestep": np.asarray(
                [float(backend.model.opt.timestep)], dtype=float
            ),
        }

    @property
    def state(self) -> dict[str, np.ndarray]:
        backend = self._require_started()
        minimal = self.state_minimal()
        model_state = backend.state(minimal["qpos"], minimal["qvel"])
        state = {
            **minimal,
            "tcp_pose": np.zeros(7),
            "tcp_vel": np.zeros(6),
            "flange_pose": np.zeros(7),
            "ft_sensor_raw": np.zeros(6),
            "ext_wrench_in_tcp": np.zeros(6),
            "ext_wrench_in_world": np.zeros(6),
            "ext_wrench_in_tcp_raw": np.zeros(6),
            "ext_wrench_in_world_raw": np.zeros(6),
            "model_backend": np.asarray(["mujoco"], dtype=object),
            "mujoco_velocity_source": np.asarray(
                [self.mujoco_velocity_source], dtype=object
            ),
        }
        state.update(model_state)
        return state

    def step(self, torque: np.ndarray | list[float]) -> None:
        backend = self._require_started()
        tau = np.asarray(torque, dtype=float)
        if tau.shape != (self.dof,):
            raise ValueError(f"Expected torque shape {(self.dof,)}, got {tau.shape}")

        if self.mujoco_realtime:
            now = time.perf_counter()
            last_wall_time = self._last_wall_time or now
            target_time = last_wall_time + float(backend.model.opt.timestep)
            sleep_time = target_time - now
            if sleep_time > 0:
                time.sleep(sleep_time)
            self._last_wall_time = time.perf_counter()

        if self._joint_limited is not None:
            backend.model.jnt_limited[:] = (
                self._joint_limited if self.enable_soft_limits else 0
            )

        self._mujoco.mj_forward(backend.model, backend.data)
        tau_total = tau.copy()
        if self.enable_gravity_comp:
            tau_total = tau_total + backend.gravity(backend.data.qpos)
        backend.data.qfrc_applied[:] = tau_total
        self._mujoco.mj_step(backend.model, backend.data)
        self._mujoco.mj_forward(backend.model, backend.data)
        self._last_torque = tau.copy()
        self._last_total_torque = tau_total.copy()
        self._sync_viewer()
