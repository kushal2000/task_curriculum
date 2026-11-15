from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from aioflexiv.robot import FlexivRobotInterface


class RobotStateTests(unittest.TestCase):
    def test_state_exposes_force_torque_fields_as_arrays(self) -> None:
        raw_state = {
            "timestamp": (1, 2),
            "qpos": [0.0] * 7,
            "qvel": [0.0] * 7,
            "dq": [0.0] * 7,
            "theta": [0.0] * 7,
            "dtheta": [0.0] * 7,
            "tau": [0.0] * 7,
            "tau_des": [0.0] * 7,
            "tau_ext": [0.0] * 7,
            "tcp_pose": [0.0] * 7,
            "tcp_vel": [0.0] * 6,
            "flange_pose": [0.0] * 7,
            "ft_sensor_raw": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "ext_wrench_in_tcp": [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            "ext_wrench_in_world": [13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
            "ext_wrench_in_tcp_raw": [19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
            "ext_wrench_in_world_raw": [25.0, 26.0, 27.0, 28.0, 29.0, 30.0],
            "ee": np.eye(4),
            "jac": np.zeros((6, 7)),
            "mm": np.eye(7),
            "coriolis": [0.0] * 7,
            "gravity": [0.0] * 7,
        }
        ctrl = SimpleNamespace(read_once_full=lambda link_name, timeout_ms: raw_state)

        robot = FlexivRobotInterface.__new__(FlexivRobotInterface)
        robot._ctrl = ctrl
        robot._last_torque = None
        robot.link_name = "flange"

        state = robot.state

        for key in (
            "ft_sensor_raw",
            "ext_wrench_in_tcp",
            "ext_wrench_in_world",
            "ext_wrench_in_tcp_raw",
            "ext_wrench_in_world_raw",
        ):
            self.assertIsInstance(state[key], np.ndarray)
            self.assertEqual(state[key].shape, (6,))

        np.testing.assert_allclose(state["ft_sensor_raw"], raw_state["ft_sensor_raw"])
        np.testing.assert_allclose(
            state["ext_wrench_in_tcp"], raw_state["ext_wrench_in_tcp"]
        )
        np.testing.assert_allclose(
            state["ext_wrench_in_world_raw"], raw_state["ext_wrench_in_world_raw"]
        )

    def test_state_minimal_uses_motor_side_dtheta_for_control_qvel(self) -> None:
        raw_state = SimpleNamespace(
            timestamp=(1, 2),
            q=[0.0] * 7,
            dq=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            dtheta=[11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
            theta=[0.0] * 7,
            tau=[0.0] * 7,
            tau_des=[0.0] * 7,
            tau_ext=[0.0] * 7,
        )
        ctrl = SimpleNamespace(read_once=lambda timeout_ms: raw_state)

        robot = FlexivRobotInterface.__new__(FlexivRobotInterface)
        robot._ctrl = ctrl
        robot._last_torque = None

        state = robot.state_minimal()

        np.testing.assert_allclose(state["qvel"], raw_state.dtheta)
        np.testing.assert_allclose(state["dq"], raw_state.dq)
        np.testing.assert_allclose(state["dtheta"], raw_state.dtheta)

    def test_mujoco_state_uses_backend_with_configured_velocity_source(self) -> None:
        raw_state = SimpleNamespace(
            timestamp=(1, 2),
            q=[0.0] * 7,
            dq=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            dtheta=[11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
            theta=[0.0] * 7,
            tau=[0.0] * 7,
            tau_des=[0.0] * 7,
            tau_ext=[0.0] * 7,
        )
        ctrl = SimpleNamespace(read_once=lambda timeout_ms: raw_state)

        class FakeBackend:
            def __init__(self) -> None:
                self.qpos = None
                self.qvel = None

            def state(self, qpos, qvel):
                self.qpos = np.asarray(qpos, dtype=float)
                self.qvel = np.asarray(qvel, dtype=float)
                return {
                    "ee": np.eye(4),
                    "jac": np.zeros((6, 7)),
                    "mm": np.eye(7),
                    "coriolis": np.zeros(7),
                    "gravity": np.zeros(7),
                }

        backend = FakeBackend()
        robot = FlexivRobotInterface.__new__(FlexivRobotInterface)
        robot._ctrl = ctrl
        robot._last_torque = None
        robot.model_backend = "mujoco"
        robot.mujoco_velocity_source = "dtheta"
        robot._mujoco_backend = backend

        state = robot.state

        np.testing.assert_allclose(backend.qpos, raw_state.q)
        np.testing.assert_allclose(backend.qvel, raw_state.dtheta)
        np.testing.assert_allclose(state["qvel"], raw_state.dtheta)
        np.testing.assert_allclose(state["dq"], raw_state.dq)
        self.assertEqual(state["model_backend"][0], "mujoco")


if __name__ == "__main__":
    unittest.main()
