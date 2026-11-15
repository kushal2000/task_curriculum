from __future__ import annotations

import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from aioflexiv.__main__ import cmd_tool_calibrate, cmd_tool_list, cmd_tool_load
from aioflexiv.config import save_last_robot_sn
from aioflexiv.controller import FlexivController
from aioflexiv.robot import FlexivRobotInterface
from aioflexiv.tools import ToolPayload, read_active_tool_payload, tool_payload_from_params


class _FakeActiveTorqueControl:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    def info(self) -> dict:
        return {"dof": 7, "tau_max": [1.0] * 7, "model_name": "Rizon4"}


class _FakeRt:
    ActiveTorqueControl = _FakeActiveTorqueControl


class _FakeParams:
    def __init__(
        self,
        *,
        mass: float = 0.0,
        com=None,
        inertia=None,
        tcp_location=None,
    ) -> None:
        self.mass = mass
        self.CoM = list(com if com is not None else [0.0, 0.0, 0.0])
        self.inertia = list(inertia if inertia is not None else [0.0] * 6)
        self.tcp_location = list(
            tcp_location if tcp_location is not None else [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        )


class _FakeRobot:
    def __init__(self, serial: str, *args) -> None:
        self.serial = serial
        self.idle_mode = SimpleNamespace(name="IDLE")

    def info(self):
        return SimpleNamespace(serial_num=self.serial, has_FT_sensor=False)

    def mode(self):
        return self.idle_mode


class _FakeTool:
    def __init__(self) -> None:
        self.active = "Flange"
        self.saved = {
            "gripper": _FakeParams(
                mass=0.2,
                tcp_location=[0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
            )
        }
        self.updated: tuple[str, _FakeParams] | None = None

    def name(self):
        return self.active

    def list(self):
        return list(self.saved)

    def exist(self, name: str):
        return name in self.saved or name == "Flange"

    def params(self, name: str | None = None):
        return self.saved[name or self.active]

    def Switch(self, name: str):
        if not self.exist(name):
            raise RuntimeError(f"missing tool {name}")
        self.active = name

    def Add(self, name: str, params: _FakeParams):
        self.saved[name] = params

    def Update(self, name: str, params: _FakeParams):
        self.saved[name] = params
        self.updated = (name, params)

    def CalibratePayloadParams(self, tool_mounted: bool):
        return _FakeParams(
            mass=1.5,
            com=[0.01, 0.02, 0.03],
            inertia=[0.1, 0.2, 0.3, 0.01, 0.02, 0.03],
            tcp_location=[0.0] * 7,
        )


class _FakeFlexivRdk:
    Mode = SimpleNamespace(IDLE=SimpleNamespace(name="IDLE"))

    def __init__(self, tool: _FakeTool) -> None:
        self.tool = tool

    def Robot(self, serial: str, *args):
        return _FakeRobot(serial, *args)

    def Tool(self, robot: _FakeRobot):
        return self.tool


class ToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._old_config = os.environ.get("AIOFLEXIV_CONFIG")
        os.environ["AIOFLEXIV_CONFIG"] = str(Path(self._tmpdir.name) / "config.json")

    def tearDown(self) -> None:
        if self._old_config is None:
            os.environ.pop("AIOFLEXIV_CONFIG", None)
        else:
            os.environ["AIOFLEXIV_CONFIG"] = self._old_config
        self._tmpdir.cleanup()

    def test_flexiv_controller_passes_tool_to_robot_interface(self) -> None:
        controller = FlexivController("robot", tool="gripper")
        self.assertEqual(controller.robot.tool, "gripper")

    def test_flexiv_controller_applies_tool_to_existing_robot_interface(self) -> None:
        robot = FlexivRobotInterface("robot")
        controller = FlexivController(robot, tool="gripper")

        self.assertIs(controller.robot, robot)
        self.assertEqual(robot.tool, "gripper")

    def test_robot_start_switches_tool_before_rt_start(self) -> None:
        robot = FlexivRobotInterface(
            "robot",
            tool="gripper",
            network_interface_whitelist=["192.168.2.10"],
            verbose=True,
            model_backend="rdk",
        )
        with (
            patch("aioflexiv.robot.switch_active_tool") as switch_active_tool,
            patch("aioflexiv.robot.load_rt", return_value=_FakeRt()),
        ):
            robot.start()

        switch_active_tool.assert_called_once_with(
            "robot",
            "gripper",
            network_interface_whitelist=["192.168.2.10"],
            verbose=True,
        )

    def test_tool_load_switches_active_tool(self) -> None:
        fake_tool = _FakeTool()
        fake_rdk = _FakeFlexivRdk(fake_tool)
        args = SimpleNamespace(
            name="gripper",
            robot_sn="robot",
            network_interface=None,
            verbose=False,
        )

        with patch.dict(sys.modules, {"flexivrdk": fake_rdk}):
            stdout = StringIO()
            with redirect_stdout(stdout):
                status = cmd_tool_load(args)

        self.assertEqual(status, 0)
        self.assertEqual(fake_tool.active, "gripper")
        self.assertIn("Loaded", stdout.getvalue())

    def test_tool_payload_from_rdk_params(self) -> None:
        params = _FakeParams(
            mass=1.2,
            com=[0.01, 0.02, 0.03],
            inertia=[0.1, 0.2, 0.3, 0.01, 0.02, 0.03],
            tcp_location=[0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0],
        )

        payload = tool_payload_from_params(params, name="gripper")

        self.assertEqual(payload.name, "gripper")
        self.assertEqual(payload.mass, 1.2)
        self.assertEqual(payload.com, (0.01, 0.02, 0.03))
        self.assertEqual(payload.inertia, (0.1, 0.2, 0.3, 0.01, 0.02, 0.03))
        self.assertEqual(
            payload.tcp_location,
            (0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0),
        )

    def test_read_active_tool_payload_uses_current_rdk_tool(self) -> None:
        fake_tool = _FakeTool()
        fake_tool.active = "gripper"
        fake_rdk = _FakeFlexivRdk(fake_tool)

        with patch.dict(sys.modules, {"flexivrdk": fake_rdk}):
            payload = read_active_tool_payload("robot")

        self.assertEqual(payload.name, "gripper")
        self.assertEqual(payload.mass, 0.2)

    def test_robot_start_passes_active_tool_payload_to_mujoco_backend(self) -> None:
        payload = ToolPayload(
            name="gripper",
            mass=0.2,
            com=(0.0, 0.0, 0.03),
            inertia=(0.01, 0.01, 0.01, 0.0, 0.0, 0.0),
            tcp_location=(0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0),
        )
        robot = FlexivRobotInterface("robot", model_backend="mujoco")

        with (
            patch("aioflexiv.robot.read_active_tool_payload", return_value=payload),
            patch("aioflexiv.robot.MujocoModelBackend") as backend_cls,
            patch("aioflexiv.robot.load_rt", return_value=_FakeRt()),
        ):
            robot.start()

        backend_cls.assert_called_once()
        self.assertIs(backend_cls.call_args.kwargs["tool_payload"], payload)

    def test_tool_list_uses_saved_serial_when_robot_sn_omitted(self) -> None:
        save_last_robot_sn("saved-robot")
        fake_tool = _FakeTool()
        fake_rdk = _FakeFlexivRdk(fake_tool)
        args = SimpleNamespace(
            robot_sn=None,
            network_interface=None,
            verbose=False,
            json=False,
        )

        with patch.dict(sys.modules, {"flexivrdk": fake_rdk}):
            stdout = StringIO()
            with redirect_stdout(stdout):
                status = cmd_tool_list(args)

        self.assertEqual(status, 0)
        self.assertIn("saved-robot", stdout.getvalue())

    def test_tool_calibrate_preserves_existing_tcp_when_updating(self) -> None:
        fake_tool = _FakeTool()
        fake_rdk = _FakeFlexivRdk(fake_tool)
        args = SimpleNamespace(
            name="gripper",
            robot_sn="robot",
            network_interface=None,
            verbose=False,
            tcp_location=None,
            unmounted_pass=False,
            skip_unmounted_pass=True,
            load=False,
            yes=True,
        )

        with patch.dict(sys.modules, {"flexivrdk": fake_rdk}):
            stdout = StringIO()
            with redirect_stdout(stdout):
                status = cmd_tool_calibrate(args)

        self.assertEqual(status, 0)
        self.assertIsNotNone(fake_tool.updated)
        name, params = fake_tool.updated
        self.assertEqual(name, "gripper")
        self.assertEqual(params.mass, 1.5)
        self.assertEqual(params.tcp_location, [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
