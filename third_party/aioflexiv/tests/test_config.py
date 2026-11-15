from __future__ import annotations

import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from aioflexiv.__main__ import cmd_status
from aioflexiv.config import config_path, resolve_robot_sn, save_last_robot_sn
from aioflexiv.controller import FlexivController
from aioflexiv.robot import FlexivRobotInterface, MujocoRobotInterface


class _FakeActiveTorqueControl:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    def info(self) -> dict:
        return {
            "dof": 7,
            "tau_max": [1.0] * 7,
            "model_name": "Rizon4",
        }


class _FakeRt:
    ActiveTorqueControl = _FakeActiveTorqueControl


class ConfigTests(unittest.TestCase):
    def test_resolve_robot_sn_uses_saved_serial(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            old_config = os.environ.get("AIOFLEXIV_CONFIG")
            os.environ["AIOFLEXIV_CONFIG"] = str(Path(tmpdir) / "config.json")
            try:
                save_last_robot_sn("saved-robot")

                self.assertEqual(resolve_robot_sn(None), "saved-robot")
                self.assertEqual(resolve_robot_sn(" explicit-robot "), "explicit-robot")
            finally:
                if old_config is None:
                    os.environ.pop("AIOFLEXIV_CONFIG", None)
                else:
                    os.environ["AIOFLEXIV_CONFIG"] = old_config

    def test_flexiv_controller_defaults_to_saved_serial(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            old_config = os.environ.get("AIOFLEXIV_CONFIG")
            os.environ["AIOFLEXIV_CONFIG"] = str(Path(tmpdir) / "config.json")
            try:
                save_last_robot_sn("saved-robot")

                controller = FlexivController()
                none_controller = FlexivController(None)

                self.assertEqual(controller.robot.robot_sn, "saved-robot")
                self.assertEqual(none_controller.robot.robot_sn, "saved-robot")
            finally:
                if old_config is None:
                    os.environ.pop("AIOFLEXIV_CONFIG", None)
                else:
                    os.environ["AIOFLEXIV_CONFIG"] = old_config

    def test_robot_interface_start_saves_serial_after_successful_connection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            old_config = os.environ.get("AIOFLEXIV_CONFIG")
            os.environ["AIOFLEXIV_CONFIG"] = str(Path(tmpdir) / "config.json")
            try:
                robot = FlexivRobotInterface("explicit-robot", model_backend="rdk")
                with patch("aioflexiv.robot.load_rt", return_value=_FakeRt()):
                    robot.start()

                self.assertEqual(resolve_robot_sn(None), "explicit-robot")
            finally:
                if old_config is None:
                    os.environ.pop("AIOFLEXIV_CONFIG", None)
                else:
                    os.environ["AIOFLEXIV_CONFIG"] = old_config

    def test_resolve_robot_sn_errors_without_argument_or_saved_serial(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            old_config = os.environ.get("AIOFLEXIV_CONFIG")
            os.environ["AIOFLEXIV_CONFIG"] = str(Path(tmpdir) / "missing.json")
            try:
                with self.assertRaisesRegex(RuntimeError, "No robot serial number"):
                    resolve_robot_sn(None)
                self.assertFalse(config_path().exists())
            finally:
                if old_config is None:
                    os.environ.pop("AIOFLEXIV_CONFIG", None)
                else:
                    os.environ["AIOFLEXIV_CONFIG"] = old_config

    def test_flexiv_controller_mujoco_serial_selects_simulation(self) -> None:
        controller = FlexivController("mujoco", mujoco_viewer=False)

        self.assertIsInstance(controller.robot, MujocoRobotInterface)
        self.assertEqual(controller.robot.robot_sn, "mujoco")
        self.assertEqual(controller.robot.model_backend, "mujoco")

    def test_status_without_argument_or_saved_serial_fails_before_rdk_import(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            old_config = os.environ.get("AIOFLEXIV_CONFIG")
            os.environ["AIOFLEXIV_CONFIG"] = str(Path(tmpdir) / "missing.json")
            try:
                args = SimpleNamespace(
                    robot_sn=None,
                    network_interface=None,
                    verbose=False,
                    events=0,
                )
                stdout = StringIO()
                with redirect_stdout(stdout):
                    status = cmd_status(args)

                self.assertEqual(status, 2)
                self.assertIn("No robot serial number", stdout.getvalue())
            finally:
                if old_config is None:
                    os.environ.pop("AIOFLEXIV_CONFIG", None)
                else:
                    os.environ["AIOFLEXIV_CONFIG"] = old_config


if __name__ == "__main__":
    unittest.main()
