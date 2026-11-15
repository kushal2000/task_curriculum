from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import signal
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from aioflexiv.__main__ import _run_gravcomp, _serve_qpos, build_parser
from aioflexiv.controller import FlexivController
from aioflexiv.robot import MujocoRobotInterface


class _FakeActiveTorqueControl:
    """Streams a constant state and sends SIGINT after a few cycles, as Ctrl+C would."""

    instances: list[_FakeActiveTorqueControl] = []

    def __init__(self, robot_sn, network_interface_whitelist, *args) -> None:
        self.robot_sn = robot_sn
        self.network_interface_whitelist = network_interface_whitelist
        self.reads = 0
        self.writes: list[tuple] = []
        self.stopped = False
        self.instances.append(self)

    def info(self) -> dict:
        return {"dof": 7, "tau_max": [100.0] * 7, "model_name": "Rizon4"}

    def read_once(self, timeout_ms: int = 1000):
        self.reads += 1
        if self.reads == 50:
            os.kill(os.getpid(), signal.SIGINT)
        time.sleep(0.0005)
        return SimpleNamespace(
            timestamp=(0, self.reads),
            q=[0.0] * 7,
            dq=[0.1] * 7,
            theta=[0.0] * 7,
            dtheta=[0.1] * 7,
            tau=[0.0] * 7,
            tau_des=[0.0] * 7,
            tau_ext=[0.5] * 7,
        )

    def write_once(self, torques, *flags) -> None:
        self.writes.append((list(torques), *flags))

    def stop(self) -> None:
        self.stopped = True


class _FakeRt:
    ActiveTorqueControl = _FakeActiveTorqueControl


class GravcompTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._old_config = os.environ.get("AIOFLEXIV_CONFIG")
        os.environ["AIOFLEXIV_CONFIG"] = str(Path(self._tmpdir.name) / "config.json")
        _FakeActiveTorqueControl.instances.clear()

    def tearDown(self) -> None:
        if self._old_config is None:
            os.environ.pop("AIOFLEXIV_CONFIG", None)
        else:
            os.environ["AIOFLEXIV_CONFIG"] = self._old_config
        self._tmpdir.cleanup()

    def test_parser_rejects_negative_damping_and_bad_friction_scale(self) -> None:
        parser = build_parser()
        for argv in (
            ["gravcomp", "--damping", "-1"],
            ["gravcomp", "--friction-comp-scale", "101"],
        ):
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(argv)

    @unittest.skipIf(importlib.util.find_spec("mujoco") is None, "mujoco is not installed")
    def test_gravcomp_streams_damping_torque_until_sigint(self) -> None:
        args = build_parser().parse_args(
            [
                "gravcomp",
                "robot",
                "--damping",
                "2",
                "--tool",
                "gripper",
                "--friction-comp-scale",
                "40",
                "--network-interface",
                "192.168.2.10",
            ]
        )

        with (
            patch("aioflexiv.robot.load_rt", return_value=_FakeRt()),
            patch("aioflexiv.robot.switch_active_tool") as switch_active_tool,
            patch("aioflexiv.robot.read_active_tool_payload", side_effect=RuntimeError),
            patch("warnings.warn"),
        ):
            stdout = StringIO()
            with redirect_stdout(stdout):
                status = args.func(args)

        self.assertEqual(status, 0)
        self.assertIn("Stopped", stdout.getvalue())
        switch_active_tool.assert_called_once_with(
            "robot",
            "gripper",
            network_interface_whitelist=["192.168.2.10"],
            verbose=False,
        )
        (ctrl,) = _FakeActiveTorqueControl.instances
        self.assertEqual(ctrl.network_interface_whitelist, ["192.168.2.10"])
        self.assertTrue(ctrl.stopped)
        torques, gravity_comp, soft_limits, friction_comp_scale = ctrl.writes[-1]
        # No stiffness: -kd * dtheta, minus the tau_ext recorded at start.
        np.testing.assert_allclose(torques, [-2.0 * 0.1 - 0.5] * 7)
        self.assertTrue(gravity_comp)
        self.assertTrue(soft_limits)
        self.assertEqual(friction_comp_scale, 40.0)

    @unittest.skipIf(importlib.util.find_spec("mujoco") is None, "mujoco is not installed")
    def test_gravcomp_holds_the_simulated_arm_until_sigint(self) -> None:
        robot = MujocoRobotInterface(mujoco_viewer=False, mujoco_realtime=False)
        controller = FlexivController(robot)
        controller.kp = 0.0
        controller.kd = 2.0

        async def run() -> None:
            asyncio.get_running_loop().call_later(0.3, os.kill, os.getpid(), signal.SIGINT)
            await _run_gravcomp(controller)

        with redirect_stdout(StringIO()):
            asyncio.run(run())

        self.assertFalse(robot.started)
        self.assertGreater(controller.state["timestamp"][1], 0)
        np.testing.assert_allclose(
            controller.state["qpos"], controller.initial_qpos, atol=1e-6
        )

    def test_serve_qpos_answers_with_the_latest_state(self) -> None:
        controller = SimpleNamespace(state={"qpos": np.array([0.1, -0.2, 0.3])})

        async def get(port: int, path: str) -> tuple[bytes, dict]:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(f"GET {path} HTTP/1.1\r\nHost: localhost\r\n\r\n".encode())
            response = await reader.read()
            writer.close()
            head, _, body = response.partition(b"\r\n\r\n")
            return head.split(b"\r\n", 1)[0], json.loads(body)

        async def run() -> list[tuple[bytes, dict]]:
            server = await asyncio.start_server(
                lambda reader, writer: _serve_qpos(controller, reader, writer),
                "127.0.0.1",
                0,
            )
            port = server.sockets[0].getsockname()[1]
            try:
                return [await get(port, "/qpos"), await get(port, "/state")]
            finally:
                server.close()

        (qpos_status, qpos_body), (other_status, _) = asyncio.run(run())

        self.assertEqual(qpos_status, b"HTTP/1.1 200 OK")
        self.assertEqual(qpos_body, {"qpos": [0.1, -0.2, 0.3]})
        self.assertEqual(other_status, b"HTTP/1.1 404 Not Found")


if __name__ == "__main__":
    unittest.main()
