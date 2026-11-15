from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from aioflexiv.mujoco_model import (
    MUJOCO_TOOL_PAYLOAD_BODY_NAME,
    MujocoModelBackend,
    default_mujoco_model_path,
    default_mujoco_scene_path,
)
from aioflexiv.tools import ToolPayload


@unittest.skipIf(importlib.util.find_spec("mujoco") is None, "mujoco is not installed")
class MujocoModelBackendTests(unittest.TestCase):
    def test_default_model_path_follows_robot_model_name(self) -> None:
        self.assertEqual(
            default_mujoco_model_path("Rizon4").name,
            "flexiv_rizon4.xml",
        )
        self.assertEqual(
            default_mujoco_model_path("Rizon4S").name,
            "flexiv_rizon4s.xml",
        )
        self.assertEqual(default_mujoco_scene_path("Rizon4").name, "scene.xml")

    def test_backend_computes_kinematics_and_dynamics_for_bundled_models(self) -> None:
        for model_name in ("Rizon4", "Rizon4S"):
            with self.subTest(model_name=model_name):
                backend = MujocoModelBackend(default_mujoco_model_path(model_name))
                state = backend.state(np.zeros(7), np.zeros(7))

                self.assertEqual(state["ee"].shape, (4, 4))
                self.assertEqual(state["jac"].shape, (6, 7))
                self.assertEqual(state["mm"].shape, (7, 7))
                self.assertEqual(state["coriolis"].shape, (7,))
                self.assertEqual(state["gravity"].shape, (7,))
                self.assertEqual(state["mujoco_frame_type"][0], "site")
                self.assertEqual(state["mujoco_frame_name"][0], "attachment_site")

                np.testing.assert_allclose(state["ee"][3], [0.0, 0.0, 0.0, 1.0])
                mass_eigs = np.linalg.eigvalsh(0.5 * (state["mm"] + state["mm"].T))
                self.assertGreater(float(mass_eigs[0]), 0.0)

    def test_rizon4s_rdk_flange_and_physical_tool_mount_frames(self) -> None:
        import mujoco

        backend = MujocoModelBackend(default_mujoco_model_path("Rizon4S"))
        model = backend.model
        data = backend.data
        mujoco.mj_forward(model, data)

        flange_id = model.site("attachment_site").id
        tool_mount_id = model.site("tool_mount_site").id
        expected_position = np.array([0.0, 0.0, 0.124])
        np.testing.assert_allclose(
            model.site_pos[flange_id], expected_position, atol=1.0e-10
        )
        np.testing.assert_allclose(
            model.site_pos[tool_mount_id], expected_position, atol=1.0e-10
        )

        flange_rotation = np.asarray(data.site_xmat[flange_id]).reshape(3, 3)
        tool_mount_rotation = np.asarray(data.site_xmat[tool_mount_id]).reshape(
            3, 3
        )
        np.testing.assert_allclose(
            flange_rotation.T @ tool_mount_rotation,
            np.diag([-1.0, -1.0, 1.0]),
            atol=2.0e-9,
        )

    def test_tool_payload_is_compiled_into_model_dynamics(self) -> None:
        payload = ToolPayload(
            name="gripper",
            mass=1.2,
            com=(0.0, 0.0, 0.05),
            inertia=(0.01, 0.01, 0.01, 0.0, 0.0, 0.0),
            tcp_location=(0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0),
        )
        qpos = np.zeros(7)
        qvel = np.zeros(7)

        bare = MujocoModelBackend(default_mujoco_model_path("Rizon4"))
        loaded = MujocoModelBackend(
            default_mujoco_model_path("Rizon4"),
            tool_payload=payload,
        )
        bare_state = bare.state(qpos, qvel)
        loaded_state = loaded.state(qpos, qvel)

        self.assertGreater(loaded.model.body(MUJOCO_TOOL_PAYLOAD_BODY_NAME).id, 0)
        self.assertEqual(loaded_state["mujoco_tool_name"][0], "gripper")
        self.assertEqual(loaded_state["mujoco_tool_mass"][0], 1.2)
        self.assertGreater(float(np.linalg.norm(loaded_state["mm"] - bare_state["mm"])), 0.0)

    def test_tool_payload_patches_scene_includes(self) -> None:
        payload = ToolPayload(
            name="gripper",
            mass=1.0,
            com=(0.0, 0.0, 0.04),
            inertia=(0.01, 0.01, 0.01, 0.0, 0.0, 0.0),
            tcp_location=(0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0),
        )

        backend = MujocoModelBackend(
            default_mujoco_scene_path("Rizon4"),
            tool_payload=payload,
        )

        self.assertGreater(backend.model.body(MUJOCO_TOOL_PAYLOAD_BODY_NAME).id, 0)


if __name__ == "__main__":
    unittest.main()
