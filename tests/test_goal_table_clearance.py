"""reset.goal_table_clearance_m: goals are lifted until the object clears the table top.

Kit-free: goal_table_clearance.py needs only numpy and torch.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
Rotation = pytest.importorskip("scipy.spatial.transform").Rotation

from conftest import load_module  # noqa: E402

gtc = load_module("isaacsimenvs/tasks/play/utils/goal_table_clearance.py")


def _urdf(tmp_path, collisions: str):
    path = tmp_path / "object.urdf"
    path.write_text(f'<robot name="o"><link name="object">{collisions}</link></robot>')
    return path


# A box handle along x and a cylinder head across its end, the way generate_objects.py writes a
# hammer: the head's axis turned onto y by rpy (-pi/2, 0, 0), x_offset = handle_x / 2 + radius.
HANDLE = (0.20, 0.03, 0.02)
HEAD_R, HEAD_LEN = 0.015, 0.08
HAMMER = (
    f'<collision><origin xyz="0 0 0" rpy="0 0 0"/><geometry><box size="{" ".join(map(str, HANDLE))}"/>'
    f"</geometry></collision>"
    f'<collision><origin xyz="{HANDLE[0] / 2 + HEAD_R} 0 0" rpy="{-math.pi / 2} 0 0"/>'
    f'<geometry><cylinder radius="{HEAD_R}" length="{HEAD_LEN}"/></geometry></collision>'
)


def test_corners_bound_handle_and_head(tmp_path) -> None:
    corners = gtc.collision_corners(_urdf(tmp_path, HAMMER))
    assert corners.shape == (gtc.NUM_CORNERS, 3)
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    # The head, as the capsule the importer makes, spans length + 2 r along y.
    half_head = HEAD_LEN / 2 + HEAD_R
    np.testing.assert_allclose(lo, (-HANDLE[0] / 2, -half_head, -HEAD_R), atol=1e-6)
    np.testing.assert_allclose(hi, (HANDLE[0] / 2 + 2 * HEAD_R, half_head, HEAD_R), atol=1e-6)


def test_handle_only_object_repeats_its_corners(tmp_path) -> None:
    corners = gtc.collision_corners(_urdf(tmp_path, HAMMER.split("</collision>")[0] + "</collision>"))
    assert corners.shape == (gtc.NUM_CORNERS, 3)
    np.testing.assert_array_equal(corners[:8], corners[8:])


def test_floor_at_identity_and_on_end(tmp_path) -> None:
    corners = torch.as_tensor(gtc.collision_corners(_urdf(tmp_path, HAMMER)))[None]
    top, clearance = torch.tensor([0.28]), 0.01
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    # Flat: the head's capsule (radius 0.015) reaches lowest.
    assert gtc.min_goal_z(identity, corners, top, clearance).item() == pytest.approx(0.28 + 0.01 + HEAD_R)
    # Stood on its handle end (x down, +90 deg about y): the handle's far end is lowest.
    on_end = torch.tensor([[math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0]])
    assert gtc.min_goal_z(on_end, corners, top, clearance).item() == pytest.approx(
        0.28 + 0.01 + HANDLE[0] / 2 + 2 * HEAD_R, abs=1e-6)


def test_lifted_goals_clear_the_table_at_any_orientation(tmp_path) -> None:
    """Put the object at the floor z for random orientations and rotate its corners with scipy:
    every corner is at or above top + clearance, and the lowest one is on it."""
    corners_np = gtc.collision_corners(_urdf(tmp_path, HAMMER)).astype(np.float64)
    n = 500
    quat_xyzw = Rotation.random(n, random_state=0).as_quat()
    quat_wxyz = torch.as_tensor(quat_xyzw[:, [3, 0, 1, 2]], dtype=torch.float32)
    corners = torch.as_tensor(corners_np, dtype=torch.float32).expand(n, -1, -1)
    top = torch.full((n,), 0.28)
    z = gtc.min_goal_z(quat_wxyz, corners, top, 0.01).numpy()
    lowest = np.array([Rotation.from_quat(q).apply(corners_np)[:, 2].min() for q in quat_xyzw]) + z
    np.testing.assert_allclose(lowest, 0.29, atol=1e-5)
    # Not vacuous: at some orientations the floor is above the Flexiv goal volume's z_min (0.35),
    # so a goal sampled there unlifted would be in the table.
    assert (z > 0.35).any()
