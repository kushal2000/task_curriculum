"""Init yaw is randomised, and everything scored survives the rotation.

The sheet used to spawn axis-aligned every episode: `reset_utils` drew a Haar-uniform SO(3)
quaternion and `ClothAsRigidObject.write_root_pose_to_sim` discarded it. The policy therefore only
ever saw one sheet heading.

Turning that on required two other fixes first, both covered here, because each silently breaks on
a rotated sheet:
  * `footprint_ratio` measured extent along a WORLD axis;
  * `_drive_goal_marker` wrote a FIXED world quaternion, which reaches the policy through
    `keypoints_rel_goal`.
"""

from __future__ import annotations

import math

import torch

from isaacsimenvs.tasks.cloth.utils.cloth_geometry import grid_mesh

SIZE, RES = 0.10, 9


def _rest() -> torch.Tensor:
    verts, _ = grid_mesh(SIZE, RES)
    return torch.tensor(verts, dtype=torch.float32)


def _yaw_rot(theta: float) -> torch.Tensor:
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _footprint_world(parts: torch.Tensor, ax: int = 0) -> float:
    """The OLD formula: extent along a world axis."""
    return float((parts[0, :, ax].max() - parts[0, :, ax].min()) / SIZE)


def _footprint_frame(parts: torch.Tensor, r: torch.Tensor, ax: int = 0) -> float:
    """The NEW formula: extent along the sheet's own fold axis."""
    axis_dir = r[:, :, ax]
    coord = torch.einsum("npi,ni->np", parts, axis_dir)
    return float((coord.amax(dim=1) - coord.amin(dim=1)) / SIZE)


ANGLES_DEG = (0, 15, 30, 45, 60, 90, 137)


def test_frame_relative_footprint_is_rotation_invariant():
    rest = _rest()
    for deg in ANGLES_DEG:
        r = _yaw_rot(math.radians(deg))
        parts = (r @ rest.T).T.unsqueeze(0)
        got = _footprint_frame(parts, r.unsqueeze(0))
        assert abs(got - 1.0) < 1e-4, f"{deg} deg -> {got}"


def test_world_axis_footprint_is_NOT_rotation_invariant():
    """Negative control: the bug this replaced, so the test above is known to discriminate.

    An unfolded sheet at 45 degrees reads sqrt(2) with no deformation whatsoever. Since
    `footprint_ratio < 0.65` is half of `is_folded`, that made a correctly folded but rotated sheet
    score as unfolded -- and it is the likely source of the 1.03-1.41 footprints seen in the render
    HUD, which were read at the time as the sheet stretching under the hand.
    """
    rest = _rest()
    worst = max(_footprint_world((_yaw_rot(math.radians(d)) @ rest.T).T.unsqueeze(0))
                for d in ANGLES_DEG)
    assert worst > 1.4, f"expected the old formula to inflate, got {worst}"
    at45 = _footprint_world((_yaw_rot(math.radians(45)) @ rest.T).T.unsqueeze(0))
    assert abs(at45 - math.sqrt(2.0)) < 1e-3, f"45 deg should read sqrt(2), got {at45}"


def test_yaw_extracted_from_uniform_so3_is_uniform():
    """The adapter takes the yaw of the existing SO(3) draw rather than opening a new RNG stream.

    That is only legitimate if the yaw marginal of a Haar-uniform rotation is itself uniform.
    """
    from isaaclab.utils.math import random_orientation

    torch.manual_seed(0)
    q = random_orientation(200_000, device="cpu")
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    hist = torch.histc(yaw, bins=12, min=-math.pi, max=math.pi) / len(yaw)
    assert float((hist - 1 / 12).abs().max()) < 0.005, hist.tolist()


def test_adapter_yaw_matches_an_explicit_rotation():
    """The adapter's inlined yaw rotation must equal composing the rotation matrix directly."""
    rest = _rest()
    for deg in ANGLES_DEG:
        theta = math.radians(deg)
        # As written in ClothAsRigidObject.write_root_pose_to_sim.
        cos_y = torch.tensor([[math.cos(theta)]])
        sin_y = torch.tensor([[math.sin(theta)]])
        rx, ry, rz = rest[:, 0].unsqueeze(0), rest[:, 1].unsqueeze(0), rest[:, 2].unsqueeze(0)
        inlined = torch.stack(
            (rx * cos_y - ry * sin_y, rx * sin_y + ry * cos_y, rz.expand(1, -1)), dim=-1
        )
        explicit = (_yaw_rot(theta) @ rest.T).T.unsqueeze(0)
        assert torch.allclose(inlined, explicit, atol=1e-6), f"{deg} deg"


def test_yaw_extraction_recovers_a_known_yaw():
    """A pure-yaw quaternion must round-trip through the adapter's atan2 extraction."""
    for deg in ANGLES_DEG:
        theta = math.radians(deg)
        # (w, x, y, z) for a rotation of theta about z.
        q = torch.tensor([[math.cos(theta / 2), 0.0, 0.0, math.sin(theta / 2)]])
        qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        got = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        assert abs(float(got) - theta) < 1e-5, f"{deg} deg -> {float(got)}"


def test_roll_and_pitch_are_dropped():
    """A sheet must stay flat: only yaw may survive, or it stands on edge or enters the table."""
    # 90 degrees about x -- pure roll, zero yaw.
    q = torch.tensor([[math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]])
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    assert abs(float(yaw)) < 1e-6, f"roll leaked into yaw: {float(yaw)}"
    rest = _rest()
    assert float(rest[:, 2].abs().max()) < 1e-6, "rest sheet should be flat in z"
