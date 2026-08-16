"""Present a ``CableObject`` through the rigid-object surface the Play task reads.

The Play task is written against a rigid manipuland: it reads a single position, orientation and
velocity for "the object" and builds keypoints, rewards and terminations from them. A cable has no
such pose — it is a chain of segments.

Rather than fork the task code, this adapter synthesises one. That choice is what keeps the
observation **140-dim and byte-identical**, so the pretrained checkpoint loads and the cable task
is measured by the same instrument as the rigid one.

The surface is small and was enumerated from the task code rather than guessed — five reads and
three writes:

    reads   data.root_pos_w, data.root_quat_w, data.root_lin_vel_w, data.root_ang_vel_w,
            data.default_mass
    writes  write_root_pose_to_sim, write_root_velocity_to_sim, set_external_force_and_torque

**Quaternion convention.** ``CableObject`` speaks xyzw (``wp.transformf``, and
``write_segment_pose_to_sim_index`` documents ``(x, y, z, w)``). The task code speaks wxyz, which
for real Isaac Lab assets is arranged by ``patches.wrap_env_assets`` / ``install_pose_write_
conversion``. This adapter is not an Isaac Lab asset, so those patches do not touch it and it must
do the conversion itself — in both directions.
"""

from __future__ import annotations

import math

import torch

__all__ = ["CableAsRigidObject"]


def _to_torch(value):
    """Isaac Lab 3.0 hands back warp-backed ProxyArrays; the task code wants torch."""
    return value.torch if hasattr(value, "torch") else value


def _xyzw_to_wxyz(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., 3:4], q[..., 0:3]], dim=-1)


def _wxyz_to_xyzw(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., 1:4], q[..., 0:1]], dim=-1)


def _frame_from_span(span: torch.Tensor) -> torch.Tensor:
    """Build a wxyz orientation whose +X axis follows the cable, as a right-handed frame.

    The cable's own geometry supplies only one axis — the direction from its first segment to its
    last. The remaining two are fixed against world up, so the frame is stable and reproducible
    rather than free to spin about the cable's length.
    """
    x = torch.nn.functional.normalize(span, dim=-1, eps=1e-8)

    up = torch.zeros_like(x)
    up[..., 2] = 1.0
    # Near-vertical cables would give a degenerate cross product; fall back to world +X there.
    degenerate = x[..., 2].abs() > 0.99
    alt = torch.zeros_like(x)
    alt[..., 0] = 1.0
    up = torch.where(degenerate.unsqueeze(-1), alt, up)

    y = torch.nn.functional.normalize(torch.cross(up, x, dim=-1), dim=-1, eps=1e-8)
    z = torch.cross(x, y, dim=-1)

    # Rotation matrix (columns x, y, z) -> wxyz quaternion, via the trace formulation.
    m00, m01, m02 = x[..., 0], y[..., 0], z[..., 0]
    m10, m11, m12 = x[..., 1], y[..., 1], z[..., 1]
    m20, m21, m22 = x[..., 2], y[..., 2], z[..., 2]

    trace = m00 + m11 + m22
    w = torch.sqrt(torch.clamp(1.0 + trace, min=1e-8)) * 0.5
    denom = torch.clamp(4.0 * w, min=1e-8)
    qx = (m21 - m12) / denom
    qy = (m02 - m20) / denom
    qz = (m10 - m01) / denom
    return torch.nn.functional.normalize(torch.stack([w, qx, qy, qz], dim=-1), dim=-1, eps=1e-8)


class _CableData:
    """The ``.data`` half of the surface. Recomputed lazily from live segment state."""

    def __init__(self, owner: "CableAsRigidObject") -> None:
        self._owner = owner

    @property
    def root_pos_w(self) -> torch.Tensor:
        return self._owner._segment_positions().mean(dim=1)

    @property
    def root_quat_w(self) -> torch.Tensor:
        pos = self._owner._segment_positions()
        return _frame_from_span(pos[:, -1] - pos[:, 0])

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        return self._owner._segment_velocities()[..., :3].mean(dim=1)

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        return self._owner._segment_velocities()[..., 3:].mean(dim=1)

    @property
    def default_mass(self) -> torch.Tensor:
        return self._owner._mass


class CableAsRigidObject:
    """Adapter over a ``CableObject``. Not a general-purpose asset — see the module docstring."""

    def __init__(
        self, cable, *, num_envs: int, length: float, thickness: float, density: float, device
    ) -> None:
        self._cable = cable
        # Passed in rather than read from `cable.num_instances`: the asset has no view until the
        # simulation initialises it, which happens after `_setup_scene` returns.
        self._num_envs = int(num_envs)
        self._device = device
        self._length = float(length)
        self._rest_quat_cache = None
        self.data = _CableData(self)

        radius = 0.5 * float(thickness)
        total_mass = float(density) * math.pi * radius**2 * self._length
        # Shaped (num_instances, 1) to match `RigidObjectData.default_mass`, which the reward
        # code indexes per environment.
        self._mass = torch.full(
            (self._num_envs, 1), total_mass, dtype=torch.float32, device=device
        )

    # -- reads ---------------------------------------------------------------------------------

    def _segment_pose(self) -> torch.Tensor:
        """(num_envs, num_segments, 7) — position then xyzw quaternion."""
        return _to_torch(self._cable.data.segment_pose_w).view(self._num_envs, -1, 7)

    def _segment_positions(self) -> torch.Tensor:
        return self._segment_pose()[..., :3]

    def _segment_velocities(self) -> torch.Tensor:
        """(num_envs, num_segments, 6) — linear then angular."""
        return _to_torch(self._cable.data.segment_velocity_w).view(
            self._num_envs, -1, 6
        )

    # -- writes --------------------------------------------------------------------------------

    def write_root_pose_to_sim(self, root_pose: torch.Tensor, env_ids=None) -> None:
        """Lay the cable out straight at the requested pose.

        ``root_pose`` is ``(N, 7)``: position then a **wxyz** quaternion, the convention the task
        code uses. A rigid object would simply be teleported; a cable has to be reconstructed, so
        the segments are placed along the pose's +X axis, centred on its position — the same
        arrangement :func:`_frame_from_span` reads back.
        """
        cable = self._cable
        n_seg = cable.num_segments
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)
        env_ids = torch.as_tensor(env_ids, device=self._device, dtype=torch.long)

        pos = root_pose[:, :3]
        quat_xyzw = _wxyz_to_xyzw(root_pose[:, 3:7])

        # Segment centres, evenly spaced along the local +X axis and centred on `pos`.
        seg_len = self._length / n_seg
        offsets = (torch.arange(n_seg, device=self._device, dtype=torch.float32) + 0.5) * seg_len
        offsets = offsets - 0.5 * self._length

        axis = _rotate_vector(quat_xyzw, torch.tensor([1.0, 0.0, 0.0], device=self._device))
        centres = pos.unsqueeze(1) + axis.unsqueeze(1) * offsets.view(1, -1, 1)

        # Compose with the segments' spawn orientation rather than replacing it. **VBD takes the
        # cable body poses used to initialise `Model.body_q` as its angular rest reference**
        # (newton-physics/newton#3847), so a segment orientation written here is interpreted as a
        # bend/twist *deviation* from the spawn pose, not as an absolute placement.
        #
        # Writing the task's quaternion directly put every segment ~180 degrees from rest -- the
        # segments spawn at (0, 0.7071, 0, 0.7071), aligning their local axis along the cable,
        # while the task's identity start pose became (0, 0, 1, 0). Twelve segments each carrying
        # a half-turn of twist against a 1.0e9 bend modulus is an enormous restoring torque, and
        # it ejected the cable at 2.1 m/s on the first step after every reset.
        #
        # The signature is worth remembering: position-independent, scaling linearly with substep
        # count (a torque re-applied each substep), and bending the cable while its span stayed
        # exactly at rest length -- orientations wrong, positions right.
        rest = self._rest_quat()[env_ids]
        pose = torch.zeros((len(env_ids), n_seg, 7), device=self._device, dtype=torch.float32)
        pose[..., :3] = centres
        pose[..., 3:] = _quat_mul(quat_xyzw.unsqueeze(1).expand(-1, n_seg, -1), rest)
        cable.write_segment_pose_to_sim_index(segment_pose=pose, env_ids=env_ids)

    def _rest_quat(self) -> torch.Tensor:
        """Per-segment spawn orientation, cached. This is VBD's angular rest reference."""
        if self._rest_quat_cache is None:
            default = _to_torch(self._cable.data.default_segment_pose_w)
            self._rest_quat_cache = default.view(self._num_envs, -1, 7)[..., 3:].clone()
        return self._rest_quat_cache

    def write_root_velocity_to_sim(self, root_velocity: torch.Tensor, env_ids=None) -> None:
        """Apply one linear/angular velocity to every segment. ``root_velocity`` is ``(N, 6)``."""
        cable = self._cable
        n_seg = cable.num_segments
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)
        env_ids = torch.as_tensor(env_ids, device=self._device, dtype=torch.long)

        vel = root_velocity.unsqueeze(1).expand(-1, n_seg, -1).contiguous()
        cable.write_segment_velocity_to_sim_index(segment_velocity=vel, env_ids=env_ids)

    def set_external_force_and_torque(self, *_args, **_kwargs) -> None:
        """Accepted and ignored.

        The only caller is the task's force/torque randomisation, which evaluation disables. It is
        a no-op rather than an error so that a DR-on run still constructs -- but note the wrench
        would be silently absent, so a DR-on cable run is not measuring what it appears to.
        """
        return None


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two xyzw quaternions, batched over leading dimensions."""
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dim=-1,
    )


def _rotate_vector(quat_xyzw: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate ``vec`` by an xyzw quaternion, batched over the leading dimension."""
    q_vec = quat_xyzw[..., :3]
    q_w = quat_xyzw[..., 3:4]
    vec = vec.expand_as(q_vec)
    t = 2.0 * torch.cross(q_vec, vec, dim=-1)
    return vec + q_w * t + torch.cross(q_vec, t, dim=-1)
