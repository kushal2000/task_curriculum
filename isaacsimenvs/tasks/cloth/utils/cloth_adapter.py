"""Present a cloth ``DeformableObject`` through the rigid-object surface the Play task reads.

The Play task is written against a rigid manipuland: one position, one orientation, one velocity.
A sheet has none of those intrinsically, so this adapter defines them from the *moving half* -- the
part the fold is about:

    position     centroid of the tracked keypoints
    orientation  identity
    velocity     mean particle velocity over the moving half

The same surface as the cable adapter, enumerated from the task code:

    reads   data.root_pos_w, data.root_quat_w, data.root_lin_vel_w, data.root_ang_vel_w,
            data.default_mass
    writes  write_root_pose_to_sim, write_root_velocity_to_sim, set_external_force_and_torque

**Orientation is deliberately identity, not a fitted frame.** The cable adapter fitted one from the
first-to-last span and it degenerated exactly when the cable was held (span deviation 1.1% flat vs
16.2% bent -- see `docs/phase4_cable_env.md`); PCA fixed the estimator but never explained the
24-segment failure. A square sheet has no principal axis at all while flat, so any fitted frame
would be noise fed to the policy as signal. The fold is scored on keypoints, which carry the
orientation information honestly.

**The pose write is a reset, not a teleport.** ``write_root_pose_to_sim`` restores the sheet flat
at the requested position -- the only pose the task ever asks for is the episode-reset pose, and a
cloth cannot be rigidly moved to an arbitrary one.
"""

from __future__ import annotations

import torch

__all__ = ["ClothAsRigidObject"]


def _mat_to_wxyz(r: torch.Tensor) -> torch.Tensor:
    """Rotation matrices ``(N, 3, 3)`` to wxyz quaternions, via the trace formulation.

    Uses the largest-diagonal branch rather than the naive trace form, which loses precision (and
    can divide by ~0) when the rotation approaches 180 degrees -- exactly the case a completed fold
    produces.
    """
    n = r.shape[0]
    q = torch.zeros((n, 4), device=r.device, dtype=r.dtype)
    trace = r[:, 0, 0] + r[:, 1, 1] + r[:, 2, 2]

    big = trace > 0
    if bool(big.any()):
        sq = torch.sqrt(trace[big] + 1.0) * 2.0
        q[big, 0] = 0.25 * sq
        q[big, 1] = (r[big, 2, 1] - r[big, 1, 2]) / sq
        q[big, 2] = (r[big, 0, 2] - r[big, 2, 0]) / sq
        q[big, 3] = (r[big, 1, 0] - r[big, 0, 1]) / sq
    rest = ~big
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        sel = rest & (r[:, i, i] >= r[:, j, j]) & (r[:, i, i] >= r[:, k, k])
        if not bool(sel.any()):
            continue
        sq = torch.sqrt(1.0 + r[sel, i, i] - r[sel, j, j] - r[sel, k, k]) * 2.0
        q[sel, 0] = (r[sel, k, j] - r[sel, j, k]) / sq
        q[sel, 1 + i] = 0.25 * sq
        q[sel, 1 + j] = (r[sel, j, i] + r[sel, i, j]) / sq
        q[sel, 1 + k] = (r[sel, k, i] + r[sel, i, k]) / sq
    # eps guards a zero-norm quaternion, which a degenerate fit can produce.
    return torch.nn.functional.normalize(q, dim=-1, eps=1e-12)


def _to_torch(value):
    """Isaac Lab 3.0 hands back warp-backed ProxyArrays; the task code wants torch."""
    return value.torch if hasattr(value, "torch") else value


class ClothAsRigidObject:
    """Adapt a cloth ``DeformableObject`` to the rigid-object reads the Play task performs."""

    def __init__(
        self,
        cloth,
        *,
        num_envs: int,
        rest_local: list,
        keypoint_idx: list[int],
        corner_idx: list[int],
        corner_rest: list,
        spawn_z: float,
        mass: float,
        device,
    ) -> None:
        self._cloth = cloth
        self._num_envs = num_envs
        self._device = device
        self._rest_local = torch.tensor(rest_local, device=device, dtype=torch.float32)
        self._kp_idx = torch.tensor(keypoint_idx, device=device, dtype=torch.long)
        self._corner_idx = torch.tensor(corner_idx, device=device, dtype=torch.long)
        self._corner_rest = torch.tensor(corner_rest, device=device, dtype=torch.float32)
        self._spawn_z = float(spawn_z)
        self._mass = float(mass)
        self.data = _ClothData(self)

    # ---------------------------------------------------------------- particle access

    def _particles(self) -> torch.Tensor:
        """``(num_envs, P, 3)`` particle positions in world coordinates."""
        return _to_torch(self._cloth.data.nodal_pos_w).view(self._num_envs, -1, 3)

    def _velocities(self) -> torch.Tensor:
        return _to_torch(self._cloth.data.nodal_vel_w).view(self._num_envs, -1, 3)

    def fit_rotation_wxyz(self) -> torch.Tensor:
        """Least-squares rotation from rest corners to current corners, as wxyz. ``(N, 4)``."""
        cur = self._particles()[:, self._corner_idx, :]                 # (N, 4, 3)
        q = cur - cur.mean(dim=1, keepdim=True)                         # centred current
        p = self._corner_rest - self._corner_rest.mean(dim=0, keepdim=True)  # centred rest (4, 3)

        # H = P^T Q, then R = V diag(1, 1, det(VU^T)) U^T. The determinant term forbids a
        # reflection: without it a badly deformed patch can fit an improper "rotation", which is a
        # mirror -- and a mirrored frame fed to a policy is worse than a wrong one.
        h = torch.einsum("ki,nkj->nij", p, q)
        # A crumpled patch can drive the corners toward collinear or coincident, leaving H rank
        # deficient. The SVD then returns an arbitrary rotation and the quaternion can normalise
        # to NaN, which reaches the policy as a non-finite observation and surfaces far away as
        # "normal expects all elements of std >= 0.0" from the action sampler. Ridge the diagonal
        # so a degenerate patch degrades to identity instead of to NaN.
        h = h + 1e-8 * torch.eye(3, device=h.device, dtype=h.dtype).unsqueeze(0)
        u, _, vh = torch.linalg.svd(h)
        v = vh.transpose(-2, -1)
        det = torch.linalg.det(v @ u.transpose(-2, -1))
        d = torch.ones((cur.shape[0], 3), device=cur.device)
        d[:, 2] = det
        r = v @ torch.diag_embed(d) @ u.transpose(-2, -1)               # (N, 3, 3)
        quat = _mat_to_wxyz(r)

        # Last line of defence: never hand a non-finite orientation to the policy. An identity
        # fallback is wrong but bounded; a NaN poisons the whole observation vector and kills the
        # run several layers away from the cause.
        bad = ~torch.isfinite(quat).all(dim=-1)
        if bool(bad.any()):
            quat = quat.clone()
            quat[bad] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=quat.device, dtype=quat.dtype)
        return quat

    def keypoints_w(self) -> torch.Tensor:
        """``(num_envs, K, 3)`` tracked keypoints on the moving half."""
        return self._particles()[:, self._kp_idx, :]

    # ---------------------------------------------------------------- writes

    def write_root_pose_to_sim(self, root_pose: torch.Tensor, env_ids=None) -> None:
        """Restore the sheet flat, centred on the requested position, at the requested YAW.

        XY and yaw of ``root_pose`` are honoured; the height is the configured spawn height, and
        roll/pitch are dropped. A cloth has no rigid pose to teleport, and the task only ever calls
        this at reset.

        **Yaw only, not the full orientation.** `reset_utils` draws a Haar-uniform SO(3) quaternion,
        which is right for a rigid tool that may land in any attitude and wrong for a sheet: roll or
        pitch would stand it on edge or bury it in the table, and the task is defined on a sheet
        lying flat. Taking the yaw of a uniform SO(3) draw is itself uniform on [0, 2*pi), so this
        gets a completely random heading without a second RNG stream and without disturbing the
        seed sequence. A `fixed_start_pose` quaternion is honoured the same way, by its yaw.

        Previously the quaternion was dropped entirely, so every episode presented the sheet
        axis-aligned and the policy never saw a fold in any other heading.
        """
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)
        env_ids = torch.as_tensor(env_ids, device=self._device, dtype=torch.long)

        # `root_pose` already has ONE ROW PER RESETTING ENV, aligned with `env_ids` -- it is not
        # indexed by absolute env id. Writing `centre[env_ids]` therefore indexes a subset tensor
        # with absolute ids and reads out of bounds as soon as a partial reset occurs: at 4 envs
        # they usually reset together so ids 0..3 happen to be valid, but at 8+ a reset of
        # e.g. envs [5, 7] indexes a 2-row tensor with 5 and 7. That surfaced as a CUDA
        # device-side assert (IndexKernel.cu "index out of bounds"), which reads like a physics
        # buffer problem and is not one.
        centre = root_pose[:, :3].clone()
        centre[:, 2] = self._spawn_z

        # Yaw from the (w, x, y, z) quaternion. Standard extraction; roll and pitch are discarded.
        qw, qx, qy, qz = root_pose[:, 3], root_pose[:, 4], root_pose[:, 5], root_pose[:, 6]
        yaw = torch.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
        cos_y = torch.cos(yaw).unsqueeze(1)   # (n, 1), broadcast over particles
        sin_y = torch.sin(yaw).unsqueeze(1)

        rest_x = self._rest_local[:, 0].unsqueeze(0)   # (1, P)
        rest_y = self._rest_local[:, 1].unsqueeze(0)
        rest_z = self._rest_local[:, 2].unsqueeze(0)
        rotated = torch.stack(
            (
                rest_x * cos_y - rest_y * sin_y,
                rest_x * sin_y + rest_y * cos_y,
                rest_z.expand(cos_y.shape[0], -1),
            ),
            dim=-1,
        )                                              # (n, P, 3)

        # (len(env_ids), P, 3): the flat rest sheet yawed, then translated to each requested centre.
        target = rotated + centre.unsqueeze(1)
        self._cloth.write_nodal_pos_to_sim_index(target, env_ids)
        self._cloth.write_nodal_velocity_to_sim_index(torch.zeros_like(target), env_ids)

    def write_root_velocity_to_sim(self, root_velocity: torch.Tensor, env_ids=None) -> None:
        """Zero the sheet's particle velocities.

        The task writes a root velocity at reset; for a sheet the meaningful equivalent is that no
        particle is moving. Applying the requested velocity to every particle would launch the
        whole sheet as a rigid body.
        """
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)
        env_ids = torch.as_tensor(env_ids, device=self._device, dtype=torch.long)
        n_particles = self._rest_local.shape[0]
        zeros = torch.zeros((len(env_ids), n_particles, 3), device=self._device)
        self._cloth.write_nodal_velocity_to_sim_index(zeros, env_ids)

    def set_external_force_and_torque(self, *args, **kwargs) -> None:
        """No-op.

        The task applies external forces to the manipuland for domain randomisation. Spreading a
        body force over a particle sheet is not equivalent to applying it to a rigid body, and
        randomisation is off for every measurement taken here -- so a silent no-op is safer than a
        wrong implementation.
        """
        return None

    def __getattr__(self, name):
        """Defer anything unlisted to the underlying cloth, so a missed read is loud."""
        return getattr(self._cloth, name)


class _ClothData:
    """The ``.data`` surface: rigid-object reads derived from the moving half."""

    def __init__(self, owner: ClothAsRigidObject) -> None:
        self._owner = owner

    @property
    def root_pos_w(self) -> torch.Tensor:
        return self._owner.keypoints_w().mean(dim=1)

    @property
    def root_quat_w(self) -> torch.Tensor:
        """Orientation of the moving half, fitted from its four corners (wxyz).

        Kabsch: the least-squares rotation taking the corners' REST offsets to their CURRENT
        offsets. Four corners span an area, so the fit is well-posed -- four points along one edge
        would be collinear and leave rotation about that edge undetermined, which is why this
        previously returned identity.

        A fold is then a rigid transform of the half (180 degrees about the crease), so the task's
        own keypoint machinery -- ``centroid + R * rest_offsets`` -- produces meaningful goal
        keypoints without any special-casing.
        """
        return self._owner.fit_rotation_wxyz()

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        return self._owner._velocities()[:, self._owner._kp_idx, :].mean(dim=1)

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        return torch.zeros((self._owner._num_envs, 3), device=self._owner._device)

    @property
    def default_mass(self) -> torch.Tensor:
        return torch.full(
            (self._owner._num_envs, 1), self._owner._mass, device=self._owner._device
        )

    def __getattr__(self, name):
        return getattr(self._owner._cloth.data, name)
