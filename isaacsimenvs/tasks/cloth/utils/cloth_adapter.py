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
        spawn_z: float,
        mass: float,
        device,
    ) -> None:
        self._cloth = cloth
        self._num_envs = num_envs
        self._device = device
        self._rest_local = torch.tensor(rest_local, device=device, dtype=torch.float32)
        self._kp_idx = torch.tensor(keypoint_idx, device=device, dtype=torch.long)
        self._spawn_z = float(spawn_z)
        self._mass = float(mass)
        self.data = _ClothData(self)

    # ---------------------------------------------------------------- particle access

    def _particles(self) -> torch.Tensor:
        """``(num_envs, P, 3)`` particle positions in world coordinates."""
        return _to_torch(self._cloth.data.nodal_pos_w).view(self._num_envs, -1, 3)

    def _velocities(self) -> torch.Tensor:
        return _to_torch(self._cloth.data.nodal_vel_w).view(self._num_envs, -1, 3)

    def keypoints_w(self) -> torch.Tensor:
        """``(num_envs, K, 3)`` tracked keypoints on the moving half."""
        return self._particles()[:, self._kp_idx, :]

    # ---------------------------------------------------------------- writes

    def write_root_pose_to_sim(self, root_pose: torch.Tensor, env_ids=None) -> None:
        """Restore the sheet flat, centred on the requested position.

        Only the XY of ``root_pose`` is honoured; the height is the configured spawn height. A
        cloth has no rigid pose to teleport, and the task only ever calls this at reset.
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

        # (len(env_ids), P, 3): the flat rest sheet translated to each requested centre.
        target = self._rest_local.unsqueeze(0) + centre.unsqueeze(1)
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
        """Identity (wxyz). See the module docstring: a fitted frame would be noise."""
        q = torch.zeros((self._owner._num_envs, 4), device=self._owner._device)
        q[:, 0] = 1.0
        return q

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
