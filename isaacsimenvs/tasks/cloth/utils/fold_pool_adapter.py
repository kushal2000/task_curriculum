"""Present a POOL of fold-shape articulations as the one cloth ``ClothEnv`` knows how to score.

The stage-2 env trains one policy on five shapes at once, a shape drawn per env per episode. Newton
cannot hold that as one articulation: ``newton.selection.ArticulationView`` raises "Articulations are
not identical" unless every world has the same joint, link and shape counts, and ``SolverMuJoCo``
builds its MuJoCo model from one template world, so per-world plate GEOMETRY cannot differ either.
So every env carries all five shapes, as five articulations, identically in every world; one is
ACTIVE on the table and the other four are PARKED -- hovering a few metres away inside that env's own
world, where nothing can touch them (worlds never collide with each other).

The task code is untouched, exactly as for the stage-1 chain: :class:`PoolAsClothMesh` reconstructs
the ``resolution x resolution`` particle cloud from the ACTIVE shape's body poses
(``fold_shapes.grid_plate_map``), and :class:`PoolAsRigidObject` is ``env.object``. Reads are the
inherited ``ClothAsRigidObject`` ones; writes place the active shape and park the rest.

Pure torch, no Isaac imports -- testable on a CPU, like ``rigid_cloth_adapter``.
"""

from __future__ import annotations

import torch

from isaacsimenvs.tasks.cloth.utils.cloth_adapter import ClothAsRigidObject, _to_torch, task_yaw
from isaacsimenvs.tasks.cloth.utils.rigid_cloth_adapter import (
    _DEFAULT_QUAT_LAYOUT,
    _quat_apply,
    _sim_quat_to_wxyz,
    _yaw_quat,
)

__all__ = ["PoolAsClothMesh", "PoolAsRigidObject"]


class PoolAsClothMesh:
    """The active shape of each env, seen as the cloth's particle grid.

    ``mappings[s]`` is ``fold_shapes.grid_plate_map(shapes[s], resolution)`` -- the same ordering
    for every shape, so the cloud has one layout whatever shape an env is running.
    """

    def __init__(
        self,
        articulations,
        *,
        mappings,
        body_prefix: str,
        num_envs: int,
        device,
        xy_scale: torch.Tensor | None = None,
    ) -> None:
        if len(articulations) != len(mappings):
            raise ValueError("one grid mapping per articulation")
        sizes = {len(m) for m in mappings}
        if len(sizes) != 1:
            raise ValueError(f"every shape must map the same grid, got sizes {sorted(sizes)}")
        self._arts = list(articulations)
        self._mappings = mappings
        self._prefix = body_prefix
        self._num_envs = int(num_envs)
        self._device = device
        self.data = _PoolMeshData(self)
        # (S, P, 3) local points, in each plate's body frame (= sheet frame at rest).
        self._local = torch.tensor(
            [[pt for _, pt in m] for m in mappings], device=device, dtype=torch.float32
        )
        self._num_particles = self._local.shape[1]
        # Per-env in-plane size factor: each env's sheets were built at their own size
        # (`fold_shape_assets.scale_builder_world`), so the grid point attached to a plate sits at
        # the scaled position in that plate's frame. (N, 1, 3) so it broadcasts over the grid.
        scale = (
            torch.ones(self._num_envs, device=device) if xy_scale is None else xy_scale.to(device)
        )
        self._scale3 = torch.stack((scale, scale, torch.ones_like(scale)), dim=-1).unsqueeze(1)
        self._body_idx: list[torch.Tensor] | None = None
        self._quat_layout = _DEFAULT_QUAT_LAYOUT
        #: Which shape each env is running. Set by ``PoolAsRigidObject.select``.
        self.active = torch.zeros(self._num_envs, device=device, dtype=torch.long)

    @property
    def num_shapes(self) -> int:
        return len(self._arts)

    @property
    def quat_layout(self) -> str:
        return self._quat_layout

    def detect_quat_layout(self, rest_local) -> str:
        """Same measurement as ``ChainAsClothMesh.detect_quat_layout``, on env 0's active shape."""
        want = torch.as_tensor(rest_local, device=self._device, dtype=torch.float32)
        want = want - want.mean(dim=0, keepdim=True)
        scores = {}
        for layout in ("xyzw", "wxyz"):
            self._quat_layout = layout
            got = self._positions()[0]
            got = got - got.mean(dim=0, keepdim=True)
            scores[layout] = float((got[:, :2] - want[:, :2]).norm(dim=-1).max())
        best = min(scores, key=scores.get)
        self._quat_layout = best
        self._quat_layout_scores = scores
        return best

    def _resolve_bodies(self) -> list[torch.Tensor]:
        """Per shape, grid vertex -> body index in THAT articulation, by name, on first use.

        Deferred and by name for the reasons ``ChainAsClothMesh._resolve_bodies`` gives: an
        articulation has no ``body_names`` until the scene is played, and the solver's body order
        need not be the file's.
        """
        if self._body_idx is not None:
            return self._body_idx
        out = []
        for art, mapping in zip(self._arts, self._mappings):
            names = list(art.body_names)
            want = [self._prefix + plate for plate, _ in mapping]
            missing = sorted(set(want) - set(names))
            if missing:
                raise RuntimeError(f"articulation is missing plates {missing}; has {names}")
            out.append(torch.tensor([names.index(w) for w in want], device=self._device))
        self._body_idx = out
        return out

    def _pose(self, art) -> tuple[torch.Tensor, torch.Tensor]:
        data = art.data
        pos = getattr(data, "body_pos_w", None)
        quat = getattr(data, "body_quat_w", None)
        if pos is None or quat is None:
            state = _to_torch(data.body_state_w)
            pos, quat = state[..., 0:3], state[..., 3:7]
        else:
            pos, quat = _to_torch(pos), _to_torch(quat)
        return pos, _sim_quat_to_wxyz(quat, self._quat_layout)

    def shape_positions(self, s: int) -> torch.Tensor:
        """``(num_envs, P, 3)`` grid of shape ``s`` in every env, active or not."""
        idx = self._resolve_bodies()[s]
        pos, quat = self._pose(self._arts[s])
        local = self._local[s].unsqueeze(0) * self._scale3
        return pos[:, idx, :] + _quat_apply(quat[:, idx, :], local)

    def _positions(self) -> torch.Tensor:
        """``(num_envs, P, 3)``: each env's ACTIVE shape. Every shape is evaluated and selected --
        five gathers of 49 points beat a data-dependent branch inside a captured step."""
        out = None
        for s in range(self.num_shapes):
            p = self.shape_positions(s)
            sel = (self.active == s).view(-1, 1, 1)
            out = p if out is None else torch.where(sel, p, out)
        return out

    def _velocities(self) -> torch.Tensor:
        """``(num_envs, P, 3)`` world velocities ``v + omega x r`` of the active shape."""
        out = None
        for s, art in enumerate(self._arts):
            data = art.data
            lin = getattr(data, "body_lin_vel_w", None)
            if lin is None:
                lin = data.body_com_lin_vel_w
            ang = getattr(data, "body_ang_vel_w", None)
            if ang is None:
                ang = data.body_com_ang_vel_w
            idx = self._resolve_bodies()[s]
            lin = _to_torch(lin)[:, idx, :]
            ang = _to_torch(ang)[:, idx, :]
            quat = self._pose(art)[1][:, idx, :]
            local = self._local[s].unsqueeze(0) * self._scale3
            v = lin + torch.cross(ang, _quat_apply(quat, local), dim=-1)
            sel = (self.active == s).view(-1, 1, 1)
            out = v if out is None else torch.where(sel, v, out)
        return out

    def write_nodal_pos_to_sim_index(self, target, env_ids) -> None:
        raise NotImplementedError("the pool is repositioned by root pose -- see PoolAsRigidObject")

    def write_nodal_velocity_to_sim_index(self, target, env_ids) -> None:
        raise NotImplementedError("the pool is stopped by zeroing root and joint velocities")


class _PoolMeshData:
    def __init__(self, owner: PoolAsClothMesh) -> None:
        self._owner = owner

    @property
    def nodal_pos_w(self) -> torch.Tensor:
        return self._owner._positions().reshape(-1, 3)

    @property
    def nodal_vel_w(self) -> torch.Tensor:
        return self._owner._velocities().reshape(-1, 3)


class PoolAsRigidObject(ClothAsRigidObject):
    """``env.object`` for the pool: places the active shape, parks the others.

    ``park_local`` is ``(S, 3)``: where each shape hovers when parked, relative to its env origin.
    The offsets must be far from everything in the env and from each other; worlds never collide, so
    they need not be far from OTHER envs. ``env_origins`` may be a tensor or a zero-argument callable
    returning one -- the env builds this object inside the scene hook, before its origins exist.
    """

    def __init__(
        self,
        mesh: PoolAsClothMesh,
        *,
        articulations,
        env_origins: torch.Tensor,
        park_local: torch.Tensor,
        centre_offset: torch.Tensor | None = None,
        **kwargs,
    ):
        super().__init__(mesh, **kwargs)
        self._mesh = mesh
        self._arts = list(articulations)
        self._origins_src = env_origins
        self._park_local = park_local.to(self._device, torch.float32)
        #: (N, 3) world-frame shift of each env's reset centre -- e.g. toward the robot, so a large
        #: sheet's far edge stays within reach. Not rotated by the yaw draw: it moves the sheet, it
        #: is not part of it.
        self._centre_offset = (
            torch.zeros((self._num_envs, 3), device=self._device)
            if centre_offset is None
            else centre_offset.to(self._device, torch.float32)
        )
        # Every plate frame sits at the sheet origin, so the root pose IS the sheet pose and the
        # offset starts at zero. Kept, and calibrated at startup like the chain's, because the
        # chain's analytically-zero-looking offset turned out to be off by twice its own value.
        self._root_offset = torch.zeros((mesh.num_shapes, 3), device=self._device)
        self._parked: list[torch.Tensor] = []
        self.select(
            torch.arange(self._num_envs, device=self._device),
            torch.zeros(self._num_envs, device=self._device, dtype=torch.long),
        )

    # ------------------------------------------------------------ selection

    def select(self, env_ids: torch.Tensor, shape_ids: torch.Tensor) -> None:
        """Make ``shape_ids[i]`` the active shape of ``env_ids[i]``. Takes effect at the next write.

        Recomputes the per-shape parked env lists here, at reset time, so :meth:`park` -- which runs
        every step -- does no data-dependent indexing of its own.
        """
        self._mesh.active[env_ids] = shape_ids.to(self._mesh.active.dtype)
        self._parked = [
            (self._mesh.active != s).nonzero(as_tuple=True)[0] for s in range(self._mesh.num_shapes)
        ]
        self._parked_counts = [int(p.numel()) for p in self._parked]

    @property
    def active(self) -> torch.Tensor:
        return self._mesh.active

    @property
    def _origins(self) -> torch.Tensor:
        src = self._origins_src
        return src() if callable(src) else src

    # ------------------------------------------------------------ writes

    def write_root_pose_to_sim(self, root_pose: torch.Tensor, env_ids=None) -> None:
        """Active shape flat at the requested XY and yaw (``ChainAsRigidObject`` semantics); the
        rest parked. Every shape of ``env_ids`` is stopped and flattened."""
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)
        env_ids = torch.as_tensor(env_ids, device=self._device, dtype=torch.long)
        n = env_ids.shape[0]

        centre = root_pose[:, :3] + self._centre_offset[env_ids]
        centre[:, 2] = self._spawn_z
        quat = _yaw_quat(task_yaw(root_pose))  # wxyz
        active = self._mesh.active[env_ids]
        park_q = torch.zeros((n, 4), device=self._device)
        park_q[:, 0] = 1.0
        for s, art in enumerate(self._arts):
            on = (active == s).unsqueeze(1)
            placed = centre + _quat_apply(quat, self._root_offset[s].expand(n, -1))
            parked = self._origins[env_ids] + self._park_local[s]
            pos = torch.where(on, placed, parked)
            q = torch.where(on, quat, park_q)
            pose = torch.cat((pos, q[:, 1:4], q[:, 0:1]), dim=-1)  # xyzw out, as documented
            _write_root_pose(art, pose, env_ids)
            self._stop(art, env_ids)

    def park(self) -> None:
        """Re-place every parked shape at its hover pose, at rest. Called once per policy step.

        A parked shape falls freely between calls -- about 1.4 mm in one 60 Hz step -- and touches
        nothing; writing it back keeps it from drifting off over an episode. Its hinges need no write:
        in free fall nothing loads them, and the reset that parked it left them flat.
        """
        for s, art in enumerate(self._arts):
            ids = self._parked[s]
            if self._parked_counts[s] == 0:
                continue
            pose = torch.zeros((ids.shape[0], 7), device=self._device)
            pose[:, :3] = self._origins[ids] + self._park_local[s]
            pose[:, 6] = 1.0
            _write_root_pose(art, pose, ids)
            art.write_root_velocity_to_sim(torch.zeros((ids.shape[0], 6), device=self._device), ids)

    def write_root_velocity_to_sim(self, root_velocity: torch.Tensor, env_ids=None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)
        env_ids = torch.as_tensor(env_ids, device=self._device, dtype=torch.long)
        for art in self._arts:
            self._stop(art, env_ids)

    def _stop(self, art, env_ids: torch.Tensor) -> None:
        art.write_root_velocity_to_sim(
            torch.zeros((env_ids.shape[0], 6), device=self._device), env_ids
        )
        zeros = torch.zeros((env_ids.shape[0], art.num_joints), device=self._device)
        _write_joints(art, zeros, zeros, env_ids, self._num_envs)
        if hasattr(art, "set_joint_position_target"):
            art.set_joint_position_target(zeros, env_ids=env_ids)

    def write_joint_positions(self, shape: int, position: torch.Tensor, env_ids=None) -> None:
        """Drive one shape's hinges, through the reset's own write path (the startup check uses it)."""
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)
        env_ids = torch.as_tensor(env_ids, device=self._device, dtype=torch.long)
        _write_joints(
            self._arts[shape], position, torch.zeros_like(position), env_ids, self._num_envs
        )

    def calibrate_root_offset(self, shape: int, delta_xy: torch.Tensor) -> None:
        """Shift one shape's root offset by a MEASURED placement error (see the chain's version)."""
        self._root_offset[shape, 0] += float(delta_xy[0])
        self._root_offset[shape, 1] += float(delta_xy[1])

    @property
    def root_offset(self) -> torch.Tensor:
        return self._root_offset

    @property
    def centre_offset(self) -> torch.Tensor:
        return self._centre_offset

    def set_external_force_and_torque(self, *args, **kwargs) -> None:
        return None


def _write_root_pose(art, pose_xyzw: torch.Tensor, env_ids: torch.Tensor) -> None:
    if hasattr(art, "write_root_pose_to_sim_index"):
        art.write_root_pose_to_sim_index(root_pose=pose_xyzw, env_ids=env_ids)
    else:
        art.write_root_pose_to_sim(pose_xyzw, env_ids)


def _write_joints(art, position, velocity, env_ids: torch.Tensor, num_envs: int) -> None:
    """``write_joint_state_to_sim``, with ``env_ids=None`` for a full write.

    Both rules are measured facts about this backend, from ``ChainAsRigidObject._write_joints``: the
    deprecated name is the one that reaches the solver, and an explicit full ``arange`` does nothing.
    """
    if int(env_ids.numel()) == num_envs:
        art.write_joint_state_to_sim(position, velocity)
    else:
        art.write_joint_state_to_sim(position, velocity, env_ids=env_ids)
