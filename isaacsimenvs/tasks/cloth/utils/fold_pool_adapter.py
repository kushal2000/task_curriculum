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
        #: Per shape, which envs carry it, as ``(S, N)`` bool. Every env carries every shape unless
        #: the env says otherwise (``set_carried``: one shape per env on PhysX). Resolved against the
        #: simulator's own instance order on first use (``_resolve_members``).
        self.carried = torch.ones((len(self._arts), self._num_envs), device=device, dtype=torch.bool)
        self._members: list[torch.Tensor] | None = None
        self._local_of: torch.Tensor | None = None
        self._identity: list[bool] = []

    def set_carried(self, carried: torch.Tensor) -> None:
        """``(S, N)`` bool: which envs carry each shape. Before first use (membership is resolved once)."""
        if self._members is not None:
            raise RuntimeError("pool membership already resolved; set_carried must come first")
        self.carried = carried.to(self._device, torch.bool)

    def _resolve_members(self) -> list[torch.Tensor]:
        """Per shape, the global env id of each of its articulation's instances, in instance order.

        Read from the simulator (``root_physx_view.prim_paths``) where it can be, never assumed: an
        articulation spanning a subset of envs indexes them 0..n_s-1 in ITS order. Must agree with
        ``carried``; with every shape in every env it must be the identity, or every per-env read and
        write of the pool would have been landing in the wrong env all along.
        """
        if self._members is not None:
            return self._members
        import re

        members, local_of, ident = [], [], []
        for s, art in enumerate(self._arts):
            want = self.carried[s].nonzero(as_tuple=True)[0]
            paths = None
            view = getattr(art, "root_physx_view", None)
            if view is not None:
                paths = getattr(view, "prim_paths", None)
            if paths is not None:
                ids = []
                for path in paths:
                    m = re.search(r"/env_(\d+)/", path)
                    if m is None:
                        raise RuntimeError(f"pool articulation {s}: cannot read an env index from {path}")
                    ids.append(int(m.group(1)))
                got = torch.tensor(ids, device=self._device, dtype=torch.long)
                if sorted(ids) != want.tolist():
                    raise RuntimeError(
                        f"pool articulation {s} has instances in {len(ids)} envs, expected the "
                        f"{int(want.numel())} that carry it"
                    )
            else:
                if int(want.numel()) != self._num_envs:
                    raise RuntimeError("a subset pool needs the simulator's prim paths to map envs")
                got = want
            members.append(got)
            lo = torch.full((self._num_envs,), -1, device=self._device, dtype=torch.long)
            lo[got] = torch.arange(got.numel(), device=self._device)
            local_of.append(lo)
            ident.append(
                int(got.numel()) == self._num_envs
                and bool((got == torch.arange(self._num_envs, device=self._device)).all())
            )
            if bool(self.carried[s].all()) and not ident[-1]:
                raise RuntimeError(
                    f"pool articulation {s} spans every env but not in env order -- every per-env "
                    f"read and write would go to the wrong env"
                )
        self._members, self._local_of, self._identity = members, torch.stack(local_of), ident
        return members

    def members(self, s: int) -> torch.Tensor:
        """Global env ids of shape ``s``'s instances, in instance order."""
        return self._resolve_members()[s]

    def local_rows(self, s: int, env_ids: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor]:
        """``(keep, local)``: which of ``env_ids`` carry shape ``s`` (``None`` = all of them) and
        their instance indices in shape ``s``'s articulation."""
        self._resolve_members()
        if self._identity[s]:
            return None, env_ids
        local = self._local_of[s][env_ids]
        keep = local >= 0
        return keep, local[keep]

    def count(self, s: int) -> int:
        return int(self._resolve_members()[s].numel())

    def _full(self, s: int, t: torch.Tensor) -> torch.Tensor:
        """An ``(n_s, ...)`` per-instance tensor of shape ``s`` as ``(N, ...)``, zero where absent."""
        self._resolve_members()
        if self._identity[s]:
            return t
        out = torch.zeros((self._num_envs, *t.shape[1:]), device=t.device, dtype=t.dtype)
        out[self._members[s]] = t
        return out

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
        """``(num_envs, P, 3)`` grid of shape ``s`` in every env, active or not (zero where an env
        does not carry it)."""
        idx = self._resolve_bodies()[s]
        pos, quat = self._pose(self._arts[s])
        local = self._local[s].unsqueeze(0) * self._scale3[self.members(s)]
        return self._full(s, pos[:, idx, :] + _quat_apply(quat[:, idx, :], local))

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
            local = self._local[s].unsqueeze(0) * self._scale3[self.members(s)]
            v = self._full(s, lin + torch.cross(ang, _quat_apply(quat, local), dim=-1))
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
        #: Per shape, ``(N, num_joints)`` hinge angles the active shape starts the episode at (see
        #: :meth:`set_start_joints`); zero = flat. Created on first use, when joint counts exist.
        self._start_q: dict[int, torch.Tensor] = {}
        self._parked: list[torch.Tensor] = []
        self._parked_local: list[torch.Tensor] | None = None
        self.select(torch.arange(self._num_envs, device=self._device), self.first_carried())

    # ------------------------------------------------------------ selection

    def select(self, env_ids: torch.Tensor, shape_ids: torch.Tensor) -> None:
        """Make ``shape_ids[i]`` the active shape of ``env_ids[i]``. Takes effect at the next write.

        Recomputes the per-shape parked env lists here, at reset time, so :meth:`park` -- which runs
        every step -- does no data-dependent indexing of its own. An env can only run a shape it
        carries (``PoolAsClothMesh.carried``).
        """
        shape_ids = shape_ids.to(self._mesh.active.device, self._mesh.active.dtype)
        env_ids = torch.as_tensor(env_ids, device=self._mesh.active.device, dtype=torch.long)
        carried = self._mesh.carried
        if not bool(carried.all()) and not bool(carried[shape_ids, env_ids].all()):
            raise ValueError("select: an env was asked to run a pool shape it does not carry")
        self._mesh.active[env_ids] = shape_ids
        parked = [
            ((self._mesh.active != s) & carried[s]).nonzero(as_tuple=True)[0]
            for s in range(self._mesh.num_shapes)
        ]
        self._parked = parked
        self._parked_counts = [int(p.numel()) for p in parked]
        self._parked_local = None  # instance indices: resolved lazily, once the simulator exists

    def first_carried(self) -> torch.Tensor:
        """``(N,)`` each env's lowest-index carried shape: 0 everywhere unless one shape per env."""
        return self._mesh.carried.to(torch.int8).argmax(dim=0).to(self._mesh.active.device, torch.long)

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
            keep, local = self._mesh.local_rows(s, env_ids)
            if keep is not None:
                pose = pose[keep]
            if local.numel():
                self._write_root_pose(art, pose, local)
            self._stop(s, env_ids)

    def park(self) -> None:
        """Re-place every parked shape at its hover pose, at rest. Called once per policy step.

        A parked shape falls freely between calls -- about 1.4 mm in one 60 Hz step -- and touches
        nothing; writing it back keeps it from drifting off over an episode. Its hinges need no write:
        in free fall nothing loads them, and the reset that parked it left them flat. With one shape
        per env nothing is ever parked and this writes nothing.
        """
        if self._parked_local is None:
            self._parked_local = [self._mesh.local_rows(s, p)[1] for s, p in enumerate(self._parked)]
        for s, art in enumerate(self._arts):
            if self._parked_counts[s] == 0:
                continue
            ids, local = self._parked[s], self._parked_local[s]
            pose = torch.zeros((ids.shape[0], 7), device=self._device)
            pose[:, :3] = self._origins[ids] + self._park_local[s]
            pose[:, 6] = 1.0
            self._write_root_pose(art, pose, local)
            art.write_root_velocity_to_sim(torch.zeros((ids.shape[0], 6), device=self._device), local)

    def write_root_velocity_to_sim(self, root_velocity: torch.Tensor, env_ids=None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)
        env_ids = torch.as_tensor(env_ids, device=self._device, dtype=torch.long)
        for s in range(len(self._arts)):
            self._stop(s, env_ids)

    def _stop(self, s: int, env_ids: torch.Tensor) -> None:
        """Shape ``s`` at rest in ``env_ids``: flat, or -- where it is the active shape -- at its start
        angles. The reset writes root pose and then root velocity, and both stop every shape, so the
        start angles must come from here or the second write would flatten them again."""
        art = self._arts[s]
        keep, local = self._mesh.local_rows(s, env_ids)
        if keep is not None:
            env_ids = env_ids[keep]
        if local.numel() == 0:
            return
        art.write_root_velocity_to_sim(torch.zeros((env_ids.shape[0], 6), device=self._device), local)
        zeros = torch.zeros((env_ids.shape[0], art.num_joints), device=self._device)
        q = zeros
        if s in self._start_q:
            on = (self._mesh.active[env_ids] == s).unsqueeze(1)
            q = torch.where(on, self._start_q[s][env_ids], zeros)
        self._write_joints(art, q, zeros, local)
        if hasattr(art, "set_joint_position_target"):
            art.set_joint_position_target(zeros, env_ids=local)

    def set_start_joints(self, shape: int, env_ids: torch.Tensor, position: torch.Tensor) -> None:
        """Hinge angles ``shape`` starts at in ``env_ids`` when it is their active shape; applied by
        the next root-pose or root-velocity write (the reset). Zeros = flat."""
        art = self._arts[shape]
        if shape not in self._start_q:
            self._start_q[shape] = torch.zeros((self._num_envs, art.num_joints), device=self._device)
        self._start_q[shape][env_ids] = position.to(self._device, torch.float32)

    def write_joint_positions(self, shape: int, position: torch.Tensor, env_ids=None) -> None:
        """Drive one shape's hinges, through the reset's own write path (the startup check uses it).
        ``position`` has one row per entry of ``env_ids``; envs not carrying the shape are skipped."""
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)
        env_ids = torch.as_tensor(env_ids, device=self._device, dtype=torch.long)
        keep, local = self._mesh.local_rows(shape, env_ids)
        if keep is not None:
            position = position[keep]
        self._write_joints(self._arts[shape], position, torch.zeros_like(position), local)

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

    # The two sim writes, as methods so a backend with other conventions can override them
    # (``physx_adapters``); these are the Newton ones.

    def _write_root_pose(self, art, pose_xyzw: torch.Tensor, env_ids: torch.Tensor) -> None:
        _write_root_pose(art, pose_xyzw, env_ids)

    def _write_joints(self, art, position, velocity, env_ids: torch.Tensor) -> None:
        """``env_ids`` are the articulation's own instance indices (see ``local_rows``)."""
        _write_joints(art, position, velocity, env_ids, self._mesh.count(self._arts.index(art)))


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
        # A full write takes rows in instance order; mapped instance ids need not arrive sorted.
        order = torch.argsort(env_ids)
        art.write_joint_state_to_sim(position[order], velocity[order])
    else:
        art.write_joint_state_to_sim(position, velocity, env_ids=env_ids)
