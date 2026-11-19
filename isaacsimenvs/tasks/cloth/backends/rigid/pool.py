"""The POOL of fold-shape articulations, as one sheet per env.

One policy trains on several shapes at once. Each shape is its own articulation; an env carries every
shape (one ACTIVE on the table, the others PARKED -- hovering metres away, where nothing can touch
them) or, with ``fold_pool.one_shape_per_env``, just one.

:class:`PlatePool` reads the ``resolution x resolution`` grid of the ACTIVE shape from its body poses
(``fold_shapes.grid_plate_map``) -- the points the task scores, the same layout for every shape --
and writes resets: the active shape placed, the rest parked.

Isaac Lab 2.3 on PhysX: wxyz quaternions on reads and writes; ``write_joint_state_to_sim`` takes
``env_ids`` normally. The read layout is still MEASURED at startup (``detect_quat_layout``); the
default only matters before that. Pure torch, no Isaac imports -- testable on a CPU.
"""

from __future__ import annotations

import torch

__all__ = ["PlatePool", "to_torch"]


def to_torch(value):
    """Isaac Lab 3.0 hands back warp-backed ProxyArrays; the task code wants torch."""
    return value.torch if hasattr(value, "torch") else value


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate ``v`` by wxyz quaternion ``q``, both ``(..., 3)`` / ``(..., 4)``. Local, to stay kit-free."""
    w = q[..., 0:1]
    xyz = q[..., 1:4]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)



def _sim_quat_to_wxyz(q: torch.Tensor, layout: str) -> torch.Tensor:
    """``(..., 4)`` quaternion as the articulation API reports it (``layout``), as wxyz.

    The resting value is ambiguous between the two layouts -- ``(0, 0, 0, 1)`` is the xyzw identity
    and also the wxyz half-turn about z -- and reading the wrong one mirrors every plate (a 100 mm
    sheet read as 47 mm). Hence the measurement in ``PlatePool.detect_quat_layout``.
    """
    if layout == "wxyz":
        return q
    if layout != "xyzw":
        raise ValueError(f"unknown quaternion layout {layout!r}; expected 'xyzw' or 'wxyz'")
    return torch.cat((q[..., 3:4], q[..., 0:3]), dim=-1)



def _yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
    """``(n,)`` yaw angles to ``(n, 4)`` wxyz quaternions about ``+z``."""
    half = 0.5 * yaw
    out = torch.zeros((yaw.shape[0], 4), device=yaw.device, dtype=yaw.dtype)
    out[:, 0] = torch.cos(half)
    out[:, 3] = torch.sin(half)
    return out



class PlatePool:
    """Every pool shape's articulation in every env, presented as one sheet per env.

    ``mappings[s]`` is ``fold_shapes.grid_plate_map(shapes[s], resolution)`` -- the same ordering
    for every shape, so the grid has one layout whatever shape an env is running. ``park_local`` is
    ``(S, 3)``: where each shape hovers when parked, relative to its env origin -- far from
    everything in the env and from each other (worlds never collide, so not from OTHER envs).
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
        park_local: torch.Tensor,
        centre_offset: torch.Tensor | None = None,
        spawn_z: float,
        env_origins,
        carried: torch.Tensor | None = None,
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
        # (S, P, 3) local points, in each plate's body frame (= sheet frame at rest).
        self._local = torch.tensor(
            [[pt for _, pt in m] for m in mappings], device=device, dtype=torch.float32
        )
        self._num_particles = self._local.shape[1]
        # Per-env in-plane size factor: each env's sheets were built at their own size, so the grid
        # point attached to a plate sits at the scaled position in that plate's frame. (N, 1, 3) so
        # it broadcasts over the grid.
        scale = (
            torch.ones(self._num_envs, device=device) if xy_scale is None else xy_scale.to(device)
        )
        self._scale3 = torch.stack((scale, scale, torch.ones_like(scale)), dim=-1).unsqueeze(1)
        self._body_idx: list[torch.Tensor] | None = None
        self._quat_layout = "wxyz"
        #: Which shape each env is running. Set by :meth:`select`.
        self.active = torch.zeros(self._num_envs, device=device, dtype=torch.long)
        #: Per shape, which envs carry it, as ``(S, N)`` bool. Every env carries every shape unless
        #: ``carried`` says otherwise (one shape per env). Resolved against the simulator's own
        #: instance order on first use (``_resolve_members``).
        self.carried = (
            torch.ones((len(self._arts), self._num_envs), device=device, dtype=torch.bool)
            if carried is None
            else carried.to(device, torch.bool)
        )
        self._members: list[torch.Tensor] | None = None
        self._local_of: torch.Tensor | None = None
        self._identity: list[bool] = []

        self._spawn_z = float(spawn_z)
        #: ``env_origins`` may be a tensor or a zero-argument callable returning one -- the backend
        #: builds the pool inside the scene hook, before its origins exist.
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
        # offset starts at zero. Kept, and calibrated at startup, because an analytically-zero
        # offset once turned out to be off by twice its own value.
        self._root_offset = torch.zeros((len(self._arts), 3), device=self._device)
        #: Per shape, ``(N, num_joints)`` hinge angles the active shape starts the episode at (see
        #: :meth:`set_start_joints`); zero = flat. Created on first use, when joint counts exist.
        self._start_q: dict[int, torch.Tensor] = {}
        self._parked: list[torch.Tensor] = []
        self._parked_local: list[torch.Tensor] | None = None
        self.select(torch.arange(self._num_envs, device=self._device), self.first_carried())

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
        """Which layout the articulation API reports, MEASURED against the analytic flat grid on env
        0's active shape (the resting value is ambiguous; see ``_sim_quat_to_wxyz``)."""
        want = torch.as_tensor(rest_local, device=self._device, dtype=torch.float32)
        want = want - want.mean(dim=0, keepdim=True)
        scores = {}
        for layout in ("xyzw", "wxyz"):
            self._quat_layout = layout
            got = self.positions()[0]
            got = got - got.mean(dim=0, keepdim=True)
            scores[layout] = float((got[:, :2] - want[:, :2]).norm(dim=-1).max())
        best = min(scores, key=scores.get)
        self._quat_layout = best
        self._quat_layout_scores = scores
        return best

    def _resolve_bodies(self) -> list[torch.Tensor]:
        """Per shape, grid vertex -> body index in THAT articulation, by name, on first use.

        Deferred and by name: an articulation has no ``body_names`` until the scene is played, and
        the solver's body order need not be the file's.
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
            state = to_torch(data.body_state_w)
            pos, quat = state[..., 0:3], state[..., 3:7]
        else:
            pos, quat = to_torch(pos), to_torch(quat)
        return pos, _sim_quat_to_wxyz(quat, self._quat_layout)

    def shape_positions(self, s: int) -> torch.Tensor:
        """``(num_envs, P, 3)`` grid of shape ``s`` in every env, active or not (zero where an env
        does not carry it)."""
        idx = self._resolve_bodies()[s]
        pos, quat = self._pose(self._arts[s])
        local = self._local[s].unsqueeze(0) * self._scale3[self.members(s)]
        return self._full(s, pos[:, idx, :] + _quat_apply(quat[:, idx, :], local))

    def positions(self) -> torch.Tensor:
        """``(num_envs, P, 3)``: each env's ACTIVE shape. Every shape is evaluated and selected --
        five gathers of 49 points beat a data-dependent branch inside a captured step."""
        out = None
        for s in range(self.num_shapes):
            p = self.shape_positions(s)
            sel = (self.active == s).view(-1, 1, 1)
            out = p if out is None else torch.where(sel, p, out)
        return out

    def velocities(self) -> torch.Tensor:
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
            lin = to_torch(lin)[:, idx, :]
            ang = to_torch(ang)[:, idx, :]
            quat = self._pose(art)[1][:, idx, :]
            local = self._local[s].unsqueeze(0) * self._scale3[self.members(s)]
            v = self._full(s, lin + torch.cross(ang, _quat_apply(quat, local), dim=-1))
            sel = (self.active == s).view(-1, 1, 1)
            out = v if out is None else torch.where(sel, v, out)
        return out

    # ------------------------------------------------------------ selection

    def select(self, env_ids: torch.Tensor, shape_ids: torch.Tensor) -> None:
        """Make ``shape_ids[i]`` the active shape of ``env_ids[i]``. Takes effect at the next write.

        Recomputes the per-shape parked env lists here, at reset time, so :meth:`park` -- which runs
        every step -- does no data-dependent indexing of its own. An env can only run a shape it
        carries (``carried``).
        """
        shape_ids = shape_ids.to(self.active.device, self.active.dtype)
        env_ids = torch.as_tensor(env_ids, device=self.active.device, dtype=torch.long)
        carried = self.carried
        if not bool(carried.all()) and not bool(carried[shape_ids, env_ids].all()):
            raise ValueError("select: an env was asked to run a pool shape it does not carry")
        self.active[env_ids] = shape_ids
        parked = [
            ((self.active != s) & carried[s]).nonzero(as_tuple=True)[0]
            for s in range(self.num_shapes)
        ]
        self._parked = parked
        self._parked_counts = [int(p.numel()) for p in parked]
        self._parked_local = None  # instance indices: resolved lazily, once the simulator exists

    def first_carried(self) -> torch.Tensor:
        """``(N,)`` each env's lowest-index carried shape: 0 everywhere unless one shape per env."""
        return self.carried.to(torch.int8).argmax(dim=0).to(self.active.device, torch.long)

    @property
    def _origins(self) -> torch.Tensor:
        src = self._origins_src
        return src() if callable(src) else src

    # ------------------------------------------------------------ writes

    def place(self, env_ids: torch.Tensor, centre_xy: torch.Tensor, yaw: torch.Tensor) -> None:
        """Lay the active shape of ``env_ids`` at ``centre_xy`` (n, 2, world) and heading ``yaw``
        (n,), at the spawn height, at rest -- at its start angles (:meth:`set_start_joints`), else
        flat; park the rest of each env's shapes, flat and at rest."""
        env_ids = torch.as_tensor(env_ids, device=self._device, dtype=torch.long)
        n = env_ids.shape[0]

        centre = torch.zeros((n, 3), device=self._device)
        centre[:, :2] = centre_xy
        centre = centre + self._centre_offset[env_ids]
        centre[:, 2] = self._spawn_z
        quat = _yaw_quat(yaw)  # wxyz
        active = self.active[env_ids]
        park_q = torch.zeros((n, 4), device=self._device)
        park_q[:, 0] = 1.0
        for s, art in enumerate(self._arts):
            on = (active == s).unsqueeze(1)
            placed = centre + _quat_apply(quat, self._root_offset[s].expand(n, -1))
            parked = self._origins[env_ids] + self._park_local[s]
            pose = torch.cat((torch.where(on, placed, parked), torch.where(on, quat, park_q)), dim=-1)
            keep, local = self.local_rows(s, env_ids)
            if keep is not None:
                pose = pose[keep]
            if local.numel():
                art.write_root_pose_to_sim(pose, local)
            self._stop(s, env_ids)

    def park(self) -> None:
        """Re-place every parked shape at its hover pose, at rest. Called once per policy step.

        A parked shape falls freely between calls -- about 1.4 mm in one 60 Hz step -- and touches
        nothing; writing it back keeps it from drifting off over an episode. Its hinges need no write:
        in free fall nothing loads them, and the reset that parked it left them flat. With one shape
        per env nothing is ever parked and this writes nothing.
        """
        if self._parked_local is None:
            self._parked_local = [self.local_rows(s, p)[1] for s, p in enumerate(self._parked)]
        for s, art in enumerate(self._arts):
            if self._parked_counts[s] == 0:
                continue
            ids, local = self._parked[s], self._parked_local[s]
            pose = torch.zeros((ids.shape[0], 7), device=self._device)
            pose[:, :3] = self._origins[ids] + self._park_local[s]
            pose[:, 3] = 1.0  # identity, wxyz
            art.write_root_pose_to_sim(pose, local)
            art.write_root_velocity_to_sim(torch.zeros((ids.shape[0], 6), device=self._device), local)

    def _stop(self, s: int, env_ids: torch.Tensor) -> None:
        """Shape ``s`` at rest in ``env_ids``: flat, or -- where it is the active shape -- at its start
        angles."""
        art = self._arts[s]
        keep, local = self.local_rows(s, env_ids)
        if keep is not None:
            env_ids = env_ids[keep]
        if local.numel() == 0:
            return
        art.write_root_velocity_to_sim(torch.zeros((env_ids.shape[0], 6), device=self._device), local)
        zeros = torch.zeros((env_ids.shape[0], art.num_joints), device=self._device)
        q = zeros
        if s in self._start_q:
            on = (self.active[env_ids] == s).unsqueeze(1)
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
        keep, local = self.local_rows(shape, env_ids)
        if keep is not None:
            position = position[keep]
        self._write_joints(self._arts[shape], position, torch.zeros_like(position), local)

    def calibrate_root_offset(self, shape: int, delta_xy: torch.Tensor) -> None:
        """Shift one shape's root offset by a MEASURED placement error."""
        self._root_offset[shape, 0] += float(delta_xy[0])
        self._root_offset[shape, 1] += float(delta_xy[1])

    @property
    def root_offset(self) -> torch.Tensor:
        return self._root_offset

    @property
    def centre_offset(self) -> torch.Tensor:
        return self._centre_offset

    # PhysX: wxyz in, ``env_ids`` as given.

    def _write_joints(self, art, position, velocity, env_ids: torch.Tensor) -> None:
        """``env_ids`` are the articulation's own instance indices (see ``local_rows``)."""
        art.write_joint_state_to_sim(position, velocity, env_ids=env_ids)
