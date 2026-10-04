"""Present a hinged-slat ``Articulation`` as the cloth ``ClothEnv`` already knows how to score.

``ClothEnv`` reads its manipuland through exactly two surfaces:

    ``self._cloth``   a deformable, for ``data.nodal_pos_w`` / ``data.nodal_vel_w``
    ``self.object``   a ``ClothAsRigidObject``, for the rigid-object reads the Play task performs

Both are about a **particle cloud on a ``resolution x resolution`` grid**. A rigid chain has bodies,
so :class:`ChainAsClothMesh` reconstructs that cloud from ``body_pos_w`` / ``body_quat_w`` by
rigidly attaching each grid vertex to the slat that contains it
(:func:`rigid_cloth.sheet_grid_slat_map`). The cloud has the same length, the same ordering and the
same units as the VBD sheet's, so ``fold_error``, ``footprint_ratio``, ``_stationary_frame``, the
Kabsch orientation fit, the goal marker and the success criterion all carry over **unmodified** --
and `cloth_env.py` is not touched.

That identity is the point. Two runs that share reward, goal, observation layout and success test and
differ only in the manipuland's dynamics are comparable; two runs that also differ in how the fold is
measured are not, and the difference would be invisible in the learning curves.

**What is genuinely different, and is overridden rather than inherited.** A chain HAS a rigid root
pose, so reset writes one instead of teleporting a particle cloud: :class:`ChainAsRigidObject`
replaces ``write_root_pose_to_sim`` with a root-pose + zero-joint write. The inherited version
builds a flat yawed cloud and pushes it through ``write_nodal_pos_to_sim_index``, which for an
articulation would mean decoding a pose back out of the cloud -- a round trip that can only lose
information the articulation already has exactly.
"""

from __future__ import annotations

import torch

from isaacsimenvs.tasks.cloth.utils import rigid_cloth as rc
from isaacsimenvs.tasks.cloth.utils.cloth_adapter import ClothAsRigidObject, _to_torch, task_yaw

__all__ = ["ChainAsClothMesh", "ChainAsRigidObject"]


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate ``v`` by wxyz quaternion ``q``, both ``(..., 3)`` / ``(..., 4)``.

    Local rather than imported from ``isaaclab.utils.math`` so this module -- and its tests -- run
    without Isaac Sim. The whole adapter is pure torch on purpose: the grid mapping and the fold
    geometry are the parts most likely to be silently wrong, and they are testable on a CPU.
    """
    w = q[..., 0:1]
    xyz = q[..., 1:4]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


#: Component order the articulation API reports body orientations in. ``"xyzw"`` is what Isaac Lab 3.0
#: documents ("quaternion orientation in (x, y, z, w)") and the default here; ``"wxyz"`` is the older
#: convention. :meth:`ChainAsClothMesh.detect_quat_layout` overrides it per-run by MEASUREMENT, because
#: the resting value ``(-0.0, 0.0, 0.0, 1.0)`` is genuinely ambiguous -- it is the xyzw identity and
#: also the wxyz half-turn about z, and the two differ by exactly the mirror that made a 100 mm sheet
#: report a 47 mm cloud.
_DEFAULT_QUAT_LAYOUT = "xyzw"


def _sim_quat_to_wxyz(q: torch.Tensor, layout: str = _DEFAULT_QUAT_LAYOUT) -> torch.Tensor:
    """``(..., 4)`` quaternion from the Isaac Lab 3.0 articulation API, as wxyz.

    **Isaac Lab 3.0's articulation API is xyzw, not wxyz.** Its own docstrings say so --
    "the cartesian position and quaternion orientation in (x, y, z, w)" on
    ``write_root_pose_to_sim_index`` in both ``base_articulation.py`` and the Newton backend -- and the
    chain confirms it: every slat's resting orientation reads ``(-0.0, 0.0, 0.0, 1.0)``, which is the
    xyzw identity.

    Read as wxyz that same array is ``w = 0, z = 1``: a **180 degree rotation about z**. Applied to
    each grid vertex's offset inside its slat, it mirrored the sheet about its own centre line, and the
    emulated cloud came out **47 mm wide across a 100 mm sheet** (76.5 mm for the 17-slat chains --
    both reproduced to 0.1 mm by working the mirror through by hand). Nothing raised: a 47 mm cloud is
    a perfectly plausible *crumpled sheet*, so it read as a physics problem, and the hunt went to
    spawn height, armature, damping and self-collision before the convention.

    Local rather than ``isaaclab.utils.math`` so this module stays kit-free and CPU-testable.
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


class ChainAsClothMesh:
    """A slat ``Articulation`` seen as the cloth's particle grid.

    Implements the four calls ``ClothAsRigidObject`` makes on a deformable, and nothing else --
    anything it misses is meant to surface as an ``AttributeError`` rather than as a plausible wrong
    number.
    """

    def __init__(
        self,
        articulation,
        *,
        spec: rc.ChainSpec,
        resolution: int,
        num_envs: int,
        device,
    ) -> None:
        self._art = articulation
        self._spec = spec
        self._resolution = int(resolution)
        self._num_envs = int(num_envs)
        self._device = device
        self.data = _ChainMeshData(self)

        # The grid-to-slat map is pure geometry and is computed now, so a bad `resolution` or a
        # non-square sheet still raises at construction rather than at the first step.
        self._mapping = rc.sheet_grid_slat_map(spec, resolution)
        self._local = torch.tensor(
            [pt for _, pt in self._mapping], device=device, dtype=torch.float32
        )  # (P, 3) in each slat's own link frame
        self._num_particles = self._local.shape[0]
        self._body_idx: torch.Tensor | None = None
        self._quat_layout = _DEFAULT_QUAT_LAYOUT

    # ------------------------------------------------------------ quaternion layout

    @property
    def quat_layout(self) -> str:
        return self._quat_layout

    def detect_quat_layout(self, rest_local) -> str:
        """Pick the component order that reproduces the KNOWN flat grid, and return it.

        Not a guess and not a constant, because the observation cannot settle it: a resting slat reads
        ``(-0.0, 0.0, 0.0, 1.0)``, which is simultaneously the xyzw identity and the wxyz half-turn
        about z. Isaac Lab 3.0 documents xyzw, but the two readings differ by a 180 degree in-plane
        rotation of every grid vertex within its slat, and BOTH produce a plausible cloud: one a 47 mm
        "crumpled" sheet, the other a correctly-sized sheet facing backwards. Neither raises.

        So the ambiguity is resolved against ground truth instead. The caller guarantees the chain is at
        its flat rest pose; the analytic grid for that pose is known exactly; only one layout reproduces
        it. Whichever wins is recorded, so a future API change is a printed line rather than a silent
        mirror.

        ``rest_local`` is the analytic flat grid, ``(P, 3)`` in the sheet frame and in this cloud's
        ordering.
        """
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

    # ------------------------------------------------------------ body resolution

    def _resolve_bodies(self) -> torch.Tensor:
        """Slat index -> solver body index, resolved by NAME on first use and cached.

        **Deferred deliberately.** The chain is created inside ``pre_clone_scene_hook``, i.e. during
        ``_setup_scene``, and an Isaac Lab ``Articulation`` has no ``_data`` until the scene has been
        played -- ``body_names`` raises ``AttributeError`` there. Resolving at construction therefore
        cannot work, and the first version of this class did exactly that.

        Resolution is by name because ``body_names`` order is the SOLVER's, not the URDF's: the
        importer may reorder, and Newton's body list has been observed to differ from the URDF's
        declaration order. Matching by position instead would silently attach the grid to the wrong
        bodies, which reads as a sheet that bends in the wrong places rather than as an error.
        """
        if self._body_idx is not None:
            return self._body_idx
        names = [rc.slat_link_name(i) for i in range(self._spec.num_slats)]
        body_names = list(self._art.body_names)
        missing = [n for n in names if n not in body_names]
        if missing:
            raise RuntimeError(
                f"articulation is missing {len(missing)} slat links (first: {missing[0]}); "
                f"has {body_names}"
            )
        slat_to_body = [body_names.index(n) for n in names]
        self._body_idx = torch.tensor(
            [slat_to_body[slat] for slat, _ in self._mapping],
            device=self._device,
            dtype=torch.long,
        )
        return self._body_idx

    # ---------------------------------------------------------------- reads

    def _body_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``(N, B, 3)`` positions and ``(N, B, 4)`` wxyz orientations of every body.

        Falls back to slicing ``body_state_w`` because the Newton boundary layer is selective about
        what it converts: ``patches.TORCH_ATTRS`` lists ``body_state_w`` and does **not** list
        ``body_pos_w`` or ``body_quat_w``. Those two may therefore be absent, or be warp-backed, on
        an articulation the patch layer never wrapped -- which this one is, since it is held under
        ``scene.articulations["rigid_cloth"]`` rather than one of the five names the patch knows.
        """
        data = self._art.data
        pos = getattr(data, "body_pos_w", None)
        quat = getattr(data, "body_quat_w", None)
        if pos is None or quat is None:
            state = _to_torch(data.body_state_w)
            pos, quat = state[..., 0:3], state[..., 3:7]
        else:
            pos, quat = _to_torch(pos), _to_torch(quat)
        # Both accessors hand back xyzw on this backend -- verified on the live solver, where
        # `body_pos_w`, `body_quat_w`, `body_link_pose_w` and `body_state_w` all report the resting
        # slat as (-0.0, 0.0, 0.0, 1.0).
        return pos, _sim_quat_to_wxyz(quat, self._quat_layout)

    def _positions(self) -> torch.Tensor:
        """``(num_envs, P, 3)`` world positions of the emulated particle grid."""
        idx = self._resolve_bodies()
        pos_all, quat_all = self._body_pose()
        pos = pos_all[:, idx, :]     # (N, P, 3)
        quat = quat_all[:, idx, :]   # (N, P, 4)
        local = self._local.unsqueeze(0).expand(pos.shape[0], -1, -1)
        return pos + _quat_apply(quat, local)

    def _velocities(self) -> torch.Tensor:
        """``(num_envs, P, 3)`` world velocities, ``v + omega x r``.

        The cross term is not decoration: a folding slat's far edge moves almost entirely by
        rotation, so dropping it would report the fastest part of the sheet as the slowest, and
        ``root_lin_vel_w`` is one of the policy's observations.
        """
        art = self._art.data
        # Isaac Lab exposes both the link-frame and the COM-frame velocity and has renamed them
        # across versions; `body_lin_vel_w` is the link-frame one, which is the frame `_local` is
        # expressed in. Fall back rather than silently mixing frames.
        lin = getattr(art, "body_lin_vel_w", None)
        if lin is None:
            lin = art.body_com_lin_vel_w
        ang = getattr(art, "body_ang_vel_w", None)
        if ang is None:
            ang = art.body_com_ang_vel_w
        idx = self._resolve_bodies()
        lin = _to_torch(lin)[:, idx, :]
        ang = _to_torch(ang)[:, idx, :]
        quat = self._body_pose()[1][:, idx, :]
        local = self._local.unsqueeze(0).expand(lin.shape[0], -1, -1)
        return lin + torch.cross(ang, _quat_apply(quat, local), dim=-1)

    # ---------------------------------------------------------------- writes

    def write_nodal_pos_to_sim_index(self, target, env_ids) -> None:
        raise NotImplementedError(
            "a chain is repositioned by its root pose, not by its particle cloud -- "
            "ChainAsRigidObject.write_root_pose_to_sim does it directly"
        )

    def write_nodal_velocity_to_sim_index(self, target, env_ids) -> None:
        raise NotImplementedError(
            "a chain is stopped by zeroing root and joint velocities -- see "
            "ChainAsRigidObject.write_root_velocity_to_sim"
        )


class _ChainMeshData:
    """The ``.data`` surface a deformable presents, backed by articulation body poses."""

    def __init__(self, owner: ChainAsClothMesh) -> None:
        self._owner = owner

    @property
    def nodal_pos_w(self) -> torch.Tensor:
        return self._owner._positions().reshape(-1, 3)

    @property
    def nodal_vel_w(self) -> torch.Tensor:
        return self._owner._velocities().reshape(-1, 3)

    def __getattr__(self, name):
        """Anything else is the articulation's own -- joint states included."""
        return getattr(self._owner._art.data, name)


class ChainAsRigidObject(ClothAsRigidObject):
    """``ClothAsRigidObject`` with the two writes that a rigid chain can do honestly.

    Every read -- centroid position, Kabsch orientation, keypoint velocity, mass -- is inherited
    unchanged, so the policy's observation of the manipuland is computed by the same code for both
    manipulands.
    """

    def __init__(self, mesh: ChainAsClothMesh, *, articulation, spec: rc.ChainSpec, **kwargs):
        super().__init__(mesh, **kwargs)
        self._art = articulation
        self._spec = spec
        # The articulation's ROOT LINK is `slat_<root>`, whose frame sits at that slat's centre --
        # not at the sheet's centre. `chain_frames` puts it at `centres[root]`, which for an odd
        # mid-hinge chain is ~3 mm off-centre and for `box3-mid` is 25 mm off-centre. Writing the
        # sheet's requested centre straight into the root pose would offset the whole sheet by that
        # much, every reset, in a direction that rotates with the yaw draw.
        self._root_offset = torch.tensor(
            [rc.chain_frames(spec, None)[spec.root][0], 0.0, 0.0],
            device=self._device,
            dtype=torch.float32,
        )

    def write_root_pose_to_sim(self, root_pose: torch.Tensor, env_ids=None) -> None:
        """Restore the chain flat, centred on the requested XY, at the requested yaw.

        Same contract as the cloth's: XY and yaw are honoured, height is the configured spawn
        height, roll and pitch are dropped (they would stand the sheet on edge). Unlike the cloth's,
        this is an exact rigid placement -- the chain is flat because every joint is written to
        zero, not because a cloud of points was recomputed.
        """
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)
        env_ids = torch.as_tensor(env_ids, device=self._device, dtype=torch.long)

        centre = root_pose[:, :3].clone()
        centre[:, 2] = self._spawn_z

        # The cloth's own decode, so a reset draw puts the chain where it would have put the cloth.
        yaw = task_yaw(root_pose)
        quat = _yaw_quat(yaw)

        pose = torch.zeros((env_ids.shape[0], 7), device=self._device, dtype=torch.float32)
        pose[:, :3] = centre + _quat_apply(quat, self._root_offset.expand(env_ids.shape[0], -1))
        # **xyzw going out, as documented, and xyzw coming back.** The root pose is a
        # `wp.transformf`, whose quaternion is (x, y, z, w); the Newton backend writes it straight
        # into the free joint's `joint_q[3:7]`. `scripts/analysis/rigid_cloth_yaw_probe.py` measured
        # it from slat POSITIONS, with no quaternion convention involved: an xyzw write turned the
        # chain to exactly the requested 0 / 45 / 90 / -120 deg, and the slats read back the same.
        #
        # This used to send the wxyz `_yaw_quat` unconverted. Combined with the decode above, a
        # pinned request (identity, i.e. 180 deg after `task_yaw`) came out as the identity by
        # accident, and every other heading came out as a ROLL about the chain's long axis -- so the
        # chain never turned in the plane, and at a random reset it was flipped over instead.
        pose[:, 3] = quat[:, 1]
        pose[:, 4] = quat[:, 2]
        pose[:, 5] = quat[:, 3]
        pose[:, 6] = quat[:, 0]

        self._write_root_pose(pose, env_ids)
        self._stop(env_ids)

    # ------------------------------------------------------------ sim writes

    def _write_root_pose(self, pose_xyzw: torch.Tensor, env_ids: torch.Tensor) -> None:
        """Write a root pose, preferring the CURRENT API over the deprecated alias.

        ``pose_xyzw`` is ``(n, 7)``: position, then an **(x, y, z, w)** quaternion, which is what
        ``write_root_pose_to_sim_index`` documents and what the backend does. An earlier version of
        this adapter concluded the opposite ("writes are wxyz") from a test whose pose was being
        overwritten before it was ever simulated; `scripts/analysis/rigid_cloth_yaw_probe.py`
        settles it from slat positions alone.
        """
        art = self._art
        if hasattr(art, "write_root_pose_to_sim_index"):
            art.write_root_pose_to_sim_index(root_pose=pose_xyzw, env_ids=env_ids)
        else:
            art.write_root_pose_to_sim(pose_xyzw, env_ids)

    def _write_joints(
        self, position: torch.Tensor, velocity: torch.Tensor, env_ids: torch.Tensor
    ) -> None:
        """Write joint state, preferring the current API for the same reason as the root pose.

        A dropped joint write is worse than a dropped yaw: ``_write_flat_joints`` is how a reset
        un-folds the chain, so a chain that finished an episode folded would START the next one
        folded -- already at the goal, on an env the task believes it has reset.

        **``write_joint_state_to_sim`` is the one that works**, even though it is the deprecated name.
        The Newton articulation overrides it (the base class only raises ``NotImplementedError``),
        whereas ``write_joint_position_to_sim_index`` -- the current, documented entry point -- is
        accepted and silently does nothing: driving the hinges to 90 degrees through it left them at
        0.03 degrees, while the same request through this call reached 90.05 degrees. Verified both
        ways round on the live solver, so this is a measured fact about the backend and not a
        preference for the older API.
        """
        # `env_ids=None` when the write covers every env. Passing an explicit full `arange` instead
        # was measured to do NOTHING -- hinges driven to 90 degrees stayed at 0.07 degrees over 120
        # steps -- while the identical call with `env_ids=None` reached 90.05 degrees. The partial
        # case is exercised by `RigidClothEnv`'s startup check, because a partial reset is what
        # training actually does and a silent failure there would leave a folded chain folded at the
        # start of the next episode.
        if env_ids is not None and int(env_ids.numel()) == self._num_envs:
            self._art.write_joint_state_to_sim(position, velocity)
        else:
            self._art.write_joint_state_to_sim(position, velocity, env_ids=env_ids)

    def calibrate_root_offset(self, delta_xy: torch.Tensor) -> None:
        """Shift the root offset by a MEASURED placement error.

        The offset starts at ``chain_frames(spec)[root].x``, which is where the root slat's frame sits
        in the sheet frame and is the analytically correct correction. On the live solver it was wrong
        by exactly ``-2 x`` that value -- 51.00 mm for ``box3-mid``, 11.76 mm for ``box-mid-odd``,
        uniform across every slat, with the chain's internal spacing matching the URDF to 0.01 mm. So
        the chain's *shape* is imported exactly and only its placement disagrees.

        Measured on the live solver, the placement error is **linear in the offset with slope -1**:

            offset -25.5 mm -> centroid error +51.00 mm
            offset -76.5 mm -> centroid error +102.00 mm      (d err / d offset = -1)
            offset -178.5 mm -> centroid error +204.00 mm

        so one step lands exactly, and it converges on ``-chain_frames(spec)[root].x`` -- the opposite
        sign to the analytic derivation. Rather than flip a sign to fit, the env measures the error at
        startup and applies it here, then re-measures and refuses to run if it did not close. That is
        robust to whatever the importer does next, and it fails loudly instead of silently: a sheet
        placed 51 mm off still folds, still scores, and still trains -- against a fold target built
        around the wrong place -- so nothing downstream would report it.
        """
        self._root_offset[0] += float(delta_xy[0])
        self._root_offset[1] += float(delta_xy[1])

    @property
    def root_offset(self) -> torch.Tensor:
        return self._root_offset

    def write_root_velocity_to_sim(self, root_velocity: torch.Tensor, env_ids=None) -> None:
        """Bring the chain to rest -- root AND joints.

        Zeroing only the root leaves the hinges spinning, so the sheet "resets" and then flails.
        The task's requested velocity is deliberately discarded for the same reason the cloth
        discards it: applying it would launch the manipuland at reset.

        Writes the flat pose again rather than reading the joint positions back: the task only
        calls this straight after ``write_root_pose_to_sim``, which already flattened the chain, so
        there is nothing to preserve and one fewer read of state that has not been simulated yet.
        """
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)
        env_ids = torch.as_tensor(env_ids, device=self._device, dtype=torch.long)
        self._stop(env_ids)

    def _stop(self, env_ids: torch.Tensor) -> None:
        """Zero the root velocity and put every hinge at rest, flat, without reading state back."""
        zeros6 = torch.zeros((env_ids.shape[0], 6), device=self._device, dtype=torch.float32)
        self._art.write_root_velocity_to_sim(zeros6, env_ids)
        self._write_flat_joints(env_ids)

    def _write_flat_joints(self, env_ids: torch.Tensor) -> None:
        nj = self._art.num_joints
        zeros_j = torch.zeros((env_ids.shape[0], nj), device=self._device, dtype=torch.float32)
        self._write_joints(zeros_j, zeros_j, env_ids)
        # The hinges are passive, but Isaac Lab still holds a position TARGET per joint and it is
        # not reset by `write_joint_state_to_sim`. Left at a previous episode's value, a drive with
        # any stiffness would immediately pull the sheet back into last episode's fold.
        if hasattr(self._art, "set_joint_position_target"):
            self._art.set_joint_position_target(zeros_j, env_ids=env_ids)

    def write_joint_positions(self, position: torch.Tensor, env_ids=None) -> None:
        """Drive the hinges to a given configuration, through the reset's own write path.

        Exposed so the fold probe cannot pass while the reset fails, or vice versa: both would be
        equally silent, and they are the same operation.
        """
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)
        env_ids = torch.as_tensor(env_ids, device=self._device, dtype=torch.long)
        self._write_joints(position, torch.zeros_like(position), env_ids)

    def set_external_force_and_torque(self, *args, **kwargs) -> None:
        """No-op, as for the cloth: randomisation is off for every measurement taken here.

        A chain *could* take a body force honestly, on one slat. Which slat is a modelling choice
        with no cloth counterpart, so the honest thing is to keep the two manipulands' behaviour
        identical here rather than to give one a mechanic the other lacks.
        """
        return None
