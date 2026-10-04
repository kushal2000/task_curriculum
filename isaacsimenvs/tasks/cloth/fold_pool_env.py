"""FoldPoolEnv -- stage 2: the half-fold task on a shape drawn per episode from the fold-shape pool.

Subclasses :class:`RigidClothEnv` and inherits everything that is about the TASK or the PHYSICS of a
rigid sheet -- fold targets, reward, success, observations (``ClothEnv``), slat friction, the 7 mm
contact margin, removal of plate-plate contact, physics DR (``RigidClothEnv``). Those all key on
model labels with the ``slat_`` prefixes, which the pool's USDs reuse, so they cover every pool
shape at once. What is replaced is what assumed ONE chain:

    ``_install_cloth``                    five articulations from ``fold_shape_assets`` USDs
    ``_calibrate_and_verify_placement``   the chain's startup proofs, run once per shape
    ``_reset_idx``                        draw each env's shape, then the inherited reset places it
    ``_pre_physics_step``                 re-park the unused shapes every step
    ``_init_fold_targets``                lift by the pool's ply gap (0 mm), and scale every rest
                                          quantity to the env's own sheet size
    ``_stationary_frame`` / ``_folded_w`` / ``_sync_observed_keypoints`` / ``footprint_ratio`` /
    ``fold_fraction``                     ClothEnv's versions, with a per-env rest geometry
    ``_write_object_scale``               each env's own size and felt thickness
    ``_get_dones``                        add per-shape fold rates to the episode logs

**Why five articulations per env.** Newton cannot vary an articulation's topology across worlds
(``ArticulationView``: "Articulations are not identical"), and ``SolverMuJoCo`` builds the MuJoCo
model from one template world. So every world holds all five shapes; the drawn one is on the table
and the rest hover ``fold_pool.park_distance`` away, untouchable -- worlds never collide. See
``utils/fold_pool_adapter.py``.

**Per-env size** (``fold_pool.size_range``) is the one geometric difference between worlds, and it is
set while the scene is REPLICATED (a per-world builder hook, ``_register_size_hook``): Newton ignores
a later change of a mesh's scale. So sizes are drawn once per env, not per episode, and a startup
check confirms each env's plate mass matches its draw.

    .venv_isaaclab3/bin/python isaacsimenvs/train.py \\
        --task Isaacsimenvs-FoldPool-Direct-v0 --agent rl_games_sapg_cfg_entry_point --headless

The policy is NOT told which shape it has: the observation layout is the cloth task's 140-dim one,
so a stage-1 checkpoint loads unchanged. Each shape differs only in where the sheet can bend.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import torch

from isaacsimenvs.tasks.cloth.cloth_env import ClothEnv
from isaacsimenvs.tasks.cloth.fold_pool_env_cfg import FoldPoolEnvCfg
from isaacsimenvs.tasks.cloth.rigid_cloth_env import RigidClothEnv, wp_to_torch
from isaacsimenvs.tasks.cloth.utils import fold_shape_assets as fsa
from isaacsimenvs.tasks.cloth.utils import fold_shapes as fs
from isaacsimenvs.tasks.cloth.utils.cloth_geometry import corner_indices, grid_mesh
from isaacsimenvs.tasks.cloth.utils.fold_pool_adapter import PoolAsClothMesh, PoolAsRigidObject

__all__ = ["FoldPoolEnv"]


class FoldPoolEnv(RigidClothEnv):
    cfg: FoldPoolEnvCfg

    def __init__(self, cfg: FoldPoolEnvCfg, render_mode: str | None = None, **kwargs):
        p = cfg.fold_pool
        # Set BEFORE super().__init__: the scene hook (`_install_cloth`) and the first reset both run
        # inside it.
        self._pool = [fs.shape(n) for n in p.shapes]
        w = [1.0] * len(self._pool) if p.weights is None else [float(x) for x in p.weights]
        self._pool_weights = torch.tensor(w, dtype=torch.float32)
        thickness = {sh.thickness for sh in self._pool}
        if len(thickness) != 1:
            raise ValueError(f"pool shapes disagree on thickness {sorted(thickness)}")
        slab0 = thickness.pop()
        # `RigidClothCfg.resolved_contact_margin` reads the VARIANT's slab thickness, which is not the
        # manipuland here. Resolve the sentinel against the pool's own (nominal) thickness instead.
        r = cfg.rigid_cloth
        if r.contact_margin < 0.0:
            r.contact_margin = max(0.5 * float(cfg.cloth.thickness) - 0.5 * slab0, 0.0)
        # Per-env size and slab thickness, drawn ONCE (see FoldPoolCfg.size_range): the sheets are
        # built at these sizes during replication, so they cannot change afterwards.
        n = int(cfg.scene.num_envs)
        rng = np.random.default_rng(int(p.size_seed))
        self._env_size_np = rng.uniform(float(p.size_range[0]), float(p.size_range[1]), n)
        self._env_slab_np = rng.uniform(float(p.thickness_range[0]), float(p.thickness_range[1]), n)
        self._xy_np = self._env_size_np / float(cfg.cloth.size)
        self._z_np = self._env_slab_np / slab0
        self._scaled_worlds = 0
        try:
            super().__init__(cfg, render_mode, **kwargs)
        finally:
            self._unregister_size_hook()
        if self._size_hook is not None and self._scaled_worlds != self.num_envs:
            raise RuntimeError(
                f"[fold_pool] the size hook resized {self._scaled_worlds} worlds for "
                f"{self.num_envs} envs -- per-env sheet sizes did not reach the simulator"
            )
        self._pool_weights = self._pool_weights.to(self.device)
        # Draws start now, at the trainer's first reset. Until then every env holds the shape the
        # startup proofs left on the table -- placed, so the cloud and the scene agree.
        self._pool_ready = True

    # ------------------------------------------------------------------ construction

    def _install_cloth(self) -> None:
        """Spawn every pool shape as its own articulation, all parked, and the pool adapter."""
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import Articulation, ArticulationCfg
        from isaaclab.sim.spawners.from_files import UsdFileCfg

        c, r, p = self.cfg.cloth, self.cfg.rigid_cloth, self.cfg.fold_pool
        if c.fold_axis != "x":
            raise ValueError(
                f"the pool's half fold carries +x over onto -x; cloth.fold_axis must be 'x', got "
                f"{c.fold_axis!r}"
            )
        for sh in self._pool:
            if abs(sh.size - c.size) > 1e-9:
                raise ValueError(f"{sh.name} is {sh.size} m square, the task's sheet is {c.size} m")

        verts, _indices = grid_mesh(c.size, c.resolution)
        self._cloth_rest_local = verts
        self._cloth_kp_idx = corner_indices(c.resolution, c.fold_axis)
        self._neutralise_rigid_object()
        self._reshape_goal_marker()

        # Same actuator rule as the chain (see RigidClothEnv._install_cloth): always present, since it
        # is the only place damping and armature can be set, and built with a negligible stiffness
        # when DR will draw one, so the position gain exists to be written.
        stiffness = float(r.joint_stiffness)
        rd = r.randomization
        if rd.enabled and float(rd.hinge_stiffness[1]) > 0.0 and stiffness == 0.0:
            stiffness = 1.0e-9

        park = torch.tensor(
            [
                [0.0, p.park_distance + s * p.park_spacing, p.park_height]
                for s in range(len(self._pool))
            ],
            dtype=torch.float32,
        )
        usd_dir = Path(tempfile.mkdtemp(prefix="fold_pool_"))
        arts = []
        for s, sh in enumerate(self._pool):
            usd = fsa.write_shape_usd(sh, usd_dir / f"{sh.name}.usda", density=r.density)
            cfg = ArticulationCfg(
                prim_path=f"/World/envs/env_.*/FoldPool{s}",
                spawn=UsdFileCfg(usd_path=str(usd)),
                # Every shape starts parked; the first reset puts the drawn one on the table.
                init_state=ArticulationCfg.InitialStateCfg(pos=tuple(float(v) for v in park[s])),
                actuators={
                    "hinges": ImplicitActuatorCfg(
                        joint_names_expr=[f"{fsa.JOINT_PREFIX}.*"],
                        stiffness=stiffness,
                        damping=r.joint_damping,
                        armature=r.joint_armature,
                        effort_limit_sim=None,
                    )
                },
            )
            art = Articulation(cfg)
            self.scene.articulations[f"fold_pool_{s}"] = art
            arts.append(art)
        self._pool_arts = arts
        self._register_size_hook()

        # Per-env geometry, as tensors: in-plane factor, size [m], and the reset-centre shift that
        # keeps a large sheet's far side within the arm's reach.
        self._xy_scale = torch.tensor(self._xy_np, device=self.device, dtype=torch.float32)
        self._env_size = torch.tensor(self._env_size_np, device=self.device, dtype=torch.float32)
        reach = torch.tensor([*p.reach_shift, 0.0], device=self.device, dtype=torch.float32)
        growth = (self._env_size - float(c.size)).clamp(min=0.0) * (0.5 * math.sqrt(2.0))
        centre_offset = growth.unsqueeze(1) * reach.unsqueeze(0)

        mesh = PoolAsClothMesh(
            arts,
            mappings=[fs.grid_plate_map(sh, c.resolution) for sh in self._pool],
            body_prefix=fsa.BODY_PREFIX,
            num_envs=self.num_envs,
            device=self.device,
            xy_scale=self._xy_scale,
        )
        self._cloth = mesh
        spawn_z = float(self.cfg.reset.table_reset_z) + c.start_height
        self.object = PoolAsRigidObject(
            mesh,
            articulations=arts,
            env_origins=lambda: self.scene.env_origins,
            park_local=park,
            centre_offset=centre_offset,
            num_envs=self.num_envs,
            rest_local=verts,
            keypoint_idx=self._cloth_kp_idx,
            corner_idx=self._cloth_kp_idx,
            corner_rest=[verts[i] for i in self._cloth_kp_idx],
            spawn_z=spawn_z,
            mass=r.density * c.size * c.size,
            device=self.device,
        )
        self._write_object_scale()
        print(
            "[fold_pool] pool: "
            + "; ".join(f"{sh.name} ({sh.describe()})" for sh in self._pool)
            + f"\n[fold_pool] draw weights {self._pool_weights.tolist()}, parked at +y "
            f"{p.park_distance} m, z {p.park_height} m; USDs in {usd_dir}"
            + f"\n[fold_pool] per-env sheet size {self._env_size_np.min() * 100:.1f}-"
            f"{self._env_size_np.max() * 100:.1f} cm, slab {self._env_slab_np.min() * 1e3:.2f}-"
            f"{self._env_slab_np.max() * 1e3:.2f} mm, max reach shift {float(growth.max()) * 100:.1f} cm",
            flush=True,
        )

    # ------------------------------------------------------------------ per-env size

    _size_hook = None

    def _register_size_hook(self) -> None:
        """Resize every world's pool sheets to its env's draw, as Newton replicates the scene.

        A per-world builder hook runs once per env, after that env's copy of the scene has been
        added and before the model is finalized -- the only point at which plate geometry can
        still differ between worlds. Skipped entirely when nothing is randomised, which keeps
        Isaac Lab's batched replication path.
        """
        if np.allclose(self._xy_np, 1.0) and np.allclose(self._z_np, 1.0):
            return
        import newton
        from isaaclab_newton.physics import NewtonManager

        xy, z = self._xy_np, self._z_np
        cursor = {"body": 0, "joint": 0, "shape": 0}
        free = newton.JointType.FREE

        def new_ids(labels, worlds, key, world):
            start, end = cursor[key], len(labels)
            cursor[key] = end
            return [
                i
                for i in range(start, end)
                if worlds[i] == world and "/FoldPool" in (labels[i] or "")
            ]

        def hook(builder, world, _pos, _quat) -> None:
            bodies = new_ids(builder.body_label, builder.body_world, "body", world)
            joints = [
                j
                for j in new_ids(builder.joint_label, builder.joint_world, "joint", world)
                if builder.joint_type[j] != free
            ]
            shapes = new_ids(builder.shape_label, builder.shape_world, "shape", world)
            if not bodies:
                return
            fsa.scale_builder_world(
                builder, bodies, joints, shapes, float(xy[world]), float(z[world])
            )
            self._scaled_worlds += 1

        NewtonManager._per_world_builder_hooks.append(hook)
        self._size_hook = hook

    def _unregister_size_hook(self) -> None:
        if self._size_hook is None:
            return
        from isaaclab_newton.physics import NewtonManager

        if self._size_hook in NewtonManager._per_world_builder_hooks:
            NewtonManager._per_world_builder_hooks.remove(self._size_hook)

    def _write_object_scale(self) -> None:
        """``object_scales`` per env: this env's sheet, not the nominal one."""
        c = self.cfg.cloth
        base = float(self.cfg.reward.object_base_size)
        size = torch.tensor(self._env_size_np, device=self.device, dtype=torch.float32)
        # The thickness the hand meets is slab + 2 x contact margin -- by construction the cloth's
        # `thickness` at the nominal slab -- so a thicker slab ADDS its difference to it.
        slab = torch.tensor(self._env_slab_np, device=self.device, dtype=torch.float32)
        felt = float(c.thickness) + slab - float(self._pool[0].thickness)
        self._object_scale_per_env[:] = torch.stack((size, size, felt), dim=-1) / base

    # ------------------------------------------------------------------ episodes

    def _draw(self, env_ids: torch.Tensor) -> None:
        draw = torch.multinomial(self._pool_weights, int(env_ids.numel()), replacement=True)
        self.object.select(env_ids, draw.to(self.object.active.device))

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
        if getattr(self, "_pool_ready", False):
            self._draw(ids)
        super()._reset_idx(env_ids)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        super()._pre_physics_step(actions)
        self.object.park()

    def _init_fold_targets(self) -> None:
        """The pool's ply gap as the target lift -- 0 mm for mid-plane hinges."""
        if self.cfg.fold_pool.fold_lift_from_ply_gap:
            gaps = {sh.ply_gap for sh in self._pool}
            if len(gaps) != 1:
                raise ValueError(
                    f"pool shapes disagree on ply gap {sorted(gaps)}; the fold target is built once "
                    f"for every env, so a mixed pool needs a per-env lift"
                )
            gap = gaps.pop()
            print(
                f"[fold_pool] fold target lift {self.cfg.cloth.self_contact_radius * 1e3:.2f} -> "
                f"{gap * 1e3:.2f} mm (the pool's ply gap)",
                flush=True,
            )
            self.cfg.cloth.self_contact_radius = gap
        ClothEnv._init_fold_targets(self)
        # ClothEnv builds the rest geometry once, for a `cloth.size` sheet. Every quantity below is
        # that geometry scaled in-plane by the env's own factor (heights -- the fold lift -- are not
        # lengths in the sheet's plane and stay as they are).
        s3 = self._scale3()
        self._corner_rest_offsets = self._corner_rest_offsets.unsqueeze(0) * s3
        self._mv_folded_rest = self._mv_folded_rest.unsqueeze(0) * s3
        self._kp_folded_rest = self._kp_folded_rest.unsqueeze(0) * s3
        self._st_rest_centroid = self._st_rest_centroid.unsqueeze(0) * s3[:, 0]
        self._st_rest_centred = self._st_rest_centred.unsqueeze(0) * s3

    def _scale3(self) -> torch.Tensor:
        """``(N, 1, 3)``: in-plane factor on x and y, 1 on z."""
        s = self._xy_scale
        return torch.stack((s, s, torch.ones_like(s)), dim=-1).unsqueeze(1)

    # The four ClothEnv methods that read the shared rest geometry, each the same computation with
    # a per-env rest. Kept as close to the originals as the extra batch axis allows.

    def _stationary_frame(self, parts: torch.Tensor):
        cur = parts[:, self._stationary_idx, :]
        centroid = cur.mean(dim=1)
        q = cur - centroid.unsqueeze(1)
        h = torch.einsum("nki,nkj->nij", self._st_rest_centred, q)
        h = h + 1e-8 * torch.eye(3, device=h.device, dtype=h.dtype).unsqueeze(0)
        u, _, vh = torch.linalg.svd(h)
        v = vh.transpose(-2, -1)
        d = torch.ones((cur.shape[0], 3), device=cur.device, dtype=cur.dtype)
        d[:, 2] = torch.linalg.det(v @ u.transpose(-2, -1))
        r = v @ torch.diag_embed(d) @ u.transpose(-2, -1)
        bad = ~torch.isfinite(r).all(dim=-1).all(dim=-1)
        if bool(bad.any()):
            r = r.clone()
            r[bad] = torch.eye(3, device=r.device, dtype=r.dtype)
        return r, centroid

    def _folded_w(self, folded_rest: torch.Tensor) -> torch.Tensor:
        r, centroid = self._stationary_frame(self._particles_w())
        local = folded_rest - self._st_rest_centroid.unsqueeze(1)
        return torch.einsum("nij,nmj->nmi", r, local) + centroid.unsqueeze(1)

    def _sync_observed_keypoints(self) -> None:
        self._keypoint_offsets[:] = self._corner_rest_offsets
        self._keypoint_offsets_fixed[:] = self._corner_rest_offsets

    def footprint_ratio(self) -> torch.Tensor:
        ax = 0 if self.cfg.cloth.fold_axis == "x" else 1
        parts = self._particles_w()
        r, _ = self._stationary_frame(parts)
        coord = torch.einsum("npi,ni->np", parts, r[:, :, ax])
        return (coord.amax(dim=1) - coord.amin(dim=1)) / self._env_size

    def fold_fraction(self) -> torch.Tensor:
        rest = (
            torch.tensor(
                [self._cloth_rest_local[i] for i in self._cloth_kp_idx],
                device=self.device,
                dtype=torch.float32,
            ).unsqueeze(0)
            * self._scale3()
        )
        start = (
            rest
            + torch.tensor(
                [0.0, 0.0, float(self.cfg.reset.table_reset_z) + self.cfg.cloth.start_height],
                device=self.device,
            )
            + (self.scene.env_origins + self.object.centre_offset).unsqueeze(1)
        )
        span = (start - self.fold_targets_w()).norm(dim=-1).amax(dim=-1).clamp(min=1e-6)
        return (1.0 - self.fold_error() / span).clamp(0.0, 1.0)

    def _get_dones(self):
        terminated, truncated = super()._get_dones()
        entered = self._termination_reasons["fold"]
        active = self.object.active
        for s, sh in enumerate(self._pool):
            on = active == s
            # Averaged over finished episodes by the logger: `shape_<n>` is the share of episodes on
            # that shape, `fold_<n>` the share that were that shape AND folded. Their ratio is the
            # per-shape fold rate.
            self._termination_reasons[f"shape_{sh.name}"] = on
            self._termination_reasons[f"fold_{sh.name}"] = entered & on
        return terminated, truncated

    # ------------------------------------------------------------------ startup proofs

    def _calibrate_and_verify_placement(self) -> None:
        """``RigidClothEnv``'s placement / shape / yaw / joint-write proofs, once per pool shape.

        Each shape in turn is made active in EVERY env; the others are parked. All checks fatal, for
        the reasons the chain's version gives -- every failure they catch otherwise trains silently.
        """
        obj = self.object
        n = self.num_envs
        dt = float(self.cfg.sim.dt)
        origins = self.scene.env_origins
        all_ids = torch.arange(n, device=self.device)
        want = torch.tensor(self._cloth_rest_local, device=self.device, dtype=torch.float32)
        # (N, P, 3): each env's own flat sheet, centred.
        ref = (want - want.mean(dim=0, keepdim=True)).unsqueeze(0) * self._scale3()
        target = origins + obj.centre_offset

        # The per-env sizes reached the SIMULATOR, in the right env: total plate mass per env is
        # every pool shape at density x size^2. A world/env mix-up in the build hook would show up
        # here and nowhere else -- the cloud above is built from the same scale table it would be
        # checked against.
        mass = wp_to_torch(self._model.body_mass)[self._slat_bodies].sum(dim=1)
        want_mass = len(self._pool) * float(self.cfg.rigid_cloth.density) * self._env_size**2
        mass_err = float(((mass - want_mass).abs() / want_mass).max())
        if mass_err > 1e-3:
            raise RuntimeError(
                f"[fold_pool] per-env plate mass is off by up to {mass_err * 100:.2f}% of the size "
                f"draw -- the sheets were not built at their envs' sizes"
            )

        def place(yaw: float) -> torch.Tensor:
            pose = torch.zeros((n, 7), device=self.device, dtype=torch.float32)
            pose[:, :3] = origins
            pose[:, 3] = math.cos(0.5 * yaw)
            pose[:, 6] = math.sin(0.5 * yaw)
            # The CLASS method, bypassing the instance wrapper -- as RigidClothEnv does.
            type(obj).write_root_pose_to_sim(obj, pose)
            obj.write_root_velocity_to_sim(torch.zeros((n, 6), device=self.device))
            self.sim.step(render=False)
            self.scene.update(dt)
            return self._particles_w()

        def rotated(yaw: float) -> torch.Tensor:
            return torch.stack(
                (
                    ref[..., 0] * math.cos(yaw) - ref[..., 1] * math.sin(yaw),
                    ref[..., 0] * math.sin(yaw) + ref[..., 1] * math.cos(yaw),
                ),
                dim=-1,
            )

        def centred(cloud: torch.Tensor) -> torch.Tensor:
            return cloud - cloud.mean(dim=1, keepdim=True)

        for s, sh in enumerate(self._pool):
            obj.select(all_ids, torch.full((n,), s, device=self.device, dtype=torch.long))
            cloud = place(0.0)
            if s == 0:
                # Env 0's OWN sheet: with size DR it is not the nominal grid.
                layout = self._cloth.detect_quat_layout((want.unsqueeze(0) * self._scale3())[0])
                sc = self._cloth._quat_layout_scores
                print(
                    f"[fold_pool] quaternion layout = {layout} (flat-grid error: xyzw "
                    f"{sc['xyzw'] * 1e3:.2f} mm, wxyz {sc['wxyz'] * 1e3:.2f} mm)",
                    flush=True,
                )
            residual = None
            for _attempt in range(3):
                cloud = place(0.0)
                err = (cloud.mean(dim=1) - target)[:, :2]
                residual = float(err.norm(dim=-1).max())
                if residual < 1.0e-4:
                    break
                obj.calibrate_root_offset(s, err[0])
            if residual is None or residual > 1.0e-4:
                raise RuntimeError(
                    f"[fold_pool] {sh.name}: centroid {residual * 1e3:.2f} mm off after 3 passes"
                )

            from isaacsimenvs.tasks.cloth.utils.cloth_adapter import _to_torch

            art = self._pool_arts[s]
            hinge = math.degrees(float(_to_torch(art.data.joint_pos).abs().max()))
            if hinge > 1.0:
                raise RuntimeError(
                    f"[fold_pool] {sh.name} is not flat at reset: hinge {hinge:.2f} deg"
                )

            cloud = place(0.0)
            shape_err = float((centred(cloud)[..., :2] - ref[..., :2]).norm(dim=-1).max())
            yaw_err = 0.0
            cent_err = 0.0
            for yaw in (0.5 * math.pi, math.radians(-120.0)):
                cy = place(yaw)
                yaw_err = max(
                    yaw_err, float((centred(cy)[..., :2] - rotated(yaw)).norm(dim=-1).max())
                )
                cent_err = max(cent_err, float((cy.mean(dim=1) - target)[:, :2].norm(dim=-1).max()))
            if shape_err > 1.0e-3 or yaw_err > 1.0e-3 or cent_err > 1.0e-3:
                raise RuntimeError(
                    f"[fold_pool] {sh.name}: cloud is not the placed flat sheet -- shape "
                    f"{shape_err * 1e3:.2f} mm, yawed shape {yaw_err * 1e3:.2f} mm, yawed centroid "
                    f"{cent_err * 1e3:.2f} mm (tolerance 1 mm); layout {self._cloth.quat_layout}"
                )

            # The parked shapes really are parked: nowhere near this env's table.
            for k in range(len(self._pool)):
                if k == s:
                    continue
                far = (self._cloth.shape_positions(k).mean(dim=1) - origins).norm(dim=-1).min()
                if float(far) < 0.5 * self.cfg.fold_pool.park_distance:
                    raise RuntimeError(
                        f"[fold_pool] {self._pool[k].name} should be parked but is "
                        f"{float(far):.2f} m from its env origin"
                    )

            self._verify_pool_joint_writes(s)
            print(
                f"[fold_pool] {sh.name}: placed within {residual * 1e6:.0f} um, flat cloud within "
                f"{shape_err * 1e6:.0f} um, yawed within {yaw_err * 1e6:.0f} um, root offset "
                f"{[round(float(v) * 1e3, 2) for v in obj.root_offset[s]]} mm",
                flush=True,
            )
        self._yaw_reset_applied = True

    def _verify_pool_joint_writes(self, s: int) -> None:
        """A partial joint write reaches the solver and only the envs it names (see the chain's)."""
        from isaacsimenvs.tasks.cloth.utils.cloth_adapter import _to_torch

        n = self.num_envs
        if n < 2:
            return
        art = self._pool_arts[s]
        nj = int(art.num_joints)
        half = torch.arange(n // 2, device=self.device)
        test = 0.3
        q = torch.full((half.shape[0], nj), test, device=self.device, dtype=torch.float32)
        self.object.write_joint_positions(s, q, env_ids=half)
        self.sim.step(render=False)
        self.scene.update(float(self.cfg.sim.dt))
        got = _to_torch(art.data.joint_pos)
        moved = float(got[half].abs().max())
        untouched = float(got[n // 2 :].abs().max())
        self.object.write_joint_positions(s, torch.zeros((n, nj), device=self.device))
        self.sim.step(render=False)
        self.scene.update(float(self.cfg.sim.dt))
        if moved < 0.5 * test:
            raise RuntimeError(
                f"[fold_pool] {self._pool[s].name}: partial joint write did not reach the solver "
                f"({math.degrees(moved):.2f} deg of {math.degrees(test):.1f})"
            )
        if untouched > math.radians(5.0):
            raise RuntimeError(
                f"[fold_pool] {self._pool[s].name}: partial joint write leaked "
                f"{math.degrees(untouched):.2f} deg into unnamed envs"
            )
