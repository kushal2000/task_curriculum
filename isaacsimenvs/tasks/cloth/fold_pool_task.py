"""FoldPoolTaskMixin -- the stage-2 shape pool, on whichever simulator the env is built on.

Everything ``FoldPoolEnv`` does that is about the POOL rather than about Newton: one articulation per
shape, the active one placed and the rest parked, the per-env shape draw, the per-env sheet size
carried through every rest quantity of the fold, the per-shape startup proofs and logs. Shared by
``FoldPoolEnv`` (Newton) and ``physx_envs.FoldPoolPhysxEnv``.

A concrete env supplies:

    ``_draw_env_sizes(cfg, n)``            per-env sheet size and slab (default: ``size_seed`` draws)
    ``_pool_spawn(s, shape, dir, park)``  shape ``s``'s spawner (default: one nominal-size USD)
    ``_pool_after_spawn()``               make each env's sheets its own size, when ``_pool_spawn``
                                          did not (Newton: a per-world builder hook)
    ``_pool_build_done()``                runs once construction ends, raised or not
    ``_pool_plate_mass()``                ``(num_envs,)`` total plate mass per env, read back from
                                          the simulator for the size check
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import torch

from isaacsimenvs.tasks.cloth.cloth_task import ClothTaskMixin
from isaacsimenvs.tasks.cloth.utils import fold_shape_assets as fsa
from isaacsimenvs.tasks.cloth.utils import fold_shapes as fs
from isaacsimenvs.tasks.cloth.utils.cloth_geometry import corner_indices, grid_mesh
from isaacsimenvs.tasks.cloth.utils.fold_pool_adapter import PoolAsClothMesh, PoolAsRigidObject

__all__ = ["FoldPoolTaskMixin"]


class FoldPoolTaskMixin:
    """The fold-shape pool as the cloth task's manipuland. Mix in before the chain env."""

    _pool_mesh_cls = PoolAsClothMesh
    _pool_object_cls = PoolAsRigidObject
    #: Whether this backend can build envs that carry different shapes (``fold_pool.one_shape_per_env``).
    _supports_one_shape_per_env = False

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
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
        self._env_size_np, self._env_slab_np = self._draw_env_sizes(cfg, n)
        self._env_shape_np = None
        if p.one_shape_per_env:
            if not self._supports_one_shape_per_env:
                raise ValueError(
                    f"fold_pool.one_shape_per_env: {type(self).__name__} needs every env to carry the "
                    f"same articulations (Newton); use the PhysX task"
                )
            S = len(self._pool)
            if p.weights is None:
                self._env_shape_np = np.arange(n) % S
            else:
                w = np.asarray([float(x) for x in p.weights])
                self._env_shape_np = np.random.default_rng(int(p.size_seed) + 1).choice(S, n, p=w / w.sum())
        self._xy_np = self._env_size_np / float(cfg.cloth.size)
        self._z_np = self._env_slab_np / slab0
        self._scaled_worlds = 0
        try:
            super().__init__(cfg, render_mode, **kwargs)
        finally:
            self._pool_build_done()
        self._check_pool_build()
        self._pool_weights = self._pool_weights.to(self.device)
        self._init_start_folds()
        # Draws start now, at the trainer's first reset. Until then every env holds the shape the
        # startup proofs left on the table -- placed, so the cloud and the scene agree.
        self._pool_ready = True

    # ------------------------------------------------------------------ backend hooks

    def _draw_env_sizes(self, cfg, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Per-env sheet size and slab thickness [m], drawn once from ``fold_pool.size_seed``.

        Runs before ``super().__init__``, so it reads the ``cfg`` it is handed, not ``self.cfg``.
        """
        p = cfg.fold_pool
        rng = np.random.default_rng(int(p.size_seed))
        size = rng.uniform(float(p.size_range[0]), float(p.size_range[1]), n)
        slab = rng.uniform(float(p.thickness_range[0]), float(p.thickness_range[1]), n)
        return size, slab

    def _pool_spawn(self, s: int, shape, usd_dir: Path, park_local):
        """The spawner for pool shape ``s`` (its prims in every env), or ``None`` if this has authored
        them already. Default: one nominal-size USD for every env, resized later by the backend."""
        from isaaclab.sim.spawners.from_files import UsdFileCfg

        path = usd_dir / f"{shape.name}.usda"
        usd = fsa.write_shape_usd(shape, path, density=self.cfg.rigid_cloth.density)
        return UsdFileCfg(usd_path=str(usd))

    def _pool_after_spawn(self) -> None:
        raise NotImplementedError

    def _pool_articulation_cls(self):
        """The class each pool shape's articulation is built with."""
        from isaaclab.assets import Articulation

        return Articulation

    def _pool_build_done(self) -> None:
        pass

    def _check_pool_build(self) -> None:
        pass

    def _pool_plate_mass(self) -> torch.Tensor:
        raise NotImplementedError

    # ------------------------------------------------------------------ construction

    def _install_cloth(self) -> None:
        """Spawn every pool shape as its own articulation, all parked, and the pool adapter."""
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import ArticulationCfg

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
            cfg = ArticulationCfg(
                prim_path=f"/World/envs/env_.*/FoldPool{s}",
                spawn=self._pool_spawn(s, sh, usd_dir, park[s]),
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
            art = self._pool_articulation_cls()(cfg)
            self.scene.articulations[f"fold_pool_{s}"] = art
            arts.append(art)
        self._pool_arts = arts
        self._pool_after_spawn()

        # Per-env geometry, as tensors: in-plane factor, size [m], and the reset-centre shift that
        # keeps a large sheet's far side within the arm's reach.
        self._xy_scale = torch.tensor(self._xy_np, device=self.device, dtype=torch.float32)
        self._env_size = torch.tensor(self._env_size_np, device=self.device, dtype=torch.float32)
        reach = torch.tensor([*p.reach_shift, 0.0], device=self.device, dtype=torch.float32)
        growth = (self._env_size - float(c.size)).clamp(min=0.0) * (0.5 * math.sqrt(2.0))
        centre_offset = growth.unsqueeze(1) * reach.unsqueeze(0)

        mesh = self._pool_mesh_cls(
            arts,
            mappings=[fs.grid_plate_map(sh, c.resolution) for sh in self._pool],
            body_prefix=fsa.BODY_PREFIX,
            num_envs=self.num_envs,
            device=self.device,
            xy_scale=self._xy_scale,
        )
        if self._env_shape_np is not None:
            shape_of = torch.as_tensor(self._env_shape_np, device=self.device)
            mesh.set_carried(
                shape_of.unsqueeze(0) == torch.arange(len(self._pool), device=self.device).unsqueeze(1)
            )
        self._cloth = mesh
        spawn_z = float(self.cfg.reset.table_reset_z) + c.start_height
        self.object = self._pool_object_cls(
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
            + (
                "\n[fold_pool] ONE shape per env, fixed: "
                + ", ".join(
                    f"{sh.name} {int((self._env_shape_np == k).sum())}" for k, sh in enumerate(self._pool)
                )
                + " envs"
                if self._env_shape_np is not None
                else "\n[fold_pool] every env carries every shape; one drawn per episode"
            )
            + f"\n[fold_pool] per-env sheet size {self._env_size_np.min() * 100:.1f}-"
            f"{self._env_size_np.max() * 100:.1f} cm, slab {self._env_slab_np.min() * 1e3:.2f}-"
            f"{self._env_slab_np.max() * 1e3:.2f} mm, max reach shift {float(growth.max()) * 100:.1f} cm",
            flush=True,
        )

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
        if self._env_shape_np is not None:
            # One shape per env: the env's own, every episode.
            draw = self.object.first_carried()[env_ids.to(self.object.active.device)]
        else:
            draw = torch.multinomial(self._pool_weights, int(env_ids.numel()), replacement=True)
        self.object.select(env_ids, draw.to(self.object.active.device))
        if self._start_tables is not None:
            self._draw_start_folds(env_ids, draw.to(self.device))

    # ------------------------------------------------------------------ partly folded starts

    def _init_start_folds(self) -> None:
        """Per shape, its start folds as ``(F, num_joints)`` hinge directions plus ``(F, num_joints)``
        part indices (which of the start's independently drawn angles each hinge takes), and the
        startup proof. ``init_fold_starts: single`` has one part per start; ``other`` uses
        ``FoldShape.start_poses`` and may leave a shape with no start (``F = 0``: always flat).

        Off (``_start_tables = None``) unless ``fold_pool.init_fold_prob > 0``.
        """
        p = self.cfg.fold_pool
        self._start_folded = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._start_tables = None
        if float(p.init_fold_prob) <= 0.0:
            return
        prefix = fsa.JOINT_PREFIX
        tables, parts, names = [], [], []
        for s, sh in enumerate(self._pool):
            joints = list(self._pool_arts[s].joint_names)
            if p.init_fold_starts == "other":
                poses = sh.start_poses()
            else:
                poses = {n: (tuple(d),) for n, d in sh.start_folds().items()}
                if not poses:
                    raise ValueError(f"[fold_pool] {sh.name} has no single-stage fold to start from")
            t = torch.zeros((len(poses), len(joints)), device=self.device)
            c = torch.zeros((len(poses), len(joints)), device=self.device, dtype=torch.long)
            for k, pose in enumerate(poses.values()):
                for i, part in enumerate(pose):
                    for child in part:
                        t[k, joints.index(f"{prefix}{child}")] = 1.0
                        c[k, joints.index(f"{prefix}{child}")] = i
            tables.append(t)
            parts.append(c)
            names.append(list(poses))
        self._start_tables = tables
        self._start_parts = parts
        self._start_names = names
        self._start_num_parts = max([int(c.max()) + 1 for c, t in zip(parts, tables) if t.numel()] or [1])
        lo, hi = (float(v) for v in p.init_fold_deg)
        print(
            f"[fold_pool] partly folded starts: p={float(p.init_fold_prob):g}, {lo:g}-{hi:g} deg, "
            + "; ".join(f"{sh.name}: {n}" for sh, n in zip(self._pool, names)),
            flush=True,
        )
        self._calibrate_start_signs()
        self._verify_start_folds(math.radians(hi))

    def _draw_start_folds(self, env_ids: torch.Tensor, draw: torch.Tensor) -> None:
        p = self.cfg.fold_pool
        n = int(env_ids.numel())
        folded = torch.rand(n, device=self.device) < float(p.init_fold_prob)
        lo, hi = (math.radians(float(v)) for v in p.init_fold_deg)
        ang = torch.empty((n, self._start_num_parts), device=self.device).uniform_(lo, hi)
        ang *= folded.unsqueeze(1)
        for s, (table, part) in enumerate(zip(self._start_tables, self._start_parts)):
            on = draw == s
            ids = env_ids[on]
            if ids.numel() == 0:
                continue
            if table.shape[0] == 0:  # no start but the target's: flat
                folded &= ~on
                self.object.set_start_joints(s, ids, torch.zeros((ids.numel(), table.shape[1]), device=self.device))
                continue
            k = torch.randint(table.shape[0], (ids.numel(),), device=self.device)
            self.object.set_start_joints(s, ids, table[k] * torch.gather(ang[on], 1, part[k]))
        self._start_folded[env_ids] = folded

    def _fold_index(self, s: int, F: int) -> torch.Tensor:
        """``(N,)`` which of shape ``s``'s ``F`` start folds each env tests: its RANK among the envs
        carrying ``s``, mod ``F``. Not ``env % F``: with one shape per env (shape = ``env % S``) that
        aliases -- every box2x8 env (S=5, F=5) got the same fold and the other four went untested."""
        k = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        ids = self._cloth.members(s)
        k[ids] = torch.arange(ids.numel(), device=self.device) % F
        return k

    def _reset_all_to(self, s: int, q: torch.Tensor, dy: float = 0.0) -> None:
        """Every env on shape ``s`` at hinge angles ``q``, flat yaw-0 at its origin shifted ``dy`` in
        y, via the reset path. ``dy < 0`` is away from the robot: the startup checks of the start
        folds use -0.5 m, off the table's far edge in free air, so that a tall flap at spawn height
        cannot touch the hand resting at its home pose."""
        obj, n = self.object, self.num_envs
        all_ids = torch.arange(n, device=self.device)
        pose = torch.zeros((n, 7), device=self.device, dtype=torch.float32)
        pose[:, :3] = self.scene.env_origins
        pose[:, 1] += dy
        pose[:, 3] = 1.0
        if self._env_shape_np is None:
            obj.select(all_ids, torch.full((n,), s, device=self.device, dtype=torch.long))
        else:  # every env keeps its own shape; only the envs carrying `s` are measured
            obj.select(all_ids, obj.first_carried())
        obj.set_start_joints(s, all_ids, q)
        type(obj).write_root_pose_to_sim(obj, pose)
        obj.write_root_velocity_to_sim(torch.zeros((n, 6), device=self.device))

    def _calibrate_start_signs(self, probe_deg: float = 15.0) -> None:
        """Which joint-angle sign lifts each start fold UP, measured: ``fold_shapes`` says +, but the
        simulator's joint frame decides (PhysX turned +120 deg box2 starts DOWN through the table).

        Each fold is written at +``probe_deg`` (env ``i`` takes fold ``i % F``) and stepped once; the
        sheet's highest point rises for an up fold, its lowest point sinks for a down one. A fold whose
        envs disagree, or that moves neither way, is fatal.
        """
        n, dt = self.num_envs, float(self.cfg.sim.dt)
        all_ids = torch.arange(n, device=self.device)
        for s, (sh, table) in enumerate(zip(self._pool, self._start_tables)):
            F = table.shape[0]
            if F == 0:
                continue
            self._reset_all_to(s, torch.zeros((n, table.shape[1]), device=self.device), dy=-0.5)
            self.sim.step(render=False)
            self.scene.update(dt)
            z0 = self._particles_w()[..., 2]
            kof = self._fold_index(s, F)
            self._reset_all_to(s, table[kof] * math.radians(probe_deg), dy=-0.5)
            self.sim.step(render=False)
            self.scene.update(dt)
            z = self._particles_w()[..., 2]
            rise = z.amax(dim=1) - z0.amax(dim=1)
            sink = z0.amin(dim=1) - z.amin(dim=1)
            self.object.set_start_joints(s, all_ids, torch.zeros((n, table.shape[1]), device=self.device))
            carried = self._cloth.carried[s]
            signs = []
            for k in range(F):
                on = (kof == k) & carried
                if not bool(on.any()):
                    raise RuntimeError(
                        f"[fold_pool] {sh.name}: start fold {k} has no env to measure its direction in "
                        f"({int(carried.sum())} envs carry the shape, {F} folds)"
                    )
                up = float((rise[on] > sink[on]).float().mean())
                moved = float(torch.maximum(rise[on], sink[on]).median())
                if moved < 1.0e-3 or 0.05 < up < 0.95:
                    raise RuntimeError(
                        f"[fold_pool] {sh.name}: start fold {k} at +{probe_deg:g} deg moved the sheet "
                        f"{moved * 1e3:.2f} mm, up in {up * 100:.0f}% of envs -- no consistent direction"
                    )
                signs.append(1.0 if up >= 0.95 else -1.0)
            table *= torch.tensor(signs, device=self.device).unsqueeze(1)
            print(f"[fold_pool] {sh.name}: start-fold joint signs (up) {signs}", flush=True)
        self._reset_all_to(0, torch.zeros((n, self._start_tables[0].shape[1]), device=self.device))

    def _verify_start_folds(self, angle: float, tol_deg: float = 2.0, share: float = 0.95) -> None:
        """Every start fold, at the LARGEST start angle, through the reset's own write path: one
        physics step later the hinges are where they were asked to be (a wrong joint order or a
        write that does not reach the solver fails here; the direction is ``_calibrate_start_signs``).

        One step, and a share rather than every env, on purpose: the sheet spawns 15 cm above the
        table (``cloth.start_height``) at the robot's home height, and a tall flap held there for
        longer meets the hand in some envs -- which no real reset does, as the reset drops the sheet
        at once. Measured on PhysX through real resets: every start lands within 0.00 deg.
        """
        from isaacsimenvs.tasks.cloth.utils.cloth_adapter import _to_torch

        n, dt = self.num_envs, float(self.cfg.sim.dt)
        all_ids = torch.arange(n, device=self.device)
        for s, (sh, table) in enumerate(zip(self._pool, self._start_tables)):
            if table.shape[0] == 0:
                continue
            art = self._pool_arts[s]
            want = table[self._fold_index(s, table.shape[0])] * angle
            self._reset_all_to(s, want, dy=-0.5)
            self.sim.step(render=False)
            self.scene.update(dt)
            # Instance order: row i of the articulation is env members[i].
            rows = self._cloth.members(s)
            d = _to_torch(art.data.joint_pos) - want[rows]
            # Wrapped: a full 180 deg start may read back as -180.
            err = torch.rad2deg(torch.atan2(torch.sin(d), torch.cos(d)).abs().amax(dim=1))
            ok = float((err < tol_deg).float().mean())
            size = self._env_size[rows]
            corr = float(torch.corrcoef(torch.stack((err, size)))[0, 1]) if err.numel() > 2 else 0.0
            print(
                f"[fold_pool] {sh.name}: start folds at {math.degrees(angle):.0f} deg land within "
                f"{tol_deg:g} deg in {ok * 100:.1f}% of envs (error median {float(err.median()):.3f}, "
                f"p95 {float(err.quantile(0.95)):.3f}, max {float(err.max()):.3f} deg; corr with "
                f"sheet size {corr:+.2f})",
                flush=True,
            )
            if ok < share:
                raise RuntimeError(
                    f"[fold_pool] {sh.name}: start folds land within {tol_deg:g} deg in only "
                    f"{ok * 100:.1f}% of envs (need {share * 100:.0f}%)"
                )
        # Back to the startup state: every env on shape 0, flat, no start angles left anywhere. The
        # first trainer reset redraws.
        for s, table in enumerate(self._start_tables):
            self.object.set_start_joints(s, all_ids, torch.zeros((n, table.shape[1]), device=self.device))
        self._reset_all_to(0, torch.zeros((n, self._start_tables[0].shape[1]), device=self.device))
        self.sim.step(render=False)
        self.scene.update(dt)

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
        # The CLOTH's version, skipping the chain's (which would lift by one variant's ply gap).
        ClothTaskMixin._init_fold_targets(self)
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
        if getattr(self, "_start_tables", None) is not None:
            # As above: `start_folded` is the share of episodes that started partly folded,
            # `fold_start_folded` the share that did AND folded.
            self._termination_reasons["start_folded"] = self._start_folded
            self._termination_reasons["fold_start_folded"] = entered & self._start_folded
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        reward = super()._get_rewards()
        # How many steps a fold takes. The logger averages `episode_final` over finished episodes,
        # so `fold_steps` (the episode length if it folded, else 0) divided by `done_fold` is the
        # mean length of a folded episode; `fold_steps_flat_start` / `fold_flat_start` the same for
        # episodes that started flat. Not `done_*`: eval/episodes.py reads those as booleans.
        # `episode_length_buf` already counts this step (DirectRLEnv.step increments it before
        # `_get_dones`), so it equals eval's `per_env_length` for an episode ending here.
        entered = self._termination_reasons["fold"].bool()
        flat_fold = entered & ~self._start_folded
        length = self.episode_length_buf.float()
        final = self.extras["episode_final"]
        final["fold_steps"] = entered.float() * length
        final["fold_flat_start"] = flat_fold.float()
        final["fold_steps_flat_start"] = flat_fold.float() * length
        return reward

    # ------------------------------------------------------------------ startup proofs

    def _calibrate_and_verify_placement(self) -> None:
        """``RigidClothEnv``'s placement / shape / yaw / joint-write proofs, once per pool shape.

        Each shape in turn is made active in every env that CARRIES it -- every env, unless one shape
        per env -- and checked there; the others are parked. All checks fatal, for the reasons the
        chain's version gives -- every failure they catch otherwise trains silently.
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
        mass = self._pool_plate_mass()
        carried = self._cloth.carried
        want_mass = carried.sum(dim=0).float() * float(self.cfg.rigid_cloth.density) * self._env_size**2
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

        one_shape = self._env_shape_np is not None
        if one_shape:
            obj.select(all_ids, obj.first_carried())
        for s, sh in enumerate(self._pool):
            ids = carried[s].nonzero(as_tuple=True)[0]  # the envs this shape is checked in
            if not one_shape:
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
                err = (cloud[ids].mean(dim=1) - target[ids])[:, :2]
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

            cloud = place(0.0)[ids]
            shape_err = float((centred(cloud)[..., :2] - ref[ids][..., :2]).norm(dim=-1).max())
            yaw_err = 0.0
            cent_err = 0.0
            for yaw in (0.5 * math.pi, math.radians(-120.0)):
                cy = place(yaw)[ids]
                yaw_err = max(
                    yaw_err, float((centred(cy)[..., :2] - rotated(yaw)[ids]).norm(dim=-1).max())
                )
                cent_err = max(
                    cent_err, float((cy.mean(dim=1) - target[ids])[:, :2].norm(dim=-1).max())
                )
            if shape_err > 1.0e-3 or yaw_err > 1.0e-3 or cent_err > 1.0e-3:
                raise RuntimeError(
                    f"[fold_pool] {sh.name}: cloud is not the placed flat sheet -- shape "
                    f"{shape_err * 1e3:.2f} mm, yawed shape {yaw_err * 1e3:.2f} mm, yawed centroid "
                    f"{cent_err * 1e3:.2f} mm (tolerance 1 mm); layout {self._cloth.quat_layout}"
                )

            # The parked shapes really are parked: nowhere near this env's table. (With one shape per
            # env nothing is parked.)
            for k in range(len(self._pool)):
                rows = (carried[k] & (obj.active != k)).nonzero(as_tuple=True)[0]
                if k == s or rows.numel() == 0:
                    continue
                far = (
                    (self._cloth.shape_positions(k)[rows].mean(dim=1) - origins[rows]).norm(dim=-1).min()
                )
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
        self._verify_arm_clear()

    def _verify_arm_clear(self, steps: int = 30, tol_deg: float = 2.0) -> None:
        """The arm holds its home pose: nothing in the scene presses on it.

        A table extended toward the robot (0.6 x 0.7 m) once sat in the arm's elbow, which rests
        below the table top. Every reset then knocked the arm away, it stayed pinned at the table
        edge, and two training arrays ran to completion without one fold. Nothing else fails on
        that: the sheet is placed correctly and the episodes simply time out.
        """
        from isaacsimenvs.tasks.cloth.utils.cloth_adapter import _to_torch
        from isaacsimenvs.tasks.play.utils.reset_utils import _reset_table_pose

        robot = self.robot
        dt = float(self.cfg.sim.dt)
        # The table is moved to its working height only by a reset; this runs before the first one.
        _reset_table_pose(self, torch.arange(self.num_envs, device=self.device))
        # The sheets out of the way first, as the start-fold checks do: the placement checks above
        # leave every env's sheet at its origin, at the hand's height, and one dropping onto the
        # fingers reads as a drifting arm (7-shape pool: 0.7 deg in every run, 6 and 13 deg twice).
        self._reset_all_to(0, torch.zeros((self.num_envs, int(self._pool_arts[0].num_joints)), device=self.device), dy=-0.5)
        arm = [i for i, name in enumerate(robot.joint_names) if name.startswith("iiwa14")]
        home = _to_torch(robot.data.default_joint_pos).clone()
        robot.write_joint_state_to_sim(home, torch.zeros_like(home))
        robot.set_joint_position_target(home)
        for _ in range(steps):
            robot.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt)
        drift = math.degrees(float((_to_torch(robot.data.joint_pos) - home)[:, arm].abs().max()))
        if drift > tol_deg:
            raise RuntimeError(
                f"[fold_pool] the arm cannot hold its home pose: a joint drifted {drift:.1f} deg in "
                f"{steps} steps (tolerance {tol_deg}) -- something in the scene, most likely the "
                f"table ({self.cfg.assets.table_urdf}), is pressing on it"
            )
        print(f"[fold_pool] arm holds its home pose ({drift:.2f} deg drift)", flush=True)

    def _verify_pool_joint_writes(self, s: int) -> None:
        """A partial joint write reaches the solver and only the envs it names (see the chain's)."""
        from isaacsimenvs.tasks.cloth.utils.cloth_adapter import _to_torch

        # In the articulation's own instance order: rows [0, m/2) are `half`, the rest are not.
        ids = self._cloth.members(s)
        m = int(ids.numel())
        if m < 2:
            return
        art = self._pool_arts[s]
        nj = int(art.num_joints)
        half = ids[: m // 2]
        test = 0.3
        q = torch.full((half.shape[0], nj), test, device=self.device, dtype=torch.float32)
        self.object.write_joint_positions(s, q, env_ids=half)
        self.sim.step(render=False)
        self.scene.update(float(self.cfg.sim.dt))
        got = _to_torch(art.data.joint_pos)
        moved = float(got[: m // 2].abs().max())
        untouched = float(got[m // 2 :].abs().max())
        self.object.write_joint_positions(s, torch.zeros((m, nj), device=self.device), env_ids=ids)
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
