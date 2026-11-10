"""ClothEnv — fold half a cloth sheet onto the other half.

Subclasses :class:`PlayNewtonEnv`, so the robot, table, action space and coupled solver are the
ones already measured on the cable task. Only the manipuland changes.

The sheet is authored as a flat ``UsdGeom.Mesh`` grid before the environments are replicated --
`DeformableObject` infers ``surface`` (cloth) from a ``UsdGeom.Mesh`` and ``volume`` from a
``UsdGeom.TetMesh``, so the prim type *is* the switch.

State is the position of a few keypoints on the moving half; the goal is their mirror image across
the fold line. See ``utils/cloth_geometry.py`` for both, kept separate so the fold is testable
without a simulator.

The task -- fold targets, reward, success, the goal marker, the finite guard -- is
``cloth_task.ClothTaskMixin``, shared with the PhysX envs. This class adds only the VBD sheet.
"""

from __future__ import annotations

from isaacsimenvs.newton import patches  # noqa: F401
from isaacsimenvs.tasks.cloth.cloth_env_cfg import ClothEnvCfg
from isaacsimenvs.tasks.cloth.cloth_task import ClothTaskMixin
from isaacsimenvs.tasks.cloth.utils.cloth_adapter import ClothAsRigidObject
from isaacsimenvs.tasks.cloth.utils.cloth_geometry import (
    corner_indices,
    grid_mesh,
)
from isaacsimenvs.tasks.play_newton.play_newton_env import PlayNewtonEnv

__all__ = ["ClothEnv"]


class ClothEnv(ClothTaskMixin, PlayNewtonEnv):
    """Play task with a VBD cloth sheet as the manipuland."""

    cfg: ClothEnvCfg

    def __init__(self, cfg: ClothEnvCfg, render_mode: str | None = None, **kwargs):
        kappa = float(cfg.cloth.plastic_yield_curvature)
        self._plasticity = None
        restore = self._install_plasticity_at_capture(kappa) if kappa > 0.0 else None
        restore_sdf = self._install_full_surface_sdf(cfg) if cfg.cloth.full_surface_contact else None
        c = cfg.cloth
        restore_mu = (
            self._install_cloth_friction(cfg) if (c.self_contact_mu > 0.0 or c.table_contact_mu > 0.0) else None
        )
        try:
            super().__init__(cfg, render_mode, **kwargs)
        finally:
            for undo in (restore, restore_sdf, restore_mu):
                if undo is not None:
                    undo()
        if restore_mu is not None:
            from isaaclab_newton.physics import NewtonManager
            from newton._src.solvers.coupled.solver_coupled import SolverCoupled

            SolverCoupled._cloth_friction_apply(NewtonManager._solver)  # (3)
        if kappa > 0.0 and self._plasticity is None:  # nothing was captured, so nothing installed it
            self._install_plasticity(kappa)

    @staticmethod
    def _install_cloth_friction(cfg: ClothEnvCfg):
        """Cloth-only shape frictions for ``cloth.self_contact_mu`` / ``cloth.table_contact_mu``.

        ``build_physics`` sets ``model.soft_contact_mu`` to the cloth-on-cloth value, which VBD's
        self-contact reads raw. Cloth-vs-shape contacts use ``sqrt(soft_contact_mu * shape_mu)``,
        so the cloth solver's ModelView gets its OWN ``shape_material_mu``: the parent's value
        scaled by ``mu_cal / mu_cc`` (``mu_cal`` = the preset's ``soft_contact_mu``), which keeps
        every hand pairing where the preset calibrated it, and ``T**2 / mu_cc`` on the table's
        shapes when ``table_contact_mu`` = T. The rigid solver keeps the parent's frictions.

        Timing matters: the task writes its frictions (robot 0.5, fingertips 1.5) into the PARENT
        model after start-up and ``notify_solver`` then re-projects shape attributes into every
        view (``SolverCoupled._refresh_model_view_overrides``), which would overwrite a rescale
        done once at construction. So: (1) ``NewtonVBDManager._create_solver`` is wrapped to make
        sure the view owns a copy before the solver (and any CUDA graph) exists; (2) the rescale is
        re-applied after every view refresh, IN PLACE so captured graphs keep the same pointer;
        (3) ``ClothEnv.__init__`` applies it once more at the end. Returns the undo for (1)."""
        import re

        import numpy as np
        import newton
        import warp as wp
        from isaaclab_contrib.deformable.vbd_manager import NewtonVBDManager
        from newton._src.solvers.coupled.solver_coupled import SolverCoupled

        c = cfg.cloth
        mu_cal = float(c.soft_contact_mu)
        mu_cc = float(c.self_contact_mu) if c.self_contact_mu > 0.0 else mu_cal
        table_t = float(c.table_contact_mu)

        def apply(solver) -> None:
            for entry in getattr(solver, "_entries", {}).values():
                if not isinstance(entry.solver, newton.solvers.SolverVBD):
                    continue
                view = entry.view
                parent = view.parent
                if "shape_material_mu" not in view.overrides:
                    raise RuntimeError("cloth view has no own shape_material_mu; rescaling would write the rigid model")
                pmu = parent.shape_material_mu.numpy().astype(np.float64)
                if view.shape_material_mu.shape[0] != pmu.shape[0]:
                    raise RuntimeError("cloth view shape layout differs from the model's; cannot map frictions")
                new = pmu * (mu_cal / mu_cc)
                labels = list(parent.body_label)
                sb = parent.shape_body.numpy()
                # Same match as the proxy set's `/World/envs/env_.*/Table` (a regex SEARCH: the table
                # body may sit below the Table prim).
                table = np.array([b >= 0 and re.search(r"/World/envs/env_[^/]+/Table", str(labels[b])) is not None
                                  for b in sb], dtype=bool)
                if table_t > 0.0:
                    if not table.any():
                        raise RuntimeError("cloth.table_contact_mu is set but no table shapes matched; body labels like "
                                           f"{[str(l) for l in labels[:3]]} ... {[str(l) for l in labels if 'able' in str(l)][:4]}")
                    new[table] = table_t**2 / mu_cc
                view.shape_material_mu.assign(wp.array(new.astype(np.float32), dtype=float, device=view.shape_material_mu.device))
                eff = np.sqrt(mu_cc * new)
                hand = (~table) & (sb >= 0)
                print(f"[cloth] friction: cloth-on-cloth {mu_cc:.3f} (model.soft_contact_mu "
                      f"{float(view.soft_contact_mu):.3f}); cloth-on-hand {sorted({round(float(v), 3) for v in eff[hand]})}; "
                      f"cloth-on-table {sorted({round(float(v), 3) for v in eff[table]})} ({int(table.sum())} shapes)",
                      flush=True)

        # (2) persistent, process-wide: re-apply after every view refresh.
        if not getattr(SolverCoupled, "_cloth_friction_wrapped", False):
            orig_refresh = SolverCoupled._refresh_model_view_overrides

            def _refresh(self, flags):
                orig_refresh(self, flags)
                fn = getattr(SolverCoupled, "_cloth_friction_apply", None)
                if fn is not None:
                    fn(self)

            SolverCoupled._refresh_model_view_overrides = _refresh
            SolverCoupled._cloth_friction_wrapped = True
        SolverCoupled._cloth_friction_apply = staticmethod(apply)

        # (1) the view must own its array before the solver is built.
        orig = NewtonVBDManager.__dict__["_create_solver"]

        def _create(cls, model, solver_cfg):
            if hasattr(model, "_cow_array"):
                model._cow_array("shape_material_mu")
            return orig.__get__(None, cls)(model, solver_cfg)

        NewtonVBDManager._create_solver = classmethod(_create)

        def undo() -> None:
            NewtonVBDManager._create_solver = orig

        return undo

    @staticmethod
    def _install_full_surface_sdf(cfg: ClothEnvCfg):
        """Give every mesh shape the cloth can touch a volume SDF, before ``ModelBuilder.finalize``.

        ``cloth.full_surface_contact`` makes the proxy pipeline sample each participating rigid
        mesh's SDF for edge/face contacts, and Newton raises on a mesh without one. Participating =
        the proxy bodies (``ClothCfg.proxy_body_patterns``) plus static shapes that collide with
        particles; the coupler clears collision flags on every other shape. Wraps
        ``NewtonManager._prepare_builder_for_finalize`` (the coupler manager reaches it via
        ``super()``); returns the undo."""
        import re

        import newton
        from isaaclab_newton.physics import NewtonManager

        c = cfg.cloth
        pats = [re.compile(p) for p in c.proxy_body_patterns()]
        mesh_types = (int(newton.GeoType.MESH), int(newton.GeoType.CONVEX_MESH))
        collide_particles = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        res = int(c.full_surface_sdf_resolution) or None
        orig = NewtonManager.__dict__["_prepare_builder_for_finalize"]

        def _prep(cls, builder) -> None:
            orig.__get__(None, cls)(builder)
            body_label = builder.body_label
            n, kinds = 0, {}
            for s in range(builder.shape_count):
                if int(builder.shape_type[s]) not in mesh_types:
                    continue
                b = int(builder.shape_body[s])
                if b >= 0:
                    lab = body_label[b] or ""
                    if not any(p.search(lab) for p in pats):
                        continue
                    key = lab.rsplit("/", 1)[-1]
                elif int(builder.shape_flags[s]) & collide_particles:
                    key = "static:" + str((builder.shape_label[s] or "").rsplit("/", 1)[-1])
                else:
                    continue
                builder.shape_force_sdf[s] = True
                if res is not None:
                    builder.shape_sdf_max_resolution[s] = res
                n += 1
                kinds[key] = kinds.get(key, 0) + 1
            per_world = {k: v for k, v in sorted(kinds.items())[:24]}
            print(f"[cloth] full_surface_contact: SDF on {n} mesh shapes (max_resolution "
                  f"{res or 'default'}); by body: {per_world}", flush=True)

        NewtonManager._prepare_builder_for_finalize = classmethod(_prep)

        def undo() -> None:
            NewtonManager._prepare_builder_for_finalize = orig

        return undo

    def _install_plasticity_at_capture(self, kappa: float):
        """Install plasticity right before Newton's first CUDA graph capture, so the graph holds it.

        Installing after `super().__init__` and capturing again does not work: the re-captured
        coupled step fails to instantiate ("invalid argument"), the back-to-back capture problem
        `NewtonManager.start_simulation` documents. Every start runs `_capture_or_defer_graph` once
        the model and solver exist, graph or not. Returns the undo."""
        from isaaclab_newton.physics import NewtonManager

        orig = NewtonManager.__dict__["_capture_or_defer_graph"]
        env = self

        def capture(cls):
            if env._plasticity is None:
                env._install_plasticity(kappa)
            return orig.__func__(cls)

        NewtonManager._capture_or_defer_graph = classmethod(capture)
        return lambda: setattr(NewtonManager, "_capture_or_defer_graph", orig)

    def _install_plasticity(self, kappa: float) -> None:
        """Wrap the coupled solver's cloth entry so every VBD substep updates the rest angles."""
        import numpy as np
        import torch
        from isaaclab_newton.physics import NewtonManager

        from isaacsimenvs.tasks.cloth.utils.plasticity import BendingPlasticity

        model = NewtonManager._model
        cloth_solver = NewtonManager._solver.solver("cloth")
        # SolverCoupled runs the VBD entry on a ModelView with its OWN projected edge_rest_angle:
        # the yield update must write that array (the model's alone never reaches the solver, which
        # left every env sheet purely elastic), mirrored to the model's for refreshes and readers.
        views = [e.view for e in getattr(NewtonManager._solver, "_entries", {}).values() if e.solver is cloth_solver]
        solver_rest = views[0].edge_rest_angle if views else None
        plast = BendingPlasticity(model, kappa, solver_rest_angle=solver_rest)
        plast.wrap_solver(cloth_solver)
        print(f"[cloth] plasticity writes the cloth solver's edge_rest_angle "
              f"({'its own view copy, mirrored to the model' if plast.mirror is not None else 'shared with the model'})",
              flush=True)
        ei = model.edge_indices.numpy()
        world = model.particle_world.numpy()[ei[:, 2]]
        world[(ei[:, 0] < 0) | (ei[:, 1] < 0)] = -1  # boundary edges never bend
        # `self.device` may not exist yet: this can run inside `super().__init__`.
        self._edge_env = torch.as_tensor(world, device=str(model.device), dtype=torch.long)
        self._plast_resets_logged = 0
        self._plasticity = plast
        w = plast.hinge_width[world >= 0]
        print(f"[cloth] plasticity: yield curvature {kappa:g} /m = {np.degrees(kappa * w.max()):.1f} deg per grid "
              f"cell, {np.degrees(kappa * w.min()):.1f} deg per diagonal, {int((world >= 0).sum())} hinges", flush=True)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if getattr(self, "_plasticity", None) is None:  # off, or a reset inside super().__init__
            return
        import torch
        import warp as wp

        dev = self._edge_env.device
        ids = torch.arange(self.num_envs, device=dev) if env_ids is None else \
            torch.as_tensor(env_ids, device=dev, dtype=torch.long).reshape(-1)
        edges = torch.nonzero(torch.isin(self._edge_env, ids)).reshape(-1)
        # Log the first few resets of envs that have run, as evidence the substep update is live.
        log = self._plast_resets_logged < 3 and edges.numel() and getattr(self, "common_step_counter", 0) > 0
        if log:
            rest = wp.to_torch(self._plasticity.model.edge_rest_angle)[edges]
            before = (float(rest.abs().max()) * 57.2958, int((rest.abs() > 0.1745).sum()))
        self._plasticity.reset(wp.from_torch(edges.to(torch.int32)))
        if log:
            after = float(wp.to_torch(self._plasticity.model.edge_rest_angle)[edges].abs().max()) * 57.2958
            print(f"[cloth] plasticity reset of {ids.numel()} envs: largest set angle {before[0]:.1f} deg, {before[1]} hinges "
                  f"past 10 deg; after reset {after:.1f} deg", flush=True)
            self._plast_resets_logged += 1

    @staticmethod
    def build_physics_cfg(cfg: "ClothEnvCfg"):
        """Coupled MJWarp (robot) + VBD (cloth) solve, sized for this env count."""
        return cfg.cloth.build_physics(cfg.newton, int(cfg.scene.num_envs))

    #: Runs at the end of `setup_scene`, before replication -- which is also when Newton imports
    #: the stage. A mesh authored after that point is invisible to the solver.
    pre_clone_scene_hook = staticmethod(lambda env: env._install_cloth())

    def _install_cloth(self) -> None:
        """Author the sheet and register it as a deformable, in place of the rigid tool."""
        # Re-validate here, not only in `ClothCfg.__post_init__`. Hydra constructs the dataclass
        # with its DEFAULTS -- which is when `__post_init__` runs and passes -- and only then
        # assigns the CLI overrides onto the instance. So `env.cloth.thickness=0.03` walks straight
        # past the thickness/spacing guard: it ran against thickness 0.012 and never saw 0.03.
        # That is how a 30 mm radius ended up on a 12.5 mm grid, every particle overlapping its
        # neighbours, which the solver expresses as instability rather than as a config error.
        self.cfg.cloth._check_thickness()
        self.cfg.cloth.check_task()
        if self.cfg.cloth.task == "mixed":
            raise NotImplementedError("cloth.task=mixed is for the PhysX pool and chain; the VBD sheet runs fold or unfold")
        from isaaclab.assets import DeformableObject, DeformableObjectCfg
        from isaaclab.sim.utils import find_matching_prims
        from pxr import Sdf, UsdGeom, UsdShade

        c = self.cfg.cloth
        verts, indices = grid_mesh(c.size, c.resolution)
        self._cloth_rest_local = verts
        # Corners of the moving half, not points along one edge: four points spanning an area
        # determine a rotation; four collinear points do not.
        self._cloth_kp_idx = corner_indices(c.resolution, c.fold_axis)

        # The inherited rigid tool is still spawned by `setup_scene` -- the asset pipeline, USD
        # cache and `object_scales` observation are all built around it. Make it inert, or it is a
        # live rigid body loose in the scene. (On the cable task this was visible for hours as a
        # hammer lying on the floor beside the table.)
        self._neutralise_rigid_object()
        self._reshape_goal_marker()

        prim_path = "/World/envs/env_.*/Cloth"
        spawn_z = float(self.cfg.reset.table_reset_z) + c.start_height

        # Author the mesh directly: `DeformableObject` reads vertices and indices off the prim, and
        # a `UsdGeom.Mesh` is what makes it a *surface* (cloth) rather than a volume.
        stage = None
        for env_prim in find_matching_prims("/World/envs/env_.*"):
            stage = env_prim.GetStage()
            mesh_path = env_prim.GetPath().AppendChild("Cloth")
            mesh = UsdGeom.Mesh.Define(stage, mesh_path)
            # Height is baked into the POINTS, not applied as a transform and not left to
            # `init_state.pos`. Measured: init_state alone left the sheet on the ground (z=0.006),
            # and a translate op on top of it put the sheet at 1.15 m and then exploded to 70 m.
            # Baking it is the one placement that cannot compose with another.
            mesh.CreatePointsAttr([(v[0], v[1], v[2] + spawn_z) for v in verts])
            mesh.CreateFaceVertexIndicesAttr(list(indices))
            mesh.CreateFaceVertexCountsAttr([3] * (len(indices) // 3))
            mesh.CreateDisplayColorAttr([tuple(c.color)])
            # Doubled-sided: a cloth folded over itself is viewed from both sides, and a
            # single-sided sheet renders as half-invisible once flipped.
            mesh.CreateDoubleSidedAttr(True)
            # NO translate op here. `DeformableObject` reads these points and then applies
            # `init_state.pos` on top, so a transform on the prim composes with it and the sheet
            # spawns at twice the intended height -- which is exactly what it did.

            # `DeformableObject` reads its material off a *bound* physics material carrying
            # `newton:*` attributes, and refuses the prim outright without one. It identifies the
            # material by the presence of `newton:density`, so that attribute is what makes this a
            # Newton deformable material rather than an ordinary shading material.
            mat_path = env_prim.GetPath().AppendChild("ClothMaterial")
            mat = UsdShade.Material.Define(stage, mat_path)
            mp = mat.GetPrim()
            for name, value in (
                ("newton:density", c.density),
                ("newton:particleRadius", c.particle_radius),
                ("newton:triKe", c.tri_ke),
                ("newton:triKa", c.tri_ka),
                ("newton:triKd", c.tri_kd),
                ("newton:edgeKe", c.edge_ke),
                ("newton:edgeKd", c.edge_kd),
            ):
                mp.CreateAttribute(name, Sdf.ValueTypeNames.Float).Set(float(value))

            binding = UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
            binding.GetDirectBindingRel("physics").SetTargets([mat_path])

        cloth_cfg = DeformableObjectCfg(
            prim_path=prim_path,
            spawn=None,  # the prim is authored above; nothing to spawn
            # Identity: the height is already in the points.
            init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        )
        cloth = DeformableObject(cloth_cfg)
        self.scene.deformable_objects["cloth"] = cloth
        self._cloth = cloth

        # `PlayEnv` holds the manipuland as `env.object` and every downstream module reads it from
        # there. Without this the policy observes the *inert rigid tool* -- which this env has just
        # made kinematic and collision-free -- while the sheet sits in the scene as a bystander.
        area_density = c.density * c.thickness  # surface density -> sheet mass
        self.object = ClothAsRigidObject(
            cloth,
            num_envs=self.num_envs,
            # LOCAL coordinates (z=0). `write_root_pose_to_sim` adds a centre whose z is
            # `spawn_z`, so baking the height in here too would place the sheet at 2 x spawn_z --
            # which it did, at 1.06 m instead of 0.53 m.
            rest_local=verts,
            keypoint_idx=self._cloth_kp_idx,
            corner_idx=self._cloth_kp_idx,
            corner_rest=[verts[i] for i in self._cloth_kp_idx],
            spawn_z=spawn_z,
            mass=area_density * c.size * c.size,
            device=self.device,
            reset_offset_xy=c.reset_offset_xy,
            # task: unfold -- reset with the + half folded over, the plies one self-contact radius
            # apart (where cloth-on-cloth contact settles; see `_init_fold_targets`).
            fold_start=(
                (0 if c.fold_axis == "x" else 1, list(c.unfold_start_deg), c.self_contact_radius)
                if c.task == "unfold" else None
            ),
        )
        self._write_object_scale()

        print(
            f"[cloth] {c.resolution}x{c.resolution} sheet, {c.size:.3f} m, "
            f"{c.num_particles} particles, {len(indices) // 3} triangles, "
            f"keypoints {self._cloth_kp_idx}",
            flush=True,
        )
