"""The fold task on a deformable VBD cloth sheet, on Newton (Isaac Lab 3, ``.venv/isaaclab3``).

Where rigid-body-trained policies are evaluated (or fine-tuned). The task -- hooks, obs, reward,
success -- is ``utils/task_env.ClothTaskEnv``, shared with the rigid env; this file is the Newton
side: the Isaac Lab 3 / Newton fixes it needs (``newton/patches.py``), the scene
(``utils/scene_newton.py``), reading a ``ClothState`` from it, the reset writes, and the cloth's
own solver hooks (friction rescale, full-surface SDF, bending plasticity, the contact-based touch
test for ``fold.release.touch=contact``).

**Quaternions**: Isaac Lab 3.0 stores them xyzw; ``_read_state`` converts once to the utils' wxyz
and the writes pass xyzw.
"""

from __future__ import annotations

from isaacsimenvs.newton import compat  # isort: skip

compat.install_isaaclab3_compat()

import torch  # noqa: E402

from isaacsimenvs.newton import patches  # noqa: E402
from isaacsimenvs.tasks.cloth.deformable_env_cfg import ClothEnvCfg  # noqa: E402
from isaacsimenvs.tasks.cloth.utils import geometry, quat, reset_utils, scene_newton  # noqa: E402
from isaacsimenvs.tasks.cloth.utils.plasticity import BendingPlasticity  # noqa: E402
from isaacsimenvs.tasks.cloth.utils.robot import ROBOT  # noqa: E402
from isaacsimenvs.tasks.cloth.utils.state import ClothState  # noqa: E402
from isaacsimenvs.tasks.cloth.utils.task_env import ClothTaskEnv  # noqa: E402

__all__ = ["ClothEnv"]


def _t(x) -> torch.Tensor:
    """Isaac Lab 3.0 asset data are warp-backed ProxyArrays; the utils want torch."""
    return x.torch if hasattr(x, "torch") else x


def _install_newton_fixes(cfg: ClothEnvCfg) -> None:
    """The generic Isaac Lab 3 / Newton fixes, before ``_setup_scene``. Idempotent.

    Newton's solver is the coupled MJWarp (robot) + VBD (cloth) solve; Isaac Lab synthesises
    Newton actuators from the Lab actuator configs (the USD drives carry the gains, see
    ``scene_newton``). The rest are ``newton/patches.py``'s: actuator gains broadcast to (N, J),
    armature / friction written as 2.3 wrote them, the implicit actuators' bookkeeping torque kept
    out of the sim, the robot's gravity as MuJoCo gravcomp, condim on the fingertips.
    """
    cfg.sim.physics = cfg.cloth.build_physics(cfg.newton, int(cfg.scene.num_envs), ROBOT)
    cfg.sim.use_newton_actuators = True
    patches.install_actuator_broadcast()
    patches.install_armature_writeback()
    patches.install_implicit_effort_fix()
    if not cfg.use_gravcomp:
        raise ValueError("use_gravcomp=false needs the old object-gravity wrench path, which is gone")
    patches.install_newton_gravcomp()
    patches.install_newton_condim(int(cfg.newton.condim), tuple(ROBOT.fingertip_link_names))


class ClothEnv(ClothTaskEnv):
    """Fold the ``+x`` half of a VBD cloth sheet onto the ``-x`` half."""

    cfg: ClothEnvCfg

    def __init__(self, cfg: ClothEnvCfg, render_mode: str | None = None, **kwargs):
        c = cfg.cloth
        c.check_thickness()
        if cfg.fold.task == "mixed":
            raise NotImplementedError("fold.task=mixed: the VBD sheet runs fold or unfold")
        _install_newton_fixes(cfg)
        kappa = float(c.plastic_yield_curvature)
        self._plastic_set_curvature = float(c.plastic_set_curvature)
        self._plasticity = None
        undo = [
            self._install_plasticity_at_capture(kappa) if kappa > 0.0 else None,
            self._install_full_surface_sdf(cfg) if c.full_surface_contact else None,
            self._install_cloth_friction(cfg) if (c.self_contact_mu > 0.0 or c.table_contact_mu > 0.0) else None,
        ]
        try:
            super().__init__(cfg, render_mode, **kwargs)
        finally:
            for u in undo:
                if u is not None:
                    u()
        # Newton floors degenerate inertia at finalisation, which makes the hand's massless
        # placeholder links heavier than finger bones; restore before the solver re-reads the model.
        patches.restore_placeholder_inertia(self)
        scene_newton.set_base_materials(self, ROBOT)
        patches.notify_solver(self)  # MuJoCo caches gains and shape materials at build
        if undo[2] is not None:
            from isaaclab_newton.physics import NewtonManager
            from newton._src.solvers.coupled.solver_coupled import SolverCoupled

            SolverCoupled._cloth_friction_apply(NewtonManager._solver)
        if kappa > 0.0 and self._plasticity is None:  # nothing was captured, so nothing installed it
            self._install_plasticity(kappa)
        print(f"[cloth] VBD sheet {c.resolution}x{c.resolution}, {c.size:.3f} m, particle radius "
              f"{c.particle_radius * 1e3:.1f} mm, self-contact radius {c.self_contact_radius * 1e3:.1f} mm", flush=True)

    # ------------------------------------------------------------------ scene

    def _setup_scene(self) -> None:
        from isaacsimenvs.utils.setup_timer import stamp

        c = self.cfg.cloth
        self._dims = torch.tensor([[c.size, c.size, 2.0 * c.particle_radius]], device=self.device).expand(self.num_envs, 3).clone()
        stamp("scene: start")
        rest = geometry.grid_points(c.size, c.size, c.resolution)
        scene_newton.build(self, ROBOT, rest, geometry.grid_triangles(c.resolution))
        stamp("scene: built and replicated")

    def _point_resolution(self) -> int:
        return int(self.cfg.cloth.resolution)

    def _ply_gap(self) -> torch.Tensor:
        """Two plies settle one SELF-contact radius apart (measured 2.5 mm at 2 mm); the particle
        radius is the cloth-RIGID offset and has no say in a fold."""
        return torch.full((self.num_envs,), float(self.cfg.cloth.self_contact_radius), device=self.device)

    # ------------------------------------------------------------------ reads

    def _read_state(self) -> ClothState:
        r = self.robot.data
        bs = _t(r.body_state_w)
        body_pos, body_quat = bs[..., 0:3], quat.to_wxyz(bs[..., 3:7])
        pts = _t(self._cloth.data.nodal_pos_w).view(self.num_envs, -1, 3)
        vel = _t(self._cloth.data.nodal_vel_w).view(self.num_envs, -1, 3)
        return ClothState(
            joint_pos=_t(r.joint_pos), joint_vel=_t(r.joint_vel),
            palm_pos_w=body_pos[:, self._palm_id], palm_quat_w=body_quat[:, self._palm_id],
            palm_vel_w=bs[:, self._palm_id, 7:13],
            fingertip_pos_w=body_pos[:, self._fingertip_ids], fingertip_quat_w=body_quat[:, self._fingertip_ids],
            hand_pos_w=body_pos[:, self._hand_body_ids],
            points_w=pts, points_vel_w=vel,
            table_top_z=self.scene.env_origins[:, 2] + self._table_z + self._table_half_height,
        )

    def _read_joint_pos(self) -> torch.Tensor:
        return _t(self.robot.data.joint_pos)

    def _robot_defaults(self):
        d = self.robot.data
        return list(d.joint_names), _t(d.default_joint_pos).clone(), _t(d.joint_pos_limits).clone()

    def _body_ids(self, names: list[str]) -> list[int]:
        ids, found = self.robot.find_bodies(names, preserve_order=True)
        if list(found) != list(names):
            raise RuntimeError(f"robot bodies {names} resolved to {found}")
        return list(ids)

    def _body_ids_regex(self, regex: str) -> list[int]:
        ids, _ = self.robot.find_bodies(regex)
        if not ids:
            raise ValueError(f"fold.release.contact_bodies={regex!r} matched no robot body")
        return list(ids)

    # ------------------------------------------------------------------ writes

    def _set_robot_targets(self, pos_target, effort) -> None:
        self.robot.set_joint_position_target(pos_target)
        self.robot.set_joint_effort_target(effort)

    def _write_robot(self, ids, joint_pos, joint_vel) -> None:
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=ids)

    def _write_table(self, ids, table_z) -> None:
        pose = torch.zeros(ids.numel(), 7, device=self.device)
        pose[:, :2] = torch.tensor(self.cfg.reset.table_reset_xy, device=self.device, dtype=torch.float32)
        pose[:, 2] = table_z
        pose[:, :3] += self.scene.env_origins[ids]
        pose[:, 6] = 1.0  # xyzw identity
        self.table.write_root_pose_to_sim(pose, env_ids=ids)

    def _write_sheet(self, ids, xy, yaw, angle) -> None:
        """The sheet flat (or the + half turned up by ``angle``), yawed, ``cloth.start_height`` above
        this episode's table centre -- it drops onto the table -- every particle at rest."""
        local = geometry.folded_start(self._sheet_geom, angle, ids)
        world = quat.apply(quat.yaw(yaw).unsqueeze(1), local)
        centre = torch.zeros(ids.numel(), 3, device=self.device)
        centre[:, :2] = xy + torch.tensor(self.cfg.reset.table_reset_xy, device=self.device, dtype=torch.float32)
        centre[:, 2] = self._table_z[ids] + float(self.cfg.cloth.start_height)
        target = world + (centre + self.scene.env_origins[ids]).unsqueeze(1)
        self._cloth.write_nodal_pos_to_sim_index(target, ids)
        self._cloth.write_nodal_velocity_to_sim_index(torch.zeros_like(target), ids)

    def _randomize(self, ids) -> None:
        """Per reset: the reset sheets' plastic set back to the build-time rest angles."""
        if getattr(self, "_plasticity", None) is None:
            return
        import warp as wp

        edges = torch.nonzero(torch.isin(self._edge_env, ids.to(self._edge_env.device))).reshape(-1)
        self._plasticity.reset(wp.from_torch(edges.to(torch.int32)))

    # ------------------------------------------------------------------ the cloth's solver hooks

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
        pats = [re.compile(p) for p in c.proxy_body_patterns(ROBOT)]
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
        from isaaclab_newton.physics import NewtonManager

        model = NewtonManager._model
        cloth_solver = NewtonManager._solver.solver("cloth")
        # SolverCoupled runs the VBD entry on a ModelView with its OWN projected edge_rest_angle:
        # the yield update must write that array (the model's alone never reaches the solver, which
        # left every env sheet purely elastic), mirrored to the model's for refreshes and readers.
        views = [e.view for e in getattr(NewtonManager._solver, "_entries", {}).values() if e.solver is cloth_solver]
        solver_rest = views[0].edge_rest_angle if views else None
        kappa_set = float(getattr(self, "_plastic_set_curvature", 0.0))
        plast = BendingPlasticity(model, kappa, solver_rest_angle=solver_rest, set_curvature=kappa_set)
        if kappa_set > 0.0:
            print(f"[cloth] two-stage plasticity: set grows only past {kappa_set:g} /m "
                  f"(yield {kappa:g} /m)", flush=True)
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

    # ------------------------------------------------------------------ fold.release.touch=contact

    def _cloth_contact_buffer(self):
        """``(contacts, entry)``: the hand-cloth contacts the cloth (VBD) entry steps on, and that
        entry. Under ``SolverCoupledProxy`` they come from the proxy pipeline built on the cloth view
        (``get_proxy_contacts(<rigid entry>, "cloth")``; particle / edge / face records, shape and
        particle ids global, body ids LOCAL to the cloth view). ``(None, entry)`` before the first
        step or without a proxy pipeline."""
        from isaaclab_newton.physics import NewtonManager

        solver = NewtonManager._solver
        cloth = solver.solver("cloth")
        name = next((k for k, e in getattr(solver, "_entries", {}).items() if e.solver is cloth), None)
        if name is None:
            return None, None
        entry = solver._entries[name]
        for src, dst in getattr(solver, "_proxy_collision_configs", {}):
            if dst == name:
                return solver.get_proxy_contacts(src, dst), entry
        return getattr(solver, "_entry_contact_buffers", {}).get(name), entry

    def hand_cloth_touching(self) -> torch.Tensor:
        """``(num_envs,)`` bool: any robot shape is in contact with the env's cloth this step
        (``fold.release.touch=contact``). Reads the cloth solver's own contact records and evaluates
        each one's gap as VBD's body-particle / edge / face contact does (``-penetration_depth``:
        ``n . (x - bx) - radius - shape_margin``, ``x`` the barycentric point over the record's
        particles, ``bx`` the body-side point at the current body pose); touching = gap <
        ``release_touch_gap``. A non-finite gap reads as touching."""
        import warp as wp
        from isaaclab_newton.physics import NewtonManager

        out = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        buf, entry = self._cloth_contact_buffer()
        if buf is None or buf.soft_contact_max == 0:
            return out
        model, state = NewtonManager._model, NewtonManager._state_0
        if self._touch_cache is None:
            import re

            view, dev = entry.view, self.device
            labels = list(model.body_label)
            sb_local = view.shape_body.numpy()  # body ids local to the cloth view
            l2g = entry.body_local_to_global.numpy() if entry.body_local_to_global is not None else None
            sb = [int(l2g[b]) if (l2g is not None and b >= 0) else int(b) for b in sb_local]
            robot = [b >= 0 and re.search(r"/Robot/", str(labels[b])) is not None for b in sb]
            if not any(robot):
                raise RuntimeError("fold.release.touch=contact: no robot shape found in the cloth view")
            if int(view.particle_count) != int(model.particle_count):
                raise RuntimeError("fold.release.touch=contact assumes the cloth view keeps the model's particle ids")
            margin = view.shape_margin
            margin = wp.to_torch(margin).float() if margin is not None and margin.shape[0] > 0 \
                else torch.zeros(len(sb), device=dev)
            self._touch_cache = {
                "robot": torch.tensor(robot, dtype=torch.bool, device=dev),
                "shape_body": torch.tensor(sb, dtype=torch.long, device=dev),  # GLOBAL body ids
                "margin": margin,
                "world": wp.to_torch(model.particle_world).long(),
                "radius": wp.to_torch(model.particle_radius).float(),
            }
            print(f"[cloth] release_touch=contact: {sum(robot)} robot shapes in the cloth view, gap < "
                  f"{float(self.cfg.fold.release.touch_gap) * 1e3:.1f} mm counts as touching", flush=True)
        tc = self._touch_cache
        m = int(buf.soft_contact_max)
        count = wp.to_torch(buf.soft_contact_count).long()  # (1,), stays on device
        shape = wp.to_torch(buf.soft_contact_shape).long()
        sel = (torch.arange(m, device=self.device) < count) & (shape >= 0)
        sel &= tc["robot"][shape.clamp(min=0)]
        ids = sel.nonzero(as_tuple=True)[0]
        if ids.numel() == 0:
            return out
        shape = shape[ids]
        corners = wp.to_torch(buf.soft_contact_indices).view(m, 3)[ids].long()
        bary = wp.to_torch(buf.soft_contact_barycentric).view(m, 3)[ids].float()
        normal = wp.to_torch(buf.soft_contact_normal).view(m, 3)[ids].float()
        body_pos = wp.to_torch(buf.soft_contact_body_pos).view(m, 3)[ids].float()
        pq = wp.to_torch(state.particle_q).view(-1, 3).float()
        ok = corners >= 0
        cc = corners.clamp(min=0)
        x = (bary.unsqueeze(-1) * pq[cc] * ok.unsqueeze(-1)).sum(1)
        radius = torch.where(ok, tc["radius"][cc], torch.zeros_like(bary)).amax(1)
        body = tc["shape_body"][shape]
        bq = wp.to_torch(state.body_q).view(-1, 7).float()[body.clamp(min=0)]
        p, qv, qw = bq[:, 0:3], bq[:, 3:6], bq[:, 6:7]  # warp transform: p, then quat xyzw
        t = 2.0 * torch.cross(qv, body_pos, dim=-1)
        bx = body_pos + qw * t + torch.cross(qv, t, dim=-1) + p
        gap = (normal * (x - bx)).sum(-1) - radius - tc["margin"][shape]
        self._touch_last_gap = gap  # diagnostics (cloth_touch_probe)
        self._touch_last_env = tc["world"][cc[:, 0]]
        touch = ~(gap >= float(self.cfg.fold.release.touch_gap))  # NaN -> touching
        env = self._touch_last_env
        env = env[touch & (env >= 0) & (env < self.num_envs)]
        out[env] = True
        return out
