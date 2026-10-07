"""RigidClothEnv — the cloth-folding task with a hinged rigid chain as the manipuland.

Subclasses :class:`~isaacsimenvs.tasks.cloth.cloth_env.ClothEnv` and overrides exactly **two**
hooks. Everything else -- the fold targets, the keypoint reward, the progress ratchet, the goal
marker, the finite-value guard, the success criterion, the observation layout -- is inherited and
runs the same code on the same quantities.

    ``build_physics_cfg``   drop the coupled MJWarp + VBD solve for plain MJWarp
    ``_install_cloth``      spawn an articulation instead of authoring a deformable mesh

That is the whole point of doing it this way rather than writing a second env. A rigid chain is
being proposed as a *substitute* for the VBD sheet, and the only claim worth testing is whether the
policy learns the same task faster. If the two envs scored the fold differently -- even slightly, even
only in the reward shaping -- a difference in learning curves would be unattributable, and the
comparison would quietly measure the reward change instead of the physics change. Sharing
``cloth_env.py`` verbatim makes that class of error impossible rather than unlikely.

The bridge is ``utils/rigid_cloth_adapter.py``: it reconstructs the sheet's ``resolution x
resolution`` particle grid from the slats' ``body_pos_w`` / ``body_quat_w``, so the inherited task
code never learns that its cloth is made of boxes.

**Assets are baked, not converted here.** The Newton stack replaces URDF conversion with a
content-addressed cache lookup that raises on a miss, so a chain must be baked once under Isaac Sim
before a run can load it::

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python -m isaacsimenvs.newton.usd_cache \\
        --populate --rigid_cloth_variants

Then, per variant::

    .venv_isaaclab3/bin/python isaacsimenvs/train.py \\
        --task Isaacsimenvs-RigidCloth-Direct-v0 --agent rl_games_sapg_cfg_entry_point \\
        --headless env.rigid_cloth.variant=box3-mid

The chain itself -- construction, the particle-grid adapter, the startup proofs, the fold target at
its own ply gap -- is ``rigid_cloth_task.RigidClothTaskMixin``, shared with the PhysX env. This class
is the Newton side: the USD cache lookup, and every contact and randomisation write, which go into
``newton.Model`` arrays and MJWarp's own model.
"""

from __future__ import annotations

import math

import torch

from isaacsimenvs.tasks.cloth.cloth_env import ClothEnv
from isaacsimenvs.tasks.cloth.rigid_cloth_env_cfg import RigidClothEnvCfg
from isaacsimenvs.tasks.cloth.rigid_cloth_task import RigidClothTaskMixin
from isaacsimenvs.tasks.cloth.utils import rigid_cloth as rc
from isaacsimenvs.tasks.cloth.utils import rigid_cloth_assets as assets


def wp_to_torch(array) -> torch.Tensor:
    """A torch VIEW of a warp array -- same memory, so writes reach the solver."""
    import warp as wp

    return wp.to_torch(array)

__all__ = ["RigidClothEnv"]


class RigidClothEnv(RigidClothTaskMixin, ClothEnv):
    """Cloth-folding task whose manipuland is a chain of hinged rigid slats."""

    cfg: RigidClothEnvCfg

    @staticmethod
    def build_physics_cfg(cfg: "RigidClothEnvCfg"):
        """Plain MJWarp -- there is no deformable left to couple to.

        ``ClothEnv`` returns a coupled MJWarp + VBD ``NewtonCfg`` with a staggered proxy mapping,
        which exists solely because a particle sheet cannot share a solver with a rigid robot. A
        hinged chain is rigid, so it joins the robot in MJWarp and the whole coupling layer --
        proxy bodies, substepping, contact buffers, AVBD ramping -- goes away. That is where the
        speed-up this model is being proposed for comes from; it is not a tuning difference.
        """
        return cfg.newton.build(int(cfg.scene.num_envs))

    pre_clone_scene_hook = staticmethod(lambda env: env._install_cloth())

    def _chain_usd(self, urdf_path, work_dir) -> str:
        """The baked USD, by cache lookup -- the kit-less stack cannot convert (``assets.convert_variant``)."""
        return assets.convert_variant(urdf_path, work_dir)

    def _setup_chain_physics(self) -> None:
        self._index_chain_model()
        self._apply_slat_friction()
        self._apply_contact_margin()
        self._apply_contact_stiffness()

    # ------------------------------------------------------------------ contact physics

    def _index_chain_model(self) -> None:
        """Per-env index tensors into Newton's flat model arrays, built once.

        Every contact and DR write below is an in-place write into ``newton.Model`` arrays through
        ``wp.to_torch`` views. In place is not a style choice: the step is a captured CUDA graph, so
        a reassigned array would never be seen by the solver, while new VALUES in the same memory
        are -- the same property ``patches.notify_solver`` relies on for the fingertip friction.
        """
        import numpy as np

        from isaacsimenvs.newton import compat

        model = self.sim.physics_manager.get_model()
        self._model = model
        n = self.num_envs

        body_labels = [label.rsplit("/", 1)[-1] for label in model.body_label]
        body_world = model.body_world.numpy()
        shape_body = model.shape_body.numpy()
        fingertip_names = set(compat.play_module("scene_utils").FINGERTIP_LINK_NAMES)

        def per_world(indices, world_of, what: str) -> torch.Tensor:
            rows = [[] for _ in range(n)]
            for i in indices:
                rows[int(world_of[i])].append(int(i))
            counts = {len(r) for r in rows}
            if len(counts) != 1 or 0 in counts:
                raise RuntimeError(
                    f"[rigid_cloth] {what}: expected the same nonzero count in every env, got "
                    f"{sorted(counts)}"
                )
            return torch.tensor(rows, device=self.device, dtype=torch.long)

        slat_bodies = [b for b, lab in enumerate(body_labels) if lab.startswith(rc.SLAT_LINK_PREFIX)]
        self._slat_bodies = per_world(slat_bodies, body_world, "slat bodies")

        is_slat_body = np.zeros(len(body_labels), dtype=bool)
        is_slat_body[slat_bodies] = True
        is_tip_body = np.array([any(f in lab for f in fingertip_names) for lab in body_labels])
        shape_world = np.array([body_world[b] if b >= 0 else -1 for b in shape_body])
        slat_shapes = [s for s, b in enumerate(shape_body) if b >= 0 and is_slat_body[b]]
        tip_shapes = [s for s, b in enumerate(shape_body) if b >= 0 and is_tip_body[b]]
        self._slat_shapes = per_world(slat_shapes, shape_world, "slat shapes")
        self._tip_shapes = per_world(tip_shapes, shape_world, "fingertip shapes")

        joint_labels = [label.rsplit("/", 1)[-1] for label in model.joint_label]
        joint_world = model.joint_world.numpy()
        qd_start = model.joint_qd_start.numpy()
        hinge_dofs = []
        dof_world = {}
        for j, lab in enumerate(joint_labels):
            if lab.startswith(rc.SLAT_JOINT_PREFIX):
                d = int(qd_start[j])
                hinge_dofs.append(d)
                dof_world[d] = joint_world[j]
        self._hinge_dofs = per_world(hinge_dofs, dof_world, "hinge DOFs")

        self._mu = wp_to_torch(model.shape_material_mu)
        self._margin = wp_to_torch(model.shape_margin)
        print(
            f"[rigid_cloth] indexed per env: {self._slat_bodies.shape[1]} slat bodies, "
            f"{self._slat_shapes.shape[1]} slat shapes, {self._tip_shapes.shape[1]} fingertip "
            f"shapes, {self._hinge_dofs.shape[1]} hinge DOFs",
            flush=True,
        )

    def _solver(self):
        solver = getattr(self.sim.physics_manager, "_solver", None)
        if solver is None:
            raise RuntimeError("[rigid_cloth] no MJWarp solver on the physics manager")
        return solver

    def _notify(self, flags: int) -> None:
        self._solver().notify_model_changed(flags)

    def _apply_slat_friction(self) -> None:
        """Give every slat contact the friction the cloth has in the same contact.

        Newton's VBD mixes a particle's ``soft_contact_mu`` (0.25) with a shape's by geometric mean,
        so the sheet feels ``sqrt(0.25 * 0.5) = 0.354`` against the table and the robot links and
        ``sqrt(0.25 * 1.5) = 0.612`` against a fingertip. MJWarp resolves a contact by geom
        PRIORITY first and by ``max`` only on a tie (``contact_params``), and under the tie a slat
        could not go below the table's 0.5 or the fingertip's 1.5 -- it felt 0.5 / 1.5 before this.

        With ``friction_priority``: slats at priority 1 carry the table-side value and win every
        slat contact except against a fingertip; fingertips at priority 2 carry the fingertip-side
        value and win that one. Both are then exact. The side effect is fingertip-TABLE friction,
        which the fingertip now also wins (1.5 -> 0.612); see ``RigidClothCfg.friction_priority``.
        """
        import newton

        if not bool(getattr(self.cfg.assets, "apply_material_properties", True)):
            return
        r = self.cfg.rigid_cloth
        self._mu[self._slat_shapes.reshape(-1)] = float(r.friction)
        if r.friction_priority:
            self._mu[self._tip_shapes.reshape(-1)] = float(r.fingertip_friction)
            import warp as wp

            # Shape -> geom by inverting `mjc_geom_to_newton_shape`, NOT by reading
            # `newton_shape_to_mjc_geom`: that one array faults on any host-side access after the
            # scene is built (a torch view: illegal memory access; `.numpy()` / `wp.copy`:
            # segfault), while every other solver array reads normally. Init-time only.
            solver = self._solver()
            geom_to_shape = solver.mjc_geom_to_newton_shape.numpy()[0]
            geom_of = {int(s): g for g, s in enumerate(geom_to_shape) if s >= 0}
            try:
                slat_geoms = [geom_of[int(s)] for s in self._slat_shapes[0].tolist()]
                tip_geoms = [geom_of[int(s)] for s in self._tip_shapes[0].tolist()]
            except KeyError as exc:
                raise RuntimeError(f"[rigid_cloth] shape {exc} has no MuJoCo geom") from exc
            slat_geoms = __import__("numpy").array(slat_geoms)
            tip_geoms = __import__("numpy").array(tip_geoms)
            self._slat_geoms, self._tip_geoms = slat_geoms, tip_geoms
            if (slat_geoms < 0).any() or (tip_geoms < 0).any():
                raise RuntimeError("[rigid_cloth] a slat or fingertip shape has no MuJoCo geom")
            # Geoms are shared by every world; priority is not a per-world field.
            arr = solver.mjw_model.geom_priority
            prio = arr.numpy()
            if prio.ndim == 1:
                prio[slat_geoms] = 1
                prio[tip_geoms] = 2
            else:
                prio[:, slat_geoms] = 1
                prio[:, tip_geoms] = 2
            arr.assign(wp.array(prio, dtype=arr.dtype, device=arr.device))
            self._geom_priority = (int(prio.reshape(-1, prio.shape[-1])[0, slat_geoms[0]]),
                                   int(prio.reshape(-1, prio.shape[-1])[0, tip_geoms[0]]))
        self._notify(int(newton.ModelFlags.SHAPE_PROPERTIES))
        print(
            f"[rigid_cloth] friction: slat-table/link {r.friction:.3f}, slat-fingertip "
            f"{r.fingertip_friction if r.friction_priority else self.cfg.assets.finger_tip_friction:.3f}"
            f" ({'by priority' if r.friction_priority else 'max rule; table side is max(slat, 0.5)'})",
            flush=True,
        )

    def _apply_contact_margin(self) -> None:
        """Put the chain's contact surfaces where the cloth's are (see ``RigidClothCfg.contact_margin``)."""
        import newton

        r = self.cfg.rigid_cloth
        m = r.resolved_contact_margin(self.cfg.cloth.thickness)
        self._slat_margin_base = self._margin[self._slat_shapes].clone()
        self._contact_margin = m
        if m <= 0.0:
            return
        self._margin[self._slat_shapes.reshape(-1)] = (self._slat_margin_base + m).reshape(-1)
        self._drop_slat_self_contacts()
        self._notify(int(newton.ModelFlags.SHAPE_PROPERTIES))
        print(
            f"[rigid_cloth] contact margin +{m * 1e3:.2f} mm on every slat shape "
            f"(base {float(self._slat_margin_base.mean()) * 1e3:.2f} mm)",
            flush=True,
        )

    def _apply_contact_stiffness(self) -> None:
        """Every shape's material ke / kd (see ``RigidClothCfg.contact_ke``); 0 keeps Newton's."""
        import newton

        r = self.cfg.rigid_cloth
        if r.contact_ke <= 0.0 and r.contact_kd <= 0.0:
            return
        if r.contact_ke <= 0.0 or r.contact_kd <= 0.0:
            raise ValueError("[rigid_cloth] contact_ke and contact_kd must be set together")
        wp_to_torch(self._model.shape_material_ke).fill_(float(r.contact_ke))
        wp_to_torch(self._model.shape_material_kd).fill_(float(r.contact_kd))
        self._notify(int(newton.ModelFlags.SHAPE_PROPERTIES))
        solref = wp_to_torch(self._solver().mjw_model.geom_solref).reshape(-1, 2)
        print(
            f"[rigid_cloth] contact ke {r.contact_ke:g} kd {r.contact_kd:g} on every shape -> MuJoCo "
            f"solref {solref.min(0).values.tolist()}..{solref.max(0).values.tolist()}",
            flush=True,
        )

    def _drop_slat_self_contacts(self) -> None:
        """Remove slat-slat pairs from the collision pipeline's explicit pair list, in place.

        Margins are summed per contact, so with a margin on every slat two plies would be held
        ``2 * margin`` apart -- a folded ``box3-mid`` could never close. A chain's joint limits
        already stop its plies at their geometric contact, which is where the cloth's settle.

        The list is referenced by the captured step, so it cannot be shortened. Each removed pair
        is instead retargeted at the same slat in the NEXT env, which is ``env_spacing`` away and
        can never overlap: the broad phase rejects it on bounding boxes and it generates nothing.
        """
        from isaaclab_newton.physics.newton_manager import NewtonManager

        pipeline = getattr(NewtonManager, "_collision_pipeline", None)
        pairs_wp = getattr(pipeline, "shape_pairs_filtered", None) if pipeline is not None else None
        if pairs_wp is None:
            raise RuntimeError(
                "[rigid_cloth] no explicit collision pair list to edit; slat-slat contacts cannot "
                "be removed, and with a contact margin they would hold the plies apart"
            )
        n = self.num_envs
        if n < 2:
            raise RuntimeError("[rigid_cloth] dropping slat self-contact needs at least 2 envs")
        pairs = wp_to_torch(pairs_wp)
        if pairs.dim() == 1:
            pairs = pairs.view(-1, 2)
        total = int(self._model.shape_count)
        world_of = torch.full((total,), -1, device=self.device, dtype=torch.long)
        slot_of = torch.full((total,), -1, device=self.device, dtype=torch.long)
        s = self._slat_shapes
        world_of[s.reshape(-1)] = torch.arange(n, device=self.device).repeat_interleave(s.shape[1])
        slot_of[s.reshape(-1)] = torch.arange(s.shape[1], device=self.device).repeat(n)
        a, b = pairs[:, 0].long(), pairs[:, 1].long()
        hit = (world_of[a] >= 0) & (world_of[b] >= 0)
        if not bool(hit.any()):
            self._dropped_slat_pairs = 0
            return
        other = s[(world_of[b[hit]] + 1) % n, slot_of[b[hit]]]
        pairs[hit, 1] = other.to(pairs.dtype)
        self._dropped_slat_pairs = int(hit.sum())
        print(
            f"[rigid_cloth] slat-slat contact removed: {self._dropped_slat_pairs} pairs "
            f"({self._dropped_slat_pairs // n} per env) retargeted across envs",
            flush=True,
        )

    # ------------------------------------------------------------------ randomisation

    def _init_physics_randomization(self) -> None:
        """Capture the matched (centre) values, then draw a first sample for every env."""
        model = self._model
        self._mass = wp_to_torch(model.body_mass)
        self._inv_mass = wp_to_torch(model.body_inv_mass)
        self._inertia = wp_to_torch(model.body_inertia)
        self._inv_inertia = wp_to_torch(model.body_inv_inertia)
        self._kd = wp_to_torch(model.joint_target_kd)
        self._ke = wp_to_torch(model.joint_target_ke)
        b = self._slat_bodies
        self._mass_base = self._mass[b].clone()
        self._inertia_base = self._inertia[b].clone()
        self._inv_inertia_base = self._inv_inertia[b].clone()
        self._slat_mu_base = float(self.cfg.rigid_cloth.friction)
        self._tip_mu_base = self._mu[self._tip_shapes].clone()
        n = self.num_envs
        #: The draw each env is currently running with, for logging and for the probe.
        self.dr_sample = {
            k: torch.ones(n, device=self.device)
            for k in ("slat_friction", "fingertip_friction", "hinge_damping", "mass")
        }
        self.dr_sample["hinge_stiffness"] = torch.zeros(n, device=self.device)
        self.dr_sample["contact_margin"] = torch.full(
            (n,), float(self._contact_margin), device=self.device
        )
        self._dr_ready = bool(self.cfg.rigid_cloth.randomization.enabled)
        if self._dr_ready:
            self._randomize_physics(torch.arange(n, device=self.device))
            rd = self.cfg.rigid_cloth.randomization
            print(
                f"[rigid_cloth] physics randomisation ON, per env at every reset: slat friction x"
                f"{tuple(rd.slat_friction_scale)}, fingertip friction x"
                f"{tuple(rd.fingertip_friction_scale)}, hinge damping x"
                f"{tuple(rd.hinge_damping_scale)} (log), hinge stiffness "
                f"{tuple(rd.hinge_stiffness)} N m/rad, contact margin "
                f"{self._contact_margin * 1e3:.1f}{rd.contact_margin_offset[0] * 1e3:+.1f}.."
                f"{rd.contact_margin_offset[1] * 1e3:+.1f} mm, mass x{tuple(rd.mass_scale)}",
                flush=True,
            )

    def _randomize_physics(self, env_ids) -> None:
        """Re-draw the chain's physics for ``env_ids``. In-place writes, then one solver notify."""
        import newton

        rd = self.cfg.rigid_cloth.randomization
        r = self.cfg.rigid_cloth
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
        n = int(ids.numel())
        if n == 0:
            return

        def uni(lo_hi) -> torch.Tensor:
            lo, hi = float(lo_hi[0]), float(lo_hi[1])
            return torch.empty(n, device=self.device).uniform_(lo, hi)

        def log_uni(lo_hi) -> torch.Tensor:
            lo, hi = math.log(float(lo_hi[0])), math.log(float(lo_hi[1]))
            return torch.exp(torch.empty(n, device=self.device).uniform_(lo, hi))

        slat_mu = self._slat_mu_base * uni(rd.slat_friction_scale)
        tip_scale = uni(rd.fingertip_friction_scale)
        damp = float(r.joint_damping) * log_uni(rd.hinge_damping_scale)
        stiff = uni(rd.hinge_stiffness)
        margin = (self._contact_margin + uni(rd.contact_margin_offset)).clamp(min=0.0)
        mass = uni(rd.mass_scale)

        shapes = self._slat_shapes[ids]
        self._mu[shapes.reshape(-1)] = slat_mu.unsqueeze(1).expand_as(shapes).reshape(-1)
        tips = self._tip_shapes[ids]
        self._mu[tips.reshape(-1)] = (self._tip_mu_base[ids] * tip_scale.unsqueeze(1)).reshape(-1)
        if self._contact_margin > 0.0:
            self._margin[shapes.reshape(-1)] = (
                self._slat_margin_base[ids] + margin.unsqueeze(1)
            ).reshape(-1)

        bodies = self._slat_bodies[ids]
        m = self._mass_base[ids] * mass.unsqueeze(1)
        self._mass[bodies.reshape(-1)] = m.reshape(-1)
        self._inv_mass[bodies.reshape(-1)] = (1.0 / m).reshape(-1)
        s3 = mass.view(n, 1, 1, 1)
        self._inertia[bodies.reshape(-1)] = (self._inertia_base[ids] * s3).reshape(-1, 3, 3)
        self._inv_inertia[bodies.reshape(-1)] = (self._inv_inertia_base[ids] / s3).reshape(-1, 3, 3)

        dofs = self._hinge_dofs[ids]
        self._kd[dofs.reshape(-1)] = damp.unsqueeze(1).expand_as(dofs).reshape(-1)
        self._ke[dofs.reshape(-1)] = stiff.unsqueeze(1).expand_as(dofs).reshape(-1)

        # Shapes through the public notify (0.06 ms). Mass and hinge gains through the solver's own
        # copy kernels ONLY: the public notify for those two flags also re-runs MuJoCo's global
        # constant computation (`set_const_0` / `set_const_fixed` / length ranges), measured at
        # 34 ms per call against an 11 ms env step -- at 1536 envs something resets every step, so
        # it would have more than tripled the step. The kernels are 0.02 / 0.06 ms and a per-world
        # readback confirms the new mass and gains reach MJWarp. What is skipped is the derived
        # constants (`body_invweight0`, which only scales the default constraint softness), which
        # stay at the nominal chain's values -- a small, deliberate approximation inside the DR.
        self._notify(int(newton.ModelFlags.SHAPE_PROPERTIES))
        solver = self._solver()
        cheap = (
            getattr(solver, "_update_model_inertial_properties", None),
            getattr(solver, "_update_joint_dof_properties", None),
        )
        if all(callable(f) for f in cheap):
            for f in cheap:
                f()
        else:
            self._notify(
                int(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
                | int(newton.ModelFlags.JOINT_DOF_PROPERTIES)
            )
        self.dr_sample["slat_friction"][ids] = slat_mu
        self.dr_sample["fingertip_friction"][ids] = self._tip_mu_base[ids, 0] * tip_scale
        self.dr_sample["hinge_damping"][ids] = damp
        self.dr_sample["hinge_stiffness"][ids] = stiff
        self.dr_sample["contact_margin"][ids] = margin
        self.dr_sample["mass"][ids] = mass
