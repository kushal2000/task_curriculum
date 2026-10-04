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
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import torch

from isaacsimenvs.tasks.cloth.cloth_env import ClothEnv
from isaacsimenvs.tasks.cloth.rigid_cloth_env_cfg import RigidClothEnvCfg
from isaacsimenvs.tasks.cloth.utils import rigid_cloth as rc
from isaacsimenvs.tasks.cloth.utils import rigid_cloth_assets as assets
from isaacsimenvs.tasks.cloth.utils.cloth_geometry import corner_indices, grid_mesh
from isaacsimenvs.tasks.cloth.utils.rigid_cloth_adapter import (
    ChainAsClothMesh,
    ChainAsRigidObject,
)


def wp_to_torch(array) -> torch.Tensor:
    """A torch VIEW of a warp array -- same memory, so writes reach the solver."""
    import warp as wp

    return wp.to_torch(array)

__all__ = ["RigidClothEnv"]


class RigidClothEnv(ClothEnv):
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

    def __init__(self, cfg: RigidClothEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # AFTER `super().__init__`, deliberately: `PlayEnv.__init__` runs the material patch at the
        # very end and that patch fills EVERY shape with `assets.robot_friction` before overriding
        # the fingertips. Writing slat friction earlier would be silently overwritten by 0.5.
        self._index_chain_model()
        self._apply_slat_friction()
        self._apply_contact_margin()
        # RC_SKIP_VERIFY=1 is for DIAGNOSING a failed verification only (the yaw probe needs a built
        # env to look at the bad state). Never set it for training or evaluation.
        if __import__("os").environ.get("RC_SKIP_VERIFY", "0") != "1":
            self._calibrate_and_verify_placement()
        else:
            print("[rigid_cloth] WARNING placement verification SKIPPED (RC_SKIP_VERIFY=1)", flush=True)
        self._init_physics_randomization()

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if getattr(self, "_dr_ready", False):
            self._randomize_physics(env_ids)

    # ------------------------------------------------------------------ placement

    def _calibrate_and_verify_placement(self) -> None:
        """Place the sheet where the task asks, and PROVE the emulated cloud is the flat sheet.

        This is the guard the whole adapter was missing. Two independent defects got all the way to a
        training launch because both of them produce a plausible-looking crumpled sheet rather than an
        error:

          * the articulation API is xyzw and the adapter read it as wxyz, so every grid vertex's offset
            inside its slat was mirrored -- a 100 mm sheet reported a 47 mm cloud;
          * the root offset placed the chain 51 mm off centre (``box3-mid``; 11.76 mm for the 17-slat
            chains), uniformly, while its internal spacing matched the URDF to 0.01 mm.

        Neither raises anything. A displaced, mirrored cloud still folds, still scores, still trains --
        against a fold target built from itself -- so the *only* way to catch this class of bug is to
        compare against ground truth the simulator does not supply: the analytic flat grid.

        Three checks, cheapest first, each fatal:

          1. **rest pose.** Every hinge within 1 deg of zero, so the comparison is against a sheet that
             really is flat and a fold has not been mistaken for a mapping error.
          2. **shape.** The cloud, centred, against ``grid_mesh`` centred -- which catches the mirror,
             a wrong ordering, a wrong resolution and a dropped slat.
          3. **placement.** The cloud's centroid against the requested centre, at zero yaw and again at
             90 deg. The yaw case is not redundant: a position-only calibration cannot see a rotation
             convention error on the write side, and a yawed sheet is what every reset actually draws.
        """
        import math

        obj = self.object
        n = self.num_envs
        dt = float(self.cfg.sim.dt)
        origins = self.scene.env_origins

        def place(yaw: float) -> torch.Tensor:
            """Ask for the sheet centred on each env origin at ``yaw``; return the cloud."""
            pose = torch.zeros((n, 7), device=self.device, dtype=torch.float32)
            pose[:, :3] = origins
            # In the layout `task_yaw` decodes, and through the CLASS method: the instance attribute
            # is wrapped by `patches.install_pose_write_conversion`, which would reorder this
            # quaternion first and turn every request into 180 deg (see `task_yaw`). That the wrapped
            # reset path lands the chain where it lands the cloth is `rigid_cloth_match_probe`'s job.
            pose[:, 0 + 3] = math.cos(0.5 * yaw)
            pose[:, 3 + 3] = math.sin(0.5 * yaw)
            type(obj).write_root_pose_to_sim(obj, pose)
            obj.write_root_velocity_to_sim(torch.zeros((n, 6), device=self.device))
            self.sim.step(render=False)
            self.scene.update(dt)
            return self._particles_w()

        # --- 0. which component order does this backend report orientations in? Settled by
        # measurement against the analytic flat grid, because the resting value is ambiguous between
        # the xyzw identity and the wxyz half-turn about z. Must run BEFORE the placement loop: the
        # wrong layout mirrors the cloud, and a mirrored cloud's centroid is still centred, so
        # calibration would converge happily on a backwards sheet.
        place(0.0)
        layout = self._cloth.detect_quat_layout(self._cloth_rest_local)
        scores = self._cloth._quat_layout_scores
        print(
            f"[rigid_cloth] quaternion layout = {layout} "
            f"(flat-grid error: xyzw {scores['xyzw'] * 1e3:.2f} mm, wxyz {scores['wxyz'] * 1e3:.2f} mm)",
            flush=True,
        )

        # --- 1. calibrate the placement at zero yaw, then confirm it closed.
        residual = None
        for attempt in range(3):
            cloud = place(0.0)
            err = (cloud.mean(dim=1) - origins)[:, :2]
            residual = float(err.norm(dim=-1).max())
            print(
                f"[rigid_cloth] placement pass {attempt}: centroid error "
                f"({float(err[0, 0]) * 1e3:+.2f}, {float(err[0, 1]) * 1e3:+.2f}) mm, "
                f"root offset {[round(float(v) * 1e3, 2) for v in obj.root_offset]} mm",
                flush=True,
            )
            if residual < 1.0e-4:
                break
            obj.calibrate_root_offset(err[0])
        if residual is None or residual > 1.0e-4:
            raise RuntimeError(
                f"[rigid_cloth] could not place the sheet: {residual * 1e3:.2f} mm centroid error "
                f"remains after 3 calibration passes. The chain's internal layout disagrees with "
                f"`chain_frames` by something other than a constant offset."
            )
        print(
            f"[rigid_cloth] placement calibrated: root offset "
            f"{[round(float(v) * 1e3, 2) for v in obj.root_offset]} mm, "
            f"residual {residual * 1e6:.1f} um",
            flush=True,
        )

        # --- 2. the rest pose really is flat, so check 3 means what it says.
        from isaacsimenvs.tasks.cloth.utils.rigid_cloth_adapter import _to_torch

        hinge = _to_torch(self._chain.data.joint_pos).abs().max()
        if math.degrees(float(hinge)) > 1.0:
            raise RuntimeError(
                f"[rigid_cloth] the chain is not flat at reset: worst hinge "
                f"{math.degrees(float(hinge)):.2f} deg. A flat rest pose is what the reset writes and "
                f"what every geometric check below assumes."
            )

        # --- 3. shape against the analytic grid, and the yawed placement.
        cloud = place(0.0)
        want = torch.tensor(self._cloth_rest_local, device=self.device, dtype=torch.float32)
        got = cloud[0] - cloud[0].mean(dim=0, keepdim=True)
        ref = want - want.mean(dim=0, keepdim=True)
        shape_err = float((got[:, :2] - ref[:, :2]).norm(dim=-1).max())
        if shape_err > 1.0e-3 and bool(int(__import__("os").environ.get("RC_DUMP", "0"))):
            from isaacsimenvs.tasks.cloth.utils.rigid_cloth_adapter import _to_torch as _t

            ch = self._chain
            names = list(ch.body_names)
            st = _t(ch.data.body_state_w)
            org = self.scene.env_origins[0]
            print("[dump] slat world pose (mm, env frame) and raw quat:", flush=True)
            for i in range(self.cfg.rigid_cloth.spec.num_slats):
                nm = rc.slat_link_name(i)
                b = names.index(nm)
                pw = st[0, b, 0:3] - org
                q = st[0, b, 3:7]
                print(
                    f"[dump]   {nm:<10} x={float(pw[0]) * 1e3:+8.2f} z={float(pw[2]) * 1e3:+8.2f} "
                    f"quat={[round(float(v), 4) for v in q]} "
                    f"chain_frames_x={rc.chain_frames(self.cfg.rigid_cloth.spec, None)[i][0] * 1e3:+8.2f}",
                    flush=True,
                )
            idx = self._cloth._resolve_bodies()
            loc = self._cloth._local
            print("[dump] first 3 grid vertices: body, local(mm), cloud(mm)", flush=True)
            for k in range(3):
                cw = cloud[0, k, :] - org
                print(
                    f"[dump]   v{k} body={names[int(idx[k])]} "
                    f"local=({float(loc[k, 0]) * 1e3:+.2f},{float(loc[k, 1]) * 1e3:+.2f}) "
                    f"cloud=({float(cw[0]) * 1e3:+.2f},{float(cw[1]) * 1e3:+.2f})",
                    flush=True,
                )

        if shape_err > 1.0e-3:
            # The four grid corners of each, in mm. A mirror, a 180 deg rotation and a transpose are
            # all visible at a glance here and are indistinguishable in a single scalar.
            res = self.cfg.cloth.resolution
            corners = [0, res - 1, res * (res - 1), res * res - 1]
            fmt = lambda t: [  # noqa: E731
                (round(float(t[i, 0]) * 1e3, 1), round(float(t[i, 1]) * 1e3, 1)) for i in corners
            ]
            raise RuntimeError(
                f"[rigid_cloth] the emulated cloud is not the flat sheet: worst vertex off by "
                f"{shape_err * 1e3:.2f} mm (tolerance 1.00 mm). Extent "
                f"{float(cloud[0][:, 0].amax() - cloud[0][:, 0].amin()) * 1e3:.1f} x "
                f"{float(cloud[0][:, 1].amax() - cloud[0][:, 1].amin()) * 1e3:.1f} mm against "
                f"{self.cfg.cloth.size * 1e3:.0f} mm square.\n"
                f"  corners got  {fmt(got)}\n"
                f"  corners want {fmt(ref)}\n"
                f"  quaternion layout in use: {self._cloth.quat_layout}"
            )

        yaw = 0.5 * math.pi
        cloud_y = place(yaw)
        err_y = float((cloud_y.mean(dim=1) - origins)[:, :2].norm(dim=-1).max())
        # A 90 deg yaw of a square sheet must swap which grid axis runs along world x. Checking the
        # ROTATION, not just the centroid: a dropped yaw would leave the centroid perfect.
        gy = cloud_y[0] - cloud_y[0].mean(dim=0, keepdim=True)
        rotated = torch.stack((-ref[:, 1], ref[:, 0]), dim=-1)
        yaw_shape_err = float((gy[:, :2] - rotated).norm(dim=-1).max())
        # And an oblique yaw, which a 90 deg test cannot distinguish from a transpose.
        ob = math.radians(-120.0)
        cloud_o = place(ob)
        go = cloud_o[0] - cloud_o[0].mean(dim=0, keepdim=True)
        rot_o = torch.stack(
            (
                ref[:, 0] * math.cos(ob) - ref[:, 1] * math.sin(ob),
                ref[:, 0] * math.sin(ob) + ref[:, 1] * math.cos(ob),
            ),
            dim=-1,
        )
        yaw_shape_err = max(yaw_shape_err, float((go[:, :2] - rot_o).norm(dim=-1).max()))
        err_y = max(err_y, float((cloud_o.mean(dim=1) - origins)[:, :2].norm(dim=-1).max()))
        if err_y > 1.0e-3 or yaw_shape_err > 1.0e-3:
            res = self.cfg.cloth.resolution
            corners = [0, res - 1, res * (res - 1), res * res - 1]
            fmt = lambda t: [  # noqa: E731
                (round(float(t[i, 0]) * 1e3, 1), round(float(t[i, 1]) * 1e3, 1)) for i in corners
            ]
            # How much did it actually turn? Recovered from the cloud rather than assumed, so
            # "turned the wrong way" and "did not turn at all" are distinguishable.
            from isaacsimenvs.tasks.cloth.utils.rigid_cloth_adapter import _to_torch as _t

            _rq = _t(self._chain.data.root_pose_w)
            _st = _t(self._chain.data.body_state_w)
            _names = list(self._chain.body_names)
            _sq = _st[0, _names.index(rc.slat_link_name(0)), 3:7]
            v = gy[corners[3], :2] - gy[corners[0], :2]
            v0 = ref[corners[3], :2] - ref[corners[0], :2]
            turned = math.degrees(
                math.atan2(float(v[1]), float(v[0])) - math.atan2(float(v0[1]), float(v0[0]))
            )
            # FATAL. This used to be a warning ("yaw is dropped on reset ... recorded as a caveat"),
            # and every chain policy trained under it only ever saw an axis-aligned sheet while the
            # VBD cloth draws a uniform heading. The cause was the adapter -- it wrote its wxyz yaw
            # quaternion into an xyzw API, which rolls the chain about its long axis instead of
            # turning it -- not the backend, and a chain that cannot be turned is not a stand-in
            # for the cloth.
            raise RuntimeError(
                f"[rigid_cloth] a yawed reset is wrong: centroid off {err_y * 1e3:.2f} mm, "
                f"shape off {yaw_shape_err * 1e3:.2f} mm, sheet actually turned {turned:.1f} deg.\n"
                f"  root_pose_w quat (xyzw) {[round(float(v), 4) for v in _rq[0, 3:7]]} "
                f"-- if this is NOT rotated the write was dropped; if it IS, the cloud is stale\n"
                f"  slat_0 quat (xyzw) {[round(float(v), 4) for v in _sq]}\n"
                f"  corners got      {fmt(gy)}\n"
                f"  corners expected {fmt(rotated)}\n"
                f"  corners at yaw 0 {fmt(ref)}\n"
                f"Every reset draws a yaw, so this is the common case, and a position-only "
                f"calibration cannot see a rotation convention error."
            )
        self._yaw_reset_applied = True
        print(
            f"[rigid_cloth] geometry verified: flat cloud within {shape_err * 1e6:.0f} um of the "
            f"analytic grid, 90 / -120 deg yaw within {yaw_shape_err * 1e6:.0f} um",
            flush=True,
        )
        self._verify_joint_writes()

    def _verify_joint_writes(self) -> None:
        """A PARTIAL joint write must reach the solver, and must touch only the envs it names.

        The reset path writes joints for a subset of envs, and on this backend that write is easy to
        get silently wrong: ``write_joint_position_to_sim_index`` is accepted and does nothing at all,
        and ``write_joint_state_to_sim`` does nothing when handed an explicit full ``arange`` while
        working perfectly with ``env_ids=None``. Both failures are invisible -- the chain simply stays
        where it was -- and the consequence is a chain that ends an episode folded and STARTS the next
        one folded, i.e. already at the goal on an env the task believes it reset.

        So: bend half the envs, confirm they bent and the rest did not, then flatten everything.
        """
        import math

        from isaacsimenvs.tasks.cloth.utils.rigid_cloth_adapter import _to_torch

        n = self.num_envs
        if n < 2:
            return
        chain = self._chain
        nj = _to_torch(chain.data.joint_pos).shape[1]
        half = torch.arange(n // 2, device=self.device)
        test = 0.3  # rad, well inside every variant's limit
        q = torch.full((half.shape[0], nj), test, device=self.device, dtype=torch.float32)

        self.object.write_joint_positions(q, env_ids=half)
        self.sim.step(render=False)
        self.scene.update(float(self.cfg.sim.dt))
        got = _to_torch(chain.data.joint_pos)
        moved = float(got[half].abs().max())
        untouched = float(got[n // 2 :].abs().max())

        # Flatten again before anything else runs, whatever the outcome.
        self.object.write_joint_positions(
            torch.zeros((n, nj), device=self.device, dtype=torch.float32)
        )
        self.sim.step(render=False)
        self.scene.update(float(self.cfg.sim.dt))

        if moved < 0.5 * test:
            raise RuntimeError(
                f"[rigid_cloth] a partial joint write did not reach the solver: asked for "
                f"{math.degrees(test):.1f} deg on {half.shape[0]} envs, got "
                f"{math.degrees(moved):.2f} deg. Every reset writes joints this way, so a chain that "
                f"ends an episode folded would start the next one folded."
            )
        if untouched > math.radians(5.0):
            raise RuntimeError(
                f"[rigid_cloth] a partial joint write leaked into envs it did not name: "
                f"{math.degrees(untouched):.2f} deg on the untouched half."
            )
        print(
            f"[rigid_cloth] joint writes verified: partial write moved "
            f"{math.degrees(moved):.1f} deg on the named envs, "
            f"{math.degrees(untouched):.2f} deg on the rest",
            flush=True,
        )

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

    # ------------------------------------------------------------------ construction

    def _install_cloth(self) -> None:
        """Spawn the chain as an articulation, in place of authoring a deformable mesh.

        Mirrors ``ClothEnv._install_cloth`` step for step -- neutralise the inherited rigid tool,
        reshape the goal marker, install the manipuland, publish its scale -- so the scene differs
        from the VBD one in the manipuland and in nothing else.
        """
        from isaaclab.assets import Articulation, ArticulationCfg
        from isaaclab.sim.spawners.from_files import UsdFileCfg

        c = self.cfg.cloth
        r = self.cfg.rigid_cloth
        spec = r.spec
        if not spec.foldable():
            raise ValueError(f"variant {r.variant!r} cannot fold: {spec.describe()}")

        # `_check_thickness` is deliberately NOT called. It guards the VBD sheet's particle radius
        # against its grid spacing -- a constraint that exists because overlapping particles are how
        # a VBD cloth expresses "too thick". A slat has a collider, not a particle radius, and
        # `ChainSpec.joint_limit` already encodes its thickness as a kinematic bend limit.

        verts, _indices = grid_mesh(c.size, c.resolution)
        self._cloth_rest_local = verts
        self._cloth_kp_idx = corner_indices(c.resolution, c.fold_axis)

        # Same two preparations the cloth makes, and for the same reasons: the inherited procedural
        # tool is still spawned by `setup_scene` (the USD cache, the asset pool and the
        # `object_scales` observation are all built around it) and would otherwise be a live rigid
        # body loose in the scene; and the goal marker would render a hammer-shaped ghost.
        self._neutralise_rigid_object()
        self._reshape_goal_marker()

        # --- the chain's own asset. URDF text is deterministic given the spec and these three
        # numbers, which is what lets the bake and this run agree on a cache key.
        urdf_dir = Path(tempfile.mkdtemp(prefix="rigid_cloth_"))
        # `write_chain_urdf_for_run` is shared with the bake and does not accept a damping, because
        # the URDF text is the USD cache key: when the env wrote `r.joint_damping` (the ACTUATOR gain)
        # and the bake wrote the pinned 0.0, the two URDFs differed by one number and every run died
        # with a cache miss before building anything.
        urdf_path = assets.write_chain_urdf_for_run(
            r.variant, urdf_dir, density=r.density, joint_limit=r.joint_limit
        )
        usd_path = assets.convert_variant(urdf_path, urdf_dir)

        spawn_z = float(self.cfg.reset.table_reset_z) + c.start_height
        art_cfg = ArticulationCfg(
            prim_path="/World/envs/env_.*/RigidCloth",
            spawn=UsdFileCfg(usd_path=usd_path),
            # The root LINK is `slat_<root>`, whose frame sits at that slat's centre rather than at
            # the sheet's -- so this is the sheet centre shifted by that offset, the same correction
            # `ChainAsRigidObject` applies on every reset. Getting it wrong here spawns the sheet
            # off-centre on step 0 and then snaps it into place at the first reset, which reads as a
            # physics glitch.
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(rc.chain_frames(spec, None)[spec.root][0], 0.0, spawn_z)
            ),
            actuators={},
        )
        from isaaclab.actuators import ImplicitActuatorCfg

        # ALWAYS present, even at zero stiffness, because this is the only place `armature` and
        # `damping` can be set. The URDF's `<dynamics>` does not survive the importer -- the Newton
        # model reports `joint_damping = 0.0` on every slat hinge however the URDF is written -- so a
        # chain without this actuator runs with an undamped, inertia-free hinge. Measured: from a
        # dead stop in free fall, `joint_vel` hit 1.3e3-4.9e3 rad/s on the FIRST step and every
        # variant diverged within 2-7 steps, taking the robot with it.
        #
        # Stiffness is a spring, not a servo: torque against a zero target, standing in for the
        # cloth's bending rigidity. Zero by default, because the cloth's own bending is nearly zero.
        # Newton picks the MuJoCo actuator TYPE from the gains at build time and it cannot change
        # afterwards: zero stiffness builds a velocity actuator, which has no position gain, so a
        # randomised hinge stiffness written later would be silently ignored. When DR draws a
        # stiffness, build with a negligible one (1e-9 N m/rad; the DR range is 1e5x larger) to get
        # the position+velocity actuator whose two gains are both per-env. The deterministic arm
        # keeps the original velocity-only drive.
        stiffness = float(r.joint_stiffness)
        rd = r.randomization
        if rd.enabled and float(rd.hinge_stiffness[1]) > 0.0 and stiffness == 0.0:
            stiffness = 1.0e-9
        art_cfg.actuators = {
            "hinges": ImplicitActuatorCfg(
                joint_names_expr=[f"{rc.SLAT_JOINT_PREFIX}.*"],
                stiffness=stiffness,
                damping=r.joint_damping,
                armature=r.joint_armature,
                effort_limit_sim=None,
            )
        }

        chain = Articulation(art_cfg)
        self.scene.articulations["rigid_cloth"] = chain
        self._chain = chain

        mesh = ChainAsClothMesh(
            chain, spec=spec, resolution=c.resolution, num_envs=self.num_envs, device=self.device
        )
        self._cloth = mesh

        # `PlayEnv` holds the manipuland as `env.object`; every downstream module reads it there.
        # Without this the policy observes the inert rigid tool while the chain is a bystander.
        self.object = ChainAsRigidObject(
            mesh,
            articulation=chain,
            spec=spec,
            num_envs=self.num_envs,
            rest_local=verts,
            keypoint_idx=self._cloth_kp_idx,
            corner_idx=self._cloth_kp_idx,
            corner_rest=[verts[i] for i in self._cloth_kp_idx],
            spawn_z=spawn_z,
            # Same total mass as the sheet: `density` is per area in both, and the chain's slats
            # sum to `density * size * span` with `span == size`.
            mass=r.density * c.size * spec.span,
            device=self.device,
        )
        self._write_object_scale()

        print(
            f"[rigid_cloth] {r.variant}: {spec.describe()}\n"
            f"[rigid_cloth] mass {r.density * c.size * spec.span * 1e3:.1f} g, "
            f"ply gap {r.ply_gap * 1e3:.2f} mm, "
            f"joint limit {math.degrees(spec.joint_limit):.1f} deg, "
            f"friction {r.friction:.3f}\n"
            f"[rigid_cloth] grid {c.resolution}x{c.resolution} on {spec.num_slats} slats, "
            f"keypoints {self._cloth_kp_idx}",
            flush=True,
        )

    # `_write_object_scale` is deliberately NOT overridden -- `ClothEnv`'s version is inherited, so
    # the chain publishes the SHEET's bounding box (size x size x cloth.thickness), not its own slab
    # thickness of 2 or 5.88 mm.
    #
    # An earlier version did override it, on the reasoning that a manipuland should not be described
    # by something else's dimensions -- the bug the cable env once had. That reasoning is right for an
    # independent task and wrong here, and the cross-evaluation is what makes the difference.
    # `object_scales` is an OBSERVATION (`obs_utils.py:301`); it does not size the reward, which takes
    # `reward.fixed_size` under `fixed_size_keypoint_reward: true`. So the only thing the override
    # bought was a policy trained on the chain seeing `[2.5, 2.5, 0.05]` and then meeting
    # `[2.5, 2.5, 0.40]` the moment it was evaluated on the cloth -- an out-of-distribution input, and
    # a transfer failure with a trivial cause that would have been reported as a physics result.
    #
    # The chain's real thickness still reaches the policy, through the physics: how the sheet drapes,
    # where the plies rest, what the fingers can pinch. It does not need to be announced twice.

    def _init_fold_targets(self) -> None:
        """Place the fold target where THIS chain's plies actually come to rest.

        ``ClothEnv._init_fold_targets`` lifts the target by ``cloth.self_contact_radius``, and its
        docstring is explicit that the number has to be where contact settles: using the wrong one
        cost the VBD sheet 14 mm of error at a perfect fold. A chain settles at its own ply gap --
        2.00 mm for ``box3-mid`` and the surface-hinge variants, 5.88 mm for the mid-hinge uniform
        ones, because a mid-plane fold spends a whole slat standing vertically as the wall.

        So the same reasoning gives a different number, and the cleanest way to apply it is to write
        the chain's ply gap into the field the parent reads. Disable with
        ``rigid_cloth.fold_lift_from_ply_gap=false`` to score every variant against the cloth's
        2 mm instead, which is the right setting for asking "does this chain satisfy the CLOTH's
        criterion" rather than "can this chain fold".
        """
        r = self.cfg.rigid_cloth
        if r.fold_lift_from_ply_gap:
            gap = r.ply_gap
            if abs(gap - self.cfg.cloth.self_contact_radius) > 1e-9:
                print(
                    f"[rigid_cloth] fold target lift {self.cfg.cloth.self_contact_radius * 1e3:.2f}"
                    f" -> {gap * 1e3:.2f} mm (this chain's ply gap)",
                    flush=True,
                )
            self.cfg.cloth.self_contact_radius = gap
        super()._init_fold_targets()
