"""ClothEnvCfg — fold half a cloth onto the other half.

Subclasses :class:`PlayNewtonEnvCfg`, so the robot, table, action space and coupled-solver
machinery are exactly the ones measured on the cable task. Only the manipuland changes, from a
rigid tool to a VBD cloth sheet.

**The task.** A square sheet lies flat on the table. Its *far* half (``+x``) is tracked by a
handful of keypoints; the goal is to bring those keypoints onto the mirror image of their rest
positions across the fold line, i.e. to fold the far half back over the near half. Success is the
usual keypoint criterion: every tracked keypoint within tolerance, held for ``success_steps``.

**Why keypoints rather than a pose.** A cloth has no rigid frame. The cable task could fake one
from its first-to-last span, and that estimator degenerated as soon as the cable bent
(``docs/phase4_cable_env.md``). A sheet has no meaningful single axis at all, so the state *is* the
keypoint set -- there is nothing to reduce it to.

**Coupling defaults are inherited from the cable work, not guessed.** ``proxy_mode="staggered"``,
``cable_substeps=2``-equivalent substepping and the damping stack are what took the cable from 0 to
~14 goals per 64 envs; `lagged` coupling pairs the hand's begin-pose with its end-velocities, an
error that scales with hand speed and which shook the cable loose during transport. A cloth is more
compliant than a cable, so the same defect should hurt at least as much here.
"""

from __future__ import annotations

# IMPORT ORDER IS LOAD-BEARING, as in `cable_env_cfg`: importing `isaaclab_physx` binds
# `isaaclab.utils.configclass` to the same-named *submodule*, shadowing the decorator. Importing
# PlayNewtonEnvCfg first runs the compat shim that re-asserts it.
from isaacsimenvs.tasks.play_newton.play_newton_env_cfg import PlayNewtonEnvCfg  # isort: skip

from isaaclab.utils import configclass  # noqa: E402

__all__ = ["ClothCfg", "ClothEnvCfg"]


@configclass
class ClothCfg:
    """Cloth geometry, material, the fold definition, and the coupling that lets the hand feel it."""

    # --- geometry -------------------------------------------------------------------------
    size: float = 0.10
    """Side length of the square sheet [m]."""

    thickness: float = 0.012
    """Cloth thickness [m], as ``2 * particle_radius``.

    **A VBD cloth is a surface: thickness IS the particle collision radius**, there is no
    volumetric dimension. That makes it bounded by grid spacing -- particles closer together than
    their diameter self-collide at rest and the sheet crumples before anything touches it. The
    constraint is ``thickness <= size / (resolution - 1)``, checked in ``__post_init__``.

    Thicker is genuinely better for *contact*: the cable was ungraspable at 10 mm and liftable at
    30 mm, and thin geometry tunnels through fingers. But it is worse for *folding*, which needs
    bending. And 30 mm on a 100 mm sheet forces a 4x4 grid (spacing 33 mm), which is a stiff mat
    rather than cloth -- genuine 30 mm material is a volumetric soft body (``add_soft_mesh``, a
    tet mesh), not a cloth.

    Default 12 mm at ``resolution=9`` (spacing 12.5 mm) sits just inside the limit: thick enough
    for the hand to feel, thin enough to fold."""

    resolution: int = 9
    """Particles per side, so the sheet is ``resolution**2`` particles and
    ``2*(resolution-1)**2`` triangles. 9 gives 81 particles / 128 triangles -- coarse, but the
    cable work showed collision geometry is the binding memory cost at scale, and a finer sheet
    buys nothing until folding works at all."""

    start_height: float = 0.30
    """Spawn height above ``reset.table_reset_z`` [m].

    **The table's collision surface is at ``table_reset_z + 0.150``, not + 0.045.** See
    ``table_half_thickness``. Everything below is measured with zero actions, so the robot never
    approaches the sheet and anything it does is spawn behaviour:

        0.150  -> spawn 0.5300, surface 0.5298: in contact. EJECTED upward at ~1.7 m/s, peaks at
                  0.68 by step 10, falls back, settles by step 30 with a 17 mm spread across a
                  12 mm sheet -- i.e. visibly wrinkled.
        0.300  -> spawn 0.6800, a genuine 15 cm drop. Falls at exactly g (predicted 0.0340 m over
                  the first 5 steps, measured 0.0349), lands at 0.5360 and settles PERFECTLY flat
                  (min = mean = max to four decimals).

    An earlier version of this docstring reported 0.150 as "settles dead flat", which was wrong: it
    was read after settling and never checked the spread. Every episode was therefore starting with
    the sheet thrown into the air and landing creased, and that was mis-attributed in turn to
    thickness, to grid resolution, and to cloth physics.

    Values below the surface are worse still, and were correctly diagnosed at the time: at 0.010 the
    sheet starts inside the table and reaches 51 m by step 80."""

    table_half_thickness: float = 0.150
    """Distance from ``reset.table_reset_z`` up to the table's COLLISION surface [m].

    **Measured, not read off the USD.** The visual box is 0.045 half-thick, and that number sat here
    for a long time -- but the sheet settles with its lowest particles at 0.5360 against a table root
    of 0.3800, putting the contact surface at 0.5298, i.e. 0.1498 above the root. Confirmed twice
    from different drop heights, and `Table/box` is the only non-robot collider in the scene, so
    that is what the cloth is resting on.

    Getting it wrong is not a cosmetic error. At 0.045 the guard below accepted
    ``start_height = 0.15``, which looked like a 10 cm drop and actually spawned the sheet 0.2 mm
    above the surface -- effectively in contact. Contact resolution then ejected it upward at
    ~1.7 m/s at the start of EVERY episode: it flew to 0.68, fell back, and settled visibly
    wrinkled (17 mm spread across a 12 mm sheet). With a real drop it falls at exactly g and lands
    perfectly flat (spread 0.0000). The crumpled spawn had been attributed to thickness, to grid
    resolution, and to cloth physics; it was this."""

    # --- material -------------------------------------------------------------------------
    density: float = 100.0
    tri_ke: float = 1.0e4
    """In-plane stretch stiffness. Softening *stretch* was catastrophic on the cable (x0.01 peaked
    at 94.9 m/s); a slack surface oscillates. Keep this stiff."""
    tri_ka: float = 1.0e4
    """In-plane shear/area stiffness."""
    tri_kd: float = 1.0e-5
    """In-plane damping. Nonzero by default here: every cable damping term defaulted to 0.0 and
    the sheet is far more compliant."""
    edge_ke: float = 0.5
    """Bending stiffness. Low -- a cloth that will not bend cannot be folded. This is the opposite
    of the cable, where a stiff bend response was the thing ringing.

    Measured with a teleported fold: at 5.0 the crease springs open after **1** step; at 0.5 it
    holds **7**. A further 10x softer (0.05) gives the same 7, so the effect saturates here and
    bending stops being the binding constraint."""
    edge_kd: float = 1.0e-2

    # --- the fold -------------------------------------------------------------------------
    fold_axis: str = "x"
    """Axis normal to the fold line, in cloth-local coordinates: the ``+`` half folds onto ``-``."""

    num_keypoints: int = 4
    """Tracked keypoints on the moving half. Four matches the task's existing keypoint machinery
    and is the minimum that pins a folded flap's position *and* orientation."""

    max_folded_footprint: float = 0.65
    """Largest sheet extent along the fold axis, as a fraction of ``size``, that still counts as
    folded.

    A second condition on success, independent of keypoint distance. A perfect fold gives ~0.5; an
    unfolded sheet gives 1.0. Keypoint proximity alone proved insufficient: a rigid slide scored
    "folds" because the targets were world-fixed, and after that was corrected a *crumpled* sheet
    still scored transiently by putting four keypoints near their targets. 0.65 leaves room for a
    real fold's drape while excluding anything close to flat."""

    keypoint_tolerance: float = 0.04
    """Per-keypoint distance for success [m]. Deliberately looser than the cable's 1.5 cm: a
    folded flap settles wherever the drape puts it, and demanding rigid-body precision from a
    sheet would measure the tolerance rather than the fold."""

    # --- coupling (defaults carried from the cable result) --------------------------------
    proxy_mode: str = "staggered"
    """See `docs/phase4_cable_env.md`. `lagged` syncs the hand's *begin* pose with its *end*
    velocities; the mismatch scales with hand speed and was the single change that took the cable
    from ~0 to 5 goals in an episode."""

    substeps: int = 2
    """Sharp optimum on the cable: 1 peaked at 37 m/s, 4/8/16 degraded monotonically. Fewer, larger
    substeps mean fewer proxy exchanges per step, which is where the energy enters."""

    vbd_iterations: int = 80
    coupler_iterations: int = 1
    """Raising this made the cable *worse* at every setting tried."""

    mass_scale: float = 1.0
    """KEEP AT 1.0. 0.05 was ~500x worse on the cable and NaN'd the robot."""

    collide_interval: int = 1
    soft_contact_ke: float = 8.0e3
    soft_contact_mu: float = 10.0
    rigid_contact_k_start: float = 1.0e2

    proxy_links: str = "hand"
    """Which hand links the cloth can feel. `tips` (five distal phalanges) makes an enclosing grasp
    geometrically impossible; a pinch grasp is arguably enough to drag a cloth, but the asymmetry
    against the rigid baseline is not worth reintroducing."""

    proxy_table: bool = True
    """The table must be in the proxy set or the sheet falls through it."""

    color: tuple[float, float, float] = (0.85, 0.35, 0.55)

    # --- buffers ---------------------------------------------------------------------------
    rigid_body_particle_contact_buffer_size: int = 65536
    """Body-to-particle soft-contact capacity **PER RIGID BODY**, not global.

    Newton's own comment on the argument is explicit: "Per-body soft-contact list capacity"
    (`newton/_src/solvers/vbd/solver_vbd.py:266`, default 256). So it must NOT scale with env count:
    the allocation is this value x body count, and body count already scales with envs. Multiplying
    it by `num_envs` as well makes the allocation quadratic -- at 512 envs that is 1,048,576 x 2,304
    bodies = 2.4e9 elements, which overflows Warp's signed-int32 array dimension and aborts before a
    single step.

    1024, inherited from the cable where the manipuland was bodies rather than particles, was far
    too small: a 32-env run died with a CUDA device-side assert, which is what an out-of-bounds
    write into a contact buffer looks like from the host."""
    rigid_body_contact_buffer_size: int = 16384
    """Body-to-body contact capacity per rigid body (Newton default 64). Also per-body: the failure
    message is literally "Per-body rigid contact buffer overflowed 87 > 64"."""
    per_particle_triangle_pairs: int = 2048
    """Triangle-pair budget per cloth particle. The budget is GLOBAL and must scale with env count
    *and* geometry: on the cable a correctly env-scaled 4.19M still overflowed once segments
    doubled.

    Raised from 512 after a 32-env run overflowed twice. That run then died with
    ``normal expects all elements of std >= 0.0`` from the policy's sampler -- dropped contacts let
    a sheet penetrate, which produced NaN in that env's observation. The policy error was the
    symptom; the overflow was the cause, and it is worth remembering that this class of failure
    surfaces far from its origin."""

    @property
    def particle_radius(self) -> float:
        """Half the thickness: what the solver actually consumes."""
        return 0.5 * self.thickness

    @property
    def num_particles(self) -> int:
        return self.resolution**2

    @property
    def spacing(self) -> float:
        return self.size / (self.resolution - 1)

    @property
    def max_resolution_for_thickness(self) -> int:
        """Finest grid whose spacing still admits this thickness without self-overlap."""
        return max(2, int(self.size / self.thickness) + 1)

    def __post_init__(self) -> None:
        """Reject a sheet that would spawn inside the table, or a thickness the grid cannot hold.

        Both failures present as violent instability rather than as configuration errors, which is
        exactly how they were first diagnosed here -- as a solver problem.
        """
        if self.start_height <= self.table_half_thickness:
            raise ValueError(
                f"cloth start_height {self.start_height:.3f} m does not clear the table: "
                f"`reset.table_reset_z` is the table CENTRE and its surface is "
                f"{self.table_half_thickness:.3f} m higher. The sheet would spawn inside the "
                f"table and be ejected in one step. Use start_height > "
                f"{self.table_half_thickness:.3f}."
            )
        self._check_thickness()

    def _check_thickness(self) -> None:
        """Reject a thickness the grid cannot hold.

        Silent self-collision at rest looks exactly like a cloth that mysteriously crumples, and
        would be diagnosed as a solver problem rather than a configuration one.
        """
        if self.thickness > self.spacing + 1e-9:
            raise ValueError(
                f"cloth thickness {self.thickness:.3f} m exceeds grid spacing "
                f"{self.spacing:.3f} m (size {self.size:.3f} / {self.resolution - 1} gaps). "
                f"Particles would overlap at rest and the sheet would crumple before contact.\n"
                f"  Either reduce resolution to <= {self.max_resolution_for_thickness} "
                f"(spacing {self.size / max(1, self.max_resolution_for_thickness - 1):.3f} m), "
                f"or reduce thickness to <= {self.spacing:.3f} m.\n"
                f"  For genuine volumetric thickness, a cloth is the wrong primitive: use a soft "
                f"body (tet mesh / add_soft_mesh) instead."
            )


    def build_physics(self, newton_cfg, num_envs: int):
        """``NewtonCfg`` for the coupled MJWarp (robot) + VBD (cloth) scene.

        The one structural difference from the cable: the VBD entry owns **particles**, not bodies.
        A cloth is a particle system, so ``all_particles=True`` is what assigns the sheet to the
        deformable solver -- a body selector would match nothing and the cloth would be integrated
        by nobody, presenting as a sheet frozen in mid-air.
        """
        from isaaclab_contrib.coupling import (
            CouplerEntryCfg,
            CouplerProxyCfg,
            CouplerProxyMappingCfg,
        )
        from isaaclab_contrib.deformable import NewtonModelCfg, VBDSolverCfg
        from isaaclab_newton.physics import (
            MJWarpSolverCfg,
            NewtonCfg,
            NewtonCollisionPipelineCfg,
            NewtonShapeCfg,
        )

        rigid_bodies = [r"/World/envs/env_.*/Robot"]
        finger = r"/World/envs/env_.*/Robot/left_(thumb|index|middle|ring|pinky)_"
        proxy_bodies = [finger + ("(DP|MP|PP)" if self.proxy_links != "tips" else "(DP)")]
        if self.proxy_links == "hand":
            # The palm body is `iiwa14_link_7`; the URDF's `left_hand_C_MC` is merged away by the
            # importer and matches nothing.
            proxy_bodies.append(
                r"/World/envs/env_.*/Robot/(iiwa14_link_7|left_thumb_MC|left_pinky_MC)"
            )
        if self.proxy_table:
            rigid_bodies.append(r"/World/envs/env_.*/Table")
            proxy_bodies.append(r"/World/envs/env_.*/Table")

        budget = self.triangle_pair_budget(newton_cfg, num_envs)

        # No VBDSolverCfg subclass here, deliberately. `class_type` carries
        # "{DIR}.vbd_manager:NewtonVBDManager" and `{DIR}` is substituted against the *defining*
        # module, so subclassing inside this package makes it hunt for
        # `isaacsimenvs.tasks.cloth.vbd_manager`. The cable subclasses it only to add
        # `rigid_body_contact_buffer_size`, which is a body-BODY buffer; a cloth is particles, so
        # the buffer that matters (`rigid_body_particle_contact_buffer_size`) is already a field.
        vbd = VBDSolverCfg(
            iterations=self.vbd_iterations,
            rigid_body_particle_contact_buffer_size=self.rigid_body_particle_contact_buffer_size,
            rigid_contact_k_start=self.rigid_contact_k_start,
        )
        # `rigid_body_contact_buffer_size` is NOT a field on VBDSolverCfg (the cable subclasses it
        # for exactly this reason), but it IS in SolverVBD's signature, and the manager passes
        # through any cfg attribute the signature accepts. Setting it on the instance therefore
        # reaches the solver without a subclass -- and without a subclass, `{DIR}` in `class_type`
        # still resolves against isaaclab_contrib rather than this package.
        #
        # It defaults to 64. A 32-env cloth run hit "Per-body rigid contact buffer overflowed
        # 87 > 64" and then died with a CUDA device-side assert. Declaring the field in this config
        # without wiring it here left it inert: it read as configured and did nothing.
        vbd.rigid_body_contact_buffer_size = self.rigid_body_contact_buffer_size

        solver = CouplerProxyCfg(
            entries=[
                CouplerEntryCfg(
                    name="rigid",
                    solver_cfg=MJWarpSolverCfg(
                        solver=newton_cfg.solver,
                        integrator=newton_cfg.integrator,
                        njmax=newton_cfg.njmax,
                        nconmax=newton_cfg.nconmax,
                        impratio=newton_cfg.impratio,
                        cone=newton_cfg.cone,
                        iterations=newton_cfg.iterations,
                        ls_iterations=newton_cfg.ls_iterations,
                        use_mujoco_contacts=newton_cfg.use_mujoco_contacts,
                        ccd_iterations=newton_cfg.ccd_iterations,
                        disable_sensors=newton_cfg.disable_sensors,
                    ),
                    bodies=rigid_bodies,
                ),
                CouplerEntryCfg(
                    name="cloth",
                    solver_cfg=vbd,
                    # Particles, not bodies -- see the docstring.
                    all_particles=True,
                    include_static_shapes=True,
                    substeps=self.substeps,
                ),
            ],
            proxies=[
                CouplerProxyMappingCfg(
                    source="rigid",
                    destination="cloth",
                    bodies=proxy_bodies,
                    mass_scale=self.mass_scale,
                    mode=self.proxy_mode,
                    collide_interval=self.collide_interval,
                    # The proxy builds its OWN pipeline and defaults max_triangle_pairs to a
                    # GLOBAL 1e6; the outer collision_cfg never reaches it. On the cable that
                    # silently dropped contacts above ~16 envs for hours.
                    collision_pipeline=NewtonCollisionPipelineCfg(
                        rigid_contact_max=newton_cfg.rigid_contact_max,
                        max_triangle_pairs=budget,
                    ),
                )
            ],
            iterations=self.coupler_iterations,
            model_cfg=NewtonModelCfg(
                soft_contact_ke=self.soft_contact_ke,
                soft_contact_mu=self.soft_contact_mu,
            ),
        )

        return NewtonCfg(
            solver_cfg=solver,
            collision_cfg=NewtonCollisionPipelineCfg(
                rigid_contact_max=newton_cfg.rigid_contact_max,
                max_triangle_pairs=budget,
            ),
            default_shape_cfg=NewtonShapeCfg(),
            num_substeps=newton_cfg.num_substeps,
            collision_decimation=newton_cfg.collision_decimation,
            use_cuda_graph=False,
        )

    def triangle_pair_budget(self, newton_cfg, num_envs: int) -> int:
        """Global triangle-pair budget for this sheet at ``num_envs``.

        Scales with particle count as well as env count. The cable overflowed a correctly
        env-scaled 4.19M as soon as its segment count doubled, and a sheet has far more triangles
        than a cable -- so geometry has to be in this product, not just ``num_envs``.
        """
        base = newton_cfg.resolve_max_triangle_pairs(num_envs)
        per_env = self.num_particles * self.per_particle_triangle_pairs
        return int(max(base, num_envs * per_env) * 1.5)


@configclass
class ClothEnvCfg(PlayNewtonEnvCfg):
    """Play task with a folding cloth in place of the rigid tool."""

    cloth: ClothCfg = ClothCfg()
