"""CableEnvCfg — goal reaching with a deformable cable instead of a rigid tool.

Subclasses :class:`PlayNewtonEnvCfg`. The task is unchanged: pick the object up and carry it to a
commanded pose, scored by keypoint distance. Only the object changes, from a procedural
handle+head tool to a VBD cable — and with it the physics, which becomes a coupled solve.
"""

from __future__ import annotations

import math

# IMPORT ORDER IS LOAD-BEARING, for the same reason as `play_newton_env_cfg`: importing
# `isaaclab_physx` binds `isaaclab.utils.configclass` to the same-named *submodule*, shadowing the
# decorator function. Importing PlayNewtonEnvCfg first runs the compat shim that re-asserts it;
# pulling in `configclass` before that yields a module and every `@configclass` below raises
# "'module' object is not callable".
from isaacsimenvs.tasks.play_newton.play_newton_env_cfg import PlayNewtonEnvCfg  # isort: skip

from isaaclab.utils import configclass  # noqa: E402

__all__ = ["CableCfg", "CableEnvCfg"]


@configclass
class CableCfg:
    """Cable geometry, material and the coupling that lets the hand feel it."""

    length: float = 0.30
    segments: int = 12
    thickness: float = 0.010
    density: float = 100.0

    rigid_rod: bool = False
    """Spawn a single rigid capsule instead of a cable, same dimensions and mass.

    The control for "is it deformability, or is it the shape?". A cable cannot be made rigid by
    reducing its segment count -- ``CableCfg`` rejects fewer than three control points
    (``segments=1`` raises ``CableCfg requires at least three positions, got 2``), and the minimum
    of 2 segments still has a bend joint at the exact midpoint, which is where a descending hand
    folds it. A rigid capsule has no internal degrees of freedom at all.

    Everything else is held fixed: same length and thickness, same mass, same ``object_scale``, and
    the same coupled solver and proxy, so deformability is the only variable."""

    keep_cable_for_coupling: bool = False
    """With ``rigid_rod``, also spawn the cable (parked clear of the workspace) so the coupled
    solver is retained.

    This is what separates deformability from the coupling. ``rigid_rod`` alone has to fall back
    to a plain MJWarp solve -- a scene with no deformable gives the VBD entry nothing to own -- so
    it varies deformability *and* solver together. Keeping an unused cable in the scene lets the
    rod be manipulated inside the *same* coupled solve the cables run under, leaving deformability
    of the manipuland as the only difference."""

    park_offset: tuple[float, float, float] = (0.0, -0.45, -0.515)
    """Where the unused cable sits, relative to the object start pose.

    Must land it on open ground, clear of everything. An earlier value put it at z = 0.33, which
    is *inside* the table's vertical span (0.23-0.53) -- the resulting interpenetration drove the
    solve to NaN, which surfaces as the policy dying with
    ``RuntimeError: normal expects all elements of std >= 0.0`` rather than as anything mentioning
    physics.

    It must also stay inside the environment's own cell. ``env_spacing`` is 1.2, so an offset of
    y = +1.0 lands the cable in the *neighbouring* environment, where that robot strikes it as
    soon as the policy starts moving -- invisible under zero actions, fatal under a policy."""

    color: tuple[float, float, float] = (0.95, 0.45, 0.10)
    """Cable render colour. Cosmetic, but the scene is otherwise uniformly blue/white and a
    10 mm cable on a blue table is invisible in a capture."""

    stretch_stiffness_scale: float = 5.0e5
    bend_stiffness_scale: float = 20.0
    """Moduli are derived from these per Isaac Lab's own ``scripts/demos/cables.py``: stretch as
    ``scale * seg_len / (pi r^2)`` and bend as ``scale * seg_len / (0.25 pi r^4)``, so the
    authored numbers stay physically meaningful as segment count or thickness change."""

    linear_damping: float = 0.0
    """Cable stretch/shear damping [N.s/m]. `CableMaterialCfg` exposes no damping, so this is
    applied to the model before finalize; unset, all four damping terms default to 0.0 and the
    cable runs completely undamped."""

    angular_damping: float = 0.0
    """Cable bend/twist damping [N.m.s/rad]. The lever that matters for few-segment cables, where
    the whole bend response concentrates in one hinge. Not free: newton#2557 reports high VBD
    cable damping visibly changing a catenary's shape, i.e. costing apparent stiffness."""

    proxy_mode: str = "lagged"
    """Proxy transfer mode: ``lagged`` or ``staggered``. Lagged uses one-step-old state across the
    coupling, a standard instability source."""

    coupler_iterations: int = 1
    """Outer coupled-solver iterations. More tightens convergence between the two entries."""

    # --- coupling -------------------------------------------------------------------------
    vbd_iterations: int = 10
    rigid_contact_k_start: float = 1.0e2
    """AVBD contact-stiffness ramp start for body-body contacts inside the VBD entry.

    Distinct from ``soft_contact_ke``, which is the *proxy* contact. The cable is a chain of rigid
    bodies (``mesh_edge_body_N``), so hand-vs-cable is a body-body contact and this is the stiffness
    that governs it. The VBD solver uses augmented-Lagrangian hard constraints for those by default
    (``rigid_contact_hard=True``, already on), and this sets where the penalty ramp begins.

    Default matches ``VBDSolverCfg``, so setting it is opt-in and changes nothing otherwise."""

    rigid_body_particle_contact_buffer_size: int = 1024
    rigid_body_contact_buffer_size: int = 4096
    """Body-BODY contact buffer. A Newton cable is bodies and joints, not particles, so this is
    the list that overflows here -- and ``VBDSolverCfg`` does not expose it (SolverVBD defaults it
    to 64). Declared on the subclass below, which is enough: the solver forwards any cfg field
    whose name matches one of its kwargs."""

    cable_substeps: int = 4
    """Substeps run by the *cable entry alone* inside one coupled step.

    This is the fix for the defect that made the coupled scene unusable: the VBD entry was
    integrating the coupled contact at the coupled step rate, and a 0.2 g segment cannot absorb
    that impulse in one step -- the cable was ejected on first fingertip contact and the robot
    intermittently went NaN. Both failures are the same impulse, resolved on whichever side is
    lighter."""

    mass_scale: float = 1.0
    """Proxy virtual-inertia scale. **Keep at 1.0.**

    ``0.05`` was recommended in the reference investigation and is ~500x worse -- 16005 N peak
    coupling force against 28 N, with the cable crushed from 0.275 m to 0.022 m -- and is itself
    the cause of the robot NaN. It only appeared to help because it moved the failure from the
    cable to the robot, where policy-driven runs died before the cable damage became visible."""

    collide_interval: int = 1
    """How often the proxy pipeline re-collides, in substeps. Suspect for the spawn-time impulse:
    it scales linearly with `num_substeps`, which is what a force applied per substep rather than
    per tick looks like."""

    soft_contact_ke: float = 8.0e3
    soft_contact_mu: float = 10.0

    supports: bool = False
    """Rest the cable on two low rails instead of flat on the table.

    The last untested hypothesis for why no cable is graspable. Every knob that governs how the
    cable *behaves* -- bend and stretch stiffness, density, contact stiffness and friction, solver
    iterations and substeps -- has been swept and none produces a single goal, while a rigid rod of
    the same size and mass scores ~9 on the identical scene. What none of them change is the
    *geometry of the grasp*: a cable lying flush on a table presents no gap to close fingers
    around, at any diameter or stiffness, so the hand can only press down on it.

    The rails sit under the cable's ends and leave the middle span clear, so there is somewhere to
    put the fingers. They are proxied like the table -- an unproxied support would let the cable
    sink straight through, and the run would read as "clearance did not help"."""

    support_height: float = 0.04
    """Gap under the middle span [m]. Wants to exceed a fingertip radius to be worth anything."""

    support_inset: float = 0.06
    """How far in from each cable end a rail sits [m]. Keeps the graspable middle clear."""

    proxy_links: str = "hand"
    """Which hand links the cable can feel: ``tips`` | ``fingers`` | ``hand``.

    The coupling is proxy-based, so the cable is touched *only* by the links named here --
    everything else on the robot does not exist as far as the VBD entry is concerned. This was
    ``tips`` (the five distal phalanges) for every measurement before 2026-08-16, which meant a
    cable could only ever be pinched fingertip-to-fingertip, while the rigid-rod control collided
    with the whole hand through MJWarp. An enclosing grasp was geometrically impossible, so that
    asymmetry sat underneath every cable result as a *precondition*, not a variable -- and a
    fingertip-only comparison says more about the coupling than about the cable.

    Hence the default is ``hand``: full phalanges (``_DP``/``_MP``/``_PP``), the palm and the
    thumb/pinky metacarpals. Keep it there for anything meant to characterise the cable.
    ``tips``/``fingers`` remain only for reproducing pre-fix runs.

    Measured at 32 envs with no contact-buffer overflow despite ~3x the contact pairs."""

    proxy_table: bool = True
    """Route the table through the proxy as a kinematic rigid body.

    Not cosmetic. A static shape belongs to exactly one coupler entry, and the cable entry claims
    them (``include_static_shapes``) so the cable can rest on the table. The side effect is that
    the *robot* then stops colliding with the table. Proxying it restores that, which a
    tabletop task needs."""

    @property
    def segment_length(self) -> float:
        return self.length / self.segments

    @property
    def radius(self) -> float:
        return 0.5 * self.thickness

    @property
    def stretch_modulus(self) -> float:
        return self.stretch_stiffness_scale * self.segment_length / (math.pi * self.radius**2)

    @property
    def bend_modulus(self) -> float:
        return self.bend_stiffness_scale * self.segment_length / (0.25 * math.pi * self.radius**4)

    def build_physics(self, newton_cfg, num_envs: int):
        """Return the ``NewtonCfg`` for the coupled scene.

        The coupler is a *solver* config and goes in ``solver_cfg``; it is not itself the physics
        config. Passing it straight to ``sim.physics`` fails later and obscurely, inside
        ``NewtonManager.initialize_solver`` reading ``cfg.num_substeps``.
        """
        from isaaclab_newton.physics import (
            MJWarpSolverCfg,
            NewtonCfg,
            NewtonCollisionPipelineCfg,
            NewtonShapeCfg,
        )

        # The rigid-rod control has no deformable in the scene, so there is nothing for the VBD
        # entry to own. Leaving the rod at the Cable prim path with the coupled solver assigns it
        # to the VBD entry, which does not integrate rigid bodies -- the rod then sits frozen and
        # the observation reports a manipuland that never moves, which reads as "the policy will
        # not pick it up".
        #
        # CONFOUND, stated rather than hidden: this control therefore runs a plain MJWarp solve
        # while the cable runs the coupled MJWarp+VBD one. It isolates deformability *and* the
        # coupling together, not deformability alone.
        solver = (
            MJWarpSolverCfg(
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
            )
            if self.rigid_rod and not self.keep_cable_for_coupling
            else self.build_solver(newton_cfg, num_envs)
        )

        return NewtonCfg(
            solver_cfg=solver,
            collision_cfg=NewtonCollisionPipelineCfg(
                rigid_contact_max=newton_cfg.rigid_contact_max,
                max_triangle_pairs=self.triangle_pair_budget(newton_cfg, num_envs),
            ),
            default_shape_cfg=NewtonShapeCfg(),
            num_substeps=newton_cfg.num_substeps,
            collision_decimation=newton_cfg.collision_decimation,
            # CUDA graph capture is off for the coupled solve: the reference ran the coupled scene
            # this way, and a graph would freeze a solver whose two entries step at different
            # rates (the cable entry runs `cable_substeps` per coupled step).
            use_cuda_graph=bool(self.rigid_rod and not self.keep_cable_for_coupling),
            deterministic_mode=newton_cfg.deterministic_mode,
        )

    def triangle_pair_budget(self, newton_cfg, num_envs: int) -> int:
        """Triangle-pair budget for this cable at ``num_envs``.

        The budget is GLOBAL and must scale with *scene geometry* as well as env count. A
        24-segment cable overflowed a correctly env-scaled 4,194,304 within a few hundred steps of
        manipulation -- twice the segments is roughly twice the collision geometry -- so the env
        count alone is not enough. The 1.5x headroom covers the transient peak when the hand closes
        around the cable, which is when the count spikes.
        """
        base = newton_cfg.resolve_max_triangle_pairs(num_envs)
        return int(base * max(1.0, self.segments / 12.0) * 1.5)

    def build_solver(self, newton_cfg, num_envs: int = 1):
        """Return a ``CouplerProxyCfg`` pairing MJWarp (robot) with VBD (cable).

        Replaces the plain ``MJWarpSolverCfg`` the Newton task uses. The robot keeps the solver
        settings measured on the rigid task, so a difference between the two is the object and the
        coupling, not a retuned solver.
        """
        from isaaclab_contrib.coupling import (
            CouplerEntryCfg,
            CouplerProxyCfg,
            CouplerProxyMappingCfg,
        )
        from isaaclab_contrib.deformable import NewtonModelCfg, VBDSolverCfg
        from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCollisionPipelineCfg

        @configclass
        class _VBDWithBodyBuffer(VBDSolverCfg):
            rigid_body_contact_buffer_size: int = self.rigid_body_contact_buffer_size
            rigid_contact_k_start: float = self.rigid_contact_k_start

        rigid_bodies = [r"/World/envs/env_.*/Robot"]
        if self.rigid_rod:
            # The rod is a rigid body and belongs to the MJWarp entry. At the `Cable` prim path it
            # would be claimed by the VBD entry, which does not integrate rigid bodies -- the rod
            # then never moves and the observation reports a manipuland frozen at its spawn pose.
            rigid_bodies.append(r"/World/envs/env_.*/Rod")
        finger_links = {
            "tips": "DP",
            "fingers": "DP|MP|PP",
            "hand": "DP|MP|PP",
        }
        if self.proxy_links not in finger_links:
            raise ValueError(
                f"proxy_links must be one of {sorted(finger_links)}, got {self.proxy_links!r}"
            )
        proxy_bodies = [
            r"/World/envs/env_.*/Robot/left_(thumb|index|middle|ring|pinky)_"
            rf"({finger_links[self.proxy_links]})",
        ]
        if self.proxy_links == "hand":
            # The palm and the two metacarpals that carry it -- an object resting against the palm
            # is held by these, not by any phalanx. The palm body is `iiwa14_link_7`
            # (`scene_utils.PALM_BODY_NAME`): the URDF's `left_hand_C_MC` link is merged away by
            # the importer and matches *no* body, so naming it here proxies nothing.
            proxy_bodies.append(
                r"/World/envs/env_.*/Robot/(iiwa14_link_7|left_thumb_MC|left_pinky_MC)"
            )
        if self.proxy_table:
            rigid_bodies.append(r"/World/envs/env_.*/Table")
            proxy_bodies.append(r"/World/envs/env_.*/Table")
        if self.supports:
            # Both lists: the rails must exist to MJWarp *and* be felt by the VBD entry. Geometry
            # alone is not contact in this coupling -- see `proxy_links`.
            rigid_bodies.append(r"/World/envs/env_.*/Support_.*")
            proxy_bodies.append(r"/World/envs/env_.*/Support_.*")

        return CouplerProxyCfg(
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
                    name="cable",
                    solver_cfg=_VBDWithBodyBuffer(
                        iterations=self.vbd_iterations,
                        rigid_body_particle_contact_buffer_size=(
                            self.rigid_body_particle_contact_buffer_size
                        ),
                    ),
                    # `bodies`, not `all_particles`: a Newton cable is rigid bodies plus joints.
                    bodies=[r"/World/envs/env_.*/Cable"],
                    include_static_shapes=True,
                    substeps=self.cable_substeps,
                ),
            ],
            proxies=[
                CouplerProxyMappingCfg(
                    source="rigid",
                    destination="cable",
                    bodies=proxy_bodies,
                    mass_scale=self.mass_scale,
                    mode=self.proxy_mode,
                    collide_interval=self.collide_interval,
                    # The proxy builds its OWN collision pipeline, and the field defaults to a
                    # bare `NewtonCollisionPipelineCfg()` -- i.e. `max_triangle_pairs = 1_000_000`,
                    # which is a GLOBAL budget, not per-env. The outer `NewtonCfg.collision_cfg`
                    # is sized correctly and never reaches here, so above ~16 envs the proxy
                    # silently drops contacts: at 64 envs the solver reported
                    # "Triangle pair buffer overflowed 1024336 > 1000000" while the configured
                    # budget was 4,194,304. Dropped contacts are exactly the hand failing to feel
                    # the cable, and this is the defect the sibling project found decisive
                    # (0.12 -> 14.28 goals/episode).
                    collision_pipeline=NewtonCollisionPipelineCfg(
                        rigid_contact_max=newton_cfg.rigid_contact_max,
                        max_triangle_pairs=self.triangle_pair_budget(newton_cfg, num_envs),
                    ),
                )
            ],
            iterations=self.coupler_iterations,
            model_cfg=NewtonModelCfg(
                soft_contact_ke=self.soft_contact_ke,
                soft_contact_mu=self.soft_contact_mu,
            ),
        )


@configclass
class CableEnvCfg(PlayNewtonEnvCfg):
    """Play goal-reaching, with a cable as the manipuland."""

    cable: CableCfg = CableCfg()

    cable_start_height: float = 0.15
    """Height above the table top at which the cable is laid out on reset."""
