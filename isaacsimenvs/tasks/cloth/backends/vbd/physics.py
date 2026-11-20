"""The Newton physics of the VBD task: the coupled MJWarp (robot) + VBD (cloth) solve, built from
``cfg.cloth`` (``config/schema/vbd_cloth.VbdClothCfg``) and ``cfg.newton`` (``NewtonSolverCfg``).

Newton-only: the dependencies are imported when ``build_physics`` is called.
"""

from __future__ import annotations

__all__ = ["build_physics", "resolve_max_triangle_pairs", "triangle_pair_budget"]


def resolve_max_triangle_pairs(newton_cfg, num_envs: int) -> int:
    """Triangle-pair budget for ``num_envs`` (``NewtonSolverCfg.max_triangle_pairs``, or sized from
    ``per_env_triangle_pairs``)."""
    if newton_cfg.max_triangle_pairs is not None:
        return int(newton_cfg.max_triangle_pairs)
    return max(1_000_000, int(num_envs) * newton_cfg.per_env_triangle_pairs)


def build_physics(cloth, newton_cfg, num_envs: int, robot_spec=None):
    """``NewtonCfg`` for the coupled MJWarp (robot) + VBD (cloth) scene.

    The one structural difference from the cable: the VBD entry owns **particles**, not bodies.
    A cloth is a particle system, so ``all_particles=True`` is what assigns the sheet to the
    deformable solver -- a body selector would match nothing and the cloth would be integrated
    by nobody, presenting as a sheet frozen in mid-air.

    ``robot_spec`` (a ``robots.RobotSpec``, ``robots.FLEXIV_WUJI_RIGHT`` when omitted) names the hand
    bodies the cloth couples to.
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

    from isaacsimenvs.robots import FLEXIV_WUJI_RIGHT

    robot_spec = robot_spec or FLEXIV_WUJI_RIGHT
    rigid_bodies = [r"/World/envs/env_.*/Robot"]
    robot = r"/World/envs/env_.*/Robot/"
    segments = ("distal",) if cloth.proxy_links == "tips" else ("distal", "middle", "proximal")
    proxy_bodies = [robot + robot_spec.finger_links_regex(*segments)]
    if cloth.proxy_links == "hand":
        palm = (robot_spec.palm_body_name, *robot_spec.palm_side_link_names)
        proxy_bodies.append(robot + "(" + "|".join(palm) + ")")
    if cloth.proxy_table:
        rigid_bodies.append(r"/World/envs/env_.*/Table")
        proxy_bodies.append(r"/World/envs/env_.*/Table")

    budget = triangle_pair_budget(cloth, newton_cfg, num_envs)

    # No VBDSolverCfg subclass here, deliberately. `class_type` carries
    # "{DIR}.vbd_manager:NewtonVBDManager" and `{DIR}` is substituted against the *defining*
    # module, so subclassing inside this package makes it hunt for
    # `isaacsimenvs.tasks.cloth.vbd_manager`. The cable subclasses it only to add
    # `rigid_body_contact_buffer_size`, which is a body-BODY buffer; a cloth is particles, so
    # the buffer that matters (`rigid_body_particle_contact_buffer_size`) is already a field.
    vbd = VBDSolverCfg(
        iterations=cloth.vbd_iterations,
        rigid_body_particle_contact_buffer_size=cloth.rigid_body_particle_contact_buffer_size,
        rigid_contact_k_start=cloth.rigid_contact_k_start,
        # Self-contact. Without these the folded flap passes through the half it lands on --
        # `particle_enable_self_contact` defaults to False and this call never set it.
        particle_enable_self_contact=cloth.enable_self_contact,
        particle_self_contact_radius=cloth.self_contact_radius,
        particle_self_contact_margin=cloth.self_contact_margin,
        particle_collision_detection_interval=cloth.self_contact_detection_interval,
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
    vbd.rigid_body_contact_buffer_size = cloth.rigid_body_contact_buffer_size

    # Same pass-through for two more `SolverVBD` arguments that `VBDSolverCfg` does not
    # declare. `_filter_solver_kwargs` filters `solver_cfg.to_dict()` against the solver
    # signature, and isaaclab's `class_to_dict` walks `obj.__dict__`, so instance attributes
    # set here do reach the solver.
    #
    #   * `particle_conservative_bound_relaxation`: Newton 1.5 defaults 0.85; SIM1 ran 0.42.
    #   * `rigid_avbd_beta`: ships 0.0, which DISABLES AVBD contact ramping and makes
    #     `rigid_contact_k_start` above a no-op. See the field docstring -- this is the one
    #     setting here that could plausibly weaken the grip rather than strengthen it.
    vbd.particle_conservative_bound_relaxation = cloth.conservative_bound_relaxation
    vbd.rigid_avbd_beta = cloth.rigid_avbd_beta

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
                substeps=cloth.substeps,
            ),
        ],
        proxies=[
            CouplerProxyMappingCfg(
                source="rigid",
                destination="cloth",
                bodies=proxy_bodies,
                mass_scale=cloth.mass_scale,
                mode=cloth.proxy_mode,
                collide_interval=cloth.collide_interval,
                # The proxy builds its OWN pipeline and defaults max_triangle_pairs to a
                # GLOBAL 1e6; the outer collision_cfg never reaches it. On the cable that
                # silently dropped contacts above ~16 envs for hours.
                collision_pipeline=NewtonCollisionPipelineCfg(
                    rigid_contact_max=newton_cfg.rigid_contact_max,
                    max_triangle_pairs=budget,
                ),
            )
        ],
        iterations=cloth.coupler_iterations,
        model_cfg=NewtonModelCfg(
            soft_contact_ke=cloth.soft_contact_ke,
            # `soft_contact_kd` was previously left at the `NewtonModelCfg` default (1.0e-2)
            # because this call only passed `ke` and `mu`.
            soft_contact_kd=cloth.soft_contact_kd,
            soft_contact_mu=cloth.soft_contact_mu,
        ),
    )

    # `CouplerProxyMappingCfg` does not declare the relaxation fields, but the coupler builds
    # `SolverCoupledProxy.Proxy(**vars(cfg))`, so instance attributes reach Newton -- the same
    # pass-through as the VBD arguments above. Set them on `solver.proxies`, NOT on the object
    # passed in: `CouplerProxyCfg` copies its list on construction, so an attribute set on the
    # original never reaches the solver (that is how the first wiring of this read 1.0).
    for proxy_mapping in solver.proxies:
        proxy_mapping.proxy_relaxation = float(cloth.proxy_relaxation)
        proxy_mapping.proxy_relaxation_mode = str(cloth.proxy_relaxation_mode)
        proxy_mapping.proxy_relaxation_min = min(float(cloth.proxy_relaxation_min), float(cloth.proxy_relaxation))

    return NewtonCfg(
        solver_cfg=solver,
        collision_cfg=NewtonCollisionPipelineCfg(
            rigid_contact_max=newton_cfg.rigid_contact_max,
            max_triangle_pairs=budget,
        ),
        default_shape_cfg=NewtonShapeCfg(),
        num_substeps=newton_cfg.num_substeps,
        collision_decimation=newton_cfg.collision_decimation,
        use_cuda_graph=bool(cloth.use_cuda_graph),
    )

def triangle_pair_budget(cloth, newton_cfg, num_envs: int) -> int:
    """Global triangle-pair budget for this sheet at ``num_envs``.

    Scales with particle count as well as env count. The cable overflowed a correctly
    env-scaled 4.19M as soon as its segment count doubled, and a sheet has far more triangles
    than a cable -- so geometry has to be in this product, not just ``num_envs``.
    """
    base = resolve_max_triangle_pairs(newton_cfg, num_envs)
    per_env = cloth.num_particles() * cloth.per_particle_triangle_pairs
    return int(max(base, num_envs * per_env) * 1.5)
