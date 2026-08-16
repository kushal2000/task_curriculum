"""Staged bring-up of the cable scene, to locate where the spawn-time impulse appears.

    scripts/newton_py -m isaacsimenvs.eval.cable_stages --stage 1
    scripts/newton_py -m isaacsimenvs.eval.cable_stages --stage 2 --substeps 2

Seven A/Bs inside the full coupled scene all returned identical results, which said the impulse
does not depend on the contact configuration. This builds the scene up instead, one element at a
time, so the first stage that misbehaves names the cause:

    1  cable alone, on a ground plane, plain VBD -- no robot, no coupler
    2  cable + the task's table
    3  cable + table through the COUPLER, still no robot -- splits "the coupler mechanism" from
       "the robot inside it"
    4  (the full env; use `Isaacsimenvs-Cable-Direct-v0` directly)

The reported number is the cable's speed one step after reset. In the full env that is ~2.1 m/s
mean / 4.5 m/s max from a standing start, and it scales with `--substeps`, which is what a force
applied per substep rather than per tick looks like.

Cable parameters default to the task's (`CableCfg`), so a difference between a stage and the env
is the *scene*, not the cable.
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Staged cable bring-up.")
    parser.add_argument("--stage", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--num_envs", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    # Defaults are Isaac Lab's own working cable demo (scripts/demos/cables.py:138-139), which
    # is the point of comparison. The task env inherits substeps=2 / iterations=10 from the rigid
    # Newton config instead.
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--dt", type=float, default=1.0 / 120.0)
    args, _ = parser.parse_known_args()
    sys.argv = [sys.argv[0]]

    import torch

    from isaaclab.assets import AssetBaseCfg, CableObjectCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import (
        GroundPlaneCfg,
        SimulationCfg,
        UsdPhysicsCollisionCfg,
        build_simulation_context,
    )
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.materials import CableMaterialCfg
    from isaaclab.sim.spawners.shapes import CableCfg as CableSpawnCfg
    from isaaclab.utils.configclass import configclass
    from isaaclab_contrib.deformable import VBDSolverCfg
    from isaaclab_newton.physics import NewtonCfg

    from isaacsimenvs.tasks.cable.cable_env_cfg import CableCfg

    c = CableCfg()
    seg = c.segment_length
    # Same layout the task spawns: `segments + 1` control points centred on the origin.
    positions = [(seg * i - c.length / 2.0, 0.0, 0.0) for i in range(c.segments + 1)]

    # Matches the task: table centre at `table_reset_z`, cable released above its top face.
    TABLE_SIZE = (0.475, 0.4, 0.3)
    TABLE_Z = 0.38
    TABLE_TOP_Z = TABLE_Z + TABLE_SIZE[2] / 2.0
    cable_z = TABLE_TOP_Z + 0.10 if args.stage >= 2 else 0.40

    @configclass
    class Stage(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/Ground", spawn=GroundPlaneCfg())
        cable = CableObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cable",
            spawn=CableSpawnCfg(
                positions=positions,
                physics_material=CableMaterialCfg(
                    thickness=c.thickness,
                    density=c.density,
                    stretch_stiffness=c.stretch_modulus,
                    bend_stiffness=c.bend_modulus,
                ),
                collision_props=[UsdPhysicsCollisionCfg(collision_enabled=True)],
            ),
            init_state=CableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, cable_z)),
        )

    scene_cfg_cls = Stage
    if args.stage >= 2:
        @configclass
        class StageWithTable(Stage):
            table = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Table",
                spawn=sim_utils.CuboidCfg(
                    size=TABLE_SIZE, collision_props=sim_utils.CollisionPropertiesCfg()
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, TABLE_Z)),
            )

        scene_cfg_cls = StageWithTable

    if args.stage >= 3:
        # The rigid entry needs an actual rigid body to own, so the table becomes kinematic --
        # which is also what the task spawns. Proxy it into the cable entry so the cable can
        # still rest on it.
        from isaaclab.assets import RigidObjectCfg
        from isaaclab_contrib.coupling import (
            CouplerEntryCfg,
            CouplerProxyCfg,
            CouplerProxyMappingCfg,
        )
        from isaaclab_contrib.deformable import NewtonModelCfg
        from isaaclab_newton.physics import MJWarpSolverCfg

        cable_cfg_obj = CableCfg()

        @configclass
        class StageCoupled(Stage):
            table = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Table",
                spawn=sim_utils.CuboidCfg(
                    size=TABLE_SIZE,
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, TABLE_Z)),
            )

        scene_cfg_cls = StageCoupled
        solver_cfg = CouplerProxyCfg(
            entries=[
                CouplerEntryCfg(
                    name="rigid",
                    solver_cfg=MJWarpSolverCfg(njmax=8192, nconmax=2048, cone="elliptic"),
                    bodies=[r"/World/envs/env_.*/Table"],
                ),
                CouplerEntryCfg(
                    name="cable",
                    solver_cfg=VBDSolverCfg(iterations=args.iterations),
                    bodies=[r"/World/envs/env_.*/Cable"],
                    include_static_shapes=True,
                    substeps=cable_cfg_obj.cable_substeps,
                ),
            ],
            proxies=[
                CouplerProxyMappingCfg(
                    source="rigid",
                    destination="cable",
                    bodies=[r"/World/envs/env_.*/Table"],
                    mass_scale=cable_cfg_obj.mass_scale,
                    collide_interval=cable_cfg_obj.collide_interval,
                )
            ],
            iterations=1,
            model_cfg=NewtonModelCfg(
                soft_contact_ke=cable_cfg_obj.soft_contact_ke,
                soft_contact_mu=cable_cfg_obj.soft_contact_mu,
            ),
        )
    else:
        solver_cfg = VBDSolverCfg(iterations=args.iterations)

    sim_cfg = SimulationCfg(
        dt=args.dt,
        device=args.device,
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(
            solver_cfg=solver_cfg,
            num_substeps=args.substeps,
            use_cuda_graph=False,
        ),
    )

    print(
        f"[stage{args.stage}] cable {c.segments} seg x {c.length} m, thickness {c.thickness}, "
        f"stretch {c.stretch_modulus:.3g}, bend {c.bend_modulus:.3g}",
        flush=True,
    )
    print(
        f"[stage{args.stage}] dt={args.dt:.6f} substeps={args.substeps} "
        f"iterations={args.iterations}  solver dt={args.dt / args.substeps * 1e3:.3f} ms",
        flush=True,
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        scene = InteractiveScene(scene_cfg_cls(num_envs=args.num_envs, env_spacing=2.0))
        sim.reset()
        scene.update(0.0)
        cable = scene["cable"]

        pose0 = cable.data.segment_pose_w.torch.clone()
        print(
            f"[stage{args.stage}] AT REST: z={float(pose0[0, :, 2].mean()):.4f} "
            f"span={float((pose0[0, -1, :3] - pose0[0, 0, :3]).norm()):.4f}",
            flush=True,
        )

        for i in range(args.steps):
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(args.dt)
            if i < 3 or i == args.steps - 1:
                vel = cable.data.segment_velocity_w.torch[0, :, :3]
                pos = cable.data.segment_pose_w.torch[0, :, :3]
                print(
                    f"[stage{args.stage}] step {i + 1:4d}  max|v|={float(vel.norm(dim=-1).max()):8.3f}  "
                    f"mean v=({float(vel[:, 0].mean()):6.2f},{float(vel[:, 1].mean()):6.2f},"
                    f"{float(vel[:, 2].mean()):6.2f})  z={float(pos[:, 2].mean()):.4f}  "
                    f"span={float((pos[-1] - pos[0]).norm()):.4f}",
                    flush=True,
                )

        final = cable.data.segment_pose_w.torch
        v1 = float(cable.data.segment_velocity_w.torch[0, :, :3].norm(dim=-1).max())
        print(
            f"[stage{args.stage}] RESULT finite={bool(torch.isfinite(final).all())} "
            f"z={float(final[0, :, 2].mean()):.4f} "
            f"span={float((final[0, -1, :3] - final[0, 0, :3]).norm()):.4f} "
            f"(rest {c.length * (c.segments - 1) / c.segments:.4f})  final max|v|={v1:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
