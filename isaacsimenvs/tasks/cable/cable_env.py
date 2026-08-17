"""CableEnv — the Play goal-reaching task with a deformable cable as the manipuland.

``PlayEnv`` -> ``PlayNewtonEnv`` (Newton construction) -> ``CableEnv`` (object swapped).

Everything defining the task is inherited: the action pipeline, the 140-dim observation, the
keypoint reward, the goal sampler and the terminations all run unchanged. Two things change:

* the object is a ``CableObject`` rather than a procedural rigid tool, and
* the physics becomes a coupled solve -- MJWarp for the robot, VBD for the cable, bridged by a
  proxy mapping that exposes the fingertips to the cable's solver.

The task code reads a rigid manipuland's pose, which a cable does not have. Rather than fork that
code, :class:`CableAsRigidObject` synthesises one from the segment chain — which is what keeps the
observation byte-identical and the pretrained checkpoint loadable.
"""

from __future__ import annotations

import torch

from isaacsimenvs.newton import compat, patches
from isaacsimenvs.tasks.play_newton.play_newton_env import PlayNewtonEnv

from .cable_env_cfg import CableEnvCfg
from .utils.cable_adapter import CableAsRigidObject

__all__ = ["CableEnv", "CableEnvCfg"]


class CableEnv(PlayNewtonEnv):
    cfg: CableEnvCfg

    _cable_to_park = None

    def __init__(self, cfg: CableEnvCfg, render_mode: str | None = None, **kwargs) -> None:
        super().__init__(cfg, render_mode, **kwargs)
        if self._cable_to_park is not None:
            self._park_cable(self._cable_to_park)

    @staticmethod
    def build_physics_cfg(cfg: CableEnvCfg):
        """Replace the single MJWarp solve with a coupled MJWarp (robot) + VBD (cable) one.

        The robot keeps the solver settings measured on the rigid task, so a difference between
        the two envs is the object and the coupling rather than a retuned solver.
        """
        n = int(cfg.scene.num_envs)
        print(
            f"[cable] triangle-pair budget "
            f"{cfg.newton.resolve_max_triangle_pairs(n):,} for num_envs={n}",
            flush=True,
        )
        return cfg.cable.build_physics(cfg.newton, n)

    #: Run at the end of `setup_scene`, before the environments are replicated -- which is also
    #: when Newton imports the stage into its model. A cable created after that point exists as a
    #: `BasisCurves` prim on the stage and is invisible to the solver, surfacing as
    #: `KeyError: No articulations matching '.../Cable/geometry/mesh_articulation'`.
    pre_clone_scene_hook = staticmethod(lambda env: env._install_cable())

    def _install_cable(self) -> None:
        """Replace the rigid object with a cable, behind the adapter.

        ``PlayEnv`` holds the manipuland as ``env.object`` and every downstream module reads it
        from there, so swapping this one attribute is the whole substitution.
        """
        from isaaclab.assets import CableObject, CableObjectCfg
        from isaaclab.sim.spawners.materials import CableMaterialCfg
        from isaaclab.sim.spawners.shapes import CableCfg as CableSpawnCfg
        from isaaclab.sim.schemas import UsdPhysicsCollisionCfg
        from isaaclab.sim.spawners.materials import PreviewSurfaceCfg

        c = self.cfg.cable
        seg = c.segment_length
        cable_cfg = CableObjectCfg(
            # Concrete regex, not `{ENV_REGEX_NS}`: that placeholder is substituted by
            # `InteractiveSceneCfg`, and this asset is constructed imperatively in `_setup_scene`
            # the way the rest of the task builds its assets, so nothing would expand it.
            prim_path="/World/envs/env_.*/Cable",
            spawn=CableSpawnCfg(
                positions=[(seg * i - c.length / 2.0, 0.0, 0.0) for i in range(c.segments + 1)],
                # Everything else in this scene renders in Newton's default blue/white, which
                # makes a 10 mm cable on a blue table nearly invisible. Colour is cosmetic but
                # it is the difference between a legible render and a useless one.
                visual_material=PreviewSurfaceCfg(diffuse_color=tuple(c.color)),
                physics_material=CableMaterialCfg(
                    thickness=c.thickness,
                    density=c.density,
                    stretch_stiffness=c.stretch_modulus,
                    bend_stiffness=c.bend_modulus,
                ),
                collision_props=[UsdPhysicsCollisionCfg(collision_enabled=True)],
            ),
            init_state=CableObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, self._cable_spawn_height())
            ),
        )

        patches.install_cable_damping(
            linear_kd=c.linear_damping, angular_kd=c.angular_damping
        )

        self._neutralise_rigid_object()
        self._reshape_goal_marker()
        if c.supports:
            self._install_supports()

        if c.rigid_rod:
            self._install_rigid_rod()
            if not c.keep_cable_for_coupling:
                return
            # Spawn the cable anyway, parked clear of the workspace, purely so the coupled solver
            # has a deformable to own. It is never the manipuland, and `env.object` must keep
            # pointing at the rod -- see the guard below.
            park = c.park_offset
            cable_cfg.init_state.pos = (
                float(park[0]),
                float(park[1]),
                self._cable_spawn_height() + float(park[2]),
            )

        cable = CableObject(cable_cfg)
        self.scene.cable_objects["cable"] = cable
        self._cable = cable

        # The task reads `env.object`; hand it the adapter rather than the raw cable -- unless
        # the rigid-rod control owns the manipuland, in which case this cable is only present to
        # give the coupled solver's VBD entry something to own. Overwriting `env.object` here was
        # a real bug: it silently turned the "rigid rod in a coupled scene" control into "cable as
        # manipuland, with a spare rod interpenetrating it", which then drove the solve to NaN.
        if c.rigid_rod:
            # Park it after construction: a pose write needs the asset initialised, which does not
            # happen until `_setup_scene` returns.
            self._cable_to_park = cable
            return

        self.object = CableAsRigidObject(
            cable,
            num_envs=self.num_envs,
            length=c.length,
            thickness=c.thickness,
            density=c.density,
            device=self.device,
            pose_axis=c.pose_axis,
        )
        self._write_object_scale()

        print(
            f"[cable] {c.segments} segments, {c.length:.3f} m x {c.thickness * 1000:.0f} mm, "
            f"stretch {c.stretch_modulus:.3g}, bend {c.bend_modulus:.3g}",
            flush=True,
        )

    def _park_cable(self, cable) -> None:
        """Move the unused cable to its parked pose.

        The spawn position alone is not enough: the cable's placement comes from a pose *write*,
        so a cable that is never written stays wherever the solver first put it.
        """
        c = self.cfg.cable
        n_seg = cable.num_segments
        seg = c.length / n_seg
        origin = self.scene.env_origins
        park = torch.tensor(c.park_offset, device=self.device, dtype=torch.float32)
        centre = origin + park + torch.tensor(
            [0.0, 0.0, self._cable_spawn_height()], device=self.device, dtype=torch.float32
        )
        offsets = (torch.arange(n_seg, device=self.device, dtype=torch.float32) + 0.5) * seg
        offsets = offsets - 0.5 * c.length
        pose = torch.zeros((self.num_envs, n_seg, 7), device=self.device, dtype=torch.float32)
        pose[..., 0] = centre[:, 0:1] + offsets.view(1, -1)
        pose[..., 1] = centre[:, 1:2]
        pose[..., 2] = centre[:, 2:3]
        # Keep the spawn orientation: it is VBD's angular rest reference (newton#3847).
        default = cable.data.default_segment_pose_w
        default = (default.torch if hasattr(default, "torch") else default).view(self.num_envs, -1, 7)
        pose[..., 3:] = default[..., 3:]
        cable.write_segment_pose_to_sim_index(segment_pose=pose)
        print(f"[cable] parked the unused cable at env-local {[round(float(v),3) for v in park]}", flush=True)

    def _install_rigid_rod(self) -> None:
        """Spawn a rigid capsule in place of the cable — the zero-DOF control.

        Matched to the cable on every axis that is not deformability: same length and thickness,
        same total mass, same ``object_scale``, same coupled solver and proxy mapping. The task
        reads it through the ordinary Isaac Lab rigid-object surface, so no adapter is needed —
        ``RigidObject`` already provides the five reads and three writes.
        """
        import math

        from isaaclab.assets import RigidObject, RigidObjectCfg
        import isaaclab.sim as sim_utils

        c = self.cfg.cable
        mass = c.density * math.pi * c.radius**2 * c.length
        rod_cfg = RigidObjectCfg(
            # Own prim path, so the coupler's VBD entry never claims it -- see build_solver.
            prim_path="/World/envs/env_.*/Rod",
            spawn=sim_utils.CapsuleCfg(
                radius=c.radius,
                # A capsule's `height` is its cylindrical section; the caps add 2r on top, so
                # subtract them to keep the overall length equal to the cable's.
                height=max(c.length - 2.0 * c.radius, 1e-4),
                axis="X",
                collision_props=sim_utils.CollisionPropertiesCfg(),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(c.color)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, self._cable_spawn_height())),
        )
        rod = RigidObject(rod_cfg)
        self.scene.rigid_objects["rod"] = rod
        self._cable = rod
        # A RigidObject already exposes the surface the task reads, so it replaces `env.object`
        # directly rather than going through CableAsRigidObject.
        self.object = rod
        self._write_object_scale()
        print(
            f"[cable] RIGID ROD control: capsule {c.length:.3f} m x {c.thickness * 1000:.0f} mm, "
            f"mass {mass * 1000:.2f} g, zero internal DOF",
            flush=True,
        )

    def _write_object_scale(self) -> None:
        """Tell the task the manipuland's real dimensions.

        ``_object_scale_per_env`` is populated by ``setup_scene`` from the *procedural rigid tool*
        pool, and nothing updates it when the cable replaces that object. Left alone, the policy
        is handed a hammer's bounding box while holding a cable — and it is worse than a wrong
        observation, because ``reset_utils`` sizes the **keypoints** from the same tensor
        (``scale * object_base_size * keypoint_scale * 0.5``). The reward and the success test
        would both be measured against a box the shape of a tool that is not in the scene, and
        changing the cable's geometry would move no number the policy or the metric ever sees.

        The convention is a bounding box normalised by ``reward.object_base_size`` (0.04 m), i.e.
        metric dimensions x 25. A cable laid along its local +X is
        ``(length, thickness, thickness)``.

        Must run inside ``_setup_scene``: ``allocate_state_buffers`` reads this tensor afterwards
        to build ``_keypoint_offsets``.
        """
        c = self.cfg.cable
        base = float(self.cfg.reward.object_base_size)
        bbox = torch.tensor(
            [c.length, c.thickness, c.thickness], device=self.device, dtype=torch.float32
        )
        scale = bbox / base
        self._object_scale_per_env[:] = scale
        print(
            f"[cable] object_scale = {[round(float(v), 4) for v in scale]} "
            f"(bbox {[round(float(v), 4) for v in bbox]} m / object_base_size {base})",
            flush=True,
        )

    def _neutralise_rigid_object(self) -> None:
        """Make the inherited rigid tool inert, so the scene has exactly one manipuland.

        ``setup_scene`` still spawns the procedural handle+head tool -- the asset pipeline, the
        USD cache and the ``object_scales`` observation are all built around it, so removing the
        spawn outright is a much larger change than it looks. But once ``env.object`` points at
        the cable, nothing resets or reads that prim any more: it is simulated, unmanaged, and
        falls off the table within a second. A stray rigid body in the scene can collide with the
        cable and with the hand, which would quietly corrupt every measurement taken here.

        Found by rendering, not by reading numbers -- it is plainly visible as a hammer lying on
        the floor beside the table, and completely invisible in the observation.

        Disabling collision and pinning the body kinematic is done on the USD *before* Newton
        imports it, which is the only point at which either can still be changed.
        """
        from pxr import Usd, UsdPhysics

        from isaaclab.sim.utils import find_matching_prims

        # Walk the whole subtree, not just direct children. The colliders sit at
        # /Object/<mesh>/collisions -- two levels down -- so a one-level loop reported
        # "0 colliders off" while silently leaving the tool fully collidable. The kinematic
        # pin below does not survive the Newton import either, so disabling collision is the
        # only guarantee that this body cannot touch the cable or the hand.
        colliders = pinned = prims = 0
        for prim in find_matching_prims("/World/envs/env_.*/Object"):
            prims += 1
            for target in Usd.PrimRange(prim):
                if target.HasAPI(UsdPhysics.CollisionAPI):
                    api = UsdPhysics.CollisionAPI(target)
                    attr = api.GetCollisionEnabledAttr() or api.CreateCollisionEnabledAttr()
                    attr.Set(False)
                    colliders += 1
                if target.HasAPI(UsdPhysics.RigidBodyAPI):
                    body = UsdPhysics.RigidBodyAPI(target)
                    attr = body.GetKinematicEnabledAttr() or body.CreateKinematicEnabledAttr()
                    attr.Set(True)
                    pinned += 1
        print(
            f"[cable] neutralised the inherited rigid tool "
            f"({prims} prims, {colliders} colliders off, {pinned} bodies pinned)",
            flush=True,
        )
        if prims and not colliders:
            raise RuntimeError(
                "found the inherited rigid tool but disabled none of its colliders -- it would "
                "stay live in the scene and corrupt every measurement. Check the prim layout."
            )

    def _install_supports(self) -> None:
        """Two static rails under the cable's ends, leaving the middle span clear.

        Kinematic so they never move, and registered in *both* the MJWarp rigid entry and the
        proxy set (see ``CableCfg.supports``) -- a rail the VBD entry cannot feel is a rail the
        cable falls through.

        The cable spawns at ``support_height`` above its usual rest so it lands on the rails
        rather than inside them; ``_describe_cable`` after reset is the check that it actually did.
        """
        import isaaclab.sim as sim_utils
        from isaaclab.assets import RigidObject, RigidObjectCfg

        c = self.cfg.cable
        half = 0.5 * c.length - c.support_inset
        for i, x in enumerate((-half, half)):
            cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Support_{i}",
                # Flat-topped, not a capsule. A cable resting on two round rails is laterally
                # unstable -- it rolls off the convex tops at the first nudge, which showed up as
                # ~26/32 episodes still ending in `fall` even once clearance was solved. A flat
                # top cradles it instead.
                spawn=sim_utils.CuboidCfg(
                    size=(0.024, 0.09, c.support_height),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.55, 0.60)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    # Box centre sits half its height below the top face the cable rests on.
                    pos=(float(x), 0.0, self._cable_spawn_height() - c.radius
                         - 0.5 * c.support_height)
                ),
            )
            self.scene.rigid_objects[f"support_{i}"] = RigidObject(cfg)
        print(
            f"[cable] supports: 2 rails at x=+/-{half:.3f} m, "
            f"{c.support_height * 1e3:.0f} mm clearance under the middle span",
            flush=True,
        )

    def _reshape_goal_marker(self) -> None:
        """Give the goal marker the cable's shape instead of the tool's.

        ``goal_viz`` spawns from the same procedural handle+head USD as the manipuland, so the
        cable env renders a hammer-shaped ghost for a goal that a cable has to reach. The pose is
        right and the success keypoints come from ``object_scale`` rather than this mesh, so it is
        a cosmetic defect -- but a misleading picture is exactly how the wrong conclusion gets
        drawn from a video, which has already happened once here with the orphaned rigid tool.

        Swaps the tool geometry for a capsule matching ``cable.length`` x ``cable.thickness``,
        authored on the USD before Newton imports it. ``collisionEnabled`` stays false: the marker
        must never touch the manipuland, and ``render_newton._reveal_goal_viz`` draws it by setting
        the VISIBLE flag, which does not require the shape to collide.
        """
        from pxr import Usd, UsdGeom, UsdPhysics

        from isaaclab.sim.utils import find_matching_prims

        c = self.cfg.cable
        radius = 0.5 * float(c.thickness)
        # Capsule height is the cylindrical span, so the round caps make up the rest of the length.
        height = max(float(c.length) - 2.0 * radius, 1e-4)

        marked = 0
        for prim in find_matching_prims("/World/envs/env_.*/GoalViz"):
            stage = prim.GetStage()
            # The rigid body sits on a child (`.../GoalViz/object_root`), not on the GoalViz root.
            # Deactivating that child removes the body the physics view resolves, so the marker
            # must be rebuilt *under* it: keep the body, replace only its geometry.
            body = next(
                (d for d in Usd.PrimRange(prim) if d.HasAPI(UsdPhysics.RigidBodyAPI)), prim
            )
            for child in body.GetChildren():
                child.SetActive(False)
            capsule = UsdGeom.Capsule.Define(stage, body.GetPath().AppendChild("cable_marker"))
            # The cable lies along object-frame +X -- the same axis `object_scale` stretches.
            capsule.CreateAxisAttr("X")
            capsule.CreateRadiusAttr(radius)
            capsule.CreateHeightAttr(height)
            UsdPhysics.CollisionAPI.Apply(capsule.GetPrim()).CreateCollisionEnabledAttr(False)
            marked += 1

        print(
            f"[cable] goal marker reshaped on {marked} prims "
            f"(capsule {c.length:.3f} m x {c.thickness * 1e3:.0f} mm)",
            flush=True,
        )

    def _cable_spawn_height(self) -> float:
        reset_cfg = self.cfg.reset
        h = float(reset_cfg.table_reset_z) + float(self.cfg.cable_start_height)
        if self.cfg.cable.supports:
            # Sit on top of the rails rather than inside them.
            h += float(self.cfg.cable.support_height)
        return h


def _describe_cable(env: CableEnv) -> dict:
    """Diagnostics for a probe or a render caption; not used by the task."""
    pos = env.object._segment_positions()
    span = (pos[:, -1] - pos[:, 0]).norm(dim=-1)
    return {
        "segments": int(env._cable.num_segments),
        "span_m": [round(float(v), 4) for v in span[: min(4, span.shape[0])]],
        "rest_span_m": round(float(env.cfg.cable.length), 4),
        "centroid_z": [round(float(v), 4) for v in pos[:, :, 2].mean(dim=1)[:4]],
    }
