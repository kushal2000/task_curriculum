"""The PhysX scene (Isaac Sim 5.1 / Isaac Lab 2.3): robot, table, ground, light, and the rigid cloths.

Robot and table go URDF -> USD (Isaac Lab's converter) -> a baked copy with the PhysX properties
authored (Play's recipe: contact offset 2 mm / rest 0 on every collider; the robot gravity-free,
8/0 solver iterations, self-collision on with the spec's filtered pairs). The cloths are authored
straight into the stage: a template in a private in-memory stage, ``Sdf.CopySpec``-ed into every env
inside one ``Sdf.ChangeBlock``, each env given its own dimensions. Each design has its own
batched articulation; the backend combines the groups into a common surface readout.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import torch

from ..assets import (
    apply_self_collision_filters,
    bake_usd,
    convert_urdf,
    position_drive,
    robot_cfg,
    urdf_box_size,
)
from . import designs


FRICTION_COMBINE_MODE = "multiply"


def pair_materials(table_s, hand_s, ratio):
    """Material coefficients giving exact cloth/table and cloth/hand friction.

    Both sides use multiply combine. Cloth static/dynamic coefficients are one,
    so table and robot coefficients equal the requested effective contact values.
    Non-neighbour cloth/cloth contacts remain enabled with effective friction one;
    their interlayer friction is not calibrated by the table/hand parameters.
    Scalars and per-environment tensors are both supported.
    """
    return {"cloth": (1.0, 1.0, 0.0), "table": (table_s, table_s * ratio, 0.0),
            "robot": (hand_s, hand_s * ratio, 0.0)}


def bind_physics_material(stage, root_path, coefficients):
    """Author multiply-combine material and bind every collider beneath this root."""
    from pxr import Sdf, Usd, UsdPhysics, UsdShade

    root = stage.GetPrimAtPath(root_path)
    material = UsdShade.Material.Define(stage, root.GetPath().AppendChild("ClothContactMaterial"))
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(float(coefficients[0]))
    api.CreateDynamicFrictionAttr(float(coefficients[1]))
    api.CreateRestitutionAttr(float(coefficients[2]))
    _apply_physx_api(material.GetPrim(), "PhysxMaterialAPI", {
        "physxMaterial:frictionCombineMode": (Sdf.ValueTypeNames.Token, FRICTION_COMBINE_MODE),
        "physxMaterial:restitutionCombineMode": (Sdf.ValueTypeNames.Token, "multiply"),
    })
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                material, bindingStrength=UsdShade.Tokens.strongerThanDescendants, materialPurpose="physics"
            )
    return material


def _bake_contact_material(path, coefficients):
    from pxr import Usd

    stage = Usd.Stage.Open(path)
    root = stage.GetDefaultPrim() or next(iter(stage.GetPseudoRoot().GetChildren()))
    bind_physics_material(stage, root.GetPath(), coefficients)
    stage.GetRootLayer().Save()


def _log(t0: float, msg: str) -> None:
    print(f"[scene] {time.perf_counter() - t0:7.1f}s  {msg}", flush=True)



def build_robot_and_table(env, spec) -> None:
    """Spawn robot, table, ground and light into every env; sets ``env.robot``, ``env.table``,
    ``env._table_half_height`` and registers both with the scene."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
    from isaaclab.sim.spawners.from_files import (
        GroundPlaneCfg,
        UsdFileCfg,
        spawn_ground_plane,
    )
    from isaaclab.sim.utils import get_current_stage

    t0 = time.perf_counter()
    a = env.cfg.assets
    tmp = Path(tempfile.mkdtemp(prefix="cloth_assets_"))
    env._tmp_asset_dir = str(tmp)
    work, bake = tmp / "usd", tmp / "baked_usd"
    work.mkdir(parents=True)

    self_collision = bool(env.cfg.robot_self_collision)
    robot_raw = convert_urdf(a.robot_urdf, work, fix_base=True, self_collision=self_collision, joint_drive=position_drive())
    if self_collision:
        apply_self_collision_filters(robot_raw, spec.self_collision_filter)
    robot_usd = bake_usd(
        robot_raw, bake, "robot",
        rigid_body={"physxRigidBody:disableGravity": True, "physxRigidBody:maxDepenetrationVelocity": 1000.0},
        articulation={"physxArticulation:enabledSelfCollisions": self_collision,
                      "physxArticulation:solverPositionIterationCount": 8,
                      "physxArticulation:solverVelocityIterationCount": 0},
    )
    table_usd = bake_usd(
        convert_urdf(a.table_urdf, work, fix_base=False), bake, "table",
        rigid_body={"physics:kinematicEnabled": True, "physxRigidBody:disableGravity": True},
        articulation={"physics:articulationEnabled": False},
    )
    cloth_cfg = env.cfg.rigid_cloth
    materials = pair_materials(cloth_cfg.table_static_friction, cloth_cfg.hand_static_friction,
                               cloth_cfg.dynamic_friction_ratio)
    _bake_contact_material(robot_usd, materials["robot"])
    _bake_contact_material(table_usd, materials["table"])
    env._robot_urdf_path, env._table_urdf_path = a.robot_urdf, a.table_urdf
    env._table_size = urdf_box_size(a.table_urdf)
    env._table_half_height = 0.5 * env._table_size[2]
    _log(t0, "robot + table USDs baked")

    stage = get_current_stage()
    for path in env.scene.env_prim_paths:
        if not stage.GetPrimAtPath(path).IsValid():
            stage.DefinePrim(path, "Xform")
    env.robot = Articulation(robot_cfg(robot_usd, spec, env.cfg.robot_root_pos, env.cfg.action.arm_effort_limit))
    env.table = RigidObject(RigidObjectCfg(prim_path="/World/envs/env_.*/Table", spawn=UsdFileCfg(usd_path=table_usd)))
    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
    light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light.func("/World/Light", light)
    env.scene.articulations["robot"] = env.robot
    env.scene.rigid_objects["table"] = env.table
    _log(t0, "robot + table spawned")


# ------------------------------------------------------------------ the cloths


def _apply_physx_api(prim, schema: str, attrs: dict) -> None:
    if schema not in prim.GetAppliedSchemas():
        prim.AddAppliedSchema(schema)
    for name, (vtype, value) in attrs.items():
        prim.CreateAttribute(name, vtype).Set(value)


class ArticulationGroup:
    """Scene stepping for a cloth batch whose rows cover only some environments.

    InteractiveScene passes global IDs to reset. The underlying Isaac articulation
    uses group-local rows. Physics updates and command writes already span its rows.
    Raw articulation state remains on ``asset``; backend readout scatters it globally.
    """

    def __init__(self, asset, global_ids, num_envs):
        self.asset = asset
        self.num_envs = num_envs
        self.set_env_ids(global_ids)

    def set_env_ids(self, global_ids):
        self.env_ids = global_ids
        self._local_ids = torch.full((self.num_envs,), -1, dtype=torch.long, device=global_ids.device)
        self._local_ids[global_ids] = torch.arange(len(global_ids), device=global_ids.device)

    @property
    def is_initialized(self):
        return self.asset.is_initialized

    def reset(self, env_ids=None):
        if env_ids is None:
            self.asset.reset()
            return
        global_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.env_ids.device)
        local = self._local_ids[global_ids]
        local = local[local >= 0]
        if local.numel():
            self.asset.reset(local)

    def write_data_to_sim(self):
        self.asset.write_data_to_sim()

    def update(self, dt):
        self.asset.update(dt)


def _is_rectangle(plate):
    """Axis-aligned rectangles can retain the inexpensive native box collider."""
    return (len(plate.vertices) == 4
            and len({round(x, 12) for x, _ in plate.vertices}) == 2
            and len({round(y, 12) for _, y in plate.vertices}) == 2)


def author_rigid_cloths(env_ids, dims, cloth_cfg, spawn_z, design, root_name) -> None:
    """Author one topology into the selected global environments, at their own sizes.

    A private USD template is copied in one change block. Rectangles use boxes;
    other convex panels use extruded polygon meshes. All body frames coincide at
    the cloth rest origin, so the same surface offsets work for every panel.
    """
    from isaaclab.sim.utils import get_current_stage
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt

    cfg = cloth_cfg
    layer = get_current_stage().GetRootLayer()
    stage = Usd.Stage.CreateInMemory()
    template = f"/{root_name}"
    precision = UsdGeom.XformOp.PrecisionDouble
    UsdGeom.Xform.Define(stage, template).AddTranslateOp(precision=precision)
    for index, plate in enumerate(design.plates):
        body_path = f"{template}/{plate.name}"
        body = UsdGeom.Xform.Define(stage, body_path).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
        mass = UsdPhysics.MassAPI.Apply(body)
        mass.CreateMassAttr(1.0)
        mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1.0))
        mass.CreatePrincipalAxesAttr(Gf.Quatf(1.0))
        _apply_physx_api(body, "PhysxRigidBodyAPI", {
            "physxRigidBody:maxDepenetrationVelocity": (Sdf.ValueTypeNames.Float, float(cfg.max_depenetration_velocity)),
        })
        if index == 0:
            UsdPhysics.ArticulationRootAPI.Apply(body)
            _apply_physx_api(body, "PhysxArticulationAPI", {
                "physxArticulation:enabledSelfCollisions": (Sdf.ValueTypeNames.Bool, True),
                "physxArticulation:solverPositionIterationCount": (Sdf.ValueTypeNames.Int, int(cfg.solver_position_iterations)),
                "physxArticulation:solverVelocityIterationCount": (Sdf.ValueTypeNames.Int, int(cfg.solver_velocity_iterations)),
            })
        collider_path = f"{body_path}/collider"
        if _is_rectangle(plate):
            collider = UsdGeom.Cube.Define(stage, collider_path)
            collider.CreateSizeAttr(1.0)
            collider.AddTranslateOp(precision=precision)
            collider.AddScaleOp(precision=precision)
        else:
            vertices, faces = designs.extrusion_mesh(plate)
            collider = UsdGeom.Mesh.Define(stage, collider_path)
            collider.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices.astype("float32")))
            collider.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
            collider.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.astype("int32").ravel()))
            collider.CreateSubdivisionSchemeAttr("none")
            UsdPhysics.MeshCollisionAPI.Apply(collider.GetPrim()).CreateApproximationAttr("convexHull")
        color = (0.29, 0.56, 0.76) if plate.center[0] < 0.0 else (0.95, 0.68, 0.22)
        collider.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*color)]))
        UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
        _apply_physx_api(collider.GetPrim(), "PhysxCollisionAPI", {
            "physxCollision:contactOffset": (Sdf.ValueTypeNames.Float, float(cfg.contact_offset)),
            "physxCollision:restOffset": (Sdf.ValueTypeNames.Float, float(cfg.rest_offset)),
        })

    for index, (name, hinge) in enumerate(zip(design.joint_names, design.hinges)):
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"{template}/{name}")
        joint.CreateBody0Rel().SetTargets([f"{template}/{hinge.parent}"])
        joint.CreateBody1Rel().SetTargets([f"{template}/{hinge.child}"])
        joint.CreateAxisAttr(UsdPhysics.Tokens.x)
        joint.CreateLocalRot0Attr(Gf.Quatf(1.0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
        joint.CreateLocalPos0Attr(Gf.Vec3f(0.0))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0))
        limits = cfg.hinge_limits_deg if index == design.fold_hinge else hinge.limits_deg
        joint.CreateLowerLimitAttr(float(limits[0]))
        joint.CreateUpperLimitAttr(float(limits[1]))
        # Neighbours share an edge; the remaining pairs retain self-contact.
        joint.CreateCollisionEnabledAttr(False)
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(0.0)
        drive.CreateDampingAttr(0.0)
    bind_physics_material(stage, template, pair_materials(0., 0., 1.)["cloth"])
    source = stage.GetRootLayer()

    def put(path, value):
        attribute = layer.GetAttributeAtPath(Sdf.Path(path))
        if attribute is None:
            raise RuntimeError(f"no attribute spec at {path}")
        attribute.default = value

    with Sdf.ChangeBlock():
        for row, env_id in enumerate(env_ids):
            env_id = int(env_id)
            root = f"/World/envs/env_{env_id}/{root_name}"
            Sdf.CreatePrimInLayer(layer, f"/World/envs/env_{env_id}")
            if not Sdf.CopySpec(source, template, layer, root):
                raise RuntimeError(f"Sdf.CopySpec {template} -> {root} failed")
            spec = design.resolve(*(float(value) for value in dims[row]))
            put(f"{root}.xformOp:translate", Gf.Vec3d(0.0, 0.0, float(spawn_z(env_id))))
            for plate in spec.plates:
                body = f"{root}/{plate.name}"
                mass, center, inertia, principal = designs.polygon_mass_properties(plate, float(cfg.density))
                put(f"{body}.physics:mass", float(mass))
                put(f"{body}.physics:centerOfMass", Gf.Vec3f(*map(float, center)))
                put(f"{body}.physics:diagonalInertia", Gf.Vec3f(*map(float, inertia)))
                put(f"{body}.physics:principalAxes", Gf.Quatf(float(principal[0]), Gf.Vec3f(*map(float, principal[1:]))))
                if _is_rectangle(plate):
                    put(f"{body}/collider.xformOp:translate", Gf.Vec3d(*map(float, center)))
                    put(f"{body}/collider.xformOp:scale", Gf.Vec3d(*(2.0 * h for h in plate.half_extents)))
                else:
                    vertices, _ = designs.extrusion_mesh(plate)
                    put(f"{body}/collider.points", Vt.Vec3fArray.FromNumpy(vertices.astype("float32")))
            for name, hinge in zip(spec.joint_names, spec.hinges):
                anchor = Gf.Vec3f(*map(float, hinge.anchor))
                rotation = Gf.Quatf(Gf.Rotation(Gf.Vec3d(1.0, 0.0, 0.0), Gf.Vec3d(*hinge.axis)).GetQuat())
                for side in (0, 1):
                    put(f"{root}/{name}.physics:localPos{side}", anchor)
                    put(f"{root}/{name}.physics:localRot{side}", rotation)


def rigid_cloth_articulation(cloth_cfg, spawn_z0, design, root_name):
    """One batched Isaac articulation for the environments using this design."""
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg

    # Nonzero seed gains ensure PhysX creates a drive; rigid_init writes exact
    # per-environment geometry-scaled material gains before the first step.
    return Articulation(ArticulationCfg(
        prim_path=f"/World/envs/env_.*/{root_name}",
        spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, float(spawn_z0))),
        actuators={"hinges": ImplicitActuatorCfg(
            joint_names_expr=list(design.joint_names),
            stiffness=1.0e-9, damping=1.0e-9,
            armature=float(cloth_cfg.hinge_armature), effort_limit_sim=None,
        )},
    ))
