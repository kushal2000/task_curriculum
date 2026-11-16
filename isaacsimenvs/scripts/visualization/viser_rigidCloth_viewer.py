"""Inspect rigid cloth approximations in Isaac Sim through a small Viser interface.

Run: .venv/isaacsim/bin/python isaacsimenvs/scripts/visualization/viser_rigidCloth_viewer.py
Open http://localhost:8080 (forward that port over SSH for a remote machine).
The default preview has a static robot. --policy runs the pretrained Flexiv–Wuji
policy through ClothEnv, with a choice of training or current real-hand setup.
Cloth geometry and hinges come from design YAML.
Geometry and mass changes rebuild the scene; hinge parameters and friction update live.
"""

from __future__ import annotations

import gc
import math
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch

from isaacsimenvs import CONFIG_DIR, REPO_ROOT
from isaacsimenvs.tasks.cloth.backends.rigid import designs

TABLE_SIZE = (0.475, 0.4, 0.04)
TABLE_POSITION = (0.0, 0.29, 0.26)


@dataclass(frozen=True)
class Settings:
    design: str = "single_hinge"
    length: float = 0.2
    width: float = 0.2
    thickness: float = 0.002
    density: float = 0.1
    bending_rigidity: float = 5e-5
    bending_viscosity: float = 5e-5
    armature: float = 0.0
    armature_scale: float = 0.0
    table_static_friction: float = 0.4
    hand_static_friction: float = 0.8
    dynamic_friction_ratio: float = 0.8
    drop_height: float = 0.06
    bend_deg: float = 90.0
    x: float = 0.0
    y: float = 0.0
    yaw_deg: float = 0.0
    sample_reset: bool = False
    x_range: tuple[float, float] = (-0.20, 0.20)
    y_range: tuple[float, float] = (-0.17, 0.17)
    yaw_range_deg: tuple[float, float] = (0.0, 360.0)


def load_static_robot(urdf_path=None, robot_spec=None, root_pos=None):
    import trimesh
    import yourdfpy

    if urdf_path is None:
        from isaacsimenvs.scripts.visualization.rigid_cloth.policy import robot_profile
        robot_spec, urdf_path, _, root_pos = robot_profile("training")
    path = REPO_ROOT / urdf_path
    robot = yourdfpy.URDF.load(
        path,
        build_scene_graph=True,
        build_collision_scene_graph=True,
        load_meshes=True,
        load_collision_meshes=True,
        filename_handler=partial(yourdfpy.filename_handler_magic, dir=path.parent),
    )
    joints = {name: float(robot_spec.arm_default_joint_pos.get(name, 0.0)) for name in robot.actuated_joint_names}
    robot.update_cfg(joints)
    root = trimesh.transformations.translation_matrix(root_pos)
    collisions = []
    for node in robot.collision_scene.graph.nodes_geometry:
        transform, geometry = robot.collision_scene.graph[node]
        mesh = robot.collision_scene.geometry[geometry].copy()
        mesh.apply_transform(root @ transform)
        collisions.append((mesh.vertices, mesh.faces))
    return robot, joints, collisions


class Physics:
    """One floating cloth articulation against static robot meshes and a table.

    Geometry changes require close() followed by construction of a new instance.
    Returned poses are body-link origins and wxyz orientations. Mesh vertices
    already contain their rest offsets. Density is areal kg/m², matching the task.
    """

    def __init__(self, settings, robot_collision_meshes, table_size, table_position, device="cuda:0", dt=1 / 120):
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import Articulation, ArticulationCfg
        from isaaclab.sim import SimulationCfg, SimulationContext
        from isaaclab.sim.utils import create_new_stage
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade, Vt

        self.sim = self.cloth = None
        self.dt, self.time = dt, 0.0
        self.table_size = np.asarray(table_size, dtype=float)
        self.table_position = np.asarray(table_position, dtype=float)
        self.table_top = self.table_position[2] + self.table_size[2] / 2
        self.dimensions = (float(settings.length), float(settings.width), float(settings.thickness))
        if not math.isfinite(settings.density) or settings.density <= 0:
            raise ValueError("Areal density must be finite and positive.")
        self.spec = designs.load_design(settings.design).resolve(*self.dimensions)
        try:
            create_new_stage()
            self.sim = SimulationContext(SimulationCfg(dt=dt, device=device, render_interval=1))
            self.stage = self.sim.stage
            UsdGeom.Xform.Define(self.stage, "/World")
            self.materials = {}
            from isaacsimenvs.tasks.cloth.backends.rigid.scene import pair_materials, FRICTION_COMBINE_MODE
            for name, coefficients in pair_materials(settings.table_static_friction, settings.hand_static_friction,
                                                     settings.dynamic_friction_ratio).items():
                mat = UsdShade.Material.Define(self.stage, f"/World/Materials/{name}")
                api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
                api.CreateStaticFrictionAttr(float(coefficients[0]))
                api.CreateDynamicFrictionAttr(float(coefficients[1]))
                api.CreateRestitutionAttr(0.0)
                PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim()).CreateFrictionCombineModeAttr(FRICTION_COMBINE_MODE)
                self.materials[name] = (mat, api)
            table = UsdGeom.Cube.Define(self.stage, "/World/Table")
            table.CreateSizeAttr(1.0)
            table.AddTranslateOp().Set(Gf.Vec3d(*map(float, self.table_position)))
            table.AddScaleOp().Set(Gf.Vec3f(*map(float, self.table_size)))
            self._collision(table.GetPrim(), "table")
            floor = UsdGeom.Cube.Define(self.stage, "/World/Floor")
            floor.CreateSizeAttr(1.0)
            floor.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05))
            floor.AddScaleOp().Set(Gf.Vec3f(6, 6, 0.1))
            self._collision(floor.GetPrim(), "table")
            for i, (vertices, faces) in enumerate(robot_collision_meshes):
                mesh = UsdGeom.Mesh.Define(self.stage, f"/World/Robot/shape_{i}")
                mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(vertices, dtype=np.float32)))
                mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
                mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(faces, dtype=np.int32).ravel()))
                UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
                self._collision(mesh.GetPrim(), "robot")
            self._build_cloth(settings)
            self.cloth = Articulation(
                ArticulationCfg(
                    prim_path="/World/Cloth",
                    spawn=None,
                    actuators={
                        "hinges": ImplicitActuatorCfg(
                            joint_names_expr=list(self.spec.joint_names),
                            stiffness=1e-9,
                            damping=1e-9,
                            armature=float(settings.armature),
                            effort_limit_sim=100.0,
                        )
                    },
                )
            )
            self.sim.reset()
            self.body_ids = [self.cloth.body_names.index(plate.name) for plate in self.spec.plates]
            self.joint_ids = [self.cloth.joint_names.index(name) for name in self.spec.joint_names]
            self.reset(settings)
        except BaseException:
            self.close()
            raise

    def _collision(self, prim, material):
        from pxr import PhysxSchema, UsdPhysics, UsdShade

        UsdPhysics.CollisionAPI.Apply(prim)
        collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        collision.CreateContactOffsetAttr(0.001)
        collision.CreateRestOffsetAttr(0.0)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            self.materials[material][0], UsdShade.Tokens.weakerThanDescendants, "physics"
        )

    def _build_cloth(self, settings):
        """Author the same plate geometry and hinge graph used by the cloth environment."""
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, Vt

        UsdGeom.Xform.Define(self.stage, "/World/Cloth")
        # Body frames coincide at rest; geometry and COM offsets come from the design.
        for index, plate in enumerate(self.spec.plates):
            body_path = f"/World/Cloth/{plate.name}"
            body = UsdGeom.Xform.Define(self.stage, body_path).GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(body)
            physx = PhysxSchema.PhysxRigidBodyAPI.Apply(body)
            physx.CreateMaxDepenetrationVelocityAttr(1.0)
            mass, center, inertia, principal = designs.polygon_mass_properties(plate, float(settings.density))
            props = UsdPhysics.MassAPI.Apply(body)
            props.CreateMassAttr(mass)
            props.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, center)))
            props.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, inertia)))
            props.CreatePrincipalAxesAttr(Gf.Quatf(float(principal[0]), Gf.Vec3f(*map(float, principal[1:]))))
            if index == 0:
                UsdPhysics.ArticulationRootAPI.Apply(body)
                art = PhysxSchema.PhysxArticulationAPI.Apply(body)
                art.CreateEnabledSelfCollisionsAttr(True)
                art.CreateSolverPositionIterationCountAttr(16)
                art.CreateSolverVelocityIterationCountAttr(4)
            vertices, faces = designs.extrusion_mesh(plate)
            mesh = UsdGeom.Mesh.Define(self.stage, f"{body_path}/mesh")
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.astype(np.int32).ravel()))
            mesh.CreateSubdivisionSchemeAttr("none")
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
            self._collision(mesh.GetPrim(), "cloth")
        for name, hinge in zip(self.spec.joint_names, self.spec.hinges):
            joint = UsdPhysics.RevoluteJoint.Define(self.stage, f"/World/Cloth/{name}")
            joint.CreateBody0Rel().SetTargets([f"/World/Cloth/{hinge.parent}"])
            joint.CreateBody1Rel().SetTargets([f"/World/Cloth/{hinge.child}"])
            joint.CreateAxisAttr("X")
            rotation = Gf.Quatf(Gf.Rotation(Gf.Vec3d(1, 0, 0), Gf.Vec3d(*hinge.axis)).GetQuat())
            anchor = Gf.Vec3f(*map(float, hinge.anchor))
            joint.CreateLocalRot0Attr(rotation)
            joint.CreateLocalRot1Attr(rotation)
            joint.CreateLocalPos0Attr(anchor)
            joint.CreateLocalPos1Attr(anchor)
            joint.CreateLowerLimitAttr(float(hinge.limits_deg[0]))
            joint.CreateUpperLimitAttr(float(hinge.limits_deg[1]))
            joint.CreateCollisionEnabledAttr(False)
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(0.0)
            drive.CreateDampingAttr(0.0)

    def reset(self, settings):
        """Drop with only the central hinge folded; all hinges spring toward flat."""
        from isaacsimenvs.scripts.visualization.rigid_cloth.reset import sample_reset_pose

        dimensions = torch.tensor([self.dimensions], device=self.cloth.device)
        xy, yaw = sample_reset_pose(settings, dimensions, table_size=self.table_size)
        self.update_material(settings)
        root = self.cloth.data.default_root_state.clone()
        root[:, :2] = xy + torch.as_tensor(self.table_position[:2], device=self.cloth.device)
        root[:, 2] = self.table_top + settings.drop_height + settings.thickness / 2
        root[:, 3:7] = 0.0
        root[:, 3], root[:, 6] = torch.cos(yaw / 2), torch.sin(yaw / 2)
        root[:, 7:] = 0
        q = torch.zeros_like(self.cloth.data.default_joint_pos)
        q[:, self.joint_ids[self.spec.fold_hinge]] = math.radians(settings.bend_deg)
        self.cloth.write_root_state_to_sim(root)
        self.cloth.write_joint_state_to_sim(q, torch.zeros_like(q))
        self.cloth.set_joint_position_target(torch.zeros_like(q))
        self.cloth.set_joint_velocity_target(torch.zeros_like(q))
        self.cloth.reset()
        self.cloth.write_data_to_sim()
        self.sim.forward()
        self.cloth.update(self.dt)
        self.time = 0.0

    def update_material(self, settings):
        from isaacsimenvs.tasks.cloth.backends.rigid.bending import hinge_geometry_factors
        from isaacsimenvs.tasks.cloth.backends.rigid.inertia import hinge_inertias
        from isaacsimenvs.tasks.cloth.backends.rigid.scene import pair_materials
        factors = hinge_geometry_factors(self.spec)
        for parameter, method in (("bending_rigidity", "stiffness"), ("bending_viscosity", "damping")):
            values = torch.as_tensor(getattr(settings, parameter) * factors,
                                     device=self.cloth.device, dtype=torch.float32)[None]
            getattr(self.cloth, f"write_joint_{method}_to_sim")(values, joint_ids=self.joint_ids)
        armature = settings.armature + settings.armature_scale * hinge_inertias(self.spec, settings.density)
        self.cloth.write_joint_armature_to_sim(
            torch.as_tensor(armature, device=self.cloth.device, dtype=torch.float32)[None], joint_ids=self.joint_ids,
        )
        for name, coefficients in pair_materials(settings.table_static_friction, settings.hand_static_friction,
                                                 settings.dynamic_friction_ratio).items():
            api = self.materials[name][1]
            api.GetStaticFrictionAttr().Set(float(coefficients[0]))
            api.GetDynamicFrictionAttr().Set(float(coefficients[1]))
        view = self.cloth.root_physx_view
        mats = view.get_material_properties()
        mats[..., :2], mats[..., 2] = 1.0, 0.0
        view.set_material_properties(mats, torch.tensor([0], dtype=torch.int32, device="cpu"))

    def step(self):
        self.cloth.write_data_to_sim()
        self.sim.step(render=False)
        self.cloth.update(self.dt)
        self.time += self.dt

    def read_poses(self):
        pose = self.cloth.data.body_link_pose_w[0, self.body_ids].detach().cpu().numpy()
        return pose[:, :3].copy(), pose[:, 3:7].copy()

    def metrics(self):
        return {
            "time": self.time,
            "mass": float(self.cloth.root_physx_view.get_masses().sum()),
            "hinge_angles_deg": np.degrees(self.cloth.data.joint_pos[0, self.joint_ids].detach().cpu().numpy()),
        }

    def close(self):
        if self.sim is not None:
            # Unsubscribe Isaac Lab's standalone STOP handler before stopping; it
            # otherwise waits indefinitely for the timeline to play again.
            type(self.sim).clear_instance()
            self.sim.stop()
            self.cloth = None
            self.sim = None
            gc.collect()


def main() -> None:
    import argparse
    import queue
    import time
    from dataclasses import replace

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--design", default="single_hinge", help="Catalog YAML name or path to a design YAML.")
    parser.add_argument("--policy", action="store_true", help="Start with the pretrained Flexiv–Wuji policy enabled.")
    parser.add_argument("--robot_setup", choices=("training", "real"), default="training")
    parser.add_argument("--checkpoint", default=str(REPO_ROOT / "pretrained_policy/flexiv_wuji/model.pth"))
    parser.add_argument("--policy_config", default=str(REPO_ROOT / "pretrained_policy/flexiv_wuji/config.yaml"))
    parser.add_argument("--training_config", type=Path, default=CONFIG_DIR / "runs/rigid_body_cloth.yaml",
                        help="Training recipe supplying geometry, physics and reset ranges.")
    parser.add_argument("--sapg_expl_coef", type=float, default=0.0, help="SAPG branch coefficient, 0–50.")
    parser.add_argument("--steps", type=int, default=0, help="Exit after N preview physics steps or policy control steps; 0 runs interactively.")
    args = parser.parse_args()
    if args.sapg_expl_coef not in (0, 10, 20, 30, 40, 50):
        parser.error("--sapg_expl_coef must be one of 0, 10, 20, 30, 40, 50")
    if args.steps < 0:
        parser.error("--steps must be nonnegative")

    from isaacsimenvs.scripts.visualization.rigid_cloth.reset import (
        effective_placement_bounds, load_training_env_config, sample_reset_pose, validate_reset_settings,
    )
    try:
        training_values = load_training_env_config(args.training_config)
        initial_path = designs.resolve_design_path(args.design)
        # Validate before launching Kit; Apply reloads this same YAML after edits.
        designs.load_design(initial_path)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    catalog = {path.stem: str(path) for path in designs.discover_designs()}
    initial_label = initial_path.stem
    if initial_label in catalog and catalog[initial_label] != str(initial_path):
        initial_label = str(initial_path)
    catalog[initial_label] = str(initial_path)

    import viser
    import yourdfpy
    from isaaclab.app import AppLauncher
    from viser.extras import ViserUrdf

    app = AppLauncher(headless=True, device=args.device).app
    server, physics = None, None
    try:
        from isaacsimenvs.tasks.cloth.cloth_cfg import ClothCfg
        training_cfg = ClothCfg.for_backend("rigid")
        training_cfg.from_dict(training_values)
        training_physics = training_cfg.rigid_cloth
        dr = training_physics.randomization
        robot, joint_values = None, None
        server = viser.ViserServer(host=args.host, port=args.port)
        server.scene.set_up_direction("+z")
        server.scene.add_grid("/ground", width=3.0, height=3.0, cell_size=0.1)
        robot_view = robot_frame = table_view = targets_view = keypoints_view = errors_view = None

        @server.on_client_connect
        def set_camera(client):
            center = np.asarray(physics.table_position if physics is not None else TABLE_POSITION)
            client.camera.position = tuple(center + (0.85, -0.85, 0.67))
            client.camera.look_at = tuple(center + (0.0, 0.08, 0.15))
            client.camera.up_direction = (0.0, 0.0, 1.0)

        controls = {}
        commands = queue.SimpleQueue()
        active = Settings(design=str(initial_path), bend_deg=0.0 if args.policy else 90.0,
                          drop_height=0.002 if args.policy else 0.06,
                          bending_rigidity=float(training_physics.bending_rigidity),
                          bending_viscosity=float(training_physics.bending_viscosity),
                          armature=float(training_physics.hinge_armature),
                          density=float(np.clip(training_physics.density, *dr.density)),
                          length=float(np.mean(training_physics.length_range)),
                          width=float(np.mean(training_physics.width_range)),
                          thickness=float(np.mean(training_physics.thickness_range)),
                          table_static_friction=float(training_physics.table_static_friction),
                          hand_static_friction=float(training_physics.hand_static_friction),
                          dynamic_friction_ratio=float(training_physics.dynamic_friction_ratio),
                          sample_reset=bool(args.policy),
                          x_range=tuple(map(float, training_physics.place_x)),
                          y_range=tuple(map(float, training_physics.place_y)),
                          yaw_range_deg=tuple(map(float, training_cfg.reset.yaw_range_deg)))
        server.gui.add_markdown("## Rigid cloth\nPreview geometry or evaluate the pretrained Flexiv–Wuji policy.")
        setup_labels = {"Training setup": "training", "Real hand / cloth setup": "real"}
        with server.gui.add_folder("Robot — apply to rebuild"):
            robot_mode = server.gui.add_dropdown("Robot control", ("Static robot", "Pretrained policy"),
                                                initial_value="Pretrained policy" if args.policy else "Static robot")
            robot_setup = server.gui.add_dropdown("Robot and table setup", tuple(setup_labels),
                                                 initial_value=next(k for k, v in setup_labels.items() if v == args.robot_setup))
            setup_description = server.gui.add_markdown("")
        with server.gui.add_folder("Policy — updates live"):
            coefficient = server.gui.add_dropdown("SAPG coefficient", ("0", "10", "20", "30", "40", "50"),
                                                  initial_value=f"{args.sapg_expl_coef:g}")
            deterministic = server.gui.add_checkbox("Deterministic actions", initial_value=True)
            keypoint_display = server.gui.add_dropdown("Keypoint display", ("Policy (Play)", "Physical corners"),
                                                       initial_value="Policy (Play)")
            server.gui.add_markdown("Play prior: evaluating transfer to folding. No fold training in this checkpoint.")
            server.gui.add_markdown("**Coral:** current keypoints · **Green:** target keypoints. "
                                    "Yellow lines connect each current point to its target. "
                                    "Policy view uses Play's diagonal top/bottom pairs; Physical corners shows fold landmarks.")
        coefficient.on_update(lambda _: commands.put("policy"))
        deterministic.on_update(lambda _: commands.put("policy"))
        keypoint_display.on_update(lambda _: commands.put("display"))
        robot_setup.on_update(lambda _: commands.put("geometry"))
        with server.gui.add_folder("Geometry — apply to rebuild"):
            controls["design"] = server.gui.add_dropdown(
                "Design YAML", options=tuple(catalog), initial_value=initial_label
            )
            for name, label, low, high, step in (
                ("length", "Length (m)", *training_physics.length_range, 0.001),
                ("width", "Width (m)", *training_physics.width_range, 0.001),
                ("thickness", "Thickness (m)", *training_physics.thickness_range, 0.0001),
                ("density", "Areal density (kg/m²)", *training_physics.randomization.density, 0.001),
            ):
                controls[name] = server.gui.add_slider(
                    label, min=low, max=high, step=step, initial_value=float(np.clip(getattr(active, name), low, high))
                )
            controls["thickness"] = server.gui.add_number("Collision thickness (m)", initial_value=active.thickness, disabled=True)
            rebuild = server.gui.add_button("Apply geometry and restart")
        live_fields = ("bending_rigidity", "bending_viscosity", "armature_scale", "table_static_friction",
                       "hand_static_friction", "dynamic_friction_ratio")
        with server.gui.add_folder("Physics — updates live"):
            for name, label, bounds, step in (
                ("bending_rigidity", "Sheet bending rigidity (N m)", dr.bending_rigidity, 1e-6),
                ("bending_viscosity", "Sheet bending viscosity (N m s)", dr.bending_viscosity, 1e-6),
                ("table_static_friction", "Cloth–table static friction", dr.table_static_friction, 0.01),
                ("hand_static_friction", "Cloth–robot static friction", dr.hand_static_friction, 0.01),
                ("dynamic_friction_ratio", "Dynamic / static friction", dr.dynamic_friction_ratio, 0.01),
            ):
                low, high = bounds
                handle = controls[name] = server.gui.add_slider(
                    label, min=low, max=high, step=step,
                    initial_value=float(np.clip(getattr(active, name), low, high)),
                )
                handle.on_update(lambda _: commands.put("material"))
            controls["armature_scale"] = server.gui.add_number("Armature fraction (disabled)", initial_value=0.0, disabled=True)
            server.gui.add_markdown(f"Bounds: `{args.training_config.name}`. "
                                    "One sheet material; hinge gains scale with crease geometry. "
                                    "Friction controls show actual contact coefficients.")
        reset_fields = ("x", "y", "yaw_deg", "drop_height", "bend_deg", "sample_reset",
                        "x_range", "y_range", "yaw_range_deg")
        with server.gui.add_folder("Initial state — press reset"):
            controls["sample_reset"] = server.gui.add_dropdown(
                "Reset placement", ("Manual", "Sample training ranges"),
                initial_value="Sample training ranges" if active.sample_reset else "Manual",
            )
            for name, label in (("x_range", "X placement min / max (m)"),
                                ("y_range", "Y placement min / max (m)"),
                                ("yaw_range_deg", "Yaw min / max (degrees)")):
                controls[name] = server.gui.add_vector2(label, initial_value=getattr(active, name),
                                                       step=1.0 if name == "yaw_range_deg" else .01)
            for name, label, low, high, step in (
                ("x", "Manual center X (m)", *training_physics.place_x, 0.01),
                ("y", "Manual center Y (m)", *training_physics.place_y, 0.01),
                ("yaw_deg", "Manual yaw (degrees)", 0.0, 360.0, 1.0),
                ("drop_height", "Drop above table (m)", 0.002, 0.4, 0.002),
                ("bend_deg", "Center fold (degrees)", 0.0, 175.0, 1.0),
            ):
                controls[name] = server.gui.add_slider(
                    label, min=low, max=high, step=step, initial_value=float(np.clip(getattr(active, name), low, high))
                )
            server.gui.add_markdown("Sampling keeps the full cloth inside the selected table, with a 5 mm edge margin. "
                                    "Only yaw angles that fit are sampled. Edits apply on Reset.")
            placement_info = server.gui.add_markdown("")
            reset = server.gui.add_button("Reset cloth")

        def robot_mode_changed(_):
            controls["sample_reset"].value = ("Sample training ranges" if robot_mode.value == "Pretrained policy" else "Manual")
            commands.put("geometry")

        robot_mode.on_update(robot_mode_changed)
        running = server.gui.add_checkbox("Run simulation", initial_value=bool(args.steps))
        step_button = server.gui.add_button("Single step")
        speed = server.gui.add_slider("Playback speed", min=0.1, max=2.0, step=0.1, initial_value=1.0)
        status = server.gui.add_markdown("Building physics scene…")
        rebuild.on_click(lambda _: commands.put("rebuild"))
        reset.on_click(lambda _: commands.put("reset"))
        step_button.on_click(lambda _: commands.put("step"))
        geometry_fields = ("design", "length", "width", "thickness", "density")
        for name in geometry_fields:
            controls[name].on_update(lambda _: commands.put("geometry"))

        def settings_from_controls(fields=None):
            values = {}
            for name in controls if fields is None else fields:
                value = controls[name].value
                if name == "design":
                    values[name] = catalog[value]
                elif name == "sample_reset":
                    values[name] = value == "Sample training ranges"
                elif name in ("x_range", "y_range", "yaw_range_deg"):
                    values[name] = tuple(map(float, value))
                else:
                    values[name] = float(value)
            return replace(active, **values)

        meshes = []

        def build_meshes():
            for mesh in meshes:
                mesh.remove()
            meshes.clear()
            for i, plate in enumerate(physics.spec.plates):
                color = (74, 143, 194) if plate.center[0] < 0 else (242, 173, 56)
                vertices, faces = designs.extrusion_mesh(plate)
                meshes.append(
                    server.scene.add_mesh_simple(
                        f"/cloth/panel_{i}", vertices=vertices, faces=faces, color=color, flat_shading=True,
                    )
                )

        def selected_scene():
            from isaacsimenvs.scripts.evaluation.player import read_policy_cfg
            from isaacsimenvs.scripts.visualization.rigid_cloth.policy import robot_profile, table_position
            from isaacsimenvs.tasks.cloth.backends.assets import urdf_box_size
            profile = setup_labels[robot_setup.value]
            spec, urdf, table_urdf, root = robot_profile(profile)
            source = read_policy_cfg(args.policy_config, args.device)["env"] if profile == "training" else {}
            return spec, urdf, root, urdf_box_size(str(REPO_ROOT / table_urdf)), table_position(profile, source)

        def validate_selection(settings):
            validate_reset_settings(settings)
            designs.load_design(settings.design).resolve(settings.length, settings.width, settings.thickness)
            if robot_mode.value == "Pretrained policy":
                for path in (args.checkpoint, args.policy_config):
                    if not Path(path).is_file():
                        raise ValueError(f"Policy file does not exist: {path}")
            _, _, _, table_size, _ = selected_scene()
            dimensions = torch.tensor([[settings.length, settings.width, settings.thickness]])
            with torch.random.fork_rng(devices=[]):
                sample_reset_pose(settings, dimensions, table_size=table_size)

        active_profile = args.robot_setup

        def build_physics(settings):
            nonlocal active_profile, robot, joint_values
            active_profile = setup_labels[robot_setup.value]
            if robot_mode.value == "Static robot":
                spec, urdf, root, table_size, center = selected_scene()
                robot, joint_values, collisions = load_static_robot(urdf, spec, root)
                preview = Physics(settings, collisions, table_size, center, device=args.device)
                preview.robot_urdf, preview.robot_root_pos = REPO_ROOT / urdf, root
                return preview
            from isaacsimenvs.scripts.visualization.rigid_cloth.policy import PolicyPhysics
            return PolicyPhysics(settings, profile=active_profile,
                                 checkpoint_path=args.checkpoint, config_path=args.policy_config, training_config=args.training_config,
                                 device=args.device, expl_coef=float(coefficient.value),
                                 deterministic=deterministic.value)

        def build_scene_visuals():
            nonlocal robot_view, robot_frame, table_view, targets_view, keypoints_view, errors_view
            # Removing the root removes all robot descendants. ViserUrdf.remove()
            # also traverses them individually, which double-removes nested nodes.
            for old in (robot_frame, table_view, targets_view, keypoints_view, errors_view):
                if old is not None:
                    old.remove()
            policy_active = hasattr(physics, "player")
            if policy_active:
                path = Path(physics.robot_urdf)
                model = yourdfpy.URDF.load(path, load_collision_meshes=False,
                                          filename_handler=partial(yourdfpy.filename_handler_magic, dir=path.parent))
                root = physics.robot_root_pos
            else:
                model, root = robot, physics.robot_root_pos
            robot_frame = server.scene.add_frame("/robot", position=root, show_axes=False)
            robot_view = ViserUrdf(server, model, root_node_name="/robot")
            names = robot_view.get_actuated_joint_names()
            robot_view.update_cfg(physics.robot_joint_positions(names) if policy_active else
                                  np.asarray([joint_values[n] for n in names]))
            table_view = server.scene.add_box("/table", dimensions=tuple(physics.table_size),
                                             position=tuple(physics.table_position), color=(209, 143, 89))
            targets_view = server.scene.add_point_cloud("/fold_targets", points=np.zeros((4, 3)),
                                                       colors=(66, 220, 180), point_size=0.012,
                                                       visible=policy_active)
            keypoints_view = server.scene.add_point_cloud("/cloth_keypoints", points=np.zeros((4, 3)),
                                                         colors=(255, 100, 80), point_size=0.014,
                                                         visible=policy_active)
            errors_view = server.scene.add_line_segments("/keypoint_errors", points=np.zeros((4, 2, 3)),
                                                        colors=(255, 220, 110), line_width=2,
                                                        visible=policy_active)
            coefficient.disabled = deterministic.disabled = keypoint_display.disabled = not policy_active
            top = float(physics.table_position[2] + physics.table_size[2] / 2)
            if policy_active:
                setup_description.content = (
                    f"**{next(k for k, v in setup_labels.items() if v == active_profile)}** · `{Path(physics.robot_urdf).name}`  \n"
                    "Policy object: moving-half pose, Play keypoint order.  \n"
                    f"Keypoint scale: {physics.env.cfg.obs.keypoint_scale:g} · "
                    f"Object base size: {physics.env.cfg.reward.object_base_size:g} m  \n"
                    f"Moving-half size / base: {tuple(round(float(v), 3) for v in physics.env.task._object_scales[0])}  \n"
                    f"Table: `{Path(physics.env.cfg.assets.table_urdf).name}`  \n"
                    f"Size: {tuple(round(float(v), 3) for v in physics.table_size)} m  \n"
                    f"Table center: {tuple(round(float(v), 3) for v in physics.table_position)} m  \n"
                    f"Tabletop: {top:.3f} m world / {top - root[2]:.3f} m above robot base.  \n"
                    + ("Training limits, joint dynamics and nominal table placement. Friction follows recipe controls."
                       if active_profile == "training" else
                       "Real limits and identified dynamics; this differs from the policy's training setup.")
                )
            else:
                setup_description.content = (
                    f"**{next(k for k, v in setup_labels.items() if v == active_profile)}** · Static robot  \n"
                    f"Table size: {tuple(float(v) for v in physics.table_size)} m  \n"
                    f"Table center: {tuple(float(v) for v in physics.table_position)} m  \n"
                    f"Tabletop: {top:.3f} m world / {top - root[2]:.3f} m above robot base."
                )
            build_meshes()

        active = settings_from_controls()
        validate_selection(active)
        physics = build_physics(active)
        build_scene_visuals()
        pending_geometry = False
        geometry_error = ""
        total_steps = 0
        last_draw = 0.0
        print(f"Setup: {active_profile}; table size {tuple(physics.table_size)} m; "
              f"center {tuple(physics.table_position)} m; robot base {physics.robot_root_pos} m", flush=True)
        print(f"Rigid cloth viewer ready: http://localhost:{server.get_port()}", flush=True)
        while app.is_running():
            start = time.perf_counter()
            requested = set()
            while True:
                try:
                    requested.add(commands.get_nowait())
                except queue.Empty:
                    break
            pending_geometry |= "geometry" in requested
            if "rebuild" in requested:
                rebuild.disabled = True
                status.content = "Rebuilding the physics scene…"
                try:
                    candidate = settings_from_controls()
                    validate_selection(candidate)
                except (ValueError, OSError) as error:
                    geometry_error = f"Cannot apply geometry: {error}"
                    status.content = geometry_error
                    pending_geometry = True
                    rebuild.disabled = False
                    continue
                active = candidate
                physics.close()
                physics = None
                physics = build_physics(active)
                build_scene_visuals()
                pending_geometry = False
                geometry_error = ""
                rebuild.disabled = False
            elif "material" in requested:
                active = settings_from_controls(live_fields)
                physics.update_material(active)
            if "policy" in requested and hasattr(physics, "player"):
                physics.player.expl_coef = float(coefficient.value)
                physics.deterministic = deterministic.value
                physics.player.reset_rnn()
            if "reset" in requested:
                try:
                    candidate = settings_from_controls(reset_fields)
                    validate_reset_settings(candidate)
                    physics.reset(candidate)
                    active = candidate
                    geometry_error = ""
                except ValueError as error:
                    geometry_error = f"Cannot reset cloth: {error}"
                    status.content = geometry_error
            if running.value or "step" in requested:
                physics.step()
                total_steps += 1
            if start - last_draw >= 1 / 30 or requested:
                positions, quaternions = physics.read_poses()
                if not np.isfinite(positions).all() or not np.isfinite(quaternions).all():
                    raise RuntimeError("Cloth physics produced a non-finite pose.")
                with server.atomic():
                    for mesh, position, orientation in zip(meshes, positions, quaternions):
                        mesh.position, mesh.wxyz = position, orientation
                    if hasattr(physics, "player"):
                        robot_view.update_cfg(physics.robot_joint_positions(robot_view.get_actuated_joint_names()))
                        current, targets = physics.keypoints(scaled=keypoint_display.value == "Policy (Play)")
                        keypoints_view.points = current
                        targets_view.points = targets
                        errors_view.points = np.stack((current, targets), axis=1)
                try:
                    low, high = effective_placement_bounds(active, physics.table_size)
                    placement_info.content = (
                        f"Available placement area: X [{low[0]:.3f}, {high[0]:.3f}] m · "
                        f"Y [{low[1]:.3f}, {high[1]:.3f}] m, relative to table center."
                    )
                except ValueError as error:
                    placement_info.content = f"Sampling unavailable: {error}"
                info = physics.metrics()
                angles = ", ".join(f"{v:.1f}°" for v in info["hinge_angles_deg"])
                status.content = (
                    f"**{len(physics.spec.plates)} panels · {info['mass'] * 1000:.1f} g**  \n"
                    f"Time: {info['time']:.2f} s · Hinges: {angles}  \n"
                    + (f"Physical fold error: {info['fold_error'] * 100:.1f} cm · "
                       f"Successes: {info['successes']}/{info['episodes']} completed episodes  \n"
                       if "fold_error" in info else "")
                    + (f"Whole-robot clearance: {info['robot_clearance'] * 100:.1f} cm / 10 cm  \n"
                       if info.get("robot_clearance") is not None else "")
                    + (geometry_error or ("Geometry changes pending — press Apply." if pending_geometry else ""))
                )
                last_draw = start
            if args.steps and total_steps >= args.steps:
                break
            interval = physics.dt / float(speed.value) if running.value else 1 / 30
            time.sleep(max(0.0, interval - (time.perf_counter() - start)))
        print(f"Viewer stopped after {total_steps} viewer steps.", flush=True)
    except KeyboardInterrupt:
        pass
    except Exception:
        # Kit may swallow the pending traceback during app.close(). Print it first.
        import traceback
        traceback.print_exc()
        raise
    finally:
        try:
            if physics is not None:
                physics.close()
        finally:
            try:
                if server is not None:
                    server.stop()
            finally:
                app.close()


if __name__ == "__main__":
    main()
