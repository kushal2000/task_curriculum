"""Inspect rigid cloth approximations in Isaac Sim through a small Viser interface.

Run: .venv/isaacsim/bin/python isaacsimenvs/scripts/visualization/viser_rigidCloth_viewer.py
Open http://localhost:8080 (forward that port over SSH for a remote machine).
The robot/table are static colliders. Only the 2/4/8-panel cloth is simulated.
Geometry and mass changes rebuild the scene; hinge parameters and friction update live.
"""

from __future__ import annotations

import gc
import math
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch

from isaacsimenvs import REPO_ROOT
from isaacsimenvs.tasks.cloth.backends.robot import ROBOT, ROOT_POS, URDF

TABLE_SIZE = (0.6, 0.4, 0.3)
TABLE_POSITION = (0.0, 0.0, 0.38)


@dataclass(frozen=True)
class Settings:
    panels: int = 2
    length: float = 0.2
    width: float = 0.2
    thickness: float = 0.004
    density: float = 2.0
    stiffness: float = 0.0
    damping: float = 0.05
    armature: float = 0.001
    friction: float = 1.0
    table_friction: float = 0.5
    drop_height: float = 0.06
    bend_deg: float = 90.0
    x: float = 0.0
    y: float = 0.0


def load_static_robot():
    import trimesh
    import yourdfpy

    path = REPO_ROOT / URDF
    robot = yourdfpy.URDF.load(
        path,
        build_scene_graph=True,
        build_collision_scene_graph=True,
        load_meshes=True,
        load_collision_meshes=True,
        filename_handler=partial(yourdfpy.filename_handler_magic, dir=path.parent),
    )
    joints = {name: float(ROBOT.arm_default_joint_pos.get(name, 0.0)) for name in robot.actuated_joint_names}
    robot.update_cfg(joints)
    root = trimesh.transformations.translation_matrix(ROOT_POS)
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
    Returned poses are panel CENTERS and wxyz orientations, not the shared rest
    frame body origins. Density is areal kg/m², matching the cloth task.
    """

    def __init__(self, settings, robot_collision_meshes, table_size, table_position, device="cuda:0", dt=1 / 120):
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import Articulation, ArticulationCfg
        from isaaclab.sim import SimulationCfg, SimulationContext
        from isaaclab.sim.utils import create_new_stage
        from pxr import Gf, UsdGeom, UsdPhysics, UsdShade, Vt

        self.sim = self.cloth = None
        self.dt, self.time = dt, 0.0
        self.table_size = np.asarray(table_size, dtype=float)
        self.table_position = np.asarray(table_position, dtype=float)
        self.table_top = self.table_position[2] + self.table_size[2] / 2
        self.panels = int(settings.panels)
        self.dimensions = (float(settings.length), float(settings.width), float(settings.thickness))
        if self.panels not in (2, 4, 8) or min(self.dimensions) <= 0 or settings.density <= 0:
            raise ValueError("Use 2, 4, or 8 panels, positive dimensions and areal density.")
        try:
            create_new_stage()
            self.sim = SimulationContext(SimulationCfg(dt=dt, device=device, render_interval=1))
            self.stage = self.sim.stage
            UsdGeom.Xform.Define(self.stage, "/World")
            self.materials = {}
            for name, friction in [("cloth", settings.friction), ("table", settings.table_friction), ("robot", 0.5)]:
                mat = UsdShade.Material.Define(self.stage, f"/World/Materials/{name}")
                api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
                api.CreateStaticFrictionAttr(float(friction))
                api.CreateDynamicFrictionAttr(float(friction))
                api.CreateRestitutionAttr(0.0)
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
            self._build_strip(settings)
            self.cloth = Articulation(
                ArticulationCfg(
                    prim_path="/World/Cloth",
                    spawn=None,
                    actuators={
                        "hinges": ImplicitActuatorCfg(
                            joint_names_expr=["hinge_.*"],
                            stiffness=float(settings.stiffness),
                            damping=float(settings.damping),
                            armature=float(settings.armature),
                            effort_limit_sim=100.0,
                        )
                    },
                )
            )
            self.sim.reset()
            self.body_ids = [self.cloth.body_names.index(f"panel_{i}") for i in range(self.panels)]
            self.joint_ids = [self.cloth.joint_names.index(f"hinge_{i}") for i in range(self.panels - 1)]
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

    def _build_strip(self, s):
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

        length, width, thickness = self.dimensions
        dx = length / self.panels
        UsdGeom.Xform.Define(self.stage, "/World/Cloth")
        # Body frames coincide at rest; each cube and mass COM carry their offset.
        for i in range(self.panels):
            center = (-length / 2 + (i + 0.5) * dx, 0.0, 0.0)
            body = UsdGeom.Xform.Define(self.stage, f"/World/Cloth/panel_{i}").GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(body)
            physx = PhysxSchema.PhysxRigidBodyAPI.Apply(body)
            physx.CreateMaxDepenetrationVelocityAttr(1.0)
            mass = float(s.density) * dx * width
            inertia = mass / 12 * np.array([width**2 + thickness**2, dx**2 + thickness**2, dx**2 + width**2])
            props = UsdPhysics.MassAPI.Apply(body)
            props.CreateMassAttr(mass)
            props.CreateCenterOfMassAttr(Gf.Vec3f(*center))
            props.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, inertia)))
            props.CreatePrincipalAxesAttr(Gf.Quatf(1.0))
            if i == 0:
                UsdPhysics.ArticulationRootAPI.Apply(body)
                art = PhysxSchema.PhysxArticulationAPI.Apply(body)
                art.CreateEnabledSelfCollisionsAttr(True)
                art.CreateSolverPositionIterationCountAttr(16)
                art.CreateSolverVelocityIterationCountAttr(4)
            cube = UsdGeom.Cube.Define(self.stage, f"/World/Cloth/panel_{i}/cube")
            cube.CreateSizeAttr(1.0)
            cube.AddTranslateOp().Set(Gf.Vec3d(*center))
            cube.AddScaleOp().Set(Gf.Vec3f(dx, width, thickness))
            self._collision(cube.GetPrim(), "cloth")
        for i in range(self.panels - 1):
            joint = UsdPhysics.RevoluteJoint.Define(self.stage, f"/World/Cloth/hinge_{i}")
            joint.CreateBody0Rel().SetTargets([f"/World/Cloth/panel_{i}"])
            joint.CreateBody1Rel().SetTargets([f"/World/Cloth/panel_{i + 1}"])
            joint.CreateAxisAttr("X")
            rotation = Gf.Quatf(math.sqrt(0.5), Gf.Vec3f(0, 0, -math.sqrt(0.5)))
            anchor = Gf.Vec3f(-length / 2 + (i + 1) * dx, 0, thickness / 2)
            joint.CreateLocalRot0Attr(rotation)
            joint.CreateLocalRot1Attr(rotation)
            joint.CreateLocalPos0Attr(anchor)
            joint.CreateLocalPos1Attr(anchor)
            joint.CreateLowerLimitAttr(-90.0)
            joint.CreateUpperLimitAttr(180.0)
            joint.CreateCollisionEnabledAttr(False)
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(0.0)
            drive.CreateDampingAttr(0.0)

    def reset(self, settings):
        """Drop with only the central hinge folded; all hinges spring toward flat."""
        self.update_material(settings)
        root = self.cloth.data.default_root_state.clone()
        root[:, :3] = torch.tensor(
            [settings.x, settings.y, self.table_top + settings.drop_height + settings.thickness / 2],
            device=self.cloth.device,
        )
        root[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.cloth.device)
        root[:, 7:] = 0
        q = torch.zeros_like(self.cloth.data.default_joint_pos)
        q[:, self.joint_ids[self.panels // 2 - 1]] = math.radians(settings.bend_deg)
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
        for name in ("stiffness", "damping", "armature"):
            getattr(self.cloth, f"write_joint_{name}_to_sim")(float(getattr(settings, name)))
        for name, friction in [("cloth", settings.friction), ("table", settings.table_friction)]:
            api = self.materials[name][1]
            api.GetStaticFrictionAttr().Set(float(friction))
            api.GetDynamicFrictionAttr().Set(float(friction))
        view = self.cloth.root_physx_view
        mats = view.get_material_properties()
        mats[..., :2] = float(settings.friction)
        mats[..., 2] = 0
        view.set_material_properties(mats, torch.tensor([0], dtype=torch.int32, device="cpu"))

    def step(self):
        self.cloth.write_data_to_sim()
        self.sim.step(render=False)
        self.cloth.update(self.dt)
        self.time += self.dt

    def read_poses(self):
        pose = self.cloth.data.body_com_pose_w[0, self.body_ids].detach().cpu().numpy()
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
    parser.add_argument("--panels", type=int, choices=(2, 4, 8), default=2)
    parser.add_argument("--steps", type=int, default=0, help="Exit after N physics steps; 0 runs interactively.")
    args = parser.parse_args()
    if args.steps < 0:
        parser.error("--steps must be nonnegative")

    import viser
    from isaaclab.app import AppLauncher
    from viser.extras import ViserUrdf

    app = AppLauncher(headless=True, device=args.device).app
    server, physics = None, None
    try:
        robot, joint_values, collisions = load_static_robot()
        server = viser.ViserServer(host=args.host, port=args.port)
        server.scene.set_up_direction("+z")
        server.scene.add_grid("/ground", width=3.0, height=3.0, cell_size=0.1)
        server.scene.add_frame("/robot", position=ROOT_POS, show_axes=False)
        robot_view = ViserUrdf(server, robot, root_node_name="/robot")
        robot_view.update_cfg(np.asarray([joint_values[n] for n in robot_view.get_actuated_joint_names()]))
        server.scene.add_box("/table", dimensions=TABLE_SIZE, position=TABLE_POSITION, color=(209, 143, 89))

        @server.on_client_connect
        def set_camera(client):
            client.camera.position = (0.85, -0.85, 1.05)
            client.camera.look_at = (0.0, 0.08, 0.58)
            client.camera.up_direction = (0.0, 0.0, 1.0)

        controls = {}
        commands = queue.SimpleQueue()
        active = Settings(panels=args.panels)
        server.gui.add_markdown(
            "## Rigid cloth\nStatic robot and table. Gravity and contacts are simulated in Isaac Sim."
        )
        with server.gui.add_folder("Geometry — apply to rebuild"):
            controls["panels"] = server.gui.add_dropdown(
                "Panels along length", options=("2", "4", "8"), initial_value=str(active.panels)
            )
            for name, label, low, high, step in (
                ("length", "Length (m)", 0.08, 0.4, 0.01),
                ("width", "Width (m)", 0.08, 0.35, 0.01),
                ("thickness", "Thickness (m)", 0.001, 0.02, 0.001),
                ("density", "Areal density (kg/m²)", 0.05, 5.0, 0.05),
            ):
                controls[name] = server.gui.add_slider(
                    label, min=low, max=high, step=step, initial_value=getattr(active, name)
                )
            rebuild = server.gui.add_button("Apply geometry and restart")
        live_fields = ("stiffness", "damping", "armature", "friction")
        with server.gui.add_folder("Physics — updates live"):
            for name, label, low, high, step in (
                ("stiffness", "Hinge stiffness (N m/rad)", 0.0, 0.1, 0.0001),
                ("damping", "Hinge damping (N m s/rad)", 0.0, 0.2, 0.0001),
                ("armature", "Hinge inertia (kg m²)", 0.0, 0.01, 0.00001),
                ("friction", "Cloth friction", 0.0, 2.0, 0.01),
            ):
                handle = controls[name] = server.gui.add_slider(
                    label, min=low, max=high, step=step, initial_value=getattr(active, name)
                )
                handle.on_update(lambda _: commands.put("material"))
        reset_fields = ("x", "y", "drop_height", "bend_deg")
        with server.gui.add_folder("Initial state — press reset"):
            for name, label, low, high, step in (
                ("x", "Center X (m)", -0.35, 0.35, 0.01),
                ("y", "Center Y (m)", -0.25, 0.25, 0.01),
                ("drop_height", "Drop above table (m)", 0.002, 0.4, 0.002),
                ("bend_deg", "Center fold (degrees)", 0.0, 175.0, 1.0),
            ):
                controls[name] = server.gui.add_slider(
                    label, min=low, max=high, step=step, initial_value=getattr(active, name)
                )
            reset = server.gui.add_button("Reset cloth")
        running = server.gui.add_checkbox("Run simulation", initial_value=bool(args.steps))
        step_button = server.gui.add_button("Single step")
        speed = server.gui.add_slider("Playback speed", min=0.1, max=2.0, step=0.1, initial_value=1.0)
        status = server.gui.add_markdown("Building physics scene…")
        rebuild.on_click(lambda _: commands.put("rebuild"))
        reset.on_click(lambda _: commands.put("reset"))
        step_button.on_click(lambda _: commands.put("step"))
        geometry_fields = ("panels", "length", "width", "thickness", "density")
        for name in geometry_fields:
            controls[name].on_update(lambda _: commands.put("geometry"))

        def settings_from_controls():
            return replace(
                active,
                **{
                    name: int(handle.value) if name == "panels" else float(handle.value)
                    for name, handle in controls.items()
                },
            )

        boxes = []

        def build_boxes():
            for box in boxes:
                box.remove()
            boxes.clear()
            for i in range(active.panels):
                color = (74, 143, 194) if i < active.panels // 2 else (242, 173, 56)
                boxes.append(
                    server.scene.add_box(
                        f"/cloth/panel_{i}",
                        color=color,
                        dimensions=(active.length / active.panels, active.width, active.thickness),
                    )
                )

        physics = Physics(active, collisions, TABLE_SIZE, TABLE_POSITION, device=args.device)
        build_boxes()
        pending_geometry = False
        total_steps = 0
        last_draw = 0.0
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
                active = settings_from_controls()
                physics.close()
                physics = None
                physics = Physics(active, collisions, TABLE_SIZE, TABLE_POSITION, device=args.device)
                build_boxes()
                pending_geometry = False
                rebuild.disabled = False
            elif "material" in requested:
                active = replace(active, **{name: float(controls[name].value) for name in live_fields})
                physics.update_material(active)
            if "reset" in requested:
                active = replace(active, **{name: float(controls[name].value) for name in reset_fields})
                physics.reset(active)
            if running.value or "step" in requested:
                physics.step()
                total_steps += 1
            if start - last_draw >= 1 / 30 or requested:
                positions, quaternions = physics.read_poses()
                if not np.isfinite(positions).all() or not np.isfinite(quaternions).all():
                    raise RuntimeError("Cloth physics produced a non-finite pose.")
                with server.atomic():
                    for box, position, orientation in zip(boxes, positions, quaternions):
                        box.position, box.wxyz = position, orientation
                info = physics.metrics()
                angles = ", ".join(f"{v:.1f}°" for v in info["hinge_angles_deg"])
                status.content = (
                    f"**{active.panels} panels · {info['mass'] * 1000:.1f} g**  \n"
                    f"Time: {info['time']:.2f} s · Hinges: {angles}  \n"
                    + ("Geometry changes pending — press Apply." if pending_geometry else "")
                )
                last_draw = start
            if args.steps and total_steps >= args.steps:
                break
            interval = physics.dt / float(speed.value) if running.value else 1 / 30
            time.sleep(max(0.0, interval - (time.perf_counter() - start)))
        print(f"Viewer stopped after {total_steps} physics steps.", flush=True)
    except KeyboardInterrupt:
        pass
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
