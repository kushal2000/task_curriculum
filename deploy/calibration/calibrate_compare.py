"""Calibrate a fixed RealSense's extrinsics to the Flexiv base and check them, in one viser page.

viser shows the robot at its measured joints, the live point cloud placed by the extrinsics T_base_cam
(cam = the color optical frame: x right, y down, z forward), and a live preview of the URDF robot (orange)
ray cast from the camera over its color image. Both follow the extrinsics as you tune them:
  - Extrinsics: drag the gizmo, step x/y/z and roll/pitch/yaw, or "level to table" (turns the camera so
    the table plane in the cloud is parallel to the base x-y plane); "save" writes --extrinsics (the old
    one kept as <name>_prev.json). The "save as" box names the file (default: the --extrinsics name
    with today's date, e.g. left_camera_nov7).
  - Compare: "free drive" turns the arm's stiffness off (light damping, the robot's gravity compensation
    on) so you can move it by hand. "capture this pose" holds the arm, waits --settle_s for you to let go,
    and saves real | sim | overlay to compare/<folder>/<k>.png and all.png (the "compare folder" box,
    default <time>; a new name renames the folder at the next write); free drive then comes back on.

    python deploy/calibration/calibrate_compare.py --arm <ROBOT_SN> --extrinsics deploy/calibration/extrinsics/left_camera_nov7.json

A new camera: name a file that does not exist yet (and --serial if several cameras are plugged in). The
camera starts at --start_xyz looking at the arm: drag the gizmo to where the camera really is, "aim at
robot", "level to table", then fine-tune and "save", which creates the file.

The hand is not connected: the sim draws it at default_q, so its fingers will not match.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from deploy.calibration.realsense import RealSense  # noqa: E402
from deploy.calibration.robot_points import ARM_LINKS, RobotSurface, arm_tree, table_planes  # noqa: E402
from deploy.calibration.robot_proc import RobotProcess  # noqa: E402
from deploy.joint_reach.core import DEFAULT_SPEC, REPO, Spec  # noqa: E402

HERE = Path(__file__).resolve().parent
ROBOT_RGBA = (0.78, 0.78, 0.8, 1.0)  # one color, so the meshes take an opacity
DISPLAY_POINTS = 300_000
Q_REDRAW = np.radians(0.05)  # redraw the robot when a joint moved more (the readings jitter)


# ------------------------------------------------------------------------------------------- helpers


def pose_to_T(xyz, rpy_deg) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", rpy_deg, degrees=True).as_matrix()
    T[:3, 3] = xyz
    return T


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """T_base_cam at eye with the optical axis (z) on target and the image down (y) toward base -z."""
    z = (target - eye) / np.linalg.norm(target - eye)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x = x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-6 else np.array([1.0, 0.0, 0.0])
    T = np.eye(4)
    T[:3, :3] = np.stack([x, np.cross(z, x), z], axis=1)
    T[:3, 3] = eye
    return T


def to_viser(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y, z, w = Rotation.from_matrix(T[:3, :3]).as_quat()
    return np.array([w, x, y, z]), T[:3, 3].copy()


def from_viser(wxyz, pos) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([wxyz[1], wxyz[2], wxyz[3], wxyz[0]]).as_matrix()
    T[:3, 3] = pos
    return T


def subsample(n: int, k: int) -> np.ndarray:
    return np.arange(n) if n <= k else np.sort(np.random.default_rng(0).choice(n, k, replace=False))


def voxel_down(pts: np.ndarray, col: np.ndarray, voxel: float) -> tuple[np.ndarray, np.ndarray]:
    """One point per voxel (the mean position and color of the points in it)."""
    if voxel <= 0:
        return pts, col
    import open3d as o3d

    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.astype(np.float64)))
    pc.colors = o3d.utility.Vector3dVector(col.astype(np.float64) / 255)
    pc = pc.voxel_down_sample(voxel)
    return np.asarray(pc.points, np.float32), (np.asarray(pc.colors) * 255).astype(np.uint8)


def error_colors(d: np.ndarray, d_max: float) -> np.ndarray:
    """Green on the arm surface -> red at d_max and beyond; points far from the arm are grey."""
    a = np.clip(d / d_max, 0.0, 1.0)[:, None]
    c = (1 - a) * np.array([40, 220, 60]) + a * np.array([230, 40, 40])
    c[d > 3 * d_max] = (110, 110, 110)
    return c.astype(np.uint8)


class SimRenderer:
    """The URDF's visual meshes ray cast from a pinhole camera (open3d, no GPU): the arm light grey, the
    hand (l_* links) dark, Lambert shading with the light at the camera."""

    ARM, HAND = np.array([0.86, 0.86, 0.88]), np.array([0.18, 0.18, 0.2])

    def __init__(self, urdf: Path, joint_names: list[str]) -> None:
        import yourdfpy

        self.urdf = yourdfpy.URDF.load(str(urdf), build_scene_graph=True, load_meshes=True)
        self._order = [list(joint_names).index(j) for j in self.urdf.actuated_joint_names]
        g = self.urdf.scene.graph
        self.nodes = [(n, g.transforms.parents[n], self.urdf.scene.geometry[g[n][1]]) for n in g.nodes_geometry]
        self.lock = threading.Lock()  # the URDF's configuration is shared state

    def render(self, q: np.ndarray, T_base_cam: np.ndarray, intr: dict, scale: float = 1.0
               ) -> tuple[np.ndarray, np.ndarray]:
        """(H, W, 3) uint8 image and (H, W) bool robot mask at q (canonical joint order)."""
        import open3d as o3d

        scene, colors = o3d.t.geometry.RaycastingScene(), []
        with self.lock:
            self.urdf.update_cfg(np.asarray(q)[self._order])
            for node, link, mesh in self.nodes:
                T = self.urdf.scene.graph.get(node)[0]
                v = np.asarray(mesh.vertices) @ T[:3, :3].T + T[:3, 3]
                scene.add_triangles(o3d.core.Tensor(v.astype(np.float32)),
                                    o3d.core.Tensor(np.asarray(mesh.faces).astype(np.uint32)))
                colors.append(self.HAND if link.startswith("l_") else self.ARM)
        W, H = round(intr["width"] * scale), round(intr["height"] * scale)
        K = o3d.core.Tensor([[intr["fx"] * scale, 0, intr["cx"] * scale],
                             [0, intr["fy"] * scale, intr["cy"] * scale], [0, 0, 1]])
        rays = scene.create_rays_pinhole(K, o3d.core.Tensor(np.linalg.inv(T_base_cam)), W, H)
        hit = scene.cast_rays(rays)
        mask = np.isfinite(hit["t_hit"].numpy())
        d = rays.numpy()[..., 3:]
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        shade = 0.35 + 0.65 * np.abs(np.einsum("hwk,hwk->hw", hit["primitive_normals"].numpy(), d))
        img = np.full((H, W, 3), 245.0)
        cols = np.stack(colors)[np.where(mask, hit["geometry_ids"].numpy(), 0)]
        img[mask] = 255 * cols[mask] * shade[mask, None]
        return img.clip(0, 255).astype(np.uint8), mask


def overlay(real: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The sim robot tinted orange over the real image, with its outline."""
    from scipy.ndimage import binary_erosion

    out = real.astype(np.float32)
    out[mask] = 0.55 * out[mask] + 0.45 * np.array([255.0, 130.0, 0.0])
    out[mask & ~binary_erosion(mask, iterations=2)] = (255, 120, 0)
    return out.astype(np.uint8)


def side_by_side(real: np.ndarray, sim: np.ndarray, mask: np.ndarray, title: str) -> np.ndarray:
    """real | sim | sim over real, labeled."""
    from PIL import Image, ImageDraw, ImageFont

    H, W = real.shape[:2]
    gap = np.full((H, 8, 3), 255, np.uint8)
    img = Image.fromarray(np.concatenate([real, gap, sim, gap, overlay(real, mask)], axis=1))
    draw, font = ImageDraw.Draw(img), ImageFont.load_default(size=max(H // 24, 12))
    for x, text in ((16, "real"), (W + 24, "sim"), (2 * W + 32, "overlay")):
        box = draw.textbbox((x, 14), f"{text}  {title}", font=font)
        draw.rectangle((box[0] - 8, box[1] - 6, box[2] + 8, box[3] + 6), fill=(0, 0, 0))
        draw.text((x, 14), f"{text}  {title}", fill=(255, 255, 255), font=font)
    return np.asarray(img)


# ---------------------------------------------------------------------------------------------- app


class App:
    def __init__(self, args: argparse.Namespace) -> None:
        import viser
        from viser.extras import ViserUrdf

        self.args = args
        self.spec = Spec.load(args.spec)
        self.na = self.spec.n_arm
        urdf = REPO / json.loads(Path(args.spec).read_text())["robot_urdf"]
        self.surface = RobotSurface(urdf, self.spec.joint_names)  # for "color by distance to arm"
        self.renderer = SimRenderer(urdf, self.spec.joint_names)

        self.ext_path = Path(args.extrinsics)
        ext = json.loads(self.ext_path.read_text()) if self.ext_path.exists() else {}
        self.cam = RealSense(args.serial or ext.get("camera_serial"))
        self.cam.start()
        self.intr = self.cam.intrinsics.to_dict()
        print(f"[calib] {self.cam.name} {self.cam.serial}, extrinsics {self.ext_path}, arm {args.arm}", flush=True)
        self.robot = RobotProcess(self.spec, args.arm, args.arm_gain_scale, args.viewer, args.free_kd_scale)
        if ext:
            self.T = np.asarray(ext["T_base_cam"], float)
        else:
            print(f"[calib] no {self.ext_path} yet: the camera starts at {args.start_xyz} looking at the arm")
            self.T = look_at(np.array(args.start_xyz), self._arm_center())
        self.T_saved = self.T.copy()

        self.out = HERE / "compare" / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.poses: list[tuple[np.ndarray, np.ndarray]] = []  # (color image, q) per captured pose
        self.busy = threading.Lock()  # one capture / level / save at a time
        self.planes: list[np.ndarray] = []  # table candidates (camera frame) for "try the next plane"
        self.plane_k = 0
        self.table_pts = self.cloud = None
        self._dist_cache = None
        self._redraw = True  # set by the GUI, the cloud is redrawn by the main loop
        self._syncing = False  # set while the code moves the gizmo / numbers, so their callbacks skip

        # ---- scene
        s = self.server = viser.ViserServer(host="0.0.0.0", port=args.port)
        s.scene.set_up_direction("+z")
        s.scene.add_grid("/floor", width=3.0, height=3.0, cell_size=0.25)
        s.scene.add_frame("/base_axes", axes_length=0.3, axes_radius=0.008)
        self.robot_vis = ViserUrdf(s, urdf, root_node_name="/robot", mesh_color_override=ROBOT_RGBA)
        self.vis_order = [self.spec.joint_names.index(j) for j in self.robot_vis.get_actuated_joint_names()]
        wxyz, pos = to_viser(self.T)
        self.gizmo = s.scene.add_transform_controls("/cam", scale=0.25, wxyz=wxyz, position=pos)
        # the cloud is not a child of the gizmo (the browser would hit-test every point on each pointer
        # move): its own frame follows the gizmo instead
        self.cloud_frame = s.scene.add_frame("/cloud_frame", show_axes=False, wxyz=wxyz, position=pos)
        self.frustum = s.scene.add_camera_frustum(
            "/cam/frustum", fov=2 * np.arctan2(self.intr["height"] / 2, self.intr["fy"]),
            aspect=self.intr["width"] / self.intr["height"], scale=0.08, color=(230, 120, 20))
        self.gizmo.on_update(lambda _: self._follow_gizmo())
        self.gizmo.on_drag_end(lambda _: self._set_T(from_viser(self.gizmo.wxyz, self.gizmo.position)))

        self._build_gui()
        threading.Thread(target=self._preview_loop, daemon=True).start()
        print(f"[calib] http://localhost:{args.port}", flush=True)

    def _build_gui(self) -> None:
        g = self.server.gui
        self.hud = g.add_markdown("")
        self.status_md = g.add_markdown("")

        with g.add_folder("Compare"):
            self.free_on = g.add_checkbox("free drive", False, disabled=not self.robot.can_move,
                                          hint="no stiffness: move the arm by hand")
            self.free_on.on_update(lambda _: self._set_free())
            self.preview = g.add_image(np.zeros((90, 160, 3), np.uint8), label="sim over real", format="jpeg")
            self.preview_on = g.add_checkbox("live preview", True)
            self.out_name = g.add_text("compare folder", self.out.name,
                                       hint="compare/<this>; a rename applies at the next capture / re-render")
            g.add_button("capture this pose", color="green", disabled=not self.robot.can_move).on_click(
                lambda _: self._job(self._capture))
            g.add_button("delete last pose").on_click(lambda _: self._job(self._delete_last))
            g.add_button("re-render poses with current extrinsics").on_click(lambda _: self._job(self._write_poses))

        with g.add_folder("Extrinsics T_base_cam"):
            self.num = [g.add_number(k, 0.0, step=1e-4) for k in ("x [m]", "y [m]", "z [m]")]
            self.num += [g.add_number(k, 0.0, step=0.01) for k in ("roll [deg]", "pitch [deg]", "yaw [deg]")]
            for h in self.num:
                h.on_update(lambda _: self._from_numbers())
            self.step_t = g.add_number("move step [mm]", self.args.move_step_mm, min=0.1, max=200.0, step=0.1)
            self.step_r = g.add_number("rotate step [deg]", self.args.rotate_step_deg, min=0.01, max=45.0, step=0.01)
            self.step_axes = g.add_dropdown("step axes", ("base", "camera"), initial_value="base",
                                            hint="rotations are about the camera's center")
            for kind in ("move", "rotate"):
                for i, ax in enumerate("xyz"):
                    g.add_button_group(f"{kind} {ax}", ("-", "+")).on_click(
                        lambda e, kind=kind, i=i: self._step(kind, i, 1 if e.target.value == "+" else -1))
            g.add_button("level to table", color="blue",
                         hint="fit the table plane in the cloud and turn the camera (about its center) so the "
                              "plane is parallel to the base x-y plane").on_click(lambda _: self._job(lambda: self._level(0)))
            g.add_button("try the next plane", hint="if the cyan points are not the table").on_click(
                lambda _: self._job(lambda: self._level(+1)))
            self.table_on = g.add_checkbox("show table points", True)
            self.table_on.on_update(lambda _: self._show_table())
            g.add_button("aim at robot", hint="keep the camera position, point it at the arm").on_click(
                lambda _: self._set_T(look_at(self.T[:3, 3], self._arm_center())))
            g.add_button("reset to saved").on_click(lambda _: self._set_T(self.T_saved))
            self.save_name = g.add_text("save as", dated_name(self.ext_path.stem),
                                        hint=f"file name in {self.ext_path.parent}/ (.json added)")
            g.add_button("save extrinsics", color="red").on_click(lambda _: self._job(self._save))

        with g.add_folder("View", expand_by_default=False):
            self.live_on = g.add_checkbox("live cloud", True, hint=f"resend the cloud at {self.args.live_hz:g} Hz")
            self.voxel = g.add_slider("display voxel [mm]", 0.0, 20.0, 1.0, 3.0)
            self.point_size = g.add_slider("point size [mm]", 0.5, 8.0, 0.5, 2.0)
            self.crop = g.add_slider("max depth [m]", 0.3, 3.0, 0.05, 2.0)
            self.err_on = g.add_checkbox("color by distance to arm", False)
            self.err_max = g.add_slider("color scale [mm]", 2.0, 50.0, 1.0, 10.0)
            self.robot_alpha = g.add_slider("robot opacity", 0.0, 1.0, 0.05, 1.0)
            for h in (self.voxel, self.crop, self.err_on, self.err_max):
                h.on_update(lambda _: setattr(self, "_redraw", True))
            self.point_size.on_update(lambda _: self._set_point_size())
            self.robot_alpha.on_update(lambda _: self._set_robot_opacity())
        self._sync_numbers()

    # ------------------------------------------------------------------ extrinsics

    def _set_T(self, T: np.ndarray) -> None:
        self.T = T.copy()
        wxyz, pos = to_viser(T)
        self._syncing = True
        try:
            self.gizmo.wxyz, self.gizmo.position = wxyz, pos
            self.cloud_frame.wxyz, self.cloud_frame.position = wxyz, pos
            self._sync_numbers()
        finally:
            self._syncing = False
        self._redraw |= self.err_on.value  # the distance colors depend on T

    def _follow_gizmo(self) -> None:
        """While dragging only the cloud follows; T, the numbers and the colors update on release."""
        if not self._syncing:
            self.cloud_frame.wxyz, self.cloud_frame.position = self.gizmo.wxyz, self.gizmo.position

    def _from_numbers(self) -> None:
        if not self._syncing:
            self._set_T(pose_to_T([h.value for h in self.num[:3]], [h.value for h in self.num[3:]]))

    def _sync_numbers(self) -> None:
        rpy = Rotation.from_matrix(self.T[:3, :3]).as_euler("xyz", degrees=True)
        prev, self._syncing = self._syncing, True
        try:
            for h, v in zip(self.num, [*self.T[:3, 3], *rpy]):
                h.value = round(float(v), 5)
        finally:
            self._syncing = prev

    def _step(self, kind: str, axis: int, sign: int) -> None:
        T = self.T.copy()
        a = (np.eye(3) if self.step_axes.value == "base" else T[:3, :3])[:, axis]
        if kind == "move":
            T[:3, 3] += sign * self.step_t.value / 1000 * a
        else:
            T[:3, :3] = Rotation.from_rotvec(sign * np.radians(self.step_r.value) * a).as_matrix() @ T[:3, :3]
        self._set_T(T)

    def _level(self, step: int) -> None:
        """Turn the camera about its center so the table plane is parallel to the base x-y plane."""
        pts = self.cam.latest().points
        if step == 0 or not self.planes:
            self.planes = table_planes(pts, max_planes=5)
            if not self.planes:
                raise RuntimeError("no large plane in the cloud")
            # the table: the largest plane facing up in the image (camera y is down), else the most level one
            up = [k for k, p in enumerate(self.planes) if -p[1] > 0.5]
            z_cam = self.T[:3, :3].T @ np.array([0.0, 0.0, 1.0])
            self.plane_k = up[0] if up else int(np.argmax([p[:3] @ z_cam for p in self.planes]))
        else:
            self.plane_k = (self.plane_k + step) % len(self.planes)
        plane = self.planes[self.plane_k]
        fix = Rotation.align_vectors([[0.0, 0.0, 1.0]], [self.T[:3, :3] @ plane[:3]])[0]
        T = self.T.copy()
        T[:3, :3] = fix.as_matrix() @ T[:3, :3]
        self._set_T(T)
        on = np.abs(pts @ plane[:3] + plane[3]) < 0.01
        sel = subsample(int(on.sum()), 30_000)
        self.table_pts = self.server.scene.add_point_cloud(
            "/cloud_frame/table", pts[on][sel], (40, 200, 230), point_size=1.5 * self.point_size.value / 1000,
            precision="float32", point_shape="circle")
        self._show_table()
        self._status(f"leveled to plane {self.plane_k + 1}/{len(self.planes)} (cyan): turned "
                     f"{np.degrees(fix.magnitude()):.2f} deg. Not the table? try the next plane")

    def _arm_center(self) -> np.ndarray:
        return self.surface.sample(self.robot.q(), links=ARM_LINKS)[0].mean(0)

    def _show_table(self) -> None:
        if self.table_pts is not None:
            self.table_pts.visible = self.table_on.value

    def _save(self) -> None:
        name = self.save_name.value.strip().removesuffix(".json")
        if not name or "/" in name:
            raise ValueError(f"bad file name {name!r}")
        self.ext_path = self.ext_path.with_name(f"{name}.json")
        x, y, z, w = Rotation.from_matrix(self.T[:3, :3]).as_quat()
        out = {
            "T_base_cam": self.T.tolist(),
            "xyz": self.T[:3, 3].tolist(),
            "quat_wxyz": [w, x, y, z],
            "rpy_xyz_deg": Rotation.from_matrix(self.T[:3, :3]).as_euler("xyz", degrees=True).tolist(),
            "frames": "base: robot_urdf base_link; cam: RealSense color optical frame (x right, y down, z forward)",
            "camera_serial": self.cam.serial,
            "intrinsics": self.intr,
            "robot_urdf": json.loads(Path(self.args.spec).read_text())["robot_urdf"],
            "time": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
        self.ext_path.parent.mkdir(parents=True, exist_ok=True)
        if self.ext_path.exists():  # keep the last one
            self.ext_path.rename(self.ext_path.with_name(f"{self.ext_path.stem}_prev.json"))
        self.ext_path.write_text(json.dumps(out, indent=1))
        self.T_saved = self.T.copy()
        self._status(f"saved `{self.ext_path}`")

    # ------------------------------------------------------------------ free drive and poses

    def _set_free(self) -> None:
        """Apply the checkbox, unless a job holds the arm: a capture applies the checkbox when it ends."""
        if not self.busy.acquire(blocking=False):
            return
        try:
            if self.free_on.value != self.robot.free:
                self.robot.set_free(self.free_on.value)
        finally:
            self.busy.release()

    def _capture(self) -> None:
        try:
            self.robot.set_free(False)
            self._status(f"holding: let go of the arm and step out of the view ({self.args.settle_s:.1f} s)")
            time.sleep(self.args.settle_s)
            q0, t0 = self.robot.q(), self.cam.latest().t
            while (f := self.cam.latest()).t == t0:  # a frame taken after q0
                time.sleep(0.005)
            q1 = self.robot.q()
            if np.abs(q1 - q0).max() > np.radians(0.2):
                raise RuntimeError("the arm moved during the capture, try again")
            self.poses.append((f.image, 0.5 * (q0 + q1)))
            self._write_poses()
        finally:
            if self.free_on.value:
                self.robot.set_free(True)

    def _sync_out(self) -> None:
        """Rename the compare folder to what the "compare folder" box says."""
        name = self.out_name.value.strip()
        if not name or "/" in name:
            raise ValueError(f"bad folder name {name!r}")
        new = self.out.with_name(name)
        if new == self.out:
            return
        if new.exists():
            raise FileExistsError(f"`{new}` exists")
        if self.out.exists():
            self.out.rename(new)
        self.out = new

    def _delete_last(self) -> None:
        self._sync_out()
        if self.poses:
            self.poses.pop()
            (self.out / f"{len(self.poses)}.png").unlink(missing_ok=True)
            self._write_poses()

    def _write_poses(self) -> None:
        """<out>/<k>.png per pose and all.png, rendered with the current extrinsics."""
        from PIL import Image

        self._sync_out()
        self.out.mkdir(parents=True, exist_ok=True)
        rows = [side_by_side(img, *self.renderer.render(q, self.T, self.intr), f"pose {k}")
                for k, (img, q) in enumerate(self.poses)]
        for k, row in enumerate(rows):
            Image.fromarray(row).save(self.out / f"{k}.png")
        if rows:
            Image.fromarray(np.concatenate(rows, axis=0)).save(self.out / "all.png")
        self._status(f"{len(rows)} poses -> `{self.out}`")

    # ------------------------------------------------------------------ display

    def _set_point_size(self) -> None:
        if self.cloud is not None:
            self.cloud.point_size = self.point_size.value / 1000

    def _set_robot_opacity(self) -> None:
        for h in self.robot_vis._meshes:  # ViserUrdf has no public handle list
            h.opacity = float(self.robot_alpha.value)

    def _draw_cloud(self) -> None:
        f = self.cam.latest()
        keep = f.points[:, 2] < self.crop.value
        pts, col = f.points[keep], f.colors[keep]
        sel = subsample(len(pts), 2 * DISPLAY_POINTS)  # the voxel grid on a full cloud is slow
        pts, col = voxel_down(pts[sel], col[sel], self.voxel.value / 1000)
        sel = subsample(len(pts), DISPLAY_POINTS)
        pts, col = pts[sel], col[sel]
        if self.err_on.value:
            col = error_colors(self._distance_to_arm(pts @ self.T[:3, :3].T + self.T[:3, 3]), self.err_max.value / 1000)
        self.cloud = self.server.scene.add_point_cloud("/cloud_frame/cloud", pts, col, precision="float32",
                                                       point_size=self.point_size.value / 1000, point_shape="circle")
        self.frustum.image = f.image[::8, ::8]

    def _distance_to_arm(self, pts_base: np.ndarray) -> np.ndarray:
        q = self.robot.q()
        if self._dist_cache is None or np.abs(self._dist_cache[0] - q).max() > 1e-4:
            self._dist_cache = (q, arm_tree(self.surface, q))
        return self._dist_cache[1].query(pts_base)[0]

    def _preview_loop(self) -> None:
        """The sim robot over the live color image, with the current extrinsics."""
        s = self.args.preview_scale
        step = round(1 / s)
        while True:
            t = time.time()
            if self.preview_on.value:
                try:
                    _, mask = self.renderer.render(self.robot.q(), self.T, self.intr, scale=s)
                    real = self.cam.latest().image[::step, ::step]
                    h, w = min(real.shape[0], mask.shape[0]), min(real.shape[1], mask.shape[1])
                    self.preview.image = overlay(real[:h, :w], mask[:h, :w])
                except Exception as e:  # noqa: BLE001
                    print(f"[calib] preview: {e}", flush=True)
            time.sleep(max(0.0, 1 / self.args.preview_hz - (time.time() - t)))

    # ------------------------------------------------------------------ jobs and main loop

    def _job(self, fn) -> None:
        """Run fn in a worker thread, one at a time."""
        if not self.busy.acquire(blocking=False):
            self._status("busy, wait for the current job")
            return

        def run():
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                self._status(f"**error:** {e}")
            finally:
                self.busy.release()

        threading.Thread(target=run, daemon=True).start()

    def _status(self, msg: str) -> None:
        print(f"[calib] {msg}", flush=True)
        self.status_md.content = msg

    def run(self) -> None:
        q_drawn, hud, t_cloud = None, "", 0.0
        try:
            while True:
                q = self.robot.q()
                if q_drawn is None or np.abs(q - q_drawn).max() > Q_REDRAW:  # only send what changed
                    self.robot_vis.update_cfg(q[self.vis_order])
                    q_drawn = q
                if self._redraw or (self.live_on.value and time.time() - t_cloud > 1 / self.args.live_hz):
                    self._redraw, t_cloud = False, time.time()
                    self._draw_cloud()
                d = np.linalg.norm(self.T[:3, 3] - self.T_saved[:3, 3]) * 1000
                a = np.degrees(Rotation.from_matrix(self.T[:3, :3] @ self.T_saved[:3, :3].T).magnitude())
                text = (f"camera {self.cam.serial} · {'**FREE DRIVE**' if self.robot.free else 'arm holding'} · "
                        f"{len(self.poses)} poses · {d:.1f} mm / {a:.2f} deg from saved")
                if text != hud:
                    self.hud.content = hud = text
                time.sleep(1 / 30)
        except KeyboardInterrupt:
            pass
        finally:
            self.robot.close()
            self.cam.stop()


def dated_name(stem: str) -> str:
    """left_camera_nov1 -> left_camera_<today>, e.g. left_camera_nov7 (a trailing _prev is dropped too)."""
    base = re.sub(r"(_[a-z]{3}\d{1,2})?(_prev)?$", "", stem)
    today = datetime.date.today()
    return f"{base}_{today.strftime('%b').lower()}{today.day}"


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, help="robot serial, 'mujoco', or 'none' (no free drive / captures)")
    p.add_argument("--extrinsics", required=True,
                   help="extrinsics/<name>.json: the start pose (if it exists) and where save writes")
    p.add_argument("--serial", default=None,
                   help="RealSense serial (default: the extrinsics' camera_serial, else the only one connected)")
    p.add_argument("--start_xyz", type=float, nargs=3, default=[0.0, -1.6, 1.2],
                   help="base-frame camera position [m] to start from when --extrinsics does not exist yet")
    p.add_argument("--spec", default=str(DEFAULT_SPEC))
    p.add_argument("--arm_gain_scale", type=float, default=1.0, help="x the sim's arm kp and kd when holding")
    p.add_argument("--free_kd_scale", type=float, default=0.2,
                   help="free drive damping = this x the sim's arm kd (more: heavier to move, slower to drift)")
    p.add_argument("--settle_s", type=float, default=2.0, help="hold this long before a capture")
    p.add_argument("--live_hz", type=float, default=2.0, help="live cloud refresh rate")
    p.add_argument("--preview_hz", type=float, default=4.0, help="sim-over-real preview refresh rate")
    p.add_argument("--preview_scale", type=float, default=0.5, help="preview resolution x the camera's")
    p.add_argument("--move_step_mm", type=float, default=5.0, help="initial move x/y/z step")
    p.add_argument("--rotate_step_deg", type=float, default=0.5, help="initial rotate step")
    p.add_argument("--viewer", action="store_true", help="MuJoCo viewer for --arm mujoco")
    p.add_argument("--port", type=int, default=8091)
    return p.parse_args()


if __name__ == "__main__":
    App(_parse()).run()
