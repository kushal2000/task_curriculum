"""Side-by-side viser viewer: Kuka iiwa14 + Sharpa vs Flexiv Rizon 4s + Wuji Hand 2.

Each robot stands where the RL env puts it, ``ROBOT_BASE_POS`` behind its own copy of the env's
table, with joint sliders for arm and hand. The Flexiv's base offset and the Wuji's flange mount
are live sliders too, so the mount can be tuned against the Kuka setup by eye; separation 0
overlays the two robots on one table.

Both hands are compared in one canonical hand frame (hand_frames.py), built from their knuckles.
The readout gives the Wuji's frame relative to the Sharpa's; "Match Kuka hand" solves the Flexiv
arm's IK to zero it, so whatever residual is left is what the two kinematic chains cannot match.

"Write composed URDF" saves the arm + hand with the current mount via ``flexiv_wuji.compose_urdf``
and prints the values to paste into ``flexiv_wuji.py``.

    .venv_isaaclab3/bin/python scripts/flexiv_wuji/viser_compare.py [--port 8080]

On the cluster, forward the port: ``ssh -L 8080:localhost:8080 <node>``, then open
http://localhost:8080.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import viser
import yourdfpy
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from viser.extras import ViserUrdf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flexiv_wuji as fw  # noqa: E402
from hand_frames import HAND_KEYPOINTS, hand_frame, hand_points  # noqa: E402

TABLE_SIZE = (0.475, 0.4, 0.3)  # table_narrow.urdf's box
TABLE_COLOR = (209, 143, 89)

def make_tf(pos=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0)) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()  # URDF rpy: fixed-axis x, y, z
    T[:3, 3] = pos
    return T


def wxyz(T: np.ndarray) -> np.ndarray:
    x, y, z, w = Rotation.from_matrix(T[:3, :3]).as_quat()
    return np.array([w, x, y, z])


class JointSliders:
    """One slider per actuated joint of ``urdf``, in a GUI folder."""

    def __init__(self, server, label, urdf: yourdfpy.URDF, defaults, on_change, expand):
        self.names = list(urdf.actuated_joint_names)
        self.defaults = {n: float(defaults.get(n, 0.0)) for n in self.names}
        self.sliders = {}
        with server.gui.add_folder(label, expand_by_default=expand):
            for name in self.names:
                lo, hi = urdf.joint_map[name].limit.lower, urdf.joint_map[name].limit.upper
                init = float(np.clip(self.defaults[name], lo, hi))
                slider = server.gui.add_slider(name, min=lo, max=hi, step=1e-3, initial_value=init)
                slider.on_update(lambda _: on_change())
                self.sliders[name] = slider

    def q(self) -> np.ndarray:
        return np.array([self.sliders[n].value for n in self.names])

    def set(self, values: dict[str, float]) -> None:
        for name, value in values.items():
            self.sliders[name].value = float(value)

    def reset(self) -> None:
        self.set(self.defaults)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # the printed values matter when piped to a log

    kuka = yourdfpy.URDF.load(str(fw.KUKA_SHARPA_URDF))
    arm = yourdfpy.URDF.load(str(fw._require(fw.FLEXIV_URDF)))
    hand = yourdfpy.URDF.load(str(fw._require(fw.WUJI_URDF)))

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/ground", width=4.0, height=4.0, cell_size=0.1)

    # Scene graph: /<setup> sits at that setup's env origin (moved by the separation slider);
    # robot and table hang under it in env coordinates, as the env places them.
    roots = {name: server.scene.add_frame(f"/{name}", show_axes=False) for name in HAND_KEYPOINTS}
    tables = {
        name: server.scene.add_box(
            f"/{name}/table", color=TABLE_COLOR, dimensions=TABLE_SIZE,
            position=(0.0, 0.0, fw.TABLE_CENTER_Z),
        )
        for name in HAND_KEYPOINTS
    }
    server.scene.add_frame("/kuka_sharpa/robot", show_axes=False, position=fw.ROBOT_BASE_POS)
    flexiv_base = server.scene.add_frame("/flexiv_wuji/robot", show_axes=False)
    wuji_root = server.scene.add_frame("/flexiv_wuji/robot/wuji", show_axes=False)
    viser_kuka = ViserUrdf(server, kuka, root_node_name="/kuka_sharpa/robot")
    viser_arm = ViserUrdf(server, arm, root_node_name="/flexiv_wuji/robot")
    viser_hand = ViserUrdf(server, hand, root_node_name="/flexiv_wuji/robot/wuji")
    frame_axes = {
        name: server.scene.add_frame(f"/{name}/hand_frame", axes_length=0.08, axes_radius=0.003)
        for name in HAND_KEYPOINTS
    }

    def refresh() -> None:
        if not ready:
            return
        sep = layout_sep.value
        for name, sign in (("kuka_sharpa", -0.5), ("flexiv_wuji", 0.5)):
            roots[name].position = (sign * sep, 0.0, 0.0)
            tables[name].visible = show_tables.value and (
                sep > 0 or name == "kuka_sharpa"
            )
            frame_axes[name].visible = show_frames.value

        kuka.update_cfg(kuka_q.q())
        viser_kuka.update_cfg(kuka_q.q())
        viser_arm.update_cfg(flexiv_q.q())
        viser_hand.update_cfg(wuji_q.q())

        T_base = base_tf()
        T_flange = arm.get_transform(fw.FLANGE_LINK, arm.base_link)
        flexiv_base.position = T_base[:3, 3]
        flexiv_base.wxyz = wxyz(T_base)
        T_hand_root = T_flange @ mount_tf()
        wuji_root.position = T_hand_root[:3, 3]
        wuji_root.wxyz = wxyz(T_hand_root)

        # Hand frames in env coordinates (origin = table centre, floor level).
        T_kuka = make_tf(fw.ROBOT_BASE_POS)
        H_kuka = hand_frame(hand_points(kuka, HAND_KEYPOINTS["kuka_sharpa"], T_kuka))
        H_wuji = hand_frame(hand_points(hand, HAND_KEYPOINTS["flexiv_wuji"], T_base @ T_hand_root))
        for name, H in (("kuka_sharpa", H_kuka), ("flexiv_wuji", H_wuji)):
            frame_axes[name].position = H[:3, 3]
            frame_axes[name].wxyz = wxyz(H)

        rel = np.linalg.inv(H_kuka) @ H_wuji
        dpos_mm = rel[:3, 3] * 1000
        dang_deg = np.degrees(Rotation.from_matrix(rel[:3, :3]).magnitude())
        table_top = np.array([0.0, 0.0, fw.TABLE_CENTER_Z + TABLE_SIZE[2] / 2])
        readout.content = "\n".join([
            "**Hand frame, relative to table-top centre (m)**",
            f"- Kuka+Sharpa: `{np.round(H_kuka[:3, 3] - table_top, 3)}`",
            f"- Flexiv+Wuji: `{np.round(H_wuji[:3, 3] - table_top, 3)}`",
            "",
            "**Wuji in the Sharpa hand frame**",
            f"- position: `{np.round(dpos_mm, 1)}` mm (|d| {np.linalg.norm(dpos_mm):.1f})",
            f"- rotation: {dang_deg:.1f} deg",
        ])

    def base_tf() -> np.ndarray:
        pos = np.array(fw.ROBOT_BASE_POS) + [s.value for s in base_xyz]
        return make_tf(pos, (0.0, 0.0, base_yaw.value))

    def mount_tf() -> np.ndarray:
        return make_tf([s.value for s in mount_xyz], [s.value for s in mount_rpy])

    ready = False
    with server.gui.add_folder("Layout"):
        layout_sep = server.gui.add_slider("separation (m)", 0.0, 2.0, 0.01, 1.2)
        show_tables = server.gui.add_checkbox("tables", True)
        show_frames = server.gui.add_checkbox("hand frames", True)
    readout = server.gui.add_markdown("")
    match_button = server.gui.add_button("Match Kuka hand (Flexiv IK)")
    reset_button = server.gui.add_button("Reset joints")
    write_button = server.gui.add_button("Write composed URDF")
    with server.gui.add_folder("Flexiv base offset", expand_by_default=False):
        base_xyz = [
            server.gui.add_slider(f"base {a} (m)", -0.5, 0.5, 0.005, 0.0) for a in "xyz"
        ]
        base_yaw = server.gui.add_slider("base yaw (rad)", -np.pi, np.pi, 0.01, 0.0)
    with server.gui.add_folder("Wuji mount (flange -> l_mount)", expand_by_default=False):
        mount_xyz = [
            server.gui.add_slider(f"mount {a} (m)", -0.1, 0.1, 0.001, v)
            for a, v in zip("xyz", fw.MOUNT_XYZ)
        ]
        mount_rpy = [
            server.gui.add_slider(f"mount {a} (rad)", -np.pi, np.pi, 0.01, v)
            for a, v in zip(("roll", "pitch", "yaw"), fw.MOUNT_RPY)
        ]
    kuka_q = JointSliders(server, "Kuka + Sharpa joints", kuka, fw.KUKA_DEFAULT_ARM_POS, refresh, False)
    flexiv_q = JointSliders(server, "Flexiv arm joints", arm, fw.FLEXIV_DEFAULT_ARM_POS, refresh, False)
    wuji_q = JointSliders(server, "Wuji hand joints", hand, {}, refresh, False)
    for handle in (layout_sep, show_tables, show_frames, base_yaw, *base_xyz, *mount_xyz, *mount_rpy):
        handle.on_update(lambda _: refresh())

    @match_button.on_click
    def _(_) -> None:
        """Flexiv arm IK: put the Wuji hand frame on the Sharpa's, in env coordinates."""
        T_kuka = make_tf(fw.ROBOT_BASE_POS)
        target = hand_frame(hand_points(kuka, HAND_KEYPOINTS["kuka_sharpa"], T_kuka))
        # The hand's joints and mount are held fixed, so its frame is a constant offset from l_mount.
        T_flange_hand = mount_tf() @ hand_frame(hand_points(hand, HAND_KEYPOINTS["flexiv_wuji"]))
        T_root_target = np.linalg.inv(base_tf()) @ target
        lo = np.array([arm.joint_map[n].limit.lower for n in flexiv_q.names])
        hi = np.array([arm.joint_map[n].limit.upper for n in flexiv_q.names])
        q0 = np.clip(flexiv_q.q(), lo, hi)

        def residual(q):
            arm.update_cfg(q)
            H = arm.get_transform(fw.FLANGE_LINK, arm.base_link) @ T_flange_hand
            rel = np.linalg.inv(T_root_target) @ H
            # 1 rad of rotation is weighted like 0.1 m of position; a light pull to the start
            # keeps redundant joints from wandering.
            return np.concatenate([
                rel[:3, 3], 0.1 * Rotation.from_matrix(rel[:3, :3]).as_rotvec(), 1e-3 * (q - q0),
            ])

        sol = least_squares(residual, q0, bounds=(lo, hi), x_scale=1.0, max_nfev=400)
        flexiv_q.set(dict(zip(flexiv_q.names, sol.x)))
        refresh()
        print("[match] flexiv q:", np.round(sol.x, 4).tolist(), f"cost {sol.cost:.2e}")

    @reset_button.on_click
    def _(_) -> None:
        for sliders in (kuka_q, flexiv_q, wuji_q):
            sliders.reset()
        refresh()

    @write_button.on_click
    def _(_) -> None:
        xyz = [round(s.value, 4) for s in mount_xyz]
        rpy = [round(s.value, 4) for s in mount_rpy]
        path = fw.compose_urdf(xyz, rpy)
        print(f"[write] {path.relative_to(fw.REPO_ROOT)}")
        print(f"MOUNT_XYZ = {tuple(xyz)}")
        print(f"MOUNT_RPY = {tuple(rpy)}")
        print(f"base offset xyz {[round(s.value, 4) for s in base_xyz]}, yaw {base_yaw.value:.4f}")
        print("FLEXIV_DEFAULT_ARM_POS =", {n: round(float(v), 4) for n, v in zip(flexiv_q.names, flexiv_q.q())})

    ready = True
    refresh()
    print(f"viser on http://localhost:{args.port}")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
