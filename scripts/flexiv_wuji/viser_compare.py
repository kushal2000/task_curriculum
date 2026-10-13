"""Side-by-side viser viewer: Kuka iiwa14 + Sharpa vs Flexiv Rizon 4s + Wuji Hand 2.

Each robot stands where the RL env puts it, ``ROBOT_BASE_POS`` behind its own copy of the env's
table, with joint sliders for each arm and each hand. Each hand also has one "hand action" slider
that sets every hand joint the way the policy's hand action does (-1 = lower limit, +1 = upper,
linearly in between), for opening and closing it in one go. The Flexiv's base offset and the Wuji's flange mount
are live sliders too, so the mount can be tuned against the Kuka setup by eye; separation 0
overlays the two robots on one table.

Both hands are compared in one canonical hand frame (hand_frames.py), built from their knuckles.
The readout gives the Wuji's frame relative to the Sharpa's; "Match Kuka hand" solves the Flexiv
arm's IK to zero it, so whatever residual is left is what the two kinematic chains cannot match.

The Wuji takes its joint limits from ``flexiv_wuji.HUMAN_LIKE_URDF`` (``--wuji vendor`` for the
vendor's), so its sliders cover those ranges.

Fingertips are marked as the Play task defines them (``RobotSpec`` in isaacsimenvs/tasks/play/
robots.py). A dropdown picks which point: the policy input, the pad point it observes (fingertip
body + ``fingertip_offset``), or the reward input, the body origin the fingertip-to-object reward
distance is measured from (obs_utils, no offset). The readout gives each point in its own hand's
canonical frame.

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

sys.path.insert(0, str(fw.REPO_ROOT / "isaacsimenvs" / "tasks" / "play"))
import robots  # noqa: E402  (stdlib-only; imported by path so Isaac Lab is not needed)

FINGERS = ("index", "middle", "ring", "thumb", "pinky")  # both specs list their tips in this order
#: "robots" dropdown: label -> which setups' robot meshes are drawn (fingertips, frames and tables
#: are not affected, so hiding a robot shows its fingertip points unobstructed).
ROBOT_DISPLAY = {
    "show both": {"kuka_sharpa", "flexiv_wuji"},
    "hide Kuka + Sharpa": {"flexiv_wuji"},
    "hide Flexiv + Wuji": {"kuka_sharpa"},
    "hide both": set(),
}
KUKA_COLOR, FLEXIV_COLOR = (0.62, 0.62, 0.62), (0.35, 0.55, 0.85)
TIP_POINTS = {  # dropdown label -> (use the spec's fingertip_offset, marker colour)
    "policy input (pad: body + offset)": (True, (230, 40, 160)),
    "reward input (body origin)": (False, (40, 200, 230)),
}

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
    """One slider per actuated joint of ``urdf`` (or the given subset), in a GUI folder.

    ``hand_action`` adds a slider on top that sets every joint to ``lower + (a + 1) / 2 *
    (upper - lower)``, the map the Play task applies to its hand actions.
    """

    def __init__(self, server, label, urdf: yourdfpy.URDF, defaults, on_change, expand,
                 names=None, hand_action=False):
        self.names = list(names if names is not None else urdf.actuated_joint_names)
        self.defaults = {n: float(defaults.get(n, 0.0)) for n in self.names}
        self.sliders = {}
        self.on_change = on_change
        limits = {n: (urdf.joint_map[n].limit.lower, urdf.joint_map[n].limit.upper) for n in self.names}
        self.action = None
        with server.gui.add_folder(label, expand_by_default=expand):
            if hand_action:
                action = self.action = server.gui.add_slider(
                    "hand action (all joints)", -1.0, 1.0, 0.01, 0.0)

                @action.on_update
                def _(_) -> None:
                    a = (action.value + 1.0) / 2.0
                    self.set({n: lo + a * (hi - lo) for n, (lo, hi) in limits.items()})

            for name in self.names:
                lo, hi = urdf.joint_map[name].limit.lower, urdf.joint_map[name].limit.upper
                init = float(np.clip(self.defaults[name], lo, hi))
                slider = server.gui.add_slider(name, min=lo, max=hi, step=1e-3, initial_value=init)
                slider.on_update(lambda _: self.on_change())
                self.sliders[name] = slider

    def q(self) -> np.ndarray:
        return np.array([self.sliders[n].value for n in self.names])

    def set(self, values: dict[str, float]) -> None:
        # Each slider's own update would redraw the scene; hold that off until all are set.
        on_change, self.on_change = self.on_change, lambda: None
        try:
            for name, value in values.items():
                self.sliders[name].value = float(value)
        finally:
            self.on_change = on_change
        self.on_change()

    def reset(self) -> None:
        self.set(self.defaults)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--wuji", choices=("human_like", "vendor"), default="human_like",
                        help="which Wuji joint limits the hand sliders use")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # the printed values matter when piped to a log

    kuka = yourdfpy.URDF.load(str(fw.KUKA_SHARPA_URDF))
    arm = yourdfpy.URDF.load(str(fw._require(fw.FLEXIV_URDF)))
    hand = yourdfpy.URDF.load(str(fw._require(fw.WUJI_URDF)))
    if args.wuji == "human_like":
        # The hand is loaded on its own so the mount stays live; take its limits from the
        # human-like robot's URDF, the file the PlayFlexivWujiHumanLike task trains.
        human = yourdfpy.URDF.load(str(fw._require(fw.HUMAN_LIKE_URDF)), load_meshes=False)
        for name in hand.actuated_joint_names:
            hand.joint_map[name].limit.lower = human.joint_map[name].limit.lower
            hand.joint_map[name].limit.upper = human.joint_map[name].limit.upper
    wuji_spec = robots.FLEXIV_WUJI_HUMAN_LIKE if args.wuji == "human_like" else robots.FLEXIV_WUJI

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
    # Flat colours: viser can only change the opacity of single-colour meshes (the "robot alpha"
    # slider), not of the URDFs' own textured ones.
    viser_kuka = ViserUrdf(server, kuka, root_node_name="/kuka_sharpa/robot",
                           mesh_color_override=(*KUKA_COLOR, 1.0))
    viser_arm = ViserUrdf(server, arm, root_node_name="/flexiv_wuji/robot",
                          mesh_color_override=(*FLEXIV_COLOR, 1.0))
    viser_hand = ViserUrdf(server, hand, root_node_name="/flexiv_wuji/robot/wuji",
                           mesh_color_override=(*FLEXIV_COLOR, 1.0))
    frame_axes = {
        name: server.scene.add_frame(f"/{name}/hand_frame", axes_length=0.08, axes_radius=0.003)
        for name in HAND_KEYPOINTS
    }
    specs = {"kuka_sharpa": robots.KUKA_SHARPA, "flexiv_wuji": wuji_spec}
    tip_markers = {
        name: {
            label: [server.scene.add_icosphere(f"/{name}/fingertips/{label.split()[0]}/{f}",
                                               radius=0.006, color=color) for f in FINGERS]
            for label, (_, color) in TIP_POINTS.items()
        }
        for name in HAND_KEYPOINTS
    }

    def fingertips(urdf, spec, T_root, with_offset: bool) -> np.ndarray:
        """(5, 3) fingertip points in env coordinates; T_root places ``urdf``'s base."""
        offset = np.array([*(spec.fingertip_offset if with_offset else (0.0, 0.0, 0.0)), 1.0])
        return np.array([
            (T_root @ urdf.get_transform(link, urdf.base_link) @ offset)[:3]
            for link in spec.fingertip_link_names
        ])

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
            for label, markers in tip_markers[name].items():
                for m in markers:
                    m.visible = show_tips.value and label == tip_point.value
        for urdf_handle in (viser_kuka, viser_arm, viser_hand):
            for mesh in urdf_handle._meshes:  # viser keeps no public list of a URDF's meshes
                mesh.opacity = robot_alpha.value
        shown = ROBOT_DISPLAY[robot_display.value]
        viser_kuka.show_visual = "kuka_sharpa" in shown
        viser_arm.show_visual = viser_hand.show_visual = "flexiv_wuji" in shown

        kuka.update_cfg(kuka_cfg())
        viser_kuka.update_cfg(kuka_cfg())
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

        roots_tf = {"kuka_sharpa": (kuka, T_kuka), "flexiv_wuji": (hand, T_base @ T_hand_root)}
        tips = {
            name: {
                label: fingertips(urdf, specs[name], T_root, with_offset)
                for label, (with_offset, _) in TIP_POINTS.items()
            }
            for name, (urdf, T_root) in roots_tf.items()
        }
        for name in tips:
            for label, markers in tip_markers[name].items():
                for m, p in zip(markers, tips[name][label]):
                    m.position = p
        # The chosen point in each hand's own canonical frame, mm: comparable across the hands.
        chosen = {
            name: (np.linalg.inv(H)[:3, :3] @ (tips[name][tip_point.value] - H[:3, 3]).T).T * 1000
            for name, H in (("kuka_sharpa", H_kuka), ("flexiv_wuji", H_wuji))
        }
        tip_rows = [
            f"| {f} | `{np.round(pk).astype(int)}` | `{np.round(pw).astype(int)}` | {np.linalg.norm(pw - pk):.0f} |"
            for f, pk, pw in zip(FINGERS, chosen["kuka_sharpa"], chosen["flexiv_wuji"])
        ]
        labels = list(TIP_POINTS)
        pad_to_body_mm = {
            name: np.linalg.norm(tips[name][labels[0]] - tips[name][labels[1]], axis=1).mean() * 1000
            for name in tips
        }

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
            "",
            f"**Fingertips, {tip_point.value}, each in its own hand frame (mm)**",
            "| finger | Sharpa | Wuji | apart |",
            "|---|---|---|---:|",
            *tip_rows,
            "",
            f"Pad to body origin: Sharpa {pad_to_body_mm['kuka_sharpa']:.0f} mm, "
            f"Wuji {pad_to_body_mm['flexiv_wuji']:.0f} mm",
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
        robot_display = server.gui.add_dropdown("robots", tuple(ROBOT_DISPLAY), initial_value="show both")
        robot_alpha = server.gui.add_slider("robot alpha", 0.05, 1.0, 0.05, 1.0)
        show_tips = server.gui.add_checkbox("fingertips", True)
        tip_point = server.gui.add_dropdown("fingertip point", tuple(TIP_POINTS),
                                            initial_value=next(iter(TIP_POINTS)))
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
    kuka_arm_names = [n for n in kuka.actuated_joint_names if n in fw.KUKA_DEFAULT_ARM_POS]
    kuka_arm_q = JointSliders(server, "Kuka arm joints", kuka, fw.KUKA_DEFAULT_ARM_POS, refresh,
                              False, names=kuka_arm_names)
    sharpa_q = JointSliders(server, "Sharpa hand joints", kuka, {}, refresh, True, hand_action=True,
                            names=[n for n in kuka.actuated_joint_names if n not in kuka_arm_names])
    flexiv_q = JointSliders(server, "Flexiv arm joints", arm, fw.FLEXIV_DEFAULT_ARM_POS, refresh, False)
    wuji_q = JointSliders(server, f"Wuji hand joints ({args.wuji.replace('_', '-')} limits)", hand, {}, refresh, True, hand_action=True)

    def kuka_cfg() -> np.ndarray:
        values = {**dict(zip(kuka_arm_q.names, kuka_arm_q.q())), **dict(zip(sharpa_q.names, sharpa_q.q()))}
        return np.array([values[n] for n in kuka.actuated_joint_names])
    for handle in (layout_sep, show_tables, show_frames, show_tips, tip_point, robot_display, robot_alpha, base_yaw, *base_xyz, *mount_xyz, *mount_rpy):
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
        for sliders in (kuka_arm_q, sharpa_q, flexiv_q, wuji_q):
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
