"""Side-by-side viser viewer: the Flexiv Rizon 4s with each hand it trains with -- the Kuka's
Sharpa (flexiv_wuji.FLEXIV_SHARPA_URDF), the Wuji Hand 2 with vendor joint limits, and the Wuji with
human-like limits (flexiv_wuji.HUMAN_LIKE_URDF) -- next to SimToolReal's Kuka iiwa14 + Sharpa as the
reference (hidden by default; the "robots" dropdown shows it).

Each robot stands where the RL env puts it, ``ROBOT_BASE_POS``, behind its own copy of its task's
table -- Play's for the Kuka, the real setup's (``FLEXIV_TABLE_*``) for the Flexivs -- left to
right one ``separation`` apart (0 overlays them). The green box is the region the task samples
goal positions in (``*TARGET_VOLUME_*``). The Flexivs share the Flexiv arm sliders -- one default
pose whatever the hand -- the two Sharpas share the Sharpa hand sliders, and each Wuji has its own over its own limits. Each hand also has a "hand action" slider
that sets every hand joint the way the policy's hand action does (-1 = lower limit, +1 = upper,
linearly in between). The Flexiv's base offset and the Wuji's flange mount are live sliders, so the
mount can be tuned against the Kuka setup by eye.

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
#: The setups, left to right, one ``separation`` apart: the Kuka reference, then the three Flexivs.
SETUPS = ("kuka_sharpa", "flexiv_sharpa", "flexiv_wuji", "flexiv_wuji_human_like")
SETUP_LABELS = {
    "kuka_sharpa": "Kuka+Sharpa", "flexiv_sharpa": "Flexiv+Sharpa",
    "flexiv_wuji": "Flexiv+Wuji", "flexiv_wuji_human_like": "Flexiv+Wuji human-like",
}
FLEXIVS = {"flexiv_sharpa", "flexiv_wuji", "flexiv_wuji_human_like"}
ROBOT_DISPLAY = {
    "the three Flexivs": FLEXIVS,
    "all, with Kuka+Sharpa": set(SETUPS),
    **{f"{SETUP_LABELS[n]} only": {n} for n in SETUPS},
    "Kuka+Sharpa, Flexiv+Sharpa": {"kuka_sharpa", "flexiv_sharpa"},
    "hide all": set(),
}
#: Written above each robot.
HAND_NAMES = {
    "kuka_sharpa": "Sharpa (on the Kuka)", "flexiv_sharpa": "Sharpa",
    "flexiv_wuji": "Wuji (vendor limits)", "flexiv_wuji_human_like": "Wuji (human-like limits)",
}
SETUP_COLORS = {
    "kuka_sharpa": (0.62, 0.62, 0.62), "flexiv_sharpa": (0.4, 0.7, 0.45),
    "flexiv_wuji": (0.35, 0.55, 0.85), "flexiv_wuji_human_like": (0.85, 0.55, 0.3),
}
TIP_POINTS = {  # dropdown label -> (use the spec's fingertip_offset, marker colour)
    "policy input (pad: body + offset)": (True, (230, 40, 160)),
    "reward input (body origin)": (False, (40, 200, 230)),
}

TABLE_SIZE = (0.475, 0.4, 0.3)  # table_narrow.urdf's box
#: Each setup's task table (centre, size) and goal volume (mins, maxs), env coordinates.
KUKA_SCENE = dict(table=((0.0, 0.0, fw.TABLE_CENTER_Z), TABLE_SIZE),
                  goal=(fw.TARGET_VOLUME_MINS, fw.TARGET_VOLUME_MAXS))
FLEXIV_SCENE = dict(table=(fw.FLEXIV_TABLE_CENTER, fw.FLEXIV_TABLE_SIZE),
                    goal=(fw.FLEXIV_TARGET_VOLUME_MINS, fw.FLEXIV_TARGET_VOLUME_MAXS))
TABLE_COLOR = (209, 143, 89)
GOAL_VOLUME_COLOR = (60, 200, 90)

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
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # the printed values matter when piped to a log

    kuka = yourdfpy.URDF.load(str(fw.KUKA_SHARPA_URDF))
    arm = yourdfpy.URDF.load(str(fw._require(fw.FLEXIV_URDF)))
    hand = yourdfpy.URDF.load(str(fw._require(fw.WUJI_URDF)))
    # The Wujis are loaded hand-only so the mount stays live; the human-like one takes its limits
    # from the human-like robot's URDF, the file the PlayFlexivWujiHumanLike task trains.
    arm_h = yourdfpy.URDF.load(str(fw.FLEXIV_URDF))
    hand_h = yourdfpy.URDF.load(str(fw.WUJI_URDF))
    human = yourdfpy.URDF.load(str(fw._require(fw.HUMAN_LIKE_URDF)), load_meshes=False)
    for name in hand_h.actuated_joint_names:
        hand_h.joint_map[name].limit.lower = human.joint_map[name].limit.lower
        hand_h.joint_map[name].limit.upper = human.joint_map[name].limit.upper
    # The Rizon with the Kuka's Sharpa, as one model (its mount is fixed: FLEXIV_SHARPA_MOUNT_*).
    flexiv_sharpa = yourdfpy.URDF.load(str(fw._require(fw.FLEXIV_SHARPA_URDF)))
    keypoints = {**HAND_KEYPOINTS, "flexiv_sharpa": HAND_KEYPOINTS["kuka_sharpa"]}

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/ground", width=4.0, height=4.0, cell_size=0.1)

    # Scene graph: /<setup> sits at that setup's env origin (moved by the separation slider);
    # robot and table hang under it in env coordinates, as the env places them.
    roots = {name: server.scene.add_frame(f"/{name}", show_axes=False) for name in SETUPS}
    scenes = {name: KUKA_SCENE if name == "kuka_sharpa" else FLEXIV_SCENE for name in SETUPS}
    tables = {
        name: server.scene.add_box(
            f"/{name}/table", color=TABLE_COLOR, dimensions=scenes[name]["table"][1],
            position=scenes[name]["table"][0],
        )
        for name in SETUPS
    }
    goal_volumes = {
        name: [
            server.scene.add_box(f"/{name}/goal_volume{suffix}", color=GOAL_VOLUME_COLOR,
                                 dimensions=np.subtract(hi, lo), position=np.add(lo, hi) / 2,
                                 wireframe=wire, opacity=None if wire else 0.15, cast_shadow=False)
            for suffix, wire in (("", False), ("/edges", True))
        ]
        for name, (lo, hi) in ((n, scenes[n]["goal"]) for n in SETUPS)
    }
    server.scene.add_frame("/kuka_sharpa/robot", show_axes=False, position=fw.ROBOT_BASE_POS)
    flexiv_base = server.scene.add_frame("/flexiv_wuji/robot", show_axes=False)
    wuji_root = server.scene.add_frame("/flexiv_wuji/robot/wuji", show_axes=False)
    flexiv_h_base = server.scene.add_frame("/flexiv_wuji_human_like/robot", show_axes=False)
    wuji_h_root = server.scene.add_frame("/flexiv_wuji_human_like/robot/wuji", show_axes=False)
    flexiv_sharpa_base = server.scene.add_frame("/flexiv_sharpa/robot", show_axes=False)
    # Flat colours: viser can only change the opacity of single-colour meshes (the "robot alpha"
    # slider), not of the URDFs' own textured ones.
    col = {n: (*c, 1.0) for n, c in SETUP_COLORS.items()}
    viser_kuka = ViserUrdf(server, kuka, root_node_name="/kuka_sharpa/robot", mesh_color_override=col["kuka_sharpa"])
    viser_arm = ViserUrdf(server, arm, root_node_name="/flexiv_wuji/robot", mesh_color_override=col["flexiv_wuji"])
    viser_hand = ViserUrdf(server, hand, root_node_name="/flexiv_wuji/robot/wuji",
                           mesh_color_override=col["flexiv_wuji"])
    viser_arm_h = ViserUrdf(server, arm_h, root_node_name="/flexiv_wuji_human_like/robot",
                            mesh_color_override=col["flexiv_wuji_human_like"])
    viser_hand_h = ViserUrdf(server, hand_h, root_node_name="/flexiv_wuji_human_like/robot/wuji",
                             mesh_color_override=col["flexiv_wuji_human_like"])
    viser_flexiv_sharpa = ViserUrdf(server, flexiv_sharpa, root_node_name="/flexiv_sharpa/robot",
                                    mesh_color_override=col["flexiv_sharpa"])
    # Above the robot's hand at its default pose (the env's palm sits ~0.2 m over the table).
    for name in SETUPS:
        server.scene.add_label(f"/{name}/hand_name", HAND_NAMES[name],
                               position=(0.0, fw.ROBOT_BASE_POS[1] / 2, 1.15))
    frame_axes = {
        name: server.scene.add_frame(f"/{name}/hand_frame", axes_length=0.08, axes_radius=0.003)
        for name in SETUPS
    }
    specs = {"kuka_sharpa": robots.KUKA_SHARPA, "flexiv_wuji": robots.FLEXIV_WUJI,
             "flexiv_wuji_human_like": robots.FLEXIV_WUJI_HUMAN_LIKE,
             "flexiv_sharpa": robots.FLEXIV_SHARPA}
    tip_markers = {
        name: {
            label: [server.scene.add_icosphere(f"/{name}/fingertips/{label.split()[0]}/{f}",
                                               radius=0.006, color=color) for f in FINGERS]
            for label, (_, color) in TIP_POINTS.items()
        }
        for name in SETUPS
    }

    def fingertips(urdf, spec, T_root, with_offset: bool) -> np.ndarray:
        """(5, 3) fingertip points in env coordinates; T_root places ``urdf``'s base."""
        return np.array([
            (T_root @ urdf.get_transform(link, urdf.base_link)
             @ np.array([*(offset if with_offset else (0.0, 0.0, 0.0)), 1.0]))[:3]
            for link, offset in zip(spec.fingertip_link_names, spec.fingertip_offsets)
        ])

    def refresh() -> None:
        if not ready:
            return
        sep = layout_sep.value
        for i, name in enumerate(SETUPS):
            roots[name].position = ((i - (len(SETUPS) - 1) / 2) * sep, 0.0, 0.0)
            tables[name].visible = show_tables.value and (
                sep > 0 or name == "kuka_sharpa"
            )
            frame_axes[name].visible = show_frames.value
            for box in goal_volumes[name]:
                box.visible = show_goal_volume.value and (sep > 0 or name == "kuka_sharpa")
            for label, markers in tip_markers[name].items():
                for m in markers:
                    m.visible = show_tips.value and label == tip_point.value
        for urdf_handle in (viser_kuka, viser_arm, viser_hand, viser_arm_h, viser_hand_h, viser_flexiv_sharpa):
            for mesh in urdf_handle._meshes:  # viser keeps no public list of a URDF's meshes
                mesh.opacity = robot_alpha.value
        shown = ROBOT_DISPLAY[robot_display.value]
        viser_kuka.show_visual = "kuka_sharpa" in shown
        viser_arm.show_visual = viser_hand.show_visual = "flexiv_wuji" in shown
        viser_arm_h.show_visual = viser_hand_h.show_visual = "flexiv_wuji_human_like" in shown
        viser_flexiv_sharpa.show_visual = "flexiv_sharpa" in shown

        kuka.update_cfg(kuka_cfg())
        viser_kuka.update_cfg(kuka_cfg())
        viser_arm.update_cfg(flexiv_q.q())
        viser_hand.update_cfg(wuji_q.q())
        viser_arm_h.update_cfg(flexiv_q.q())
        hand_h.update_cfg(wuji_h_q.q())
        viser_hand_h.update_cfg(wuji_h_q.q())
        flexiv_sharpa.update_cfg(flexiv_sharpa_cfg())
        viser_flexiv_sharpa.update_cfg(flexiv_sharpa_cfg())

        T_base = base_tf()
        T_flange = arm.get_transform(fw.FLANGE_LINK, arm.base_link)
        flexiv_base.position = T_base[:3, 3]
        flexiv_base.wxyz = wxyz(T_base)
        T_hand_root = T_flange @ mount_tf()
        wuji_root.position = T_hand_root[:3, 3]
        wuji_root.wxyz = wxyz(T_hand_root)
        flexiv_h_base.position, flexiv_h_base.wxyz = T_base[:3, 3], wxyz(T_base)
        wuji_h_root.position, wuji_h_root.wxyz = T_hand_root[:3, 3], wxyz(T_hand_root)
        flexiv_sharpa_base.position = T_base[:3, 3]
        flexiv_sharpa_base.wxyz = wxyz(T_base)

        # Hand frames in env coordinates (origin = table centre, floor level).
        T_kuka = make_tf(fw.ROBOT_BASE_POS)
        H_kuka = hand_frame(hand_points(kuka, HAND_KEYPOINTS["kuka_sharpa"], T_kuka))
        H_wuji = hand_frame(hand_points(hand, HAND_KEYPOINTS["flexiv_wuji"], T_base @ T_hand_root))
        H_fs = hand_frame(hand_points(flexiv_sharpa, keypoints["flexiv_sharpa"], T_base))
        H_wuji_h = hand_frame(hand_points(hand_h, HAND_KEYPOINTS["flexiv_wuji"], T_base @ T_hand_root))
        frames = {"kuka_sharpa": H_kuka, "flexiv_sharpa": H_fs, "flexiv_wuji": H_wuji,
                  "flexiv_wuji_human_like": H_wuji_h}
        for name, H in frames.items():
            frame_axes[name].position = H[:3, 3]
            frame_axes[name].wxyz = wxyz(H)

        roots_tf = {"kuka_sharpa": (kuka, T_kuka), "flexiv_wuji": (hand, T_base @ T_hand_root),
                    "flexiv_wuji_human_like": (hand_h, T_base @ T_hand_root),
                    "flexiv_sharpa": (flexiv_sharpa, T_base)}
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
            for name, H in frames.items()
        }
        tip_rows = [
            f"| {f} | `{np.round(pk).astype(int)}` | `{np.round(pw).astype(int)}` | "
            f"{np.linalg.norm(pw - pk):.0f} |"
            for f, pk, pw in zip(FINGERS, chosen["kuka_sharpa"], chosen["flexiv_wuji"])
        ]
        # Fingertips in the env (table) frame, Flexiv+Sharpa vs Kuka+Sharpa: the same hand on two
        # arms, so this is where the arm puts it, not hand geometry.
        fs_rows = [
            f"| {f} | {np.linalg.norm(pf - pk) * 1000:.0f} |"
            for f, pk, pf in zip(FINGERS, tips["kuka_sharpa"][tip_point.value],
                                 tips["flexiv_sharpa"][tip_point.value])
        ]
        labels = list(TIP_POINTS)
        pad_to_body_mm = {
            name: np.linalg.norm(tips[name][labels[0]] - tips[name][labels[1]], axis=1).mean() * 1000
            for name in tips
        }

        def vs_kuka(H) -> str:
            rel = np.linalg.inv(H_kuka) @ H
            d = rel[:3, 3] * 1000
            return (f"`{np.round(d, 1)}` mm (|d| {np.linalg.norm(d):.1f}), "
                    f"{np.degrees(Rotation.from_matrix(rel[:3, :3]).magnitude()):.1f} deg")

        def table_top(name) -> np.ndarray:
            centre, size = scenes[name]["table"]
            return np.add(centre, (0.0, 0.0, size[2] / 2))

        readout.content = "\n".join([
            "**Hand frame, relative to its table-top centre (m)**",
            *(f"- {SETUP_LABELS[n]}: `{np.round(H[:3, 3] - table_top(n), 3)}`" for n, H in frames.items()),
            "",
            "**Hand frame in the Kuka+Sharpa hand frame**",
            *(f"- {SETUP_LABELS[n]}: {vs_kuka(frames[n])}" for n in SETUPS if n != "kuka_sharpa"),
            "",
            f"**Fingertips, {tip_point.value}, each in its own hand frame (mm)**",
            "| finger | Sharpa | Wuji | apart |",
            "|---|---|---|---:|",
            *tip_rows,
            "",
            f"Pad to body origin: Sharpa {pad_to_body_mm['kuka_sharpa']:.0f} mm, "
            f"Wuji {pad_to_body_mm['flexiv_wuji']:.0f} mm",
            "",
            f"**Flexiv+Sharpa fingertips vs Kuka+Sharpa's, on the table (mm)**",
            "| finger | apart |",
            "|---|---:|",
            *fs_rows,
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
        show_goal_volume = server.gui.add_checkbox("goal volume", True)
        robot_display = server.gui.add_dropdown("robots", tuple(ROBOT_DISPLAY), initial_value="the three Flexivs")
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
    sharpa_q = JointSliders(server, "Sharpa hand joints (Kuka and Flexiv)", kuka, {}, refresh, True, hand_action=True,
                            names=[n for n in kuka.actuated_joint_names if n not in kuka_arm_names])
    flexiv_q = JointSliders(server, "Flexiv arm joints (both Flexivs)", arm, fw.FLEXIV_DEFAULT_ARM_POS, refresh, False)
    wuji_q = JointSliders(server, "Wuji hand joints (vendor limits)", hand, {}, refresh, False, hand_action=True)
    wuji_h_q = JointSliders(server, "Wuji hand joints (human-like limits)", hand_h, {}, refresh, False,
                            hand_action=True)

    def kuka_cfg() -> np.ndarray:
        values = {**dict(zip(kuka_arm_q.names, kuka_arm_q.q())), **dict(zip(sharpa_q.names, sharpa_q.q()))}
        return np.array([values[n] for n in kuka.actuated_joint_names])

    def flexiv_sharpa_cfg() -> np.ndarray:
        """The Flexiv arm sliders and the Sharpa hand sliders: one arm pose, one hand pose."""
        values = {**dict(zip(flexiv_q.names, flexiv_q.q())), **dict(zip(sharpa_q.names, sharpa_q.q()))}
        return np.array([values[n] for n in flexiv_sharpa.actuated_joint_names])
    for handle in (layout_sep, show_tables, show_frames, show_goal_volume, show_tips, tip_point, robot_display, robot_alpha, base_yaw, *base_xyz, *mount_xyz, *mount_rpy):
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
        for sliders in (kuka_arm_q, sharpa_q, flexiv_q, wuji_q, wuji_h_q):
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
