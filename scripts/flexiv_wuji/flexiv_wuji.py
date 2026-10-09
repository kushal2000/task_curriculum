"""Flexiv Rizon 4s + left Wuji Hand 2: model paths, the hand mount, and the composed URDF.

The arm and the hand come from two vendors (``fetch_assets.sh``) and are joined here by one fixed
joint, ``flange`` -> ``l_mount``. :data:`MOUNT_XYZ` / :data:`MOUNT_RPY` are that joint's origin;
tune them in ``viser_compare.py`` and paste the printed values back here.

The Kuka iiwa14 + Sharpa model it is compared against, and the RL env's placement of robot and
table, are mirrored as constants below so this script runs without Isaac Lab.

Pure numpy + stdlib + yourdfpy, run with ``.venv_isaaclab3/bin/python``.

    python scripts/flexiv_wuji/flexiv_wuji.py            # write COMPOSED_URDF with MOUNT_*
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = REPO_ROOT / "assets" / "flexiv_wuji"

FLEXIV_URDF = ASSET_DIR / "flexiv_description" / "urdf" / "Rizon4s.urdf"
WUJI_URDF = ASSET_DIR / "wuji_hand2" / "urdf" / "left_with_mount.urdf"
COMPOSED_URDF = ASSET_DIR / "rizon4s_left_wuji.urdf"
KUKA_SHARPA_URDF = (
    REPO_ROOT
    / "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
)
TABLE_URDF = REPO_ROOT / "assets" / "urdf" / "table_narrow.urdf"

FLANGE_LINK = "flange"
WUJI_ROOT_LINK = "l_mount"
#: Link each hand's pose readout uses: the first link past the arm's mount.
HAND_BASE_LINK = {"kuka_sharpa": "left_hand_C_MC", "flexiv_wuji": "l_wrist"}

#: ``flange`` -> ``l_mount``. The mount's mating face is its z=0 plane with the hand along -z,
#: and the flange's +z points out of the arm, so roll pi puts the hand on the outside face.
#: Yaw (palm direction about the flange axis) and any adapter-plate offset are not yet measured.
MOUNT_XYZ = (0.0, 0.0, 0.0)
MOUNT_RPY = (3.141592653589793, 0.0, 0.0)

# --- The RL env (isaacsimenvs/tasks/play), mirrored so this runs without Isaac Lab -----------

#: scene_utils.build_robot_articulation_usd_cfg: robot root at (0, 0.8, 0), identity rotation.
ROBOT_BASE_POS = (0.0, 0.8, 0.0)
#: play_env_cfg.reset.table_reset_z: the table box's centre. The box is 0.3 m tall
#: (table_narrow.urdf), so its top is at 0.53.
TABLE_CENTER_Z = 0.38
#: scene_utils.ARM_DEFAULT_JOINT_POS, the pose every play-family env resets the Kuka to.
KUKA_DEFAULT_ARM_POS = {
    "iiwa14_joint_1": -1.571, "iiwa14_joint_2": 1.571, "iiwa14_joint_3": 0.0,
    "iiwa14_joint_4": 1.376, "iiwa14_joint_5": 0.0, "iiwa14_joint_6": 1.485,
    "iiwa14_joint_7": 1.308,
}
#: The Flexiv counterpart of KUKA_DEFAULT_ARM_POS: viser_compare's "Match Kuka hand" IK, which
#: puts the Wuji's canonical hand frame on the Sharpa's (0.0 mm, 0.1 deg) with the MOUNT_* above.
#: Solved from upstream's home pose (initial_positions.yaml, joint 1 turned to -pi/2 to face the
#: table); redo it if the mount or base changes.
FLEXIV_DEFAULT_ARM_POS = {
    "joint1": -1.7604, "joint2": -0.2933, "joint3": 0.0197, "joint4": 1.4941,
    "joint5": -1.6591, "joint6": 1.6462, "joint7": 0.0369,
}


def _require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run scripts/flexiv_wuji/fetch_assets.sh")
    return path


def _fmt(values) -> str:
    return " ".join(repr(float(v)) for v in values)


def _rebase_meshes(root: ET.Element, src_dir: Path, dst_dir: Path) -> None:
    """Rewrite relative mesh paths in ``root`` from ``src_dir`` to ``dst_dir``."""
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename")
        if filename and "://" not in filename and not os.path.isabs(filename):
            mesh.set("filename", os.path.relpath(src_dir / filename, dst_dir))


def compose_urdf(
    mount_xyz=MOUNT_XYZ, mount_rpy=MOUNT_RPY, out: Path = COMPOSED_URDF
) -> Path:
    """Write the Rizon 4s with the Wuji hand fixed to its flange, as one URDF.

    Upstream's ``world`` link and its fixed ``base_joint`` are dropped, so the root is
    ``base_link``, as ``iiwa14_link_0`` is the Kuka's.
    """
    arm = ET.parse(_require(FLEXIV_URDF)).getroot()
    hand = ET.parse(_require(WUJI_URDF)).getroot()
    out = Path(out)
    _rebase_meshes(arm, FLEXIV_URDF.parent, out.parent)
    _rebase_meshes(hand, WUJI_URDF.parent, out.parent)

    for el in list(arm):
        is_world = el.tag == "link" and el.get("name") == "world"
        from_world = el.tag == "joint" and el.find("parent").get("link") == "world"
        if is_world or from_world:
            arm.remove(el)

    arm_names = {el.get("name") for el in arm if el.tag in ("link", "joint")}
    clashes = arm_names & {el.get("name") for el in hand if el.tag in ("link", "joint")}
    if clashes:
        raise ValueError(f"arm and hand share link/joint names: {sorted(clashes)}")

    arm.set("name", "rizon4s_left_wuji")
    for el in hand:
        arm.append(el)
    mount = ET.SubElement(arm, "joint", name="flange_to_wuji_mount", type="fixed")
    ET.SubElement(mount, "parent", link=FLANGE_LINK)
    ET.SubElement(mount, "child", link=WUJI_ROOT_LINK)
    ET.SubElement(mount, "origin", xyz=_fmt(mount_xyz), rpy=_fmt(mount_rpy))

    ET.indent(arm, space="  ")
    out.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(arm).write(out, encoding="utf-8", xml_declaration=True)
    return out


if __name__ == "__main__":
    path = compose_urdf()
    print(f"wrote {path.relative_to(REPO_ROOT)}  (mount xyz {MOUNT_XYZ}, rpy {MOUNT_RPY})")
