"""Flexiv Rizon 4s + left Wuji Hand 2: model paths, the hand mount, and the composed URDF.

The arm and the hand come from two vendors (``fetch_assets.sh``) and are joined here by one fixed
joint, ``flange`` -> ``l_mount``. :data:`MOUNT_XYZ` / :data:`MOUNT_RPY` are that joint's origin;
tune them in ``viser_compare.py`` and paste the printed values back here.

The Kuka iiwa14 + Sharpa model it is compared against, and the RL env's placement of robot and
table, are mirrored as constants below so this script runs without Isaac Lab.

Pure numpy + stdlib + yourdfpy, run with ``.venv_isaaclab3/bin/python``.

    python scripts/flexiv_wuji/flexiv_wuji.py   # write COMPOSED_URDF and HUMAN_LIKE_URDF with MOUNT_*
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
#: The Rizon 4s with the Kuka's Sharpa hand (compose_flexiv_sharpa_urdf).
FLEXIV_SHARPA_URDF = ASSET_DIR / "rizon4s_left_sharpa.urdf"
#: COMPOSED_URDF with HUMAN_LIKE_LIMITS_DEG applied.
HUMAN_LIKE_URDF = ASSET_DIR / "rizon4s_left_wuji_human_like_joint_limits.urdf"
KUKA_SHARPA_URDF = (
    REPO_ROOT
    / "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
)
TABLE_URDF = REPO_ROOT / "assets" / "urdf" / "table_narrow.urdf"

FLANGE_LINK = "flange"
#: The Sharpa subtree's root in the Kuka URDF: its adapter plate, fixed to the iiwa's flange face
#: (iiwa14_link_ee) by joint iiwa14_sharpa. The Flexiv + Sharpa fixes it to the Rizon's flange the
#: same way.
SHARPA_MOUNT_LINK = "sharpa_mount"
KUKA_SHARPA_MOUNT_JOINT = "iiwa14_sharpa"
#: ``flange`` -> ``sharpa_mount`` on the Rizon. The Flexiv keeps FLEXIV_DEFAULT_ARM_POS whatever
#: hand it carries, so the hand is turned about the flange axis instead: at that pose this yaw
#: puts the Sharpa palm-down, 8.3 deg from the Kuka's hand orientation with its origin 47 mm off
#: (45 of it along the fingers -- the Sharpa and its plate stand further off the flange than the
#: Wuji). The Kuka's own 15 deg would leave the palm facing up. Solved, not measured on hardware.
FLEXIV_SHARPA_MOUNT_XYZ = (0.0, 0.0, 0.0)
FLEXIV_SHARPA_MOUNT_RPY = (0.0, 0.0, -3.129)
WUJI_ROOT_LINK = "l_mount"
#: Link each hand's pose readout uses: the first link past the arm's mount.
HAND_BASE_LINK = {"kuka_sharpa": "left_hand_C_MC", "flexiv_wuji": "l_wrist"}

#: ``flange`` -> ``l_mount``. The mount's mating face is its z=0 plane with the hand along -z,
#: and the flange's +z points out of the arm, so roll pi puts the hand on the outside face.
#: Yaw (palm direction about the flange axis) and any adapter-plate offset are not yet measured.
MOUNT_XYZ = (0.0, 0.0, 0.0)
MOUNT_RPY = (3.141592653589793, 0.0, 0.0)

_FINGERS = ("index_finger", "middle_finger", "ring_finger", "pinky")
#: HUMAN_LIKE_URDF's joint limits, (lower, upper) in degrees; None keeps the vendor's bound
#: exactly. The vendor allows -60 deg on every finger joint, i.e. 60 deg of backward bend no human
#: finger has: the lowers below cut it to rough human values (the knuckle bends back some, about
#: 0-45, here 20; the middle finger joint not at all; the tip joint and the thumb a little), and
#: flexion keeps the vendor's uppers. Positive is flexion on all of these (checked by FK: the
#: fingertip moves toward the palm). Finger abduction is held to +-2 deg about the vendor's zero
#: pose, as SimToolReal restricts the Sharpa's; thumb abduction keeps the vendor's range.
HUMAN_LIKE_LIMITS_DEG: dict[str, tuple[float | None, float | None]] = {
    **{f"l_{f}_mcp_flex": (-20.0, None) for f in _FINGERS},
    **{f"l_{f}_mcp_abd": (-2.0, 2.0) for f in _FINGERS},
    **{f"l_{f}_pip": (0.0, None) for f in _FINGERS},
    **{f"l_{f}_dip": (-10.0, None) for f in _FINGERS},
    "l_thumb_cmc_flex": (-20.0, None),
    "l_thumb_mcp": (-10.0, None),
    "l_thumb_ip": (-15.0, None),
}

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


def compose_flexiv_sharpa_urdf(out: Path = FLEXIV_SHARPA_URDF) -> Path:
    """Write the Rizon 4s with the Kuka's Sharpa hand on its flange, as one URDF.

    The hand is the Kuka URDF's ``sharpa_mount`` subtree verbatim -- adapter plate, restricted
    joint limits, renamed joints, elastomer pads -- so it is the hand the Kuka trains with. It is
    fixed to the Rizon's ``flange`` (+z out of the arm, like the iiwa's ``iiwa14_link_ee``) at
    FLEXIV_SHARPA_MOUNT_*.
    """
    arm = ET.parse(_require(FLEXIV_URDF)).getroot()
    kuka = ET.parse(_require(KUKA_SHARPA_URDF)).getroot()
    out = Path(out)
    _rebase_meshes(arm, FLEXIV_URDF.parent, out.parent)
    _rebase_meshes(kuka, KUKA_SHARPA_URDF.parent, out.parent)
    for el in list(arm):
        is_world = el.tag == "link" and el.get("name") == "world"
        from_world = el.tag == "joint" and el.find("parent").get("link") == "world"
        if is_world or from_world:
            arm.remove(el)

    children: dict[str, list[ET.Element]] = {}
    for j in kuka.iter("joint"):
        children.setdefault(j.find("parent").get("link"), []).append(j)
    links = {el.get("name"): el for el in kuka.iter("link")}
    hand_links, hand_joints, stack = [], [], [SHARPA_MOUNT_LINK]
    while stack:
        name = stack.pop()
        hand_links.append(links[name])
        for j in children.get(name, []):
            hand_joints.append(j)
            stack.append(j.find("child").get("link"))

    arm_names = {el.get("name") for el in arm if el.tag in ("link", "joint")}
    clashes = arm_names & {el.get("name") for el in hand_links + hand_joints}
    if clashes:
        raise ValueError(f"arm and hand share link/joint names: {sorted(clashes)}")
    kuka_mount = next(j for j in kuka.iter("joint") if j.get("name") == KUKA_SHARPA_MOUNT_JOINT)
    assert kuka_mount.find("child").get("link") == SHARPA_MOUNT_LINK

    arm.set("name", out.stem)
    subtree = {id(el) for el in hand_links + hand_joints}
    for el in kuka:  # the Kuka file's own order, so the hand reads the same in both
        if id(el) in subtree:
            arm.append(el)
    mount = ET.SubElement(arm, "joint", name="flange_to_sharpa_mount", type="fixed")
    ET.SubElement(mount, "parent", link=FLANGE_LINK)
    ET.SubElement(mount, "child", link=SHARPA_MOUNT_LINK)
    ET.SubElement(mount, "origin", xyz=_fmt(FLEXIV_SHARPA_MOUNT_XYZ), rpy=_fmt(FLEXIV_SHARPA_MOUNT_RPY))

    ET.indent(arm, space="  ")
    out.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(arm).write(out, encoding="utf-8", xml_declaration=True)
    return out


def restrict_limits(src: Path, limits_deg: dict[str, tuple[float | None, float | None]],
                    out: Path) -> Path:
    """Copy ``src`` with the joint limits in ``limits_deg`` narrowed (``None`` keeps a bound).

    Each new range must lie inside the original and contain 0, the pose the env resets the hand
    to. Bounds are written at the vendor's 4-decimal precision; the robot is renamed to the
    output's stem.
    """
    import math

    tree = ET.parse(_require(src))
    joints = {j.get("name"): j for j in tree.getroot().iter("joint")}
    missing = sorted(set(limits_deg) - set(joints))
    if missing:
        raise KeyError(f"no such joints in {src.name}: {missing}")
    for name, (lo_deg, hi_deg) in limits_deg.items():
        limit = joints[name].find("limit")
        old_lo, old_hi = float(limit.get("lower")), float(limit.get("upper"))
        lo = old_lo if lo_deg is None else math.radians(lo_deg)
        hi = old_hi if hi_deg is None else math.radians(hi_deg)
        if not old_lo <= lo <= 0.0 <= hi <= old_hi:
            raise ValueError(
                f"{name}: ({lo_deg}, {hi_deg}) deg must contain 0 and lie inside "
                f"({math.degrees(old_lo):.1f}, {math.degrees(old_hi):.1f})")
        if lo_deg is not None:
            limit.set("lower", f"{lo:.4f}")
        if hi_deg is not None:
            limit.set("upper", f"{hi:.4f}")
    tree.getroot().set("name", Path(out).stem)
    tree.write(out, encoding="utf-8", xml_declaration=True)
    return Path(out)


if __name__ == "__main__":
    path = compose_urdf()
    print(f"wrote {path.relative_to(REPO_ROOT)}  (mount xyz {MOUNT_XYZ}, rpy {MOUNT_RPY})")
    path = compose_flexiv_sharpa_urdf()
    print(f"wrote {path.relative_to(REPO_ROOT)}  (Kuka's Sharpa subtree on the Rizon flange)")
    path = restrict_limits(COMPOSED_URDF, HUMAN_LIKE_LIMITS_DEG, HUMAN_LIKE_URDF)
    print(f"wrote {path.relative_to(REPO_ROOT)}  ({len(HUMAN_LIKE_LIMITS_DEG)} joints restricted)")
