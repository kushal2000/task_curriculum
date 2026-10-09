"""The Flexiv + Wuji robot spec must agree with the models it was derived from.

Every geometric number in ``robots.FLEXIV_WUJI`` was read off the URDFs, and the hand gains off
the vendor MJCF, by hand. Nothing ties them together at runtime except the env's joint-set check,
so a mount change or a mistyped gain would otherwise go unnoticed until a policy trained badly.

Kit-free: ``robots.py`` is stdlib-only and is loaded off disk. The model files come from
``scripts/flexiv_wuji/fetch_assets.sh``; the tests skip if they are missing.
"""

from __future__ import annotations

import math
import sys
import xml.etree.ElementTree as ET

import pytest

from conftest import REPO_ROOT, load_module

robots = load_module("isaacsimenvs/tasks/play/robots.py", "play_robots")
sys.path.insert(0, str(REPO_ROOT / "scripts" / "flexiv_wuji"))
import flexiv_wuji as fw  # noqa: E402

SPEC = robots.FLEXIV_WUJI
WUJI_MJCF = fw.ASSET_DIR / "wuji_hand2" / "mjcf" / "left.xml"

needs_assets = pytest.mark.skipif(
    not fw.COMPOSED_URDF.is_file(), reason="run scripts/flexiv_wuji/fetch_assets.sh"
)


def _joints(urdf_path) -> dict[str, ET.Element]:
    return {j.get("name"): j for j in ET.parse(urdf_path).getroot().iter("joint")}


def test_specs_are_registered_and_kuka_is_unchanged_in_shape() -> None:
    assert set(robots.ROBOT_SPECS) == {"kuka_sharpa", "flexiv_wuji"}
    assert robots.KUKA_SHARPA.num_joints == 29  # the released checkpoint's action size
    assert SPEC.num_joints == 27


def test_default_pose_matches_the_viewer_constant() -> None:
    assert SPEC.arm_default_joint_pos == fw.FLEXIV_DEFAULT_ARM_POS


@needs_assets
def test_joints_and_limits_match_composed_urdf() -> None:
    joints = _joints(fw.COMPOSED_URDF)
    actuated = [n for n, j in joints.items() if j.get("type") != "fixed"]
    assert sorted(actuated) == sorted(SPEC.joint_names_canonical)
    for name, q in SPEC.arm_default_joint_pos.items():
        limit = joints[name].find("limit")
        assert float(limit.get("lower")) <= q <= float(limit.get("upper")), name
    for name in SPEC.hand_joint_names:  # hand resets to 0
        limit = joints[name].find("limit")
        assert float(limit.get("lower")) <= 0.0 <= float(limit.get("upper")), name


@needs_assets
def test_palm_and_fingertip_bodies_survive_fixed_joint_merging() -> None:
    """The importer merges fixed-joint children into their parent, so the bodies the task reads
    must each be the child of a moving joint."""
    child_joint = {j.find("child").get("link"): j for j in _joints(fw.COMPOSED_URDF).values()}
    for link in (SPEC.palm_body_name, *SPEC.fingertip_link_names):
        assert child_joint[link].get("type") == "revolute", link


@needs_assets
def test_every_moving_link_has_an_inertial() -> None:
    """Upstream's Rizon xacro emits mass/inertia outside <inertial>; fetch_assets.sh wraps them.
    Without it, parsers see no mass and Isaac Sim silently uses defaults for the whole arm."""
    root = ET.parse(fw.COMPOSED_URDF).getroot()
    moving = {j.find("child").get("link") for j in root.iter("joint") if j.get("type") != "fixed"}
    for link in root.iter("link"):
        if link.get("name") in moving | {"base_link"}:
            mass = link.find("inertial/mass")
            assert mass is not None and float(mass.get("value")) > 0, link.get("name")
        assert link.find("mass") is None, f"{link.get('name')}: loose <mass> outside <inertial>"


@needs_assets
def test_hand_gains_match_vendor_mjcf() -> None:
    root = ET.parse(WUJI_MJCF).getroot()
    default_armature = float(root.find("default/joint").get("armature"))
    armature = {
        j.get("name"): float(j.get("armature", default_armature))
        for j in root.iter("joint") if j.get("name")
    }
    servos = {a.get("joint"): a for a in root.iter("position")}
    assert set(servos) == set(SPEC.hand_joint_names)
    for name, servo in servos.items():
        assert SPEC.hand_stiffness[name] == pytest.approx(float(servo.get("kp"))), name
        assert SPEC.hand_damping[name] == pytest.approx(float(servo.get("kv"))), name
        assert SPEC.hand_armature[name] == pytest.approx(armature[name]), name


@needs_assets
def test_arm_stiffness_follows_rizon_torque_limits() -> None:
    joints = _joints(fw.COMPOSED_URDF)
    for name in SPEC.arm_joint_names:
        effort = float(joints[name].find("limit").get("effort"))
        assert SPEC.arm_stiffness[name] == pytest.approx(2.0 * effort), name


@needs_assets
def test_palm_center_is_the_kukas_carried_over_through_the_hand_frame() -> None:
    """Recompute the palm offset from the models. Fails when MOUNT_* changes without it."""
    np = pytest.importorskip("numpy")
    yourdfpy = pytest.importorskip("yourdfpy")
    from hand_frames import HAND_KEYPOINTS, hand_frame, hand_points

    kuka = yourdfpy.URDF.load(str(fw.KUKA_SHARPA_URDF))
    flexiv = yourdfpy.URDF.load(str(fw.COMPOSED_URDF))
    kuka_spec = robots.KUKA_SHARPA
    # Any pose works (it is all rigid past the palm body); use the defaults.
    kuka.update_cfg(np.array([kuka_spec.arm_default_joint_pos.get(n, 0.0) for n in kuka.actuated_joint_names]))
    flexiv.update_cfg(np.array([SPEC.arm_default_joint_pos.get(n, 0.0) for n in flexiv.actuated_joint_names]))

    def palm_body(urdf, spec):
        return urdf.get_transform(spec.palm_body_name, urdf.base_link)

    kuka_palm = palm_body(kuka, kuka_spec) @ np.array([*kuka_spec.palm_center_offset, 1.0])
    in_hand = np.linalg.inv(hand_frame(hand_points(kuka, HAND_KEYPOINTS["kuka_sharpa"]))) @ kuka_palm
    flexiv_palm = hand_frame(hand_points(flexiv, HAND_KEYPOINTS["flexiv_wuji"])) @ in_hand
    expected = (np.linalg.inv(palm_body(flexiv, SPEC)) @ flexiv_palm)[:3]
    assert np.allclose(SPEC.palm_center_offset, expected, atol=1e-3), expected.round(4)


@needs_assets
def test_fingertip_offset_is_the_sharpa_fraction_of_the_wuji_tip() -> None:
    """The Sharpa pad point is 0.02 of the 0.026 m to its tip; the Wuji's is the same share of
    its 0.0248 m (along the distal link's -z)."""
    joints = _joints(fw.COMPOSED_URDF)
    tip = -float(joints["l_index_finger_tip_fixed"].find("origin").get("xyz").split()[2])
    assert math.isclose(-SPEC.fingertip_offset[2], tip * 0.02 / 0.026, abs_tol=5e-4)
