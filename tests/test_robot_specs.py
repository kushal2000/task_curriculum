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
    assert set(robots.ROBOT_SPECS) == {
        "kuka_sharpa", "flexiv_wuji", "flexiv_wuji_human_like", "flexiv_wuji_right_human_like",
        "flexiv_sharpa"}
    assert robots.KUKA_SHARPA.num_joints == 29  # the released checkpoint's action size
    assert SPEC.num_joints == 27


def test_default_pose_matches_the_viewer_constant() -> None:
    assert SPEC.arm_default_joint_pos == fw.FLEXIV_DEFAULT_ARM_POS


def _task_cfg(task: str) -> dict:
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load((REPO_ROOT / "isaacsimenvs" / "cfg" / "task" / f"{task}.yaml").read_text())


def _box_size(urdf: str) -> tuple[float, ...]:
    sizes = {box.get("size") for box in ET.parse(REPO_ROOT / urdf).iter("box")}
    assert len(sizes) == 1, sizes
    return tuple(float(v) for v in sizes.pop().split())


def test_play_scene_matches_the_viewer_constants() -> None:
    cfg = _task_cfg("Play")
    reset = cfg["reset"]
    assert reset["table_reset_z"] == fw.TABLE_CENTER_Z
    assert tuple(reset["table_reset_xy"]) == (0.0, 0.0)
    assert tuple(reset["target_volume_mins"]) == fw.TARGET_VOLUME_MINS
    assert tuple(reset["target_volume_maxs"]) == fw.TARGET_VOLUME_MAXS


@pytest.mark.parametrize("task", ["PlayFlexivWuji", "PlayFlexivSharpa", "PlayFlexivWujiHumanLike",
                                  "PlayFlexivWujiRightHumanLike"])
def test_flexiv_scene_matches_the_viewer_constants(task) -> None:
    cfg = _task_cfg(task)
    reset = cfg["reset"]
    assert (*reset["table_reset_xy"], reset["table_reset_z"]) == fw.FLEXIV_TABLE_CENTER
    assert _box_size(cfg["assets"]["table_urdf"]) == fw.FLEXIV_TABLE_SIZE
    assert tuple(reset["target_volume_mins"]) == fw.FLEXIV_TARGET_VOLUME_MINS
    assert tuple(reset["target_volume_maxs"]) == fw.FLEXIV_TARGET_VOLUME_MAXS
    assert reset["target_volume_region_scale"] == 1.0


def test_flexiv_table_is_where_the_real_one_is() -> None:
    """Measured on the real setup, in the robot frame: the table top 0.28 m up, its near side
    0.31 m in front (the robot reaches along -y). Objects start, and the goal volume sits, where
    Play has them relative to the table top centre (but the goal volume's y_min, taken out to 0)."""
    base = fw.ROBOT_BASE_POS
    (cx, cy, cz), (_, depth, height) = fw.FLEXIV_TABLE_CENTER, fw.FLEXIV_TABLE_SIZE
    top = cz + height / 2
    assert top - base[2] == pytest.approx(0.28)
    assert base[1] - (cy + depth / 2) == pytest.approx(0.31)
    assert cx == base[0]
    play_top = fw.TABLE_CENTER_Z + _box_size("assets/urdf/table_narrow.urdf")[2] / 2
    shift = (cx, cy, top - play_top)
    moved_mins = [p + s for p, s in zip(fw.TARGET_VOLUME_MINS, shift)]
    moved_mins[1] = 0.0
    assert fw.FLEXIV_TARGET_VOLUME_MINS == pytest.approx(tuple(moved_mins))
    assert fw.FLEXIV_TARGET_VOLUME_MAXS == pytest.approx(
        tuple(p + s for p, s in zip(fw.TARGET_VOLUME_MAXS, shift)))
    play_reset, flexiv_reset = _task_cfg("Play")["reset"], _task_cfg("PlayFlexivWuji")["reset"]
    assert flexiv_reset["table_object_z_offset"] - height / 2 == pytest.approx(
        play_reset["table_object_z_offset"] - (play_top - fw.TABLE_CENTER_Z))


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
def test_hand_armature_matches_vendor_mjcf() -> None:
    """The armature is the vendor MJCF's; the gains and effort limits are Wuji's Isaac Lab drives
    (robots._WUJI_DRIVE), not the MJCF's position servos."""
    root = ET.parse(WUJI_MJCF).getroot()
    default_armature = float(root.find("default/joint").get("armature"))
    armature = {
        j.get("name"): float(j.get("armature", default_armature))
        for j in root.iter("joint") if j.get("name")
    }
    servos = {a.get("joint"): a for a in root.iter("position")}
    assert set(servos) == set(SPEC.hand_joint_names)
    for name in servos:
        assert SPEC.hand_armature[name] == pytest.approx(armature[name]), name
    assert set(SPEC.hand_effort_limit) == set(SPEC.hand_joint_names)
    assert all(e > 0 for e in SPEC.hand_effort_limit.values())


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
    for link, offset in zip(SPEC.fingertip_link_names, SPEC.fingertip_offsets):
        finger = link.removesuffix("_distal")
        tip = -float(joints[f"{finger}_tip_fixed"].find("origin").get("xyz").split()[2])
        assert offset[:2] == (0.0, 0.0), link
        assert math.isclose(-offset[2], tip * 0.02 / 0.026, abs_tol=5e-4), (link, tip * 0.02 / 0.026)


def test_sharpa_filter_is_simtoolreals_verbatim() -> None:
    """The Kuka control runs SimToolReal's own list, not a re-derivation."""
    path = REPO_ROOT.parent / "simtoolreal" / "isaacgymenvs" / "tasks" / "simtoolreal" / "adjacent_links.py"
    if not path.is_file():
        pytest.skip(f"no SimToolReal checkout at {path}")
    scope: dict = {}
    exec(path.read_text(), scope)
    upstream = scope["LEFT_SHARPA_KUKA_LINK_TO_ADJACENT_LINKS"]
    assert {k: set(v) for k, v in robots.KUKA_SHARPA.self_collision_filter.items()} == {
        k: set(v) for k, v in upstream.items()
    }


@pytest.mark.parametrize("spec", list(robots.ROBOT_SPECS.values()), ids=lambda s: s.name)
def test_self_collision_filter_is_symmetric_and_names_post_merge_bodies(spec) -> None:
    pairs = spec.self_collision_filter
    assert pairs
    for a, partners in pairs.items():
        for b in partners:
            assert a in pairs.get(b, ()), f"{a} -> {b} listed one way only"
    if spec is not SPEC or not fw.COMPOSED_URDF.is_file():
        return
    # Post-merge bodies are the base and the children of moving joints.
    joints = _joints(fw.COMPOSED_URDF).values()
    bodies = {"base_link"} | {j.find("child").get("link") for j in joints if j.get("type") != "fixed"}
    assert set(pairs) <= bodies, set(pairs) - bodies


@needs_assets
@pytest.mark.parametrize("robot", ["flexiv_wuji", "flexiv_wuji_human_like", "flexiv_wuji_right_human_like"])
def test_wuji_filter_matches_its_derivation(robot) -> None:
    """Rerun self_collision_pairs.py: jointed pairs, pairs touching at reset, pairs that meet
    across a knuckle's range. Fails when the mount, default pose, meshes or joint limits change
    without it."""
    pytest.importorskip("trimesh")
    pytest.importorskip("yourdfpy")
    import self_collision_pairs

    derived, _ = self_collision_pairs.derive(robot)
    spec = robots.ROBOT_SPECS[robot]
    assert {k: set(v) for k, v in spec.self_collision_filter.items()} == {
        k: set(v) for k, v in derived.items()
    }


@pytest.mark.skipif(not fw.HUMAN_LIKE_URDF.is_file(), reason="run scripts/flexiv_wuji/fetch_assets.sh")
def test_human_like_urdf_differs_only_in_its_table_of_limits() -> None:
    """Regenerated from flexiv_wuji.HUMAN_LIKE_LIMITS_DEG: every bound the table leaves as None,
    every joint it does not list, and everything that is not a limit is the vendor's; every range
    still contains the 0 reset pose."""
    vendor, human = (ET.parse(p).getroot() for p in (fw.COMPOSED_URDF, fw.HUMAN_LIKE_URDF))

    def without_limits(root) -> bytes:
        root = ET.fromstring(ET.tostring(root))
        root.set("name", "")
        for limit in root.iter("limit"):
            limit.attrib.pop("lower", None)
            limit.attrib.pop("upper", None)
        return ET.tostring(root)

    assert without_limits(vendor) == without_limits(human)
    old = {j.get("name"): j.find("limit") for j in vendor.iter("joint") if j.find("limit") is not None}
    new = {j.get("name"): j.find("limit") for j in human.iter("joint") if j.find("limit") is not None}
    for name, limit in new.items():
        table = fw.HUMAN_LIKE_LIMITS_DEG.get(name, (None, None))
        for i, bound in enumerate(("lower", "upper")):
            if table[i] is None:
                assert limit.get(bound) == old[name].get(bound), (name, bound)
            else:
                assert float(limit.get(bound)) == pytest.approx(math.radians(table[i]), abs=1e-4), (name, bound)
        lo, hi = float(limit.get("lower")), float(limit.get("upper"))
        assert float(old[name].get("lower")) <= lo <= 0.0 <= hi <= float(old[name].get("upper")), name


def test_human_like_spec_is_the_flexiv_wuji_but_for_its_name() -> None:
    """Its limits live in its URDF; joints, gains, geometry and pose must be FLEXIV_WUJI's."""
    import dataclasses

    human = robots.FLEXIV_WUJI_HUMAN_LIKE
    assert dataclasses.replace(human, name=SPEC.name) == SPEC


needs_flexiv_sharpa = pytest.mark.skipif(
    not fw.FLEXIV_SHARPA_URDF.is_file(), reason="run scripts/flexiv_wuji/flexiv_wuji.py"
)


@needs_flexiv_sharpa
def test_flexiv_sharpa_carries_the_kukas_hand_unchanged() -> None:
    """Same hand joints, limits and links as the Kuka's, and the same spec entries for them."""
    kuka, flexiv = _joints(fw.KUKA_SHARPA_URDF), _joints(fw.FLEXIV_SHARPA_URDF)
    spec, kuka_spec = robots.FLEXIV_SHARPA, robots.KUKA_SHARPA
    for name in kuka_spec.hand_joint_names:
        canon = lambda el: ET.canonicalize(ET.tostring(el), strip_text=True)  # noqa: E731
        assert canon(kuka[name]) == canon(flexiv[name]), name
    for field in ("hand_stiffness", "hand_damping", "hand_armature", "fingertip_link_names",
                  "fingertip_body_regex", "fingertip_offset"):
        assert getattr(spec, field) == getattr(kuka_spec, field), field
    for field in ("arm_stiffness", "arm_damping", "arm_default_joint_pos"):
        assert getattr(spec, field) == getattr(SPEC, field), field


@needs_flexiv_sharpa
def test_flexiv_sharpa_palm_centre_is_the_kukas_carried_with_the_hand() -> None:
    np = pytest.importorskip("numpy")
    yourdfpy = pytest.importorskip("yourdfpy")
    kuka = yourdfpy.URDF.load(str(fw.KUKA_SHARPA_URDF), load_meshes=False)
    flexiv = yourdfpy.URDF.load(str(fw.FLEXIV_SHARPA_URDF), load_meshes=False)
    k_spec, f_spec = robots.KUKA_SHARPA, robots.FLEXIV_SHARPA
    palm = kuka.get_transform(k_spec.palm_body_name, kuka.base_link) @ np.array([*k_spec.palm_center_offset, 1.0])
    in_hand = np.linalg.inv(kuka.get_transform("left_hand_C_MC", kuka.base_link)) @ palm
    expected = (np.linalg.inv(flexiv.get_transform(f_spec.palm_body_name, flexiv.base_link))
                @ flexiv.get_transform("left_hand_C_MC", flexiv.base_link) @ in_hand)[:3]
    assert np.allclose(f_spec.palm_center_offset, expected, atol=1e-3), expected.round(4)


@needs_flexiv_sharpa
def test_flexiv_sharpa_filter_covers_its_derivation() -> None:
    """Every pair self_collision_pairs.py finds (jointed, touching at reset, meeting across a
    knuckle) is filtered. SimToolReal's list adds two knuckle pairs the derivation does not find,
    as on the Kuka."""
    pytest.importorskip("trimesh")
    import self_collision_pairs

    derived, _ = self_collision_pairs.derive("flexiv_sharpa")
    pairs = lambda m: {frozenset((a, b)) for a, bs in m.items() for b in bs}  # noqa: E731
    assert pairs(derived) <= pairs(robots.FLEXIV_SHARPA.self_collision_filter)


needs_right = pytest.mark.skipif(
    not fw.RIGHT_HUMAN_LIKE_URDF.is_file(), reason="run scripts/flexiv_wuji/flexiv_wuji.py"
)


@needs_right
def test_right_hand_spec_mirrors_the_left() -> None:
    """Same arm, same gains on the r_ joints, joints and limits as in its URDF."""
    right, left = robots.FLEXIV_WUJI_RIGHT_HUMAN_LIKE, robots.FLEXIV_WUJI_HUMAN_LIKE
    assert right.arm_default_joint_pos == left.arm_default_joint_pos
    assert right.arm_stiffness == left.arm_stiffness and right.arm_damping == left.arm_damping
    assert list(right.hand_stiffness.values()) == list(left.hand_stiffness.values())
    joints = _joints(fw.RIGHT_HUMAN_LIKE_URDF)
    assert set(right.hand_joint_names) == {n for n, j in joints.items()
                                           if j.get("type") == "revolute" and n.startswith("r_")}
    for name, (lo, hi) in fw.HUMAN_LIKE_LIMITS_DEG_RIGHT.items():
        limit = joints[name].find("limit")
        if lo is not None:
            assert float(limit.get("lower")) == pytest.approx(math.radians(lo), abs=1e-4)
        if hi is not None:
            assert float(limit.get("upper")) == pytest.approx(math.radians(hi), abs=1e-4)


@needs_right
def test_right_palm_faces_the_table_and_mirrors_the_left_palm_centre() -> None:
    """At the default pose: positive is flexion on every right flex joint, the palm (the side the
    fingers flex towards) faces the table within 10 deg, and the palm centre is the left's mirrored
    across the hand."""
    np = pytest.importorskip("numpy")
    yourdfpy = pytest.importorskip("yourdfpy")
    from hand_frames import hand_frame

    right, left = robots.FLEXIV_WUJI_RIGHT_HUMAN_LIKE, robots.FLEXIV_WUJI_HUMAN_LIKE
    R = yourdfpy.URDF.load(str(fw.RIGHT_HUMAN_LIKE_URDF), load_meshes=False)
    L = yourdfpy.URDF.load(str(fw.HUMAN_LIKE_URDF), load_meshes=False)

    def pose(u, extra=None):
        q = {**right.arm_default_joint_pos, **(extra or {})}
        u.update_cfg(np.array([q.get(n, 0.0) for n in u.actuated_joint_names]))

    def frame(u, s):
        return hand_frame([u.get_transform(f"{s}_{k}", u.base_link)[:3, 3]
                           for k in ("wrist", "index_finger_proximal", "middle_finger_proximal", "pinky_proximal")])

    pose(R)
    H = frame(R, "r")
    # A point out on each fingertip: the distal link's origin sits on its own joint.
    def tip_of(f):
        offset = right.fingertip_offsets[right.fingertip_link_names.index(f"r_{f}_distal")]
        return (R.get_transform(f"r_{f}_distal", R.base_link) @ np.array([*offset, 1.0]))[:3]

    tip = {f: tip_of(f) for f in ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")}
    palm_side = None
    for name in fw.HUMAN_LIKE_LIMITS_DEG_RIGHT:
        if name.endswith("_abd"):
            continue
        finger = next(f for f in tip if name.startswith(f"r_{f}"))
        pose(R, {name: 0.5})
        moved = (tip_of(finger) - tip[finger]) @ H[:3, 2]
        side = np.sign(moved)
        assert palm_side in (None, side), name  # every flex joint bends the same way
        palm_side = side
    normal = palm_side * H[:3, 2]
    assert np.degrees(np.arccos(-normal[2])) < 10.0

    pose(R)
    pose(L)
    in_left = np.linalg.inv(frame(L, "l")) @ (L.get_transform("link7", L.base_link) @ np.array([*left.palm_center_offset, 1.0]))
    expected = np.linalg.inv(R.get_transform("link7", R.base_link)) @ (H @ (in_left * np.array([1, 1, -1, 1])))
    assert np.allclose(right.palm_center_offset, expected[:3], atol=1e-3), expected[:3].round(4)
