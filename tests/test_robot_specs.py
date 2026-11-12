"""The Flexiv + Wuji robot spec must agree with the models it was derived from.

Every geometric number in ``robots.FLEXIV_WUJI`` was read off the URDFs, and the hand gains off
the vendor MJCF, by hand. Nothing ties them together at runtime except the env's joint-set check,
so a mount change or a mistyped gain would otherwise go unnoticed until a policy trained badly.

Kit-free: ``robots.py`` is stdlib-only and is loaded off disk. The model files come from
``scripts/flexiv_wuji/fetch_assets.sh``; the tests skip if they are missing.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
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
        "kuka_sharpa", "flexiv_wuji", "flexiv_wuji_human_like", "flexiv_wuji_human_like_s2r",
        "flexiv_sharpa", "flexiv_wuji_human_like_sharpagain", "flexiv_sharpa_wujigain",
        "flexiv_wuji_right_human_like", "flexiv_wuji_right_real"}
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
@pytest.mark.parametrize("robot", ["flexiv_wuji", "flexiv_wuji_human_like"])
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


RIGHT_URDF = Path(__file__).resolve().parents[1] / "assets/flexiv_wuji/rizon4s_right_wuji_human_like_joint_limits.urdf"


def test_right_hand_spec_matches_its_urdf() -> None:
    """Joints, home pose inside the limits, fingertip / palm / filter bodies all exist in the right
    URDF; hand drive is the S2R one, renamed."""
    spec = robots.FLEXIV_WUJI_RIGHT_HUMAN_LIKE
    joints = _joints(RIGHT_URDF)
    actuated = [n for n, j in joints.items() if j.get("type") != "fixed"]
    assert sorted(actuated) == sorted(spec.joint_names_canonical)
    for name, q in spec.arm_default_joint_pos.items():
        limit = joints[name].find("limit")
        assert float(limit.get("lower")) <= q <= float(limit.get("upper")), name
    for name in spec.hand_joint_names:
        limit = joints[name].find("limit")
        assert float(limit.get("lower")) <= 0.0 <= float(limit.get("upper")), name
    bodies = {"base_link"} | {j.find("child").get("link") for j in joints.values() if j.get("type") != "fixed"}
    assert set(spec.fingertip_link_names) <= bodies
    assert spec.palm_body_name in bodies
    assert set(spec.self_collision_filter) <= bodies
    assert all(set(v) <= bodies for v in spec.self_collision_filter.values())
    s2r = robots.FLEXIV_WUJI_HUMAN_LIKE_S2R
    for table in ("hand_stiffness", "hand_damping", "hand_effort_limit", "hand_armature"):
        assert list(getattr(spec, table).values()) == list(getattr(s2r, table).values()), table
    assert [n[2:] for n in spec.hand_joint_names] == [n[2:] for n in s2r.hand_joint_names]


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


REAL_URDF = Path(__file__).resolve().parents[1] / "assets/flexiv_wuji/rizon4s_right_wuji_real_limits.urdf"
SYSID_JSON = Path(__file__).resolve().parents[1] / "assets/flexiv_wuji/wuji_hand2_right_sysid_fit_20261009.json"


def test_right_real_spec_is_the_sysid_fit() -> None:
    """FLEXIV_WUJI_RIGHT_REAL = the right human-like spec, with the sysid fit's armature, friction
    and passive damping, and torque limit 1.0 A x Kt; everything else unchanged."""
    import json

    if not SYSID_JSON.exists():
        pytest.skip(f"{SYSID_JSON.name} is kept out of git (.gitignore); put the fit there to run this")
    real, base = robots.FLEXIV_WUJI_RIGHT_REAL, robots.FLEXIV_WUJI_RIGHT_HUMAN_LIKE
    fit = json.loads(SYSID_JSON.read_text())
    assert fit["handedness"] == "right"
    by_joint = {v["mjcf_joint"]: v for v in fit["joints"].values()}
    # The replay's per-type friction scale (S index from the fit's joint key, e.g. "index_S3").
    per_type = {"thumb": (0.8, 0.8, 0.8, 1.0), "finger": (1.0, 0.8, 0.8, 0.6)}
    fscale = {v["mjcf_joint"]: per_type["thumb" if key.startswith("thumb") else "finger"][int(key[-1]) - 1]
              for key, v in fit["joints"].items()}
    assert set(by_joint) == set(real.hand_joint_names)
    for j, v in by_joint.items():
        assert real.hand_armature[j] == pytest.approx(v["armature"], rel=1e-3), j
        assert real.hand_friction[j] == pytest.approx(v["frictionloss"] * fscale[j], rel=1e-3), j
        assert real.hand_dynamic_friction[j] == pytest.approx(v["frictionloss"] * fscale[j], rel=1e-3), j
        assert real.hand_viscous_friction[j] == pytest.approx(v["damping"], rel=1e-3), j
        assert real.hand_effort_limit[j] == pytest.approx(1.0 * v["gear"], rel=1e-3), j
        assert real.hand_spring_stiffness[j] == pytest.approx(v["stiffness"], rel=1e-3), j
        assert real.hand_spring_ref[j] == pytest.approx(v["springref"], rel=1e-3, abs=1e-6), j
    # The checklist's example: index DIP (S4) limit 0.13 N m.
    assert real.hand_effort_limit["r_index_finger_dip"] == pytest.approx(0.135, abs=1e-3)
    changed = {"name", "hand_armature", "hand_effort_limit", "hand_friction", "hand_dynamic_friction",
               "hand_viscous_friction", "hand_spring_stiffness", "hand_spring_ref"}
    for f in dataclasses.fields(base):
        if f.name not in changed:
            assert getattr(real, f.name) == getattr(base, f.name), f.name


def test_right_real_urdf_limits() -> None:
    """The real-limits URDF differs from the human-like one only in the arm limits (5 deg inside
    on both sides) and the finger PIP upper limit (100 deg); the home pose stays inside."""
    base, real = _joints(RIGHT_URDF), _joints(REAL_URDF)
    assert set(base) == set(real)
    expect_deg = {"joint1": (-160, 160), "joint2": (-130, 130), "joint3": (-170, 170),
                  "joint4": (-107, 154), "joint5": (-170, 170), "joint6": (-80, 260), "joint7": (-170, 170)}
    pips = {f"r_{f}_pip" for f in ("index_finger", "middle_finger", "ring_finger", "pinky")}
    for name, j in real.items():
        lim, blim = j.find("limit"), base[name].find("limit")
        if lim is None:
            assert blim is None, name
            continue
        lo, hi = float(lim.get("lower")), float(lim.get("upper"))
        blo, bhi = float(blim.get("lower")), float(blim.get("upper"))
        if name in expect_deg:
            assert (math.degrees(lo), math.degrees(hi)) == pytest.approx(expect_deg[name], abs=0.01), name
            assert math.degrees(lo - blo) == pytest.approx(5.0, abs=0.01), name
            assert math.degrees(bhi - hi) == pytest.approx(5.0, abs=0.01), name
        elif name in pips:
            assert (lo, math.degrees(hi)) == (blo, pytest.approx(100.0, abs=0.01)), name
        else:
            assert (lo, hi) == (blo, bhi), name
        assert lim.get("effort") == blim.get("effort") and lim.get("velocity") == blim.get("velocity"), name
    for name, q in robots.FLEXIV_WUJI_RIGHT_HUMAN_LIKE.arm_default_joint_pos.items():
        lim = real[name].find("limit")
        # 0.1 rad: the arm reset noise (reset_dof_pos_random_interval_arm).
        assert float(lim.get("lower")) + 0.1 <= q <= float(lim.get("upper")) - 0.1, name


def test_hand_spring_effort() -> None:
    """The spring effort is ks (ref - q) per env and joint, zero on spring-free joints, None without."""
    import types

    import torch

    au = load_module("isaacsimenvs/tasks/play/utils/action_utils.py", "action_utils_spring")
    gen = torch.Generator().manual_seed(0)
    n, j = 8, 27
    ks = torch.rand(j, generator=gen) * 0.2
    ks[:7] = 0.0  # the arm: no spring
    ref = torch.randn(j, generator=gen) * 0.3
    q = torch.randn(n, j, generator=gen)
    env = types.SimpleNamespace(_spring_ks=ks, _spring_ref=ref, robot=types.SimpleNamespace(
        data=types.SimpleNamespace(joint_pos=q)))
    tau = au.hand_spring_effort(env)
    assert torch.allclose(tau, ks * (ref - q))
    assert torch.equal(tau[:, :7], torch.zeros(n, 7))
    assert au.hand_spring_effort(types.SimpleNamespace(_spring_ks=None)) is None


def test_only_right_real_has_a_spring() -> None:
    for name, spec in robots.ROBOT_SPECS.items():
        if name == "flexiv_wuji_right_real":
            assert len(spec.hand_spring_stiffness) == len(spec.hand_joint_names)
        else:
            assert not spec.hand_spring_stiffness, name
