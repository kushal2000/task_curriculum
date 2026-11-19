"""The right Flexiv + Wuji robot specs must agree with their URDFs and the hand's sysid fit.

Every geometric number in ``robots/flexiv_right.py`` was read off the URDFs, and the hand dynamics
off the vendor drive and the sysid fit, by hand. Nothing ties them together at runtime except the
env's joint-set check, so a mistyped value would otherwise go unnoticed until a policy trained badly.

Kit-free: ``robots/`` is stdlib-only and is loaded off disk. The URDFs are committed; their meshes
come from ``isaacsimenvs/scripts/robots/fetch_assets.sh``.
"""

from __future__ import annotations

import dataclasses
import math
import xml.etree.ElementTree as ET

import pytest

from isaacsimenvs.conftest import REPO_ROOT, load_module

robots = load_module("isaacsimenvs/robots/__init__.py", "robots", package=True)


def _joints(urdf_path) -> dict[str, ET.Element]:
    return {j.get("name"): j for j in ET.parse(urdf_path).getroot().iter("joint")}


SPEC = robots.FLEXIV_WUJI_RIGHT
SYSID = robots.with_sysid(SPEC)


@pytest.mark.parametrize("spec", [SPEC, SYSID], ids=lambda s: s.name)
def test_spec_shape(spec) -> None:
    assert spec.num_joints == 27
    pairs = spec.self_collision_filter
    assert pairs
    for a, partners in pairs.items():
        for b in partners:
            assert a in pairs.get(b, ()), f"{a} -> {b} listed one way only"


def test_robot_spec_switch() -> None:
    assert robots.robot_spec() is SPEC and robots.robot_spec(False) is SPEC
    assert robots.robot_spec(True) == SYSID


RIGHT_URDF = REPO_ROOT / "assets/flexiv_wuji/rizon4s_right_wuji_human_like_joint_limits.urdf"


def test_right_hand_spec_matches_its_urdf() -> None:
    """Joints, home pose inside the limits, fingertip / palm / filter bodies all exist in the
    human-like URDF (the Play task's)."""
    spec = SPEC
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
    # The vendor drive's first joint (thumb CMC flexion) and the 0.0005 / 0.0002 armature split.
    assert spec.hand_stiffness["r_thumb_cmc_flex"] == 1.354985
    assert spec.hand_armature["r_thumb_cmc_flex"] == 0.0005 and spec.hand_armature["r_thumb_ip"] == 0.0002


REAL_URDF = REPO_ROOT / "assets/flexiv_wuji/rizon4s_right_wuji_real_limits.urdf"
SYSID_JSON = REPO_ROOT / "assets/flexiv_wuji/wuji_hand2_right_sysid_fit_20261009.json"


def test_sysid_spec_is_the_sysid_fit() -> None:
    """with_sysid(FLEXIV_WUJI_RIGHT) = the spec with the sysid fit's armature, friction, passive
    damping and spring, and torque limit 1.0 A x Kt; everything else unchanged."""
    import json

    if not SYSID_JSON.exists():
        pytest.skip(f"{SYSID_JSON.name} is kept out of git (.gitignore); put the fit there to run this")
    real, base = SYSID, SPEC
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


def test_real_limits_urdf() -> None:
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
    for name, q in SPEC.arm_default_joint_pos.items():
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


def test_only_the_sysid_hand_has_a_spring() -> None:
    assert not SPEC.hand_spring_stiffness
    assert len(SYSID.hand_spring_stiffness) == len(SYSID.hand_joint_names)
