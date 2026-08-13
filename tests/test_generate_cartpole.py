"""The generated articulation must have the same DOF count at every difficulty.

That is not a style preference: Isaac Lab builds one `Articulation` view across all envs,
so a varying joint count silently drops envs from the view. Difficulty is expressed by
locking joints at runtime, never by generating a different articulation.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest


def _parse(path):
    return ET.parse(path).getroot()


def test_emits_one_joint_per_link_plus_the_slider(tmp_path, gen_cartpole):
    for n_max in (1, 2, 4, 6):
        root = _parse(
            gen_cartpole.generate_cartpole_urdf(tmp_path / f"c{n_max}.urdf", [0.25] * n_max)
        )
        joints = [j.get("name") for j in root.findall("joint")]
        assert joints[0] == gen_cartpole.CART_JOINT_NAME
        assert joints[1:] == [gen_cartpole.pole_joint_name(i) for i in range(n_max)]
        # rail + cart + one link per pole
        assert len(root.findall("link")) == n_max + 2


def test_joint_count_is_constant_across_a_length_pool(tmp_path, gen_cartpole):
    """Different link lengths, identical DOF count — the invariant the view depends on."""
    pool = [[0.25, 0.25, 0.25], [0.1, 0.5, 0.15], [0.4, 0.2, 0.3]]
    counts = {
        len(_parse(gen_cartpole.generate_cartpole_urdf(tmp_path / f"p{i}.urdf", L)).findall("joint"))
        for i, L in enumerate(pool)
    }
    assert counts == {4}


def test_masses_and_inertias_are_positive_and_scale_with_length(tmp_path, gen_cartpole):
    root = _parse(gen_cartpole.generate_cartpole_urdf(tmp_path / "c.urdf", [0.1, 0.5, 0.3]))
    masses = [float(m.get("value")) for m in root.findall(".//mass")]
    assert all(m > 0 for m in masses)
    # Last three are the poles, in declaration order; mass follows length.
    short, long_, mid = masses[-3:]
    assert short < mid < long_

    for inertia in root.findall(".//inertia"):
        assert all(float(inertia.get(k)) > 0 for k in ("ixx", "iyy", "izz"))


def test_pole_joints_carry_explicit_limits(tmp_path, gen_cartpole):
    """`reset_utils` narrows these limits to lock a joint, and PhysX can only narrow a
    limit that already exists — a `continuous` joint would have none."""
    root = _parse(gen_cartpole.generate_cartpole_urdf(tmp_path / "c.urdf", [0.25] * 3))
    for joint in root.findall("joint"):
        if joint.get("name").startswith(gen_cartpole.POLE_JOINT_PREFIX):
            assert joint.get("type") == "revolute"
            limit = joint.find("limit")
            assert limit is not None
            assert float(limit.get("upper")) > float(limit.get("lower"))


def test_rejects_degenerate_geometry(tmp_path, gen_cartpole):
    with pytest.raises(ValueError):
        gen_cartpole.generate_cartpole_urdf(tmp_path / "bad.urdf", [])
    with pytest.raises(ValueError):
        gen_cartpole.generate_cartpole_urdf(tmp_path / "bad.urdf", [0.25, -0.1])
