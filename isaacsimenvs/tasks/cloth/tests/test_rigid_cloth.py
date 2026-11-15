"""The hinged-slat rigid cloth is a valid articulation and can actually fold.

Two classes of failure are covered, both of which have already cost this repo time on the VBD
sheet:

  * **The URDF and the kinematics disagree.** ``forward_kinematics`` is used to define the fold
    target and to check the model without a simulator; if the joint origins written to disk do not
    reproduce it, every geometric claim made here is about a model that is not the one simulated.
    So the rest pose is reconstructed by walking the emitted XML and compared against the analytic
    one, rather than both being read off the same function.
  * **A "perfect" fold that is not perfect.** ``ClothEnv`` lifted its fold targets by
    ``2 * particle_radius`` where cloth settles at ``self_contact_radius``, so a geometrically
    perfect fold scored 14 mm of error -- 40% of the tolerance -- before the policy moved. The
    rigid chain has the same trap with a different constant, so the residual is asserted to be
    exactly one pitch in each of x and z, which is what ``fold_residual`` is for.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import pytest

SIZE = 0.10


@pytest.fixture(scope="module")
def rc(rigid_cloth):
    return rigid_cloth


def _tree(rc, tmp_path, **kw):
    path = rc.generate_rigid_cloth_urdf(tmp_path / "cloth.urdf", **kw)
    return ET.parse(path).getroot()


# ------------------------------------------------------------------------------- structure


def test_articulation_is_a_tree_rooted_at_the_centre(rc, tmp_path):
    root = _tree(rc, tmp_path, size=SIZE, num_slats=16)
    links = [e.get("name") for e in root.findall("link")]
    joints = root.findall("joint")

    assert len(links) == 16
    assert len(joints) == 15, "a tree over N links has exactly N-1 joints"

    children = [j.find("child").get("link") for j in joints]
    assert len(set(children)) == len(children), "a link with two parents is a loop, not a tree"
    base = set(links) - set(children)
    assert base == {rc.slat_link_name(rc.root_slat(16))}


def test_every_joint_is_a_hinge_about_y(rc, tmp_path):
    root = _tree(rc, tmp_path, size=SIZE, num_slats=16)
    for j in root.findall("joint"):
        assert j.get("type") == "revolute"
        assert j.find("axis").get("xyz") == "0 1 0"
        limit = j.find("limit")
        assert float(limit.get("lower")) == pytest.approx(-float(limit.get("upper")))


def test_total_mass_is_density_times_area(rc, tmp_path):
    root = _tree(rc, tmp_path, size=SIZE, num_slats=16, density=2.0)
    total = sum(float(m.get("value")) for m in root.iter("mass"))
    assert total == pytest.approx(2.0 * SIZE * SIZE)


def test_collider_radius_is_half_the_pitch_by_default(rc, tmp_path):
    root = _tree(rc, tmp_path, size=SIZE, num_slats=16)
    pitch = rc.slat_pitch(SIZE, 16)
    for cyl in root.iter("cylinder"):
        assert float(cyl.get("radius")) == pytest.approx(pitch / 2.0)
        assert float(cyl.get("length")) == pytest.approx(SIZE)


# ------------------------------------------------------------------------------- kinematics


def test_urdf_joint_origins_reproduce_the_analytic_rest_pose(rc, tmp_path):
    """Walk the emitted XML and rebuild the rest pose from it alone."""
    num_slats = 16
    root = _tree(rc, tmp_path, size=SIZE, num_slats=num_slats)

    origin_of: dict[str, tuple[str, float]] = {}
    for j in root.findall("joint"):
        dx = float(j.find("origin").get("xyz").split()[0])
        origin_of[j.find("child").get("link")] = (j.find("parent").get("link"), dx)

    base = rc.slat_link_name(rc.root_slat(num_slats))
    frame_x = {base: rc.slat_centres(SIZE, num_slats)[rc.root_slat(num_slats)]}
    # Repeat until closed: document order already puts parents first, but the test should not
    # depend on that -- it is the tree that is being checked, not the serialisation.
    while len(frame_x) < num_slats:
        progressed = False
        for child, (parent, dx) in origin_of.items():
            if child not in frame_x and parent in frame_x:
                frame_x[child] = frame_x[parent] + dx
                progressed = True
        assert progressed, "joint graph is disconnected from the root"

    from_urdf = [frame_x[rc.slat_link_name(i)] for i in range(num_slats)]
    analytic = [f[0] for f in rc.forward_kinematics(SIZE, num_slats)]
    assert from_urdf == pytest.approx(analytic)


def test_rest_pose_is_a_flat_sheet_of_the_requested_size(rc, tmp_path):
    num_slats = 16
    pitch = rc.slat_pitch(SIZE, num_slats)
    xs = []
    for i in range(num_slats):
        for pt in rc.slat_frame_points(SIZE, num_slats, i):
            x, y, z = rc.point_in_sheet_frame(SIZE, num_slats, i, pt)
            assert z == pytest.approx(0.0)
            assert abs(y) == pytest.approx(SIZE / 2.0)
            xs.append(x)
    assert min(xs) == pytest.approx(-SIZE / 2.0)
    assert max(xs) == pytest.approx(SIZE / 2.0)
    # No gaps and no overlaps: consecutive slat seams are exactly one pitch apart.
    seams = sorted(set(round(v, 9) for v in xs))
    assert len(seams) == num_slats + 1
    assert all(
        b - a == pytest.approx(pitch) for a, b in zip(seams, seams[1:])
    )


def test_corners_of_the_moving_half_span_an_area(rc):
    """Four collinear points cannot determine a rotation -- the defect that made `object_rot`
    read identity on the VBD sheet."""
    pts = [
        rc.point_in_sheet_frame(SIZE, 16, i, p) for i, p in rc.corner_points(SIZE, 16)
    ]
    assert len({round(p[0], 9) for p in pts}) == 2
    assert len({round(p[1], 9) for p in pts}) == 2
    # The near pair sits exactly ON the crease (x = 0) -- it is the inboard edge of the first
    # moving slat -- and the far pair at the sheet's edge.
    assert all(p[0] >= -1e-12 for p in pts), "corners must be on the moving (+x) half"
    assert sorted(round(p[0], 9) for p in pts)[:2] == [0.0, 0.0]
    assert max(p[0] for p in pts) == pytest.approx(SIZE / 2.0)


# ------------------------------------------------------------------------------- the fold


def test_a_perfect_fold_stacks_the_moving_half_one_pitch_up(rc):
    num_slats = 16
    pitch = rc.slat_pitch(SIZE, num_slats)
    angles = rc.folded_joint_angles(num_slats)
    moving = rc.half_slats(num_slats, positive=True)

    # The wall, then the returning ply: every slat past the first is upside down and level.
    frames = rc.forward_kinematics(SIZE, num_slats, angles)
    assert frames[moving[0]][2] == pytest.approx(-math.pi / 2.0)
    for i in moving[1:]:
        assert frames[i][2] == pytest.approx(-math.pi)
        assert frames[i][1] == pytest.approx(pitch)

    assert rc.fold_ply_gap(SIZE, num_slats) == pytest.approx(pitch)
    # Stationary half untouched.
    for i in rc.half_slats(num_slats, positive=False):
        assert frames[i][1] == pytest.approx(0.0)
        assert frames[i][2] == pytest.approx(0.0)


def test_fold_residual_is_one_pitch_in_each_axis(rc):
    for num_slats in (12, 16, 20):
        pitch = rc.slat_pitch(SIZE, num_slats)
        dx, dz = rc.fold_residual(SIZE, num_slats)
        assert dx == pytest.approx(pitch)
        assert dz == pytest.approx(pitch)


def test_odd_slat_counts_fold_with_no_in_plane_residual(rc):
    """Parity decides where the WALL of the fold comes from, and odd wins.

    A mid-plane fold must spend one slat standing vertically. An odd count puts a slat astride
    ``x = 0`` that belongs to neither ply, so it is the wall for free and both plies keep their full
    length. An even count puts a seam there, so the wall is taken out of the moving ply and the
    returning ply lands one width short.

    This reverses what an earlier version of this module asserted. That version derived the fold by
    bending the first two slats of the MOVING half, which takes the wall out of that ply at ANY
    parity, and so measured odd counts as a pitch worse. The wall slat is a free choice; the crease
    slat is the right one whenever it exists.
    """
    for odd in (15, 17, 21):
        assert rc.crease_slat(odd) is not None
        assert rc.fold_residual(SIZE, odd)[0] == pytest.approx(0.0, abs=1e-12)
    for even in (12, 16, 20):
        assert rc.crease_slat(even) is None
        assert rc.fold_residual(SIZE, even)[0] == pytest.approx(rc.slat_pitch(SIZE, even))


def test_the_wall_is_the_crease_slat_when_there_is_one(rc):
    """The mechanism behind the parity rule, asserted directly rather than through the residual."""
    spec = rc.uniform_chain(SIZE, 17, hinge="mid")
    assert spec.wall == spec.crease, "an odd chain must spend its crease slat, not a ply slat"
    assert spec.wall not in spec.moving and spec.wall not in spec.stationary

    even = rc.uniform_chain(SIZE, 16, hinge="mid")
    assert even.crease is None
    assert even.wall == even.moving[0], "with a seam at x=0 the wall has to come out of the ply"


def test_default_num_slats_follows_the_hinge_mode(rc):
    """The two hinge modes want OPPOSITE parity, so the default cannot be one number.

    A mid-plane fold wants a crease slat to spend as the wall (odd). A surface fold spends no slat
    but folds about a SEAM, which has to sit at x = 0 (even).
    """
    assert rc.default_num_slats("mid", 16) % 2 == 1
    assert rc.default_num_slats("surface", 16) % 2 == 0
    assert rc.default_num_slats("surface", 16) == rc.DEFAULT_NUM_SLATS
    for target in (7, 8, 16, 17):
        assert rc.fold_residual(SIZE, rc.default_num_slats("mid", target))[0] == pytest.approx(
            0.0, abs=1e-12
        )


def test_the_folded_ply_lies_over_the_stationary_half(rc):
    """A fold, not a slide: every folded point must be above the sheet's own stationary side."""
    num_slats = 16
    angles = rc.folded_joint_angles(num_slats)
    for i in rc.half_slats(num_slats, positive=True)[1:]:
        for pt in rc.slat_frame_points(SIZE, num_slats, i):
            x, _, z = rc.point_in_sheet_frame(SIZE, num_slats, i, pt, angles)
            assert z > 0.0
            assert -SIZE / 2.0 <= x <= 0.0 + 1e-9


# ------------------------------------------------------------- thickness / limit consistency


def test_default_thickness_and_joint_limit_are_the_same_decision(rc, tmp_path):
    """Collider radius pitch/2 is exactly the thickness at which 90 degrees per joint brings slats
    i and i+2 into contact -- so the default limit is not a separate guess."""
    pitch = rc.slat_pitch(SIZE, 16)
    assert rc.max_bend_for_thickness(pitch, pitch) == pytest.approx(math.pi / 2.0, abs=1e-6)

    root = _tree(rc, tmp_path, size=SIZE, num_slats=16)
    for j in root.findall("joint"):
        assert float(j.find("limit").get("upper")) == pytest.approx(math.pi / 2.0, abs=1e-6)


def test_a_thinner_collider_may_bend_further(rc):
    pitch = rc.slat_pitch(SIZE, 16)
    angles = [rc.max_bend_for_thickness(pitch, f * pitch) for f in (1.0, 0.5, 0.25)]
    assert angles == sorted(angles), "thinner slats must permit at least as much bend"
    assert angles[-1] > math.pi / 2.0


def test_plate_stiffness_scales_with_the_cube_of_fabric_thickness(rc):
    k1 = rc.plate_joint_stiffness(1.0e7, 3.0e-4, SIZE, rc.slat_pitch(SIZE, 16))
    k2 = rc.plate_joint_stiffness(1.0e7, 6.0e-4, SIZE, rc.slat_pitch(SIZE, 16))
    assert k2 / k1 == pytest.approx(8.0)


# ------------------------------------------------------------------------------------ guards


def test_rejects_configurations_that_cannot_fold(rc, tmp_path):
    with pytest.raises(ValueError):
        rc.generate_rigid_cloth_urdf(tmp_path / "a.urdf", num_slats=2)
    with pytest.raises(ValueError):
        rc.generate_rigid_cloth_urdf(tmp_path / "b.urdf", size=0.0)
    with pytest.raises(ValueError):
        rc.generate_rigid_cloth_urdf(tmp_path / "c.urdf", shape="sphere")
    with pytest.raises(ValueError):
        rc.generate_rigid_cloth_urdf(tmp_path / "d.urdf", density=-1.0)


# ----------------------------------------------------------------- the surface hinge variant
#
# Raising the axis from the slab's mid-plane to its top face is what makes a fold exact: the body
# hangs a half-thickness below its own axis, so ONE joint at 180 degrees lands the child flat on the
# parent instead of coincident with it. The price is that every hinge becomes one-directional, and
# these tests pin both halves of that trade so neither can be lost silently.


def test_surface_hinge_folds_with_a_single_joint(rc):
    angles = rc.folded_joint_angles(16, hinge="surface")
    bent = [i for i, a in enumerate(angles) if abs(a) > 1e-12]
    assert len(bent) == 1, "a surface hinge needs no wall slat, so one joint does the whole fold"
    assert abs(angles[bent[0]]) == pytest.approx(math.pi)
    assert bent[0] == rc.half_slats(16, positive=True)[0]


def test_surface_hinge_has_no_in_plane_fold_residual(rc):
    """The whole point: the folded half lands on its exact mirror, not a slat short."""
    for num_slats in (12, 16, 20):
        for thickness in (None, 0.002):
            dx, dz = rc.fold_residual(SIZE, num_slats, thickness, "surface")
            assert dx == pytest.approx(0.0, abs=1e-12)
            # The only residual left is the ply gap, which is now the slab thickness and so is
            # free to be chosen -- unlike the mid-plane hinge, where it is pinned to the pitch.
            expected = rc.slat_pitch(SIZE, num_slats) if thickness is None else thickness
            assert dz == pytest.approx(expected)


def test_surface_hinge_ply_gap_is_thickness_not_pitch(rc):
    assert rc.fold_ply_gap(SIZE, 16, 0.002, "surface") == pytest.approx(0.002)
    assert rc.fold_ply_gap(SIZE, 16, 0.002, "mid") == pytest.approx(rc.slat_pitch(SIZE, 16))


def test_surface_hinge_folded_half_covers_the_stationary_half_exactly(rc):
    num_slats, thickness = 16, 0.002
    angles = rc.folded_joint_angles(num_slats, hinge="surface")
    xs = []
    for i in rc.half_slats(num_slats, positive=True):
        for pt in rc.slat_frame_points(SIZE, num_slats, i, thickness=thickness, hinge="surface"):
            x, _y, z = rc.point_in_sheet_frame(SIZE, num_slats, i, pt, angles)
            xs.append(x)
            assert z == pytest.approx(thickness / 2.0)
    assert min(xs) == pytest.approx(-SIZE / 2.0)
    assert max(xs) == pytest.approx(0.0, abs=1e-12)


def test_surface_hinge_limits_are_one_sided_and_mirrored(rc, tmp_path):
    """The cost of the exact fold, pinned so it cannot be forgotten.

    The slab hangs below its own axis, so only one sense of rotation sweeps it clear of its parent;
    the other drives it straight through. The permitted sense mirrors across the root.
    """
    root = _tree(rc, tmp_path, size=SIZE, num_slats=16, shape="box", hinge="surface")
    seen = {"up": 0, "down": 0}
    for j in root.findall("joint"):
        child = int(j.find("child").get("link").removeprefix(rc.SLAT_LINK_PREFIX))
        lo = float(j.find("limit").get("lower"))
        hi = float(j.find("limit").get("upper"))
        assert lo == pytest.approx(0.0) or hi == pytest.approx(0.0), "a surface hinge is one-sided"
        assert (lo, hi) != (0.0, 0.0)
        if rc.fold_direction(child, 16) < 0:
            assert (lo, hi) == pytest.approx((-math.pi, 0.0))
            seen["up"] += 1
        else:
            assert (lo, hi) == pytest.approx((0.0, math.pi))
            seen["down"] += 1
    assert seen["up"] and seen["down"], "both branches must be represented"


def test_mid_hinge_limits_stay_symmetric(rc, tmp_path):
    root = _tree(rc, tmp_path, size=SIZE, num_slats=16, hinge="mid")
    for j in root.findall("joint"):
        lo, hi = float(j.find("limit").get("lower")), float(j.find("limit").get("upper"))
        assert lo == pytest.approx(-hi)
        assert hi > 0.0


def test_surface_hinge_offsets_the_body_below_its_axis(rc):
    pitch = rc.slat_pitch(SIZE, 16)
    _dx, dz_mid = rc._geom_offset(9, 16, pitch, 0.002, "mid")
    _dx, dz_surf = rc._geom_offset(9, 16, pitch, 0.002, "surface")
    assert dz_mid == 0.0
    assert dz_surf == pytest.approx(-0.001)


def test_rejects_an_unknown_hinge_mode(rc, tmp_path):
    with pytest.raises(ValueError):
        rc.generate_rigid_cloth_urdf(tmp_path / "h.urdf", hinge="edge")
    with pytest.raises(ValueError):
        rc.folded_joint_angles(16, hinge="edge")


# ------------------------------------------------------------------ the five approximations
#
# `VARIANTS` holds five specs of the same 100 mm sheet, each isolating one decision. These tests pin
# the property that makes each one worth having, so a refactor cannot quietly collapse them into
# each other.


def test_every_variant_is_a_valid_articulation(rc, tmp_path):
    for name, spec in rc.VARIANTS.items():
        root = ET.parse(rc.write_chain_urdf(spec, tmp_path / f"{name}.urdf")).getroot()
        links = [e.get("name") for e in root.findall("link")]
        joints = root.findall("joint")
        assert len(links) == spec.num_slats, name
        assert len(joints) == spec.num_slats - 1, f"{name}: a tree over N links has N-1 joints"
        children = [j.find("child").get("link") for j in joints]
        assert len(set(children)) == len(children), f"{name}: a link with two parents is a loop"
        assert set(links) - set(children) == {rc.slat_link_name(spec.root)}, name


def test_every_variant_spans_the_sheet_and_carries_its_mass(rc, tmp_path):
    for name, spec in rc.VARIANTS.items():
        assert spec.span == pytest.approx(SIZE), name
        root = ET.parse(rc.write_chain_urdf(spec, tmp_path / f"{name}.urdf", density=2.0)).getroot()
        total = sum(float(m.get("value")) for m in root.iter("mass"))
        assert total == pytest.approx(2.0 * SIZE * SIZE), name


def test_variant_mass_follows_width_not_slat_count(rc, tmp_path):
    """The crease bar is 2 mm of a 100 mm sheet, so it must carry 2% of the mass, not a third.

    Splitting mass evenly across slats is right only for a uniform chain. On `box3-mid` it would make
    a 2 mm bar as heavy as a 49 mm plate, which changes both the drape and where the sheet balances.
    """
    spec = rc.VARIANTS["box3-mid"]
    root = ET.parse(rc.write_chain_urdf(spec, tmp_path / "s.urdf", density=2.0)).getroot()
    masses = [float(m.get("value")) for m in root.iter("mass")]
    assert masses[1] / sum(masses) == pytest.approx(spec.widths[1] / SIZE)
    assert masses[0] == pytest.approx(masses[2])


def test_mid_hinge_ply_gap_is_the_walls_width(rc):
    """The generalisation that makes `box3-mid` possible: a mid-plane fold's ply gap is the width of
    the slat standing between the plies, which is the pitch only because a uniform chain has one."""
    for name in ("box-mid-odd", "cyl-mid-odd", "box3-mid"):
        spec = rc.VARIANTS[name]
        assert rc.chain_ply_gap(spec) == pytest.approx(spec.widths[spec.wall]), name
        assert rc.chain_fold_residual(spec)[1] == pytest.approx(spec.widths[spec.wall]), name


def test_only_the_thin_crease_bar_gets_a_thin_ply_gap_from_a_mid_hinge(rc):
    """Why `box3-mid` exists at all, stated as a comparison.

    A uniform mid-plane chain's ply gap is its pitch, so getting cloth-like plies that way needs an
    absurd slat count. Narrowing only the WALL gets the same gap for three bodies.
    """
    uniform = rc.VARIANTS["box-mid-odd"]
    stacked = rc.VARIANTS["box3-mid"]
    assert rc.chain_ply_gap(uniform) == pytest.approx(SIZE / 17)
    assert rc.chain_ply_gap(stacked) == pytest.approx(0.002)
    assert stacked.num_slats < uniform.num_slats
    # Both land exactly on the mirror -- the gap is the only difference.
    assert rc.chain_fold_residual(uniform)[0] == pytest.approx(0.0, abs=1e-12)
    assert rc.chain_fold_residual(stacked)[0] == pytest.approx(0.0, abs=1e-12)


def test_a_thin_uniform_mid_chain_folds_to_a_hovering_ply(rc):
    """The trap `uniform_chain`'s default thickness avoids, pinned so it stays visible.

    A mid-plane ply gap is the wall's width no matter how thin the slab is, so a 2 mm slab on a
    5.9 mm pitch folds to a ply floating 3.9 mm above the one below it -- held up by the joint limit
    rather than by contact, which is the same soft-limit dependence that makes the surface hinge
    fragile. `thickness=None` (one pitch) is what makes the fold rest on something.
    """
    thin = rc.uniform_chain(SIZE, 17, thickness=0.002, hinge="mid", shape="box")
    assert rc.chain_ply_gap(thin) - thin.thickness == pytest.approx(SIZE / 17 - 0.002)
    fat = rc.uniform_chain(SIZE, 17, hinge="mid", shape="box")
    assert rc.chain_ply_gap(fat) == pytest.approx(fat.thickness), "plies must touch"


def test_the_stacked_chain_limit_is_exactly_a_right_angle(rc):
    """`[49, 2, 49]` with a 2 mm slab: at 90 degrees per joint the two plates are exactly touching,
    so the collider geometry itself stops the fold at the pose the fold wants."""
    spec = rc.VARIANTS["box3-mid"]
    assert spec.joint_limit == pytest.approx(math.pi / 2.0, abs=1e-6)


def test_max_bend_finds_the_first_crossing_not_the_endpoint(rc):
    """A non-uniform chain's slat separation is not monotone, and a bisection anchored at pi misses it.

    `[49, 2, 49]` dives to 2 mm at 90 degrees and climbs back to 47 mm at 180. Checking only the
    endpoint reports "never touches" for a chain that collides squarely mid-range -- which would have
    emitted a 180-degree limit for `box3-mid` and let its plates pass through each other.
    """
    assert rc._max_bend(0.049, 0.002, 0.049, 0.002) == pytest.approx(math.pi / 2.0, abs=1e-6)
    # The endpoint really is clear, which is what makes the naive check wrong rather than unlucky.
    sep_at_pi = abs(0.049 / 2 - 0.002 + 0.049 / 2)
    assert sep_at_pi > 0.002


def test_two_slats_fold_with_a_surface_hinge_but_not_a_mid_one(rc, tmp_path):
    """The minimal model. A surface fold spends no wall, so two plates and one joint suffice; a
    mid-plane fold needs a wall and a returning ply, so two is not enough."""
    two = rc.VARIANTS["box2-surface"]
    assert two.num_slats == 2 and two.foldable()
    bent = [i for i, a in enumerate(rc.chain_fold_angles(two)) if abs(a) > 1e-12]
    assert bent == [1]
    assert rc.chain_fold_residual(two)[0] == pytest.approx(0.0, abs=1e-12)
    assert rc.chain_fold_residual(two)[1] == pytest.approx(two.thickness)

    mid_two = rc.ChainSpec(widths=(0.05, 0.05), size=SIZE, thickness=0.002, hinge="mid")
    assert not mid_two.foldable()
    with pytest.raises(ValueError):
        rc.write_chain_urdf(mid_two, tmp_path / "x.urdf")


def test_mid_variants_keep_both_bend_directions(rc, tmp_path):
    """The property the surface hinge trades away: a mid-plane sheet has no distinguished up, so it
    can be folded either way and works upside down."""
    for name in ("box-mid-odd", "cyl-mid-odd", "box3-mid"):
        root = ET.parse(rc.write_chain_urdf(rc.VARIANTS[name], tmp_path / f"{name}.urdf")).getroot()
        for j in root.findall("joint"):
            lo, hi = float(j.find("limit").get("lower")), float(j.find("limit").get("upper"))
            assert lo == pytest.approx(-hi) and hi > 0.0, name


def test_surface_variants_are_one_sided(rc, tmp_path):
    for name in ("box-surface", "box2-surface"):
        root = ET.parse(rc.write_chain_urdf(rc.VARIANTS[name], tmp_path / f"{name}.urdf")).getroot()
        for j in root.findall("joint"):
            lo, hi = float(j.find("limit").get("lower")), float(j.find("limit").get("upper"))
            assert (lo == pytest.approx(0.0)) != (hi == pytest.approx(0.0)), name


def test_variant_urdf_origins_reproduce_each_variants_rest_pose(rc, tmp_path):
    """The same independent-reconstruction check as the uniform case, over non-uniform widths.

    Joint origins are the parent's WIDTH (halved at the root), so a chain whose widths differ is
    exactly where an origin computed from a single pitch would go wrong.
    """
    for name, spec in rc.VARIANTS.items():
        root_el = ET.parse(rc.write_chain_urdf(spec, tmp_path / f"{name}.urdf")).getroot()
        origin_of = {
            j.find("child").get("link"): (
                j.find("parent").get("link"),
                float(j.find("origin").get("xyz").split()[0]),
            )
            for j in root_el.findall("joint")
        }
        base = rc.slat_link_name(spec.root)
        frame_x = {base: spec.centres[spec.root]}
        while len(frame_x) < spec.num_slats:
            progressed = False
            for child, (parent, dx) in origin_of.items():
                if child not in frame_x and parent in frame_x:
                    frame_x[child] = frame_x[parent] + dx
                    progressed = True
            assert progressed, f"{name}: joint graph is disconnected from the root"
        from_urdf = [frame_x[rc.slat_link_name(i)] for i in range(spec.num_slats)]
        assert from_urdf == pytest.approx([f[0] for f in rc.chain_frames(spec)]), name


def test_every_variant_lays_a_flat_sheet_with_no_gaps(rc):
    for name, spec in rc.VARIANTS.items():
        xs = []
        for i in range(spec.num_slats):
            for pt in rc.chain_points(spec, i):
                x, y, z = rc.chain_point_in_sheet_frame(spec, i, pt)
                expected_z = 0.0 if spec.hinge == "mid" else -spec.thickness / 2.0
                assert z == pytest.approx(expected_z), name
                assert abs(y) == pytest.approx(SIZE / 2.0), name
                xs.append(x)
        assert min(xs) == pytest.approx(-SIZE / 2.0), name
        assert max(xs) == pytest.approx(SIZE / 2.0), name
        seams = sorted(set(round(v, 9) for v in xs))
        assert len(seams) == spec.num_slats + 1, f"{name}: slats must abut without gaps or overlap"


def test_every_variants_folded_ply_lands_over_the_stationary_half(rc):
    """A fold, not a slide: every folded point must be above the sheet's own stationary side."""
    for name, spec in rc.VARIANTS.items():
        angles = rc.chain_fold_angles(spec)
        for i in spec.moving:
            if spec.hinge == "mid" and i == spec.wall:
                continue  # the wall is vertical, not part of either ply
            for pt in rc.chain_points(spec, i):
                x, _y, z = rc.chain_point_in_sheet_frame(spec, i, pt, angles)
                assert z > 0.0, f"{name}: slat {i} did not lift"
                assert -SIZE / 2.0 - 1e-9 <= x <= 1e-9, f"{name}: slat {i} landed off the sheet"


def test_rejects_malformed_specs(rc):
    with pytest.raises(ValueError):
        rc.ChainSpec(widths=(0.05,), size=SIZE)
    with pytest.raises(ValueError):
        rc.ChainSpec(widths=(0.05, -0.05), size=SIZE)
    with pytest.raises(ValueError):
        rc.ChainSpec(widths=(0.05, 0.05), size=SIZE, hinge="edge")
    with pytest.raises(ValueError):
        rc.ChainSpec(widths=(0.05, 0.05), size=SIZE, shape="sphere")
    with pytest.raises(ValueError):
        rc.ChainSpec(widths=(0.05, 0.05), size=SIZE, thickness=0.0)
    with pytest.raises(ValueError):
        rc.stacked_chain(wall_width=0.2)
    with pytest.raises(ValueError):
        rc.variant("no-such-variant")


# ------------------------------------------------------- fidelity against the VBD sheet it replaces


def test_the_chain_carries_the_same_mass_as_the_cloth(rc, tmp_path):
    ref = rc.CLOTH_REFERENCE
    for name, spec in rc.VARIANTS.items():
        root = ET.parse(
            rc.write_chain_urdf(spec, tmp_path / f"{name}.urdf", density=ref["density"])
        ).getroot()
        total = sum(float(m.get("value")) for m in root.iter("mass"))
        assert total == pytest.approx(ref["density"] * ref["size"] ** 2), name


def test_no_rigid_slab_can_match_both_of_the_cloths_two_thicknesses(rc):
    """The cloth presents 8 mm to rigid bodies and 2 mm to itself; a slab has one shape.

    A slab's standoff is ``thickness/2`` and two slabs face to face are ``thickness`` apart, so any
    rigid sheet obeys ``ply_gap >= 2 * standoff``. The cloth runs the ratio the other way --
    2.5 mm of ply gap against an 8 mm standoff -- so this is not a tuning failure, it is
    unrepresentable, and the ply gap is the half that gets matched because the fold is scored on it.
    """
    ref = rc.CLOTH_REFERENCE
    assert ref["measured_ply_gap"] < 2.0 * ref["particle_radius"], "the cloth inverts the ratio"

    for name, spec in rc.VARIANTS.items():
        standoff = spec.thickness / 2.0
        assert rc.chain_ply_gap(spec) >= 2.0 * standoff - 1e-12, name

    # The thin variants match the ply gap to well inside the 40 mm keypoint tolerance; none of them
    # get anywhere near reproducing the 8 mm standoff, and that is the accepted trade.
    thin = rc.VARIANTS["box3-mid"]
    assert abs(rc.chain_ply_gap(thin) - ref["measured_ply_gap"]) < 1e-3
    assert thin.thickness / 2.0 < ref["particle_radius"] / 4.0


def test_cloth_effective_friction_is_the_geometric_mean(rc):
    """Newton's VBD mixes friction as ``sqrt(mu_cloth * mu_shape)`` -- verified in
    ``newton/_src/solvers/vbd/rigid_vbd_kernels.py``, ``mixed_mu = wp.sqrt(...)``.

    So the cloth's 0.25 against the 1.5 fingertips is 0.61, not 0.25. Copying the raw 0.25 onto a
    rigid slat does NOT copy the behaviour, because MJWarp mixes by element-wise maximum instead.
    """
    ref = rc.CLOTH_REFERENCE
    assert rc.cloth_effective_friction(ref["shape_mu"]["finger_tip"]) == pytest.approx(0.612, abs=1e-3)
    assert rc.cloth_effective_friction(ref["shape_mu"]["table"]) == pytest.approx(0.354, abs=1e-3)
    # The one contact where the two mixing rules agree: slat on slat, both materials the cloth's own.
    mu = ref["soft_contact_mu"]
    assert rc.cloth_effective_friction(mu) == pytest.approx(max(mu, mu))
    # ...and the one where they cannot be reconciled: max() can never reach below either input.
    assert rc.cloth_effective_friction(1.5) < 1.5
    with pytest.raises(ValueError):
        rc.cloth_effective_friction(-1.0)


def test_contact_pairs_impose_the_cloths_mixing_rule(rc):
    """One pair per slat per external surface, each carrying the geometric mean.

    MJWarp takes the element-wise MAXIMUM of two materials, so a 0.25 sheet on a 1.5 fingertip rubs
    at 1.5 where the cloth rubs at 0.61. An explicit pair overrides that, and MJWarp honours it
    (`collision_core.py` reads `pair_friction` ahead of the per-geom maximum).
    """
    spec = rc.VARIANTS["box3-mid"]
    externals = {"finger": 1.5, "table": 0.5}
    pairs = rc.cloth_contact_pairs(spec, externals)

    assert len(pairs) == spec.num_slats * len(externals)
    assert {p[0] for p in pairs} == {rc.slat_link_name(i) for i in range(spec.num_slats)}
    for link, geom, mu in pairs:
        assert mu == pytest.approx(rc.cloth_effective_friction(externals[geom]))
        # The pair must be strictly slipperier than the maximum rule it replaces -- that is the
        # entire reason it exists.
        assert mu < max(rc.CLOTH_REFERENCE["soft_contact_mu"], externals[geom])

    with pytest.raises(ValueError):
        rc.cloth_contact_pairs(spec, {})


def test_every_external_surface_needs_a_pair_not_just_the_grippy_ones(rc):
    """The maximum rule is wrong against everything, only by different amounts."""
    cloth_mu = rc.CLOTH_REFERENCE["soft_contact_mu"]
    for shape_mu in rc.CLOTH_REFERENCE["shape_mu"].values():
        assert rc.cloth_effective_friction(shape_mu) < max(cloth_mu, shape_mu)
    # Fingertips are over-gripped ~2.5x and the table ~1.4x: same defect, different size.
    finger = rc.CLOTH_REFERENCE["shape_mu"]["finger_tip"]
    table = rc.CLOTH_REFERENCE["shape_mu"]["table"]
    assert max(cloth_mu, finger) / rc.cloth_effective_friction(finger) == pytest.approx(2.45, abs=0.05)
    assert max(cloth_mu, table) / rc.cloth_effective_friction(table) == pytest.approx(1.41, abs=0.05)


def test_slat_on_slat_needs_no_pair(rc):
    """The one contact the two mixing rules agree on -- and the one the fold depends on."""
    mu = rc.CLOTH_REFERENCE["soft_contact_mu"]
    assert rc.cloth_effective_friction(mu, mu) == pytest.approx(max(mu, mu))
