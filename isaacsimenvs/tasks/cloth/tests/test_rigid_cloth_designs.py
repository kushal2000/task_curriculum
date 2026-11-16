"""Design geometry and kinematics must preserve the same single-fold task across topologies."""

from __future__ import annotations

import math

import numpy as np
import pytest

from isaacsimenvs.tasks.cloth.backends.rigid import designs as sd


def test_the_grid_is_4096_distinct_designs_spanning_the_ranges():
    g = sd.design_grid()
    assert g.shape == (4096, 3)
    assert len({tuple(r) for r in np.round(g, 9)}) == 4096
    for col, (lo, hi) in enumerate((sd.LENGTH_RANGE, sd.WIDTH_RANGE, sd.THICKNESS_RANGE)):
        levels = np.unique(np.round(g[:, col], 9))
        assert len(levels) == 16
        assert levels[0] == pytest.approx(lo) and levels[-1] == pytest.approx(hi)


@pytest.mark.parametrize("start", [0, 4096, 8192, 20480])
def test_every_sapg_block_of_4096_envs_holds_each_design_once(start):
    d = sd.design_of(np.arange(start, start + 4096), 4096)
    assert sorted(d.tolist()) == list(range(4096))


def test_a_small_run_spans_every_dimension():
    """256 envs (a smoke test) must not all be one length: the env -> design map is permuted."""
    g = sd.design_grid()[sd.design_of(np.arange(256), 4096)]
    for col in range(3):
        assert len(np.unique(np.round(g[:, col], 9))) == 16


def test_a_half_block_holds_half_the_designs_once():
    """2 GPUs x 12288 envs: SAPG blocks of 2048 see 2048 distinct designs, not a repeat."""
    d = sd.design_of(np.arange(2048, 4096), 4096)
    assert len(set(d.tolist())) == 2048


def test_single_hinge_plates_tile_the_sheet():
    s = sd.single_hinge(0.24, 0.12, 0.01)
    lo = [np.subtract(p.center, p.half_extents) for p in s.plates]
    hi = [np.add(p.center, p.half_extents) for p in s.plates]
    assert np.min(lo, axis=0) == pytest.approx([-0.12, -0.06, -0.005])
    assert np.max(hi, axis=0) == pytest.approx([0.12, 0.06, 0.005])
    assert hi[0][0] == pytest.approx(lo[1][0]) == pytest.approx(0.0)   # they meet at the crease
    assert s.ply_gap == pytest.approx(0.01)


@pytest.mark.parametrize("dims", [(0.10, 0.30, 0.002), (0.30, 0.10, 0.020), (0.17, 0.23, 0.0092)])
def test_a_180_deg_fold_lands_every_moving_point_on_its_target(dims):
    """The fold-target rule (reflect across the crease, lift by the ply gap) is exactly where the
    hinge puts the plate: a perfect fold scores 0, and the plate sits on top, not inside."""
    s = sd.single_hinge(*dims)
    length, width, t = dims
    x = np.linspace(1e-6, length / 2, 5)
    y = np.linspace(-width / 2, width / 2, 4)
    pts = np.array([(a, b, 0.0) for a in x for b in y])
    folded = sd.plate_points(s, pts, math.pi)
    target = pts * [-1.0, 1.0, 1.0] + [0.0, 0.0, s.ply_gap]
    assert np.abs(folded - target).max() < 1e-9
    # Folding flips the plate: its TOP face comes down onto the stationary plate's top face
    # (z = t/2), touching it without overlap, and its bottom face ends up on top at 3t/2.
    top = sd.plate_points(s, pts + [0, 0, t / 2], math.pi)
    bottom = sd.plate_points(s, pts + [0, 0, -t / 2], math.pi)
    assert top[:, 2] == pytest.approx(np.full(len(pts), t / 2))
    assert bottom[:, 2] == pytest.approx(np.full(len(pts), 1.5 * t))


def test_positive_angles_lift_the_moving_plate_and_leave_the_other():
    s = sd.single_hinge(0.2, 0.2, 0.01)
    pts = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
    out = sd.plate_points(s, pts, math.radians(90))
    assert out[0, 2] > 0.09
    assert np.allclose(out[1], pts[1])


def test_a_mid_plane_hinge_would_fold_the_plate_into_the_other():
    """Negative control for the hinge height: about the mid-plane the folded points land at z = 0,
    inside the stationary plate, and miss the lifted targets by a whole thickness."""
    s = sd.single_hinge(0.2, 0.2, 0.01)
    mid = sd.SheetSpec(s.length, s.width, s.thickness, s.plates,
                       (sd.Hinge("minus", "plus", (0.0, 0.0, 0.0), (0.0, -1.0, 0.0)),), s.ply_gap)
    pts = np.array([[0.05, 0.02, 0.0]])
    folded = sd.plate_points(mid, pts, math.pi)
    target = pts * [-1.0, 1.0, 1.0] + [0.0, 0.0, s.ply_gap]
    assert np.abs(folded - target).max() == pytest.approx(0.01)


def test_mass_is_per_area_and_inertia_is_the_box_formula():
    s = sd.single_hinge(0.2, 0.1, 0.01)
    m, com, i = sd.box_mass_properties(s.plates[1], density=2.0)
    assert m == pytest.approx(2.0 * 0.1 * 0.1)
    assert com == pytest.approx([0.05, 0.0, 0.0])
    assert i == pytest.approx(m / 12 * np.array([0.1**2 + 0.01**2, 0.1**2 + 0.01**2, 0.1**2 + 0.1**2]))


@pytest.mark.parametrize("panels", [2, 4, 8])
@pytest.mark.parametrize("dims", [(0.10, 0.30, 0.002), (0.30, 0.10, 0.020), (0.17, 0.23, 0.0092)])
def test_normalized_design_scales_geometry_without_changing_topology(panels, dims):
    design = sd.strip(panels)
    unit = design.resolve(1.0, 1.0, 1.0)
    sized = design.resolve(*dims)
    assert sized.joint_names == unit.joint_names
    assert sized.fold_hinge == unit.fold_hinge
    assert len(sized.plates) == panels
    assert len(sized.hinges) == panels - 1
    for reference, actual in zip(unit.plates, sized.plates):
        assert actual.name == reference.name
        assert actual.center == pytest.approx(np.asarray(reference.center) * dims)
        assert actual.half_extents == pytest.approx(np.asarray(reference.half_extents) * dims)
    for reference, actual in zip(unit.hinges, sized.hinges):
        assert (actual.parent, actual.child, actual.axis) == (reference.parent, reference.child, reference.axis)
        assert actual.anchor == pytest.approx(np.asarray(reference.anchor) * dims)
    assert sized.ply_gap == pytest.approx(dims[2])


@pytest.mark.parametrize("panels", [2, 4, 8])
def test_strip_tiles_the_sheet_and_mass_does_not_depend_on_subdivision_or_thickness(panels):
    length, width, density = 0.24, 0.13, 2.7
    for thickness in (0.002, 0.01, 0.02):
        spec = sd.strip(panels).resolve(length, width, thickness)
        lower = np.array([np.subtract(p.center, p.half_extents) for p in spec.plates])
        upper = np.array([np.add(p.center, p.half_extents) for p in spec.plates])
        np.testing.assert_allclose(np.sort(lower[:, 0]), np.linspace(-length / 2, length / 2, panels + 1)[:-1], atol=1e-15)
        np.testing.assert_allclose(np.sort(upper[:, 0]), np.linspace(-length / 2, length / 2, panels + 1)[1:], atol=1e-15)
        np.testing.assert_allclose(lower[:, 1:], np.tile([-width / 2, -thickness / 2], (panels, 1)))
        np.testing.assert_allclose(upper[:, 1:], np.tile([width / 2, thickness / 2], (panels, 1)))
        masses = [sd.box_mass_properties(p, density) for p in spec.plates]
        assert sum(mass for mass, _, _ in masses) == pytest.approx(density * length * width)
        assert all(np.all(inertia > 0) for _, _, inertia in masses)


@pytest.mark.parametrize("panels", [2, 4, 8])
def test_unique_central_fold_hinge_and_acyclic_connected_chain(panels):
    spec = sd.strip(panels).resolve(0.2, 0.12, 0.006)
    central = [i for i, h in enumerate(spec.hinges) if abs(h.anchor[0]) < 1e-12]
    assert central == [spec.fold_hinge]
    assert len(set(spec.joint_names)) == panels - 1
    assert len(set(p.name for p in spec.plates)) == panels
    reached = {spec.plates[0].name}
    for hinge in spec.hinges:
        assert hinge.parent in reached
        assert hinge.child not in reached
        reached.add(hinge.child)
        assert hinge.axis == (0.0, -1.0, 0.0)
        assert hinge.anchor[2] == pytest.approx(spec.thickness / 2)
    assert reached == {p.name for p in spec.plates}
    assert spec.hinges[spec.fold_hinge].limits_deg[1] >= 180.0


@pytest.mark.parametrize("panels", [2, 4, 8])
def test_surface_binding_covers_all_plates_and_assigns_seams_consistently(panels):
    design = sd.strip(panels)
    centers = np.asarray([p.center for p in design.plates])
    assert design.point_owners(centers).tolist() == list(range(panels))
    # Shared boundaries are assigned once, including x=0 on the stationary half.
    seams = np.c_[np.linspace(-0.5, 0.5, panels + 1), np.zeros((panels + 1, 2))]
    owners = design.point_owners(seams)
    by_x = sorted(range(panels), key=lambda i: design.plates[i].center[0])
    assert owners.tolist() == [by_x[0], *by_x]
    assert design.plates[owners[panels // 2]].center[0] < 0.0
    with pytest.raises(ValueError):
        design.point_owners(np.array([[0.51, 0.0, 0.0]]))


@pytest.mark.parametrize("panels", [2, 4, 8])
@pytest.mark.parametrize("dims", [(0.10, 0.30, 0.002), (0.30, 0.10, 0.020)])
def test_only_central_fold_joint_reproduces_common_mirrored_targets(panels, dims):
    design = sd.strip(panels)
    spec = design.resolve(*dims)
    # Include each plate interior and every seam; no grid-resolution assumption.
    x = np.unique(np.r_[np.linspace(-0.5, 0.5, 2 * panels + 1), -0.01, 0.01])
    normalized = np.array([(a, b, 0.0) for a in x for b in (-0.5, 0.0, 0.5)])
    points = normalized * dims
    owners = design.point_owners(normalized)
    poses = {spec.plates[0].name: (np.eye(3), np.zeros(3))}
    for index, hinge in enumerate(spec.hinges):
        rotation, translation = sd.hinge_transform(hinge, math.pi if index == spec.fold_hinge else 0.0)
        parent_rotation, parent_translation = poses[hinge.parent]
        poses[hinge.child] = (
            parent_rotation @ rotation,
            parent_rotation @ translation + parent_translation,
        )
    actual = np.empty_like(points)
    for index, plate in enumerate(spec.plates):
        mask = owners == index
        rotation, translation = poses[plate.name]
        actual[mask] = points[mask] @ rotation.T + translation
    expected = points.copy()
    moving = points[:, 0] > 0.0
    expected[moving] = points[moving] * [-1.0, 1.0, 1.0] + [0.0, 0.0, dims[2]]
    np.testing.assert_allclose(actual, expected, atol=1e-12)


@pytest.mark.parametrize("name,panels", [("single_hinge", 2), ("strip_4", 4), ("strip_8", 8),
                                         ("hinge_band", 4), ("flexible_corners", 6)])
def test_yaml_catalog_partitions_a_symmetric_unit_sheet(name, panels):
    design = sd.load_design(name)
    assert len(design.plates) == panels
    assert len(design.hinges) == panels - 1
    assert design.plates[0].center[0] < 0
    assert design.hinges[design.fold_hinge].parent == design.plates[0].name
    assert sum(sd.polygon_mass_properties(p, 1)[0] for p in design.plates) == pytest.approx(1)

    def signature(vertices):
        return tuple(sorted(tuple(v) for v in np.round(vertices, 10)))

    polygons = {signature(p.vertices) for p in design.plates}
    for mirror in ([-1, 1], [1, -1]):
        assert {signature(np.asarray(p.vertices) * mirror) for p in design.plates} == polygons
    sample = np.array([(x, y, 0) for x in np.linspace(-.5, .5, 31) for y in np.linspace(-.5, .5, 31)])
    owners = design.point_owners(sample)
    assert owners.min() >= 0 and owners.max() < panels
    # Midline samples are assigned to the stationary side, including shared vertices.
    crease_owners = design.point_owners(np.c_[np.zeros(11), np.linspace(-.5, .5, 11), np.zeros(11)])
    assert all(design.plates[i].center[0] < 0 for i in crease_owners)


def test_diagonal_hinges_follow_dimensioned_edges_and_have_matching_frames():
    design = sd.load_design("flexible_corners")
    spec = design.resolve(.3, .1, .006)
    for normalized, resolved in zip(design.hinges, spec.hinges):
        edge = np.asarray(normalized.edge) * [.3, .1, .006]
        np.testing.assert_allclose(resolved.edge, edge)
        direction = edge[1] - edge[0]
        np.testing.assert_allclose(resolved.axis, direction / np.linalg.norm(direction))
        np.testing.assert_allclose(resolved.anchor, edge.mean(axis=0))
        rotation, translation = sd.hinge_transform(resolved, .7)
        # Every point along a hinge stays fixed under its own joint rotation.
        np.testing.assert_allclose(edge @ rotation.T + translation, edge, atol=1e-14)
    assert any(not np.allclose(a.axis, b.axis) for a, b in zip(design.hinges, spec.hinges))


@pytest.mark.parametrize("name", ["single_hinge", "strip_4", "strip_8", "hinge_band", "flexible_corners"])
def test_all_designs_fold_the_positive_half_to_the_same_target(name):
    design = sd.load_design(name)
    spec = design.resolve(.28, .13, .004)
    sample = np.array([(x, y, 0) for x in np.linspace(-.5, .5, 17) for y in np.linspace(-.5, .5, 13)])
    points = sample * [.28, .13, .004]
    owners = design.point_owners(sample)
    poses = {spec.plates[0].name: (np.eye(3), np.zeros(3))}
    for i, hinge in enumerate(spec.hinges):
        r, t = sd.hinge_transform(hinge, math.pi if i == spec.fold_hinge else 0)
        pr, pt = poses[hinge.parent]
        poses[hinge.child] = pr @ r, pr @ t + pt
    actual = np.empty_like(points)
    for i, plate in enumerate(spec.plates):
        r, t = poses[plate.name]
        actual[owners == i] = points[owners == i] @ r.T + t
    expected = points.copy()
    positive = sample[:, 0] > 0
    expected[positive] = points[positive] * [-1, 1, 1] + [0, 0, .004]
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def _inertia_matrix(diagonal, quaternion):
    from scipy.spatial.transform import Rotation
    rotation = Rotation.from_quat(np.asarray(quaternion)[[1, 2, 3, 0]]).as_matrix()
    assert np.linalg.det(rotation) == pytest.approx(1)
    return rotation @ np.diag(diagonal) @ rotation.T


def test_triangle_prism_has_exact_com_and_full_inertia():
    a, b, thickness, density = .3, .12, .006, 2.4
    plate = sd.Plate("triangle", ((0., 0.), (a, 0.), (0., b)), thickness)
    mass, center, diagonal, axes = sd.polygon_mass_properties(plate, density)
    assert mass == pytest.approx(density * a * b / 2)
    np.testing.assert_allclose(center, [a / 3, b / 3, 0])
    expected = mass * np.array([[b*b/18 + thickness**2/12, a*b/36, 0],
                               [a*b/36, a*a/18 + thickness**2/12, 0],
                               [0, 0, (a*a+b*b)/18]])
    np.testing.assert_allclose(_inertia_matrix(diagonal, axes), expected, atol=1e-15)
    assert np.all(diagonal > 0)
    vertices, faces = sd.extrusion_mesh(plate)
    volume = sum(np.dot(vertices[x], np.cross(vertices[y], vertices[z])) / 6 for x, y, z in faces)
    assert volume == pytest.approx(a * b * thickness / 2)
    # Closed mesh: each undirected edge occurs twice.
    from collections import Counter
    edge_counts = Counter(tuple(sorted((int(face[i]), int(face[(i+1) % 3])))) for face in faces for i in range(3))
    assert set(edge_counts.values()) == {2}


def test_rectangle_polygon_inertia_matches_existing_box_formula():
    for plate in sd.single_hinge(.3, .16, .008).plates:
        mass, center, inertia = sd.box_mass_properties(plate, 2.0)
        actual_mass, actual_center, diagonal, axes = sd.polygon_mass_properties(plate, 2.0)
        assert actual_mass == pytest.approx(mass)
        np.testing.assert_allclose(actual_center, center)
        np.testing.assert_allclose(_inertia_matrix(diagonal, axes), np.diag(inertia), atol=1e-15)


def test_corner_ownership_uses_polygon_not_bounding_box():
    design = sd.load_design("flexible_corners")
    points = np.array([[.3, .3, 0], [.49, .49, 0], [-.3, -.3, 0], [-.49, -.49, 0]])
    owners = design.point_owners(points)
    assert len(design.plates[owners[0]].vertices) > 3
    assert len(design.plates[owners[1]].vertices) == 3
    assert len(design.plates[owners[2]].vertices) > 3
    assert len(design.plates[owners[3]].vertices) == 3
    with pytest.raises(ValueError, match="outside"):
        design.point_owners(np.array([[.6, .6, 0]]))


@pytest.mark.parametrize("creases", [
    {"center": [[0, .25], [0, -.25]]},  # partial cut
    {"center": [[0, .5], [0, -.5]], "cross": [[-.5, 0], [.5, 0]]},
    {"offset": [[.1, .5], [.1, -.5]]},  # no central fold
    {"center": [[0, .5], [0, -.5]], "duplicate": [[0, -.5], [0, .5]]},
    {"center": [[0, .5], [0, -.5]], "tee": [[0, 0], [.5, 0]]},
    {"inside": [[-.1, -.1], [.1, -.1]], "second": [[.1, -.1], [0, .1]], "third": [[0, .1], [-.1, -.1]]},
    {"outside": [[0, .6], [0, -.5]]},
    {"malformed": {"a": [0, .5], "b": [0, -.5]}},
])
def test_loader_rejects_unsupported_cuts(tmp_path, creases):
    import yaml
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump({"shape": "rectangle", "creases": creases}))
    with pytest.raises(ValueError):
        sd.load_design(path)


def test_loader_reads_yaml_changes_and_ignores_simulation_status(tmp_path):
    import yaml
    path = tmp_path / "edited.yaml"
    data = {"shape": "rectangle", "creases": {"center": [[0, .5], [0, -.5]]},
            "simulation": {"status": "proposed", "design": "nonexistent"}}
    path.write_text(yaml.safe_dump(data))
    first = sd.load_design(path)
    assert len(first.plates) == 2
    data["creases"]["left"] = [[-.2, .5], [-.2, -.5]]
    data["creases"]["right"] = [[.2, .5], [.2, -.5]]
    path.write_text(yaml.safe_dump(data))
    second = sd.load_design(path)
    assert len(second.plates) == 4
    assert sorted(round(p.half_extents[0] * 2, 6) for p in second.plates) == [.2, .2, .3, .3]
    assert first.source_path == second.source_path == path.resolve()
    path.write_text("creases: [invalid: [")
    with pytest.raises(ValueError, match="invalid cloth design YAML"):
        sd.load_design(path)


def test_path_resolution_and_discovery(tmp_path, monkeypatch):
    import shutil
    path = tmp_path / "local.yaml"
    shutil.copy(sd.resolve_design_path("single_hinge"), path)
    monkeypatch.setattr(sd, "DEFAULT_DESIGN_DIR", tmp_path)
    assert sd.discover_designs() == (path,)
    assert sd.resolve_design_path("local") == path
    assert sd.resolve_design_path("local.yaml") == path
    monkeypatch.chdir(tmp_path.parent)
    assert sd.resolve_design_path(f"{tmp_path.name}/local.yaml") == path
    assert sd.load_design(path).name == "local"
