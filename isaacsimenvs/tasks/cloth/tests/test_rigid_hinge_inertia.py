"""Physical reference inertia uses full polygon tensors and every child descendant."""
from dataclasses import replace

import numpy as np
import pytest

from isaacsimenvs.tasks.cloth.backends.rigid import designs
from isaacsimenvs.tasks.cloth.backends.rigid.inertia import hinge_inertias


@pytest.mark.parametrize('dims', [(.2, .15, .004), (.12, .31, .012)])
def test_single_hinge_matches_edge_axis_box_inertia(dims):
    length, width, thickness = dims
    density = .1
    spec = designs.single_hinge(*dims)
    mass = density * length/2 * width
    expected = mass * ((length/2)**2 + thickness**2) / 3
    np.testing.assert_allclose(hinge_inertias(spec, density), [expected], rtol=1e-13)


@pytest.mark.parametrize('name', ['strip_4', 'strip_8', 'hinge_band'])
def test_strip_subtrees_equal_locked_contiguous_rectangles(name):
    spec = designs.load_design(name).resolve(.28, .17, .007)
    density = .13
    expected = []
    for hinge in spec.hinges:
        descendants = {hinge.child}
        while True:
            expanded = descendants | {h.child for h in spec.hinges if h.parent in descendants}
            if expanded == descendants:
                break
            descendants = expanded
        vertices = np.concatenate([np.asarray(p.vertices) for p in spec.plates if p.name in descendants])
        extent = np.ptp(vertices[:, 0])
        mass = density * extent * spec.width
        # Every strip subtree is a rectangle ending at its parent hinge edge.
        expected.append(mass * (extent**2 + spec.thickness**2) / 3)
    np.testing.assert_allclose(hinge_inertias(spec, density), expected, rtol=2e-13)
    reordered = replace(spec, hinges=spec.hinges[::-1])
    np.testing.assert_allclose(hinge_inertias(reordered, density), expected[::-1], rtol=2e-13)


def test_diagonal_corner_hinges_use_off_diagonal_tensor_and_correct_scaled_axes():
    spec = designs.load_design('flexible_corners').resolve(.32, .12, .006)
    density = .09
    values = hinge_inertias(spec, density)
    # Right triangular corner: distance from the hypotenuse is h times a
    # barycentric coordinate; its mean square is h²/6. Top-surface hinge adds t²/3.
    a, b = spec.length/4, spec.width/4
    altitude = a*b/np.hypot(a, b)
    corner_mass = density*a*b/2
    expected_corner = corner_mass * (altitude**2/6 + spec.thickness**2/3)
    for index, hinge in enumerate(spec.hinges):
        if index != spec.fold_hinge:
            assert len(spec.plate(hinge.child).vertices) == 3
            assert abs(hinge.axis[0]) > .1 and abs(hinge.axis[1]) > .1
            assert values[index] == pytest.approx(expected_corner, rel=1e-12)
    # Main moving body plus both descendant corners equals the unsplit half-sheet.
    half_mass = density*spec.length/2*spec.width
    assert values[spec.fold_hinge] == pytest.approx(half_mass*((spec.length/2)**2+spec.thickness**2)/3, rel=1e-12)


@pytest.mark.parametrize('path', designs.discover_designs(), ids=lambda path: path.stem)
def test_positive_and_linear_density_quartic_uniform_size_scaling(path):
    design = designs.load_design(path)
    dims = np.array([.21, .13, .005])
    reference = hinge_inertias(design.resolve(*dims), .1)
    assert reference.shape == (len(design.hinges),)
    assert np.isfinite(reference).all() and (reference > 0).all()
    np.testing.assert_allclose(hinge_inertias(design.resolve(*dims), .37), reference*3.7, rtol=1e-12)
    np.testing.assert_allclose(hinge_inertias(design.resolve(*(dims*1.7)), .1), reference*1.7**4, rtol=1e-12)


def test_axis_normalization_and_invalid_inputs():
    spec = designs.single_hinge(.2, .15, .004)
    scaled = replace(spec, hinges=(replace(spec.hinges[0], axis=(0., -7., 0.)),))
    np.testing.assert_allclose(hinge_inertias(scaled, .1), hinge_inertias(spec, .1))
    with pytest.raises(ValueError, match='density'):
        hinge_inertias(spec, 0.)
    with pytest.raises(ValueError, match='nonzero axis'):
        hinge_inertias(replace(spec, hinges=(replace(spec.hinges[0], axis=(0., 0., 0.)),)), .1)
