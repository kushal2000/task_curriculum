"""Sheet B/C map to local hinges consistently across polygon discretizations."""
from dataclasses import replace

import numpy as np
import pytest

from isaacsimenvs.tasks.cloth.backends.rigid import designs
from isaacsimenvs.tasks.cloth.backends.rigid.bending import hinge_geometry_factors


@pytest.mark.parametrize('panels', [2, 4, 8])
def test_regular_strips_have_crease_length_over_panel_spacing(panels):
    length, width = .28, .17
    spec = designs.strip(panels).resolve(length, width, .002)
    factors = hinge_geometry_factors(spec)
    np.testing.assert_allclose(factors, np.full(panels-1, width/(length/panels)), rtol=1e-13)
    # One material draw produces the complete vector; no per-joint parameter draw.
    rigidity, viscosity = 5e-5, 3e-5
    stiffness, damping = rigidity*factors, viscosity*factors
    np.testing.assert_allclose(stiffness/damping, rigidity/viscosity)
    assert np.sum(.5*stiffness*np.zeros(panels-1)**2) == 0.


def test_default_twenty_centimeter_single_hinge_maps_to_nominal_coefficients():
    factors = hinge_geometry_factors(designs.single_hinge(.2, .2, .002))
    np.testing.assert_allclose(5e-5*factors, [1e-4], rtol=1e-13)


def test_unequal_hinge_band_uses_neighboring_half_widths():
    length, width = .3, .15
    spec = designs.load_design('hinge_band').resolve(length, width, .002)
    factors = hinge_geometry_factors(spec)
    for index, hinge in enumerate(spec.hinges):
        # Band strips are .1L; outside plates are .4L. Central dual span=.1L;
        # the two band-edge dual spans=(.1L+.4L)/2=.25L.
        span = (.1 if index == spec.fold_hinge else .25)*length
        assert factors[index] == pytest.approx(width/span, rel=1e-13)


def test_diagonal_corner_factor_matches_analytic_centroids_and_reflection_symmetry():
    length, width = .32, .12
    spec = designs.load_design('flexible_corners').resolve(length, width, .002)
    factors = hinge_geometry_factors(spec)
    a, b = length/4, width/4
    half_area = length*width/2
    main_x = (half_area*length/4 - a*b*(length/2-a/3))/(half_area-a*b)
    edge_length = np.hypot(a, b)
    normal = np.array([b, a])/edge_length
    edge_point = np.array([length/2-a, width/2])
    parent_distance = normal@edge_point-normal[0]*main_x
    child_distance = a*b/(3*edge_length)
    expected_corner = edge_length/(parent_distance+child_distance)
    for index, hinge in enumerate(spec.hinges):
        expected = width/(2*main_x) if index == spec.fold_hinge else expected_corner
        assert factors[index] == pytest.approx(expected, rel=1e-13)
    assert len(set(np.round(np.delete(factors, spec.fold_hinge), 12))) == 1


def test_constant_small_curvature_energy_converges_as_strip_is_refined():
    length, width, rigidity, curvature = .24, .16, 7e-5, .2
    continuum = .5*rigidity*curvature**2*length*width
    energies = []
    for panels in (2, 4, 8):
        spacing = length/panels
        factors = hinge_geometry_factors(designs.strip(panels).resolve(length, width, .002))
        angles = np.full(panels-1, curvature*spacing)
        energy = np.sum(.5*rigidity*factors*angles**2)
        # Centroid-to-centroid dual cells exclude half a panel at both outside
        # boundaries; this discretization approaches the continuum with refinement.
        assert energy == pytest.approx(continuum*(1-1/panels), rel=1e-13)
        energies.append(energy)
    assert 0 < energies[0] < energies[1] < energies[2] < continuum


@pytest.mark.parametrize('path', designs.discover_designs(), ids=lambda path:path.stem)
def test_geometry_factors_are_positive_and_independent_of_contact_thickness(path):
    design = designs.load_design(path)
    spec = design.resolve(.21, .13, .002)
    reference = hinge_geometry_factors(spec)
    assert reference.shape == (len(spec.hinges),)
    assert np.isfinite(reference).all() and (reference > 0).all()
    np.testing.assert_allclose(hinge_geometry_factors(design.resolve(.21, .13, .02)), reference, rtol=1e-13)
    np.testing.assert_allclose(hinge_geometry_factors(design.resolve(.42, .26, .002)), reference, rtol=1e-13)
    np.testing.assert_allclose(hinge_geometry_factors(replace(spec, hinges=spec.hinges[::-1])), reference[::-1])


def test_missing_or_degenerate_crease_has_a_clear_error():
    spec = designs.single_hinge(.2, .2, .002)
    with pytest.raises(ValueError, match='endpoints'):
        hinge_geometry_factors(replace(spec, hinges=(replace(spec.hinges[0], edge=None),)))
    with pytest.raises(ValueError, match='nonzero length'):
        hinge_geometry_factors(replace(spec, hinges=(replace(spec.hinges[0], edge=((0.,0.,0.),(0.,0.,0.))),)))
