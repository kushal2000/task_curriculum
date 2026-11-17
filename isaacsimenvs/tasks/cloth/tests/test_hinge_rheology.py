"""Hinge rheology: bending from rigidity, friction hysteresis, curvature yield, creep, reset.

Runs on the warp CPU device with a two-triangle hinge (as ``test_cloth_plasticity``), so it needs
Newton but no GPU or Kit.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from isaacsimenvs.utils.testing import load_module

wp = pytest.importorskip("warp")
newton = pytest.importorskip("newton")

rh = load_module("isaacsimenvs/tasks/cloth/utils/hinge_rheology.py", "hinge_rheology")

H = 0.006
B = 1.0e-5  # N m, cotton-like


def _hinge():
    """Two right triangles sharing the edge (0,0)-(0,H) along y."""
    verts = [(0.0, 0.0, 0.0), (0.0, H, 0.0), (-H, 0.0, 0.0), (H, 0.0, 0.0)]
    idx = [2, 0, 1, 0, 3, 1]
    b = newton.ModelBuilder()
    b.add_cloth_mesh(
        pos=wp.vec3(0.0), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0.0),
        vertices=verts, indices=idx, density=1.0, tri_ke=1e2, tri_ka=1e2, tri_kd=0.0, edge_ke=1e-3, edge_kd=0.0,
    )
    return b.finalize(device="cpu")


def _interior(model):
    ei = model.edge_indices.numpy()
    return int(np.nonzero((ei[:, 0] >= 0) & (ei[:, 1] >= 0))[0][0])


class _State:
    def __init__(self, q):
        self.particle_q = q


def _at(model, phi):
    q = model.particle_q.numpy().copy()
    q[3] = (H * math.cos(phi), 0.0, H * math.sin(phi))
    return _State(wp.array(q, dtype=wp.vec3, device="cpu"))


def _setup(**kw):
    m = _hinge()
    e = _interior(m)
    r = rh.HingeRheology(m, rh.RheologyParams(bending_rigidity=B, **kw), dt=1.0 / 1440.0)
    return m, e, r


def _signed(m, e, phi):
    """Newton's signed dihedral for the bend ``phi`` (the sign convention is Newton's, not ours)."""
    plast = load_module("isaacsimenvs/tasks/cloth/utils/plasticity.py", "plasticity")
    return float(plast.dihedral_angles(_at(m, phi).particle_q, m.edge_indices)[e])


def test_width_and_ke_from_rigidity():
    """w_e = (A1 + A2) / l: two right triangles of legs H around an edge of length H -> H, and
    edge_ke = B / w_e is what the solver reads."""
    m, e, r = _setup()
    assert r.width_np[e] == pytest.approx(H, rel=1e-6)
    assert m.edge_bending_properties.numpy()[e, 0] == pytest.approx(B / H, rel=1e-5)
    # boundary edges never bend
    ei = m.edge_indices.numpy()
    assert (r.width_np[(ei[:, 0] < 0) | (ei[:, 1] < 0)] == 0.0).all()


def test_stiffness_scale_multiplies_the_solver_stiffness():
    m = _hinge()
    e = _interior(m)
    rh.HingeRheology(m, rh.RheologyParams(bending_rigidity=B, stiffness_scale=2.5), dt=1.0 / 1440.0)
    assert m.edge_bending_properties.numpy()[e, 0] == pytest.approx(2.5 * B / H, rel=1e-5)


def test_pure_elastic_returns_to_flat():
    m, e, r = _setup()
    for phi in (0.5, 2.5, 0.0):
        r.apply(_at(m, phi))
    assert abs(r.rest_angle.numpy()[e]) < 1e-7


def test_yield_on_curvature_leaves_a_crease():
    """Elastic below the yield curvature, rest trails by kappa_y * w beyond it, and reopening leaves
    that set."""
    ky = 100.0
    m, e, r = _setup(yield_curvature=ky)
    w = H
    small = 0.5 * ky * w  # half the yield angle: elastic
    r.apply(_at(m, small))
    assert abs(r.theta_p.numpy()[e]) < 1e-7
    big = 2.5
    th = _signed(m, e, big)
    r.apply(_at(m, big))
    assert abs(r.theta_p.numpy()[e]) == pytest.approx(abs(th) - ky * w, abs=1e-5)
    assert abs(r.plastic_curvature()[e]) == pytest.approx((abs(th) - ky * w) / w, rel=1e-4)
    # Released, the hinge springs back elastically: anywhere inside the yield band keeps the set.
    back = abs(th) - ky * w + 0.5 * ky * w
    r.apply(_at(m, math.copysign(back, big)))
    assert abs(r.theta_p.numpy()[e]) == pytest.approx(abs(th) - ky * w, abs=1e-5)
    # Ironed flat by force: reverse yielding leaves exactly one yield angle of crease.
    r.apply(_at(m, 0.0))
    assert abs(r.theta_p.numpy()[e]) == pytest.approx(ky * w, abs=1e-5)


def test_curvature_yield_is_the_same_on_any_mesh():
    """The same yield curvature yields at an angle proportional to the hinge width: a finer mesh needs
    a smaller angle per hinge, so the crease curvature, not the hinge angle, is the material law."""
    m, e, r = _setup(yield_curvature=300.0)
    w = r.width_np[e]
    assert float(r.yield_kappa.numpy()[e]) * w == pytest.approx(300.0 * H, rel=1e-5)


def test_friction_hysteresis():
    """Load, then unload: the friction anchor is dragged along while loading and stays put inside the
    slip band on the way back, so the combined rest angle lags (a hysteresis loop)."""
    ratio, mf = 1.0, 2.0e-4  # N m / m
    m, e, r = _setup(friction_ratio=ratio, friction_moment=mf)
    kb = B / H
    kf = ratio * kb
    slip = mf / kf
    # combined stiffness written to the solver
    assert m.edge_bending_properties.numpy()[e, 0] == pytest.approx(kb + kf, rel=1e-5)
    th = _signed(m, e, 1.5)
    r.apply(_at(m, 1.5))
    tf = float(r.theta_f.numpy()[e])
    assert abs(tf) == pytest.approx(abs(th) - slip, abs=1e-5)
    rest = float(r.rest_angle.numpy()[e])
    assert rest == pytest.approx(kf * tf / (kb + kf), abs=1e-6)  # theta_p = 0
    # Back to flat: the anchor slips back to within one slip angle of 0, not to 0.
    r.apply(_at(m, 0.0))
    assert abs(float(r.theta_f.numpy()[e])) == pytest.approx(slip, abs=1e-6)
    # A small wiggle inside the slip band does not move the anchor at all.
    before = float(r.theta_f.numpy()[e])
    r.apply(_at(m, 0.5 * slip))
    assert float(r.theta_f.numpy()[e]) == pytest.approx(before, abs=1e-7)


def test_creep_only_when_sharply_bent():
    tau, dt = 0.5, 1.0 / 1440.0
    m, e, r = _setup(creep_curvature=200.0, creep_tau=tau)
    w = H
    r.apply(_at(m, 0.5 * 200.0 * w))  # below the creep curvature: nothing
    assert abs(r.theta_p.numpy()[e]) < 1e-7
    th = _signed(m, e, 2.0)
    for _ in range(10):
        r.apply(_at(m, 2.0))
    expect = th * (1.0 - (1.0 - (1.0 - math.exp(-dt / tau))) ** 10)
    assert float(r.theta_p.numpy()[e]) == pytest.approx(expect, rel=1e-3)


def test_reset_clears_set_and_friction():
    m, e, r = _setup(yield_curvature=100.0, friction_ratio=1.0, friction_moment=1e-4)
    r.apply(_at(m, 2.5))
    assert abs(r.rest_angle.numpy()[e]) > 0.1
    r.reset([e])
    for a in (r.theta_p, r.theta_f, r.rest_angle):
        assert abs(a.numpy()[e]) < 1e-7
    r.apply(_at(m, 2.5))
    r.reset()
    for a in (r.theta_p, r.theta_f, r.rest_angle):
        assert np.abs(a.numpy()).max() < 1e-7


def test_natural_curvature_is_a_fixed_rest_imperfection():
    m, e, r = _setup(natural_curvature=5.0, imperfection_seed=3)
    rest = float(r.initial_rest.numpy()[e])
    assert rest != 0.0 and abs(rest) < 5 * 5.0 * r.width_np[e]  # within 5 sigma of 5 /m curvature
    assert float(r.rest_angle.numpy()[e]) == rest  # the solver starts from it
    r.reset()
    assert float(r.rest_angle.numpy()[e]) == rest  # reset returns to it, not to flat
    m2, _, r2 = _setup(natural_curvature=5.0, imperfection_seed=3)
    assert float(r2.initial_rest.numpy()[e]) == rest  # deterministic per seed
    assert np.abs(r2.initial_rest.numpy()[r2.width_np == 0.0]).max() == 0.0  # boundary edges untouched
