"""Bending plasticity: the yield rule, its sign convention against Newton's own, and the reset.

Runs on the warp CPU device with a two-triangle hinge, so it needs Newton but no GPU or Kit.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import load_module  # noqa: E402

wp = pytest.importorskip("warp")
newton = pytest.importorskip("newton")

plast = load_module("isaacsimenvs/tasks/cloth/utils/plasticity.py", "plasticity")

H = 0.0167


def _hinge(phi: float):
    """Two right triangles sharing the edge (0,0)-(0,H) along y; the +x one turned up by ``phi``."""
    c, s = math.cos(phi), math.sin(phi)
    verts = [(0.0, 0.0, 0.0), (0.0, H, 0.0), (-H, 0.0, 0.0), (H * c, 0.0, H * s)]
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


def _bend(model, phi):
    """Positions of the same hinge turned to ``phi`` (the +x vertex only moves)."""
    q = model.particle_q.numpy().copy()
    q[3] = (H * math.cos(phi), 0.0, H * math.sin(phi))
    return wp.array(q, dtype=wp.vec3, device="cpu")


class _State:
    def __init__(self, q):
        self.particle_q = q


def test_dihedral_matches_newton_rest_angle():
    """A mesh built pre-bent gets its rest angle from Newton's builder; our dihedral must agree in
    magnitude AND sign, or the yield rule would push the rest angle the wrong way."""
    for phi in (0.3, 1.2, -0.8):
        m = _hinge(phi)
        e = _interior(m)
        ours = plast.dihedral_angles(m.particle_q, m.edge_indices)[e]
        assert ours == pytest.approx(float(m.edge_rest_angle.numpy()[e]), abs=1e-5)
        assert abs(ours) == pytest.approx(abs(phi), abs=1e-5)


def test_yield_rule_and_reset():
    m = _hinge(0.0)
    e = _interior(m)
    width = plast.BendingPlasticity(m, math.inf).hinge_width[e]
    assert width == pytest.approx(H, rel=1e-4)  # two H^2/2 faces over an H edge
    theta_y = math.radians(30.0)
    p = plast.BendingPlasticity(m, theta_y / width)

    def rest_after(phi):
        q = _bend(m, phi)
        p.apply(_State(q))
        return float(m.edge_rest_angle.numpy()[e]), float(plast.dihedral_angles(q, m.edge_indices)[e])

    rest, theta = rest_after(math.radians(20.0))      # below yield: elastic, rest stays flat
    assert rest == pytest.approx(0.0, abs=1e-6)
    rest, theta = rest_after(math.radians(90.0))      # past yield: rest trails by theta_y
    assert abs(theta - rest) == pytest.approx(theta_y, abs=1e-5)
    assert abs(rest) == pytest.approx(math.radians(60.0), abs=1e-5)
    rest_90 = rest
    rest, _ = rest_after(math.radians(70.0))          # back off inside the band: no change
    assert rest == pytest.approx(rest_90, abs=1e-6)
    rest, theta = rest_after(0.0)                     # opened flat again: rest dragged back to theta_y
    assert abs(rest) == pytest.approx(theta_y, abs=1e-5)
    assert np.sign(rest) == np.sign(rest_90)          # a crease mark remains on the same side
    p.reset()
    assert float(m.edge_rest_angle.numpy()[e]) == pytest.approx(0.0, abs=1e-7)


def test_infinite_yield_is_elastic():
    m = _hinge(0.0)
    e = _interior(m)
    p = plast.BendingPlasticity(m, math.inf)
    p.apply(_State(_bend(m, math.radians(170.0))))
    assert float(m.edge_rest_angle.numpy()[e]) == 0.0


def test_reset_some_edges():
    """Per-env reset restores only the listed edges."""
    m = _hinge(0.0)
    e = _interior(m)
    p = plast.BendingPlasticity(m, math.radians(10.0) / H)
    p.apply(_State(_bend(m, math.radians(90.0))))
    bent = float(m.edge_rest_angle.numpy()[e])
    assert abs(bent) > 1.0
    others = [i for i in range(int(m.edge_count)) if i != e]
    p.reset(others)                                   # not this edge: unchanged
    assert float(m.edge_rest_angle.numpy()[e]) == pytest.approx(bent)
    p.reset([e])
    assert float(m.edge_rest_angle.numpy()[e]) == pytest.approx(0.0, abs=1e-7)
