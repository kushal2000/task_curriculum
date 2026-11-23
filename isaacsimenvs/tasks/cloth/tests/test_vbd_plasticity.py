"""Bending plasticity (backends/vbd/plasticity.py): the yield rule, its sign against Newton's own
dihedral, the per-edge reset and the write to a solver view's own rest angles.

A two-triangle hinge on the warp CPU device: needs Newton, no GPU or Kit.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

wp = pytest.importorskip("warp")
newton = pytest.importorskip("newton")

from isaacsimenvs.tasks.cloth.backends.vbd import plasticity as plast  # noqa: E402

H = 0.0167  # the 10 cm 7x7 grid's cell


def _hinge(phi: float):
    """Two right triangles sharing the edge (0,0)-(0,H) along y; the +x one turned up by ``phi``."""
    c, s = math.cos(phi), math.sin(phi)
    verts = [(0.0, 0.0, 0.0), (0.0, H, 0.0), (-H, 0.0, 0.0), (H * c, 0.0, H * s)]
    b = newton.ModelBuilder()
    b.add_cloth_mesh(
        pos=wp.vec3(0.0), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0.0),
        vertices=verts, indices=[2, 0, 1, 0, 3, 1], density=1.0,
        tri_ke=1e2, tri_ka=1e2, tri_kd=0.0, edge_ke=1e-3, edge_kd=0.0,
    )
    return b.finalize(device="cpu")


def _interior(model) -> int:
    ei = model.edge_indices.numpy()
    return int(np.nonzero((ei[:, 0] >= 0) & (ei[:, 1] >= 0))[0][0])


class _State:
    def __init__(self, model, phi: float):
        """The hinge's positions turned to ``phi`` (only the +x vertex moves)."""
        q = model.particle_q.numpy().copy()
        q[3] = (H * math.cos(phi), 0.0, H * math.sin(phi))
        self.particle_q = wp.array(q, dtype=wp.vec3, device="cpu")


def test_dihedral_matches_newton_rest_angle():
    """A mesh built pre-bent gets its rest angle from Newton's builder; ours must agree in magnitude
    AND sign, or the yield rule would push the rest angle the wrong way."""
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

    def rest_after(deg):
        state = _State(m, math.radians(deg))
        p.apply(state)
        return float(m.edge_rest_angle.numpy()[e]), float(plast.dihedral_angles(state.particle_q, m.edge_indices)[e])

    rest, _ = rest_after(20.0)                        # below yield: elastic, rest stays flat
    assert rest == pytest.approx(0.0, abs=1e-6)
    rest, theta = rest_after(90.0)                    # past yield: rest trails theta by theta_y
    assert abs(theta - rest) == pytest.approx(theta_y, abs=1e-5)
    assert abs(rest) == pytest.approx(math.radians(60.0), abs=1e-5)
    rest_90 = rest
    rest, _ = rest_after(70.0)                        # back inside the band: no change
    assert rest == pytest.approx(rest_90, abs=1e-6)
    rest, _ = rest_after(0.0)                         # opened flat: rest dragged back to theta_y
    assert abs(rest) == pytest.approx(theta_y, abs=1e-5)
    assert np.sign(rest) == np.sign(rest_90)          # a crease remains, on the same side
    p.reset()
    assert float(m.edge_rest_angle.numpy()[e]) == pytest.approx(0.0, abs=1e-7)


def test_infinite_yield_is_elastic():
    m = _hinge(0.0)
    e = _interior(m)
    plast.BendingPlasticity(m, math.inf).apply(_State(m, math.radians(170.0)))
    assert float(m.edge_rest_angle.numpy()[e]) == 0.0


def test_reset_some_edges():
    """A per-env reset restores only the listed edges."""
    m = _hinge(0.0)
    e = _interior(m)
    p = plast.BendingPlasticity(m, math.radians(10.0) / H)
    p.apply(_State(m, math.radians(90.0)))
    bent = float(m.edge_rest_angle.numpy()[e])
    assert abs(bent) > 1.0
    p.reset([i for i in range(int(m.edge_count)) if i != e])
    assert float(m.edge_rest_angle.numpy()[e]) == pytest.approx(bent)
    p.reset([e])
    assert float(m.edge_rest_angle.numpy()[e]) == pytest.approx(0.0, abs=1e-7)


def test_writes_the_solver_copy_and_mirrors_the_model():
    """Under the coupled solver VBD reads a ModelView's own edge_rest_angle: the update must land
    there (writing the model's alone left every sheet elastic) and be mirrored to the model's."""
    m = _hinge(0.0)
    e = _interior(m)
    solver_copy = wp.clone(m.edge_rest_angle)
    p = plast.BendingPlasticity(m, math.radians(10.0) / H, solver_rest_angle=solver_copy)
    assert p.mirror is not None
    p.apply(_State(m, math.radians(90.0)))
    assert abs(float(solver_copy.numpy()[e])) == pytest.approx(math.radians(80.0), abs=1e-5)
    assert float(m.edge_rest_angle.numpy()[e]) == pytest.approx(float(solver_copy.numpy()[e]))
    p.reset([e])
    assert float(solver_copy.numpy()[e]) == 0.0 and float(m.edge_rest_angle.numpy()[e]) == 0.0
