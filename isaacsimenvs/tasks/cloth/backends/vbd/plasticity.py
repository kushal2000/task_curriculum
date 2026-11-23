"""Bending plasticity of the VBD sheet: ``cloth.plastic_yield_curvature``. A fold bent past the yield
point keeps the excess, so a pressed fold stays partly folded instead of springing back flat.

Newton's VBD bending is purely elastic: each interior edge stores
``0.5 * edge_ke * rest_length * (theta - rest_angle)^2`` with ``rest_angle`` fixed at build (0 on a flat
sheet), so every fold springs back toward flat once nothing holds it. Real towel or denim keeps most
of a pressed fold; silk or a paper napkin springs open.

The model is the bending plasticity of Narain, Pfaff, O'Brien 2013 ("Folding and crumpling adaptive
sheets", after Gingold et al. 2004), reduced from their per-face strain tensor to one value per edge:
once the elastic part of the bend exceeds a yield curvature ``kappa``, the excess becomes permanent
at once (return mapping, no time term, no hardening). Per edge, with ``theta`` the current dihedral
angle and ``theta_y = kappa * hinge_width``:

    d = theta - rest_angle
    |d| > theta_y   ->   rest_angle = theta - sign(d) * theta_y

``hinge_width = (A1 + A2) / edge_length`` (the rest areas of the two faces) turns curvature into hinge
angle, as their Eq. 1 does for one edge on a face. On the 10 cm 7x7 grid it is the 16.7 mm cell width
for the edges along the grid and 11.8 mm for the diagonals. A crease narrower than a cell cannot exist
on this mesh, so ``kappa`` is a mesh-level value calibrated by a fold-press-release test, not a
literature one: the paper's 200 /m would need a 190 deg fold on one hinge.

Where it acts: the coupled solver runs VBD on a ModelView that may own a projected copy of
``edge_rest_angle``. The update writes the array the cloth SOLVER reads and mirrors it to the model's
(a view refresh projects the model's back over the copy); writing the model's alone left every sheet
elastic (origin/vbd_cloth_eval 3d83aea). It runs after every VBD ``step`` -- each cloth substep -- by
wrapping the solver instance, installed before Newton's CUDA graph capture so a graph holds it. A reset
puts the reset envs' rest angles back to the build's (flat).

From origin/cloth-physics ``utils/plasticity.py`` (the single-stage model; the two-stage
``set_curvature`` gate is not ported). ``kappa`` 0 = off: nothing is installed.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

__all__ = ["BendingPlasticity", "dihedral_angles", "install"]


@wp.func
def _dihedral(x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3) -> float:
    """Signed dihedral angle of edge (x2, x3) with opposite vertices x0, x1, exactly as Newton's VBD
    bending evaluates it (``evaluate_dihedral_angle_based_bending_force_hessian``)."""
    n1 = wp.cross(x2 - x0, x3 - x0)
    n2 = wp.cross(x3 - x1, x2 - x1)
    e = x3 - x2
    n1_norm = wp.length(n1)
    n2_norm = wp.length(n2)
    e_norm = wp.length(e)
    if n1_norm < 1.0e-6 or n2_norm < 1.0e-6 or e_norm < 1.0e-6:
        return 0.0
    n1_hat = n1 / n1_norm
    n2_hat = n2 / n2_norm
    return wp.atan2(wp.dot(wp.cross(n1_hat, n2_hat), e / e_norm), wp.dot(n1_hat, n2_hat))


@wp.kernel
def _dihedral_kernel(q: wp.array(dtype=wp.vec3), edge_indices: wp.array2d(dtype=wp.int32), out: wp.array(dtype=float)):
    e = wp.tid()
    if edge_indices[e, 0] == -1 or edge_indices[e, 1] == -1:
        out[e] = 0.0
        return
    out[e] = _dihedral(q[edge_indices[e, 0]], q[edge_indices[e, 1]], q[edge_indices[e, 2]], q[edge_indices[e, 3]])


@wp.kernel
def _yield_kernel(
    q: wp.array(dtype=wp.vec3),
    edge_indices: wp.array2d(dtype=wp.int32),
    yield_angle: wp.array(dtype=float),
    rest_angle: wp.array(dtype=float),
):
    e = wp.tid()
    if edge_indices[e, 0] == -1 or edge_indices[e, 1] == -1:
        return
    theta = _dihedral(q[edge_indices[e, 0]], q[edge_indices[e, 1]], q[edge_indices[e, 2]], q[edge_indices[e, 3]])
    # Newton's bending term does not wrap theta - rest; neither does this, so the two agree on what
    # "elastic part" means even for a rest angle near +-pi.
    d = theta - rest_angle[e]
    y = yield_angle[e]
    if d > y:
        rest_angle[e] = theta - y
    elif d < -y:
        rest_angle[e] = theta + y


@wp.kernel
def _restore_kernel(edge_ids: wp.array(dtype=wp.int32), initial: wp.array(dtype=float), rest_angle: wp.array(dtype=float)):
    e = edge_ids[wp.tid()]
    rest_angle[e] = initial[e]


def dihedral_angles(q, edge_indices) -> np.ndarray:
    """Per-edge dihedral angle [rad] for positions ``q`` (warp vec3 array), 0 on boundary edges."""
    out = wp.zeros(edge_indices.shape[0], dtype=float, device=q.device)
    wp.launch(_dihedral_kernel, dim=edge_indices.shape[0], inputs=[q, edge_indices, out], device=q.device)
    return out.numpy()


class BendingPlasticity:
    """Yield-curvature plasticity on the edges' rest angles.

    ``yield_curvature`` [1/m]: scalar, or one value per edge; ``inf`` never yields (elastic).
    ``solver_rest_angle``: the ``edge_rest_angle`` the cloth solver reads when it is not the model's
    (a ModelView's own copy); the update writes it and mirrors to the model's. Call :meth:`apply` after
    every solver step (or :meth:`wrap_solver` once) and :meth:`reset` when sheets are put back flat.
    """

    def __init__(self, model, yield_curvature, rest_q=None, solver_rest_angle=None) -> None:
        self.model = model
        self.edge_indices = model.edge_indices
        self.num_edges = int(model.edge_count)
        self.device = model.device
        self.rest_angle = model.edge_rest_angle if solver_rest_angle is None else solver_rest_angle
        if self.rest_angle.shape != model.edge_rest_angle.shape:
            raise ValueError("the solver's edge_rest_angle has a different edge layout from the model's")
        self.mirror = None if self.rest_angle.ptr == model.edge_rest_angle.ptr else model.edge_rest_angle
        #: The build-time rest angles, restored by `reset`.
        self.initial_rest = wp.clone(model.edge_rest_angle)
        self.hinge_width = self._hinge_width(model.particle_q.numpy() if rest_q is None else np.asarray(rest_q))
        kappa = np.broadcast_to(np.asarray(yield_curvature, dtype=np.float64), (self.num_edges,))
        with np.errstate(invalid="ignore"):  # inf * width stays inf (no yield)
            angle = np.where(np.isinf(kappa), np.inf, kappa * self.hinge_width)
        self.yield_angle = wp.array(angle.astype(np.float32), dtype=float, device=self.device)

    def _hinge_width(self, rest_q: np.ndarray) -> np.ndarray:
        """``(A1 + A2) / edge_length`` per edge on the rest positions [m]; 0 on boundary edges."""
        ei = self.edge_indices.numpy()
        w = np.zeros(self.num_edges)
        interior = (ei[:, 0] >= 0) & (ei[:, 1] >= 0)
        x0, x1, x2, x3 = (rest_q[np.maximum(ei[:, k], 0)] for k in range(4))
        a1 = 0.5 * np.linalg.norm(np.cross(x2 - x0, x3 - x0), axis=1)
        a2 = 0.5 * np.linalg.norm(np.cross(x3 - x1, x2 - x1), axis=1)
        length = np.linalg.norm(x3 - x2, axis=1)
        w[interior] = (a1 + a2)[interior] / np.maximum(length[interior], 1e-12)
        return w

    def apply(self, state) -> None:
        wp.launch(
            _yield_kernel,
            dim=self.num_edges,
            inputs=[state.particle_q, self.edge_indices, self.yield_angle, self.rest_angle],
            device=self.device,
        )
        if self.mirror is not None:
            wp.copy(self.mirror, self.rest_angle)

    def reset(self, edge_ids=None) -> None:
        """Restore the build-time rest angles of ``edge_ids`` (a warp int32 array or anything numpy
        takes; ``None`` = every edge)."""
        targets = [a for a in (self.rest_angle, self.mirror) if a is not None]
        if edge_ids is None:
            for target in targets:
                wp.copy(target, self.initial_rest)
            return
        if not isinstance(edge_ids, wp.array):
            edge_ids = wp.array(np.asarray(edge_ids, dtype=np.int32).reshape(-1), dtype=wp.int32, device=self.device)
        if edge_ids.shape[0] == 0:
            return
        for target in targets:
            wp.launch(_restore_kernel, dim=edge_ids.shape[0], inputs=[edge_ids, self.initial_rest, target], device=self.device)

    def wrap_solver(self, solver) -> None:
        """Run :meth:`apply` on the output state after every ``solver.step``. Patches the instance, so
        the coupled solver's call to its VBD entry picks it up; a graph captured afterwards holds it."""
        step = solver.step

        def step_with_plasticity(state_in, state_out, control, contacts, dt):
            step(state_in, state_out, control, contacts, dt)
            self.apply(state_out)

        solver.step = step_with_plasticity


def install(model, coupled_solver, cloth) -> BendingPlasticity | None:
    """Wrap the coupled solver's VBD entry with ``cloth.plastic_yield_curvature``; ``None`` when it is 0."""
    kappa = float(cloth.plastic_yield_curvature)
    if kappa <= 0.0:
        return None
    import newton

    entries = [e for e in coupled_solver._entries.values() if isinstance(e.solver, newton.solvers.SolverVBD)]
    if len(entries) != 1:
        raise RuntimeError(f"expected one VBD solver in the coupled solver, found {len(entries)}")
    entry = entries[0]
    plast = BendingPlasticity(model, kappa, solver_rest_angle=entry.view.edge_rest_angle)
    plast.wrap_solver(entry.solver)
    interior = plast.hinge_width > 0
    w = plast.hinge_width[interior]
    print(
        f"[cloth] plasticity: yield curvature {kappa:g} /m = {math.degrees(kappa * w.max()):.1f} deg per grid edge, "
        f"{math.degrees(kappa * w.min()):.1f} deg per diagonal, {int(interior.sum())} hinges; writes "
        f"{'the solver view copy, mirrored to the model' if plast.mirror is not None else 'the model array (shared)'}",
        flush=True,
    )
    return plast
