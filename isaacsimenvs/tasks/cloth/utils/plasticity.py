"""Bending plasticity for a VBD sheet: a fold that goes past the yield point stays partly folded.

``BendingPlasticity`` updates Newton's per-edge rest angles after every VBD substep (yield-curvature
rule, optional two-stage set gate); ``deformable_env.ClothEnv`` installs it into the coupled
solver's cloth entry and resets it per env.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

__all__ = ["BendingPlasticity", "dihedral_angles"]

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
def _dihedral_kernel(
    q: wp.array(dtype=wp.vec3), edge_indices: wp.array2d(dtype=wp.int32), out: wp.array(dtype=float)
):
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
    set_angle: wp.array(dtype=float),
    rest_angle: wp.array(dtype=float),
):
    e = wp.tid()
    if edge_indices[e, 0] == -1 or edge_indices[e, 1] == -1:
        return
    theta = _dihedral(q[edge_indices[e, 0]], q[edge_indices[e, 1]], q[edge_indices[e, 2]], q[edge_indices[e, 3]])
    d = theta - rest_angle[e]
    # Newton's bending term does not wrap theta - rest; neither does this, so the two agree on what
    # "elastic part" means even for a rest angle near +-pi.
    y = yield_angle[e]
    r = rest_angle[e]
    cand = r
    if d > y:
        cand = theta - y
    elif d < -y:
        cand = theta + y
    # Two-stage gate: growing set (|rest| increasing) needs a sharp bend, |theta| >= set angle;
    # shrinking it (toward flat) is always allowed. set_angle 0 = single-stage, as before.
    if wp.abs(cand) > wp.abs(r) and wp.abs(theta) < set_angle[e]:
        return
    rest_angle[e] = cand


@wp.kernel
def _restore_kernel(
    edge_ids: wp.array(dtype=wp.int32), initial: wp.array(dtype=float), rest_angle: wp.array(dtype=float)
):
    e = edge_ids[wp.tid()]
    rest_angle[e] = initial[e]


def dihedral_angles(q, edge_indices) -> np.ndarray:
    """Per-edge dihedral angle [rad] for positions ``q`` (warp vec3 array), 0 on boundary edges."""
    out = wp.zeros(edge_indices.shape[0], dtype=float, device=q.device)
    wp.launch(_dihedral_kernel, dim=edge_indices.shape[0], inputs=[q, edge_indices, out], device=q.device)
    return out.numpy()


class BendingPlasticity:
    """Yield-curvature plasticity on ``model.edge_rest_angle``.

    ``yield_curvature`` [1/m]: scalar, or one value per edge. ``inf`` disables yielding (the
    elastic sheet, unchanged). Call :meth:`apply` after every solver step with the new state, and
    :meth:`reset` (all edges, or the ids of some) when a sheet is put back flat.
    """

    def __init__(self, model, yield_curvature, rest_q=None, solver_rest_angle=None, set_curvature=0.0) -> None:
        """``solver_rest_angle``: the ``edge_rest_angle`` array the cloth SOLVER reads, when that is
        not ``model.edge_rest_angle`` -- under ``SolverCoupled`` the VBD entry runs on a ModelView
        that owns a projected copy, so writing the model's array alone never reaches the solver.
        The yield update then runs on the solver's array and is mirrored to the model's (which a
        view refresh projects back over the copy).

        ``set_curvature`` [1/m]: scalar or per edge; the two-stage gate (module docstring). 0 = off."""
        self.model = model
        self.edge_indices = model.edge_indices
        self.num_edges = int(model.edge_count)
        self.device = model.device
        #: The array the yield update writes (what the solver bends against) and its mirror.
        self.rest_angle = model.edge_rest_angle if solver_rest_angle is None else solver_rest_angle
        self.mirror = None if self.rest_angle.ptr == model.edge_rest_angle.ptr else model.edge_rest_angle
        if self.rest_angle.shape != model.edge_rest_angle.shape:
            raise ValueError("solver edge_rest_angle has a different edge layout from the model's")
        #: The build-time rest angles, restored by `reset`.
        self.initial_rest = wp.clone(model.edge_rest_angle)
        self.hinge_width = self._hinge_width(model.particle_q.numpy() if rest_q is None else np.asarray(rest_q))
        self.yield_angle = wp.zeros(self.num_edges, dtype=float, device=self.device)
        self.set_yield_curvature(yield_curvature)
        self.set_angle = wp.zeros(self.num_edges, dtype=float, device=self.device)
        self.set_set_curvature(set_curvature)

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

    def set_yield_curvature(self, yield_curvature) -> None:
        kappa = np.broadcast_to(np.asarray(yield_curvature, dtype=np.float64), (self.num_edges,))
        # inf * width stays inf (no yield); float32 inf compares correctly in the kernel.
        with np.errstate(invalid="ignore"):
            ang = np.where(np.isinf(kappa), np.inf, kappa * self.hinge_width)
        self.yield_angle.assign(ang.astype(np.float32))

    def set_set_curvature(self, set_curvature) -> None:
        kappa = np.broadcast_to(np.asarray(set_curvature, dtype=np.float64), (self.num_edges,))
        self.set_angle.assign((kappa * self.hinge_width).astype(np.float32))

    def apply(self, state) -> None:
        wp.launch(
            _yield_kernel,
            dim=self.num_edges,
            inputs=[state.particle_q, self.edge_indices, self.yield_angle, self.set_angle, self.rest_angle],
            device=self.device,
        )
        if self.mirror is not None:
            wp.copy(self.mirror, self.rest_angle)

    def reset(self, edge_ids=None) -> None:
        """Restore the build-time rest angles of ``edge_ids`` (a warp int32 array or anything numpy
        takes; ``None`` = every edge)."""
        if edge_ids is None:
            wp.copy(self.rest_angle, self.initial_rest)
            if self.mirror is not None:
                wp.copy(self.mirror, self.initial_rest)
            return
        if not isinstance(edge_ids, wp.array):
            edge_ids = wp.array(np.asarray(edge_ids, dtype=np.int32).reshape(-1), dtype=wp.int32, device=self.device)
        if edge_ids.shape[0] == 0:
            return
        for target in (self.rest_angle, self.mirror):
            if target is not None:
                wp.launch(_restore_kernel, dim=edge_ids.shape[0], inputs=[edge_ids, self.initial_rest, target],
                          device=self.device)

    def wrap_solver(self, solver) -> None:
        """Run :meth:`apply` on the output state after every ``solver.step`` (each VBD substep).

        Patches the instance's ``step``, so a coupled solver that calls the sub-solver's ``step``
        picks it up; a CUDA graph captured after this call contains the update."""
        step = solver.step

        def step_with_plasticity(state_in, state_out, control, contacts, dt):
            step(state_in, state_out, control, contacts, dt)
            self.apply(state_out)

        solver.step = step_with_plasticity

    @staticmethod
    def yield_angle_deg(yield_curvature: float, hinge_width: float) -> float:
        """The hinge angle [deg] at which an edge of ``hinge_width`` yields."""
        return math.degrees(yield_curvature * hinge_width)


