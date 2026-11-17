"""Hinge rheology for a VBD sheet: bending from measured rigidity, internal friction, plastic creasing.

Replaces the per-hinge angle yield of ``plasticity.py`` (kept for legacy configs) with the usual
textile picture, every parameter in material units so the same numbers hold on any mesh:

* **Elastic bending from the measured rigidity** ``B`` [N m]. A hinge of length ``l_e`` between
  faces of rest areas ``A1``, ``A2`` stands for a strip of width ``w_e = (A1 + A2) / l_e`` across it
  (Narain et al.'s hinge width), so a curvature ``kappa`` bends it by ``theta = kappa * w_e`` and the
  continuum energy ``B/2 * kappa^2 * l_e * w_e`` is Newton's ``0.5 * edge_ke * l_e * theta^2`` at
  ``edge_ke = B / w_e``. (Exact for a cylinder on a right-triangle grid: the grid edges across the
  bend each turn by ``kappa h``, the diagonals not at all.) Newton's own ``edge_ke`` is one number for
  every edge, so its bending stiffness changes with the mesh (cotton's fitted ``edge_ke`` is 4.1x
  larger on a 25x25 sheet than on 7x7); per-edge ``B / w_e`` does not. VBD at the env's 12
  iterations under-resolves soft bending (the cantilever fits needed 2-4x the continuum stiffness),
  so ``stiffness_scale`` carries a calibrated solver factor, measured, not assumed.

* **Internal friction** (Dahl's model in its elastic-slip limit, as Miguel et al. 2013 use for
  cloth hysteresis). Yarns slide on each other: a second bending branch of stiffness
  ``k_f = friction_ratio * k_b`` pulls toward a friction anchor ``theta_f``, and the anchor slips
  whenever that branch's moment per unit length exceeds ``friction_moment`` [N m / m]. A bent sheet
  therefore unbends only part way (the stuck friction holds the rest), and a loading/unloading cycle
  traces a hysteresis loop of height ``2 * friction_moment``. Small reversible deformations (a
  light touch, a wrinkle under a finger) come back to within one slip curvature,
  ``friction_moment / k_f``.

* **Plastic creasing on curvature** (Narain, Pfaff, O'Brien 2013, per hinge). When the elastic
  branch's curvature ``(theta - theta_p) / w_e`` exceeds ``yield_curvature`` [1/m] (plus
  ``hardening * |theta_p / w_e|``), the excess becomes the permanent rest angle ``theta_p``. The yield
  is on CURVATURE, so it tells a pressed fold (radius ~ the fabric thickness, curvature ~1e3 /m)
  from a loose unpressed loop or a wrinkle (radius ~5-20 mm, ~50-200 /m) by geometry alone, once the
  mesh is fine enough to resolve the loop. That is what the two-stage gate of ``plasticity.py``
  approximated on a 7x7 sheet, where both put the same angle on one hinge.

* **Creep** (optional; Gong et al. 2025, time-dependent persistent wrinkles). While a hinge is bent
  past ``creep_curvature``, its rest angle relaxes toward the current angle with time constant
  ``creep_tau`` [s], so a fold held or pressed longer keeps more set.

VBD bends each edge against ONE stiffness and ONE rest angle. Two parallel springs
``k_b (theta - theta_p) + k_f (theta - theta_f)`` are exactly one spring of stiffness ``k_b + k_f``
at rest angle ``(k_b theta_p + k_f theta_f) / (k_b + k_f)``, so the update writes those two arrays and
Newton needs no patch. Graph-capturable, like ``plasticity.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp


__all__ = ["HingeRheology", "RheologyParams", "hinge_geometry", "bending_ke_from_rigidity"]

@wp.func
def _dihedral(x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3) -> float:
    """Signed dihedral angle of edge (x2, x3), exactly as Newton's VBD bending evaluates it (the same
    function as ``plasticity._dihedral``; repeated so this module loads standalone)."""
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



@dataclass
class RheologyParams:
    """Material parameters, all independent of the mesh. ``inf`` / 0 switch a branch off."""

    bending_rigidity: float
    """B [N m]: the elastic bending stiffness per unit width (cantilever / KES B)."""
    stiffness_scale: float = 1.0
    """Solver calibration: VBD bends against ``stiffness_scale * B / w_e`` (see the module docstring)."""
    friction_ratio: float = 0.0
    """k_f / k_b: stiffness of the friction branch relative to the elastic one. 0 = no friction."""
    friction_moment: float = 0.0
    """Slip moment per unit length [N m / m]. Half the height of the bending hysteresis loop."""
    yield_curvature: float = float("inf")
    """Curvature [1/m] past which the elastic branch creases permanently. inf = never."""
    hardening: float = 0.0
    """Yield grows by ``hardening * |plastic curvature|`` (dimensionless)."""
    creep_curvature: float = float("inf")
    """Curvature [1/m] above which the rest angle creeps toward the current one. inf = no creep."""
    creep_tau: float = float("inf")
    """Creep time constant [s]."""
    natural_curvature: float = 0.0
    """Std [1/m] of a random rest curvature per interior hinge, fixed at build: real fabric never lies
    exactly flat. Without it a flat sheet under in-plane compression has no out-of-plane perturbation
    and VBD keeps it planar (no buckling) unless plastic flow happens to break the symmetry."""
    imperfection_seed: int = 0
    """Seed of the ``natural_curvature`` draw."""


def hinge_geometry(rest_q: np.ndarray, edge_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per edge: rest length ``l_e`` and hinge width ``w_e = (A1 + A2) / l_e`` [m].

    Boundary edges (one face) get ``w_e = 0``; they never bend.
    """
    ei = np.asarray(edge_indices)
    q = np.asarray(rest_q, dtype=np.float64)
    interior = (ei[:, 0] >= 0) & (ei[:, 1] >= 0)
    x0, x1, x2, x3 = (q[np.maximum(ei[:, k], 0)] for k in range(4))
    a1 = 0.5 * np.linalg.norm(np.cross(x2 - x0, x3 - x0), axis=1)
    a2 = 0.5 * np.linalg.norm(np.cross(x3 - x1, x2 - x1), axis=1)
    length = np.linalg.norm(x3 - x2, axis=1)
    w = np.zeros(len(ei))
    w[interior] = (a1 + a2)[interior] / np.maximum(length[interior], 1e-12)
    return length, w


def bending_ke_from_rigidity(rigidity, rest_q: np.ndarray, edge_indices: np.ndarray) -> np.ndarray:
    """Newton ``edge_ke`` per edge for bending rigidity ``B`` [N m] (scalar or per edge): ``B / w_e``."""
    _, w = hinge_geometry(rest_q, edge_indices)
    b = np.broadcast_to(np.asarray(rigidity, dtype=np.float64), w.shape)
    ke = np.zeros_like(w)
    ok = w > 0.0
    ke[ok] = b[ok] / w[ok]
    return ke


@wp.kernel
def _rheology_kernel(
    q: wp.array(dtype=wp.vec3),
    edge_indices: wp.array2d(dtype=wp.int32),
    width: wp.array(dtype=float),
    k_b: wp.array(dtype=float),
    k_f: wp.array(dtype=float),
    slip_moment: wp.array(dtype=float),
    yield_kappa: wp.array(dtype=float),
    hardening: wp.array(dtype=float),
    creep_kappa: wp.array(dtype=float),
    creep_rate: wp.array(dtype=float),
    theta_p: wp.array(dtype=float),
    theta_f: wp.array(dtype=float),
    rest_angle: wp.array(dtype=float),
):
    e = wp.tid()
    w = width[e]
    if edge_indices[e, 0] == -1 or edge_indices[e, 1] == -1 or w <= 0.0:
        return
    theta = _dihedral(q[edge_indices[e, 0]], q[edge_indices[e, 1]], q[edge_indices[e, 2]], q[edge_indices[e, 3]])
    kb = k_b[e]
    kf = k_f[e]

    # Plastic crease: return-map the elastic branch's curvature onto the (hardened) yield surface.
    tp = theta_p[e]
    ky = yield_kappa[e] + hardening[e] * wp.abs(tp) / w
    d = (theta - tp) / w
    if d > ky:
        tp = theta - ky * w
    elif d < -ky:
        tp = theta + ky * w
    # Creep: a hinge held sharply bent sets a little more each substep (rate = 1 - exp(-dt / tau)).
    if wp.abs(theta) / w > creep_kappa[e]:
        tp = tp + creep_rate[e] * (theta - tp)
    theta_p[e] = tp

    # Friction: the anchor follows theta once the friction branch's moment would exceed the slip moment.
    tf = theta_f[e]
    if kf > 0.0:
        slip = slip_moment[e] / kf
        if theta - tf > slip:
            tf = theta - slip
        elif theta - tf < -slip:
            tf = theta + slip
        theta_f[e] = tf

    rest_angle[e] = (kb * tp + kf * tf) / (kb + kf)


@wp.kernel
def _restore_kernel(
    edge_ids: wp.array(dtype=wp.int32),
    initial: wp.array(dtype=float),
    theta_p: wp.array(dtype=float),
    theta_f: wp.array(dtype=float),
    rest_angle: wp.array(dtype=float),
):
    e = edge_ids[wp.tid()]
    theta_p[e] = initial[e]
    theta_f[e] = initial[e]
    rest_angle[e] = initial[e]


class HingeRheology:
    """Elastic + friction + plastic + creep bending on a Newton model's edges (module docstring).

    ``params``: one :class:`RheologyParams` for every edge, or a dict of per-edge arrays with the same
    field names. ``dt`` is the solver substep the update runs after (creep only).

    ``solver_rest_angle`` / ``solver_bending_properties``: the arrays the cloth SOLVER reads when they
    are not the model's (``SolverCoupled`` runs VBD on a ``ModelView`` that may own copies); the
    update writes those and mirrors them to the model's.
    """

    def __init__(
        self,
        model,
        params,
        dt: float,
        rest_q=None,
        solver_rest_angle=None,
        solver_bending_properties=None,
    ) -> None:
        self.model = model
        self.device = model.device
        self.edge_indices = model.edge_indices
        self.num_edges = int(model.edge_count)
        ei = self.edge_indices.numpy()
        rq = model.particle_q.numpy() if rest_q is None else np.asarray(rest_q)
        self.rest_length, self.width_np = hinge_geometry(rq, ei)

        self.rest_angle = model.edge_rest_angle if solver_rest_angle is None else solver_rest_angle
        self.rest_mirror = None if self.rest_angle.ptr == model.edge_rest_angle.ptr else model.edge_rest_angle
        self.bending = model.edge_bending_properties if solver_bending_properties is None else solver_bending_properties
        self.bending_mirror = (
            None if self.bending.ptr == model.edge_bending_properties.ptr else model.edge_bending_properties
        )
        if self.rest_angle.shape[0] != self.num_edges or self.bending.shape[0] != self.num_edges:
            raise ValueError("solver edge arrays have a different edge layout from the model's")

        self.initial_rest = wp.clone(model.edge_rest_angle)
        has = (params.get("natural_curvature") if isinstance(params, dict) else getattr(params, "natural_curvature", 0.0))
        nat = self._per_edge(params, "natural_curvature") if has is not None else np.zeros(self.num_edges)
        if (nat > 0.0).any():
            ok = self.width_np > 0.0
            rng = np.random.default_rng(int(self._scalar(params, "imperfection_seed", 0)))
            rest = self.initial_rest.numpy().astype(np.float64)
            rest[ok] += self.width_np[ok] * nat[ok] * rng.standard_normal(int(ok.sum()))
            self.initial_rest.assign(rest.astype(np.float32))
            for target in (self.rest_angle, self.rest_mirror):
                if target is not None:
                    wp.copy(target, self.initial_rest)
        n = self.num_edges
        f = lambda: wp.zeros(n, dtype=float, device=self.device)  # noqa: E731
        self.width = wp.array(self.width_np.astype(np.float32), dtype=float, device=self.device)
        self.k_b, self.k_f, self.slip_moment = f(), f(), f()
        self.yield_kappa, self.hardening, self.creep_kappa, self.creep_rate = f(), f(), f(), f()
        self.theta_p = wp.clone(self.initial_rest)
        self.theta_f = wp.clone(self.initial_rest)
        self.set_params(params, dt)

    # ------------------------------------------------------------------ parameters
    @staticmethod
    def _scalar(params, name: str, default: float) -> float:
        v = params.get(name, default) if isinstance(params, dict) else getattr(params, name, default)
        return float(default if v is None else v)

    def _per_edge(self, params, name: str) -> np.ndarray:
        v = params[name] if isinstance(params, dict) else getattr(params, name)
        return np.broadcast_to(np.asarray(v, dtype=np.float64), (self.num_edges,)).copy()

    def set_params(self, params, dt: float) -> None:
        """(Re)load material parameters and write the combined bending stiffness ``k_b + k_f``."""
        w = self.width_np
        ok = w > 0.0
        b = self._per_edge(params, "bending_rigidity")
        if (params.get("stiffness_scale") if isinstance(params, dict) else getattr(params, "stiffness_scale", None)) is not None:
            b = b * self._per_edge(params, "stiffness_scale")
        kb = np.zeros(self.num_edges)
        kb[ok] = b[ok] / w[ok]
        kf = kb * self._per_edge(params, "friction_ratio")
        tau = self._per_edge(params, "creep_tau")
        with np.errstate(divide="ignore", over="ignore"):
            rate = np.where(np.isfinite(tau) & (tau > 0.0), 1.0 - np.exp(-dt / np.where(tau > 0.0, tau, 1.0)), 0.0)
        self.k_b_np, self.k_f_np = kb, kf
        for arr, val in (
            (self.k_b, kb), (self.k_f, kf), (self.slip_moment, self._per_edge(params, "friction_moment")),
            (self.yield_kappa, self._per_edge(params, "yield_curvature")),
            (self.hardening, self._per_edge(params, "hardening")),
            (self.creep_kappa, self._per_edge(params, "creep_curvature")), (self.creep_rate, rate),
        ):
            arr.assign(val.astype(np.float32))  # inf stays inf in float32 and compares correctly
        props = self.bending.numpy()
        props[:, 0] = (kb + kf).astype(np.float32)
        for target in (self.bending, self.bending_mirror):
            if target is not None:
                target.assign(props)

    # ------------------------------------------------------------------ stepping
    def apply(self, state) -> None:
        wp.launch(
            _rheology_kernel,
            dim=self.num_edges,
            inputs=[
                state.particle_q, self.edge_indices, self.width, self.k_b, self.k_f, self.slip_moment,
                self.yield_kappa, self.hardening, self.creep_kappa, self.creep_rate,
                self.theta_p, self.theta_f, self.rest_angle,
            ],
            device=self.device,
        )
        if self.rest_mirror is not None:
            wp.copy(self.rest_mirror, self.rest_angle)

    def reset(self, edge_ids=None) -> None:
        """Put ``edge_ids`` (warp int32 array or anything numpy takes; ``None`` = all) back to the
        build-time rest angle, with no set and no stuck friction."""
        if edge_ids is None:
            for a in (self.theta_p, self.theta_f, self.rest_angle, self.rest_mirror):
                if a is not None:
                    wp.copy(a, self.initial_rest)
            return
        if not isinstance(edge_ids, wp.array):
            edge_ids = wp.array(np.asarray(edge_ids, dtype=np.int32).reshape(-1), dtype=wp.int32, device=self.device)
        if edge_ids.shape[0] == 0:
            return
        for target in (self.rest_angle, self.rest_mirror):
            if target is not None:
                wp.launch(
                    _restore_kernel, dim=edge_ids.shape[0],
                    inputs=[edge_ids, self.initial_rest, self.theta_p, self.theta_f, target], device=self.device,
                )

    def wrap_solver(self, solver) -> None:
        """Run :meth:`apply` after every ``solver.step`` (each VBD substep); see ``plasticity.py``."""
        step = solver.step

        def step_with_rheology(state_in, state_out, control, contacts, dt):
            step(state_in, state_out, control, contacts, dt)
            self.apply(state_out)

        solver.step = step_with_rheology

    # ------------------------------------------------------------------ readouts
    def plastic_curvature(self) -> np.ndarray:
        """Per-edge permanent curvature ``theta_p / w_e`` [1/m] (0 on boundary edges)."""
        tp = self.theta_p.numpy().astype(np.float64)
        out = np.zeros_like(tp)
        ok = self.width_np > 0.0
        out[ok] = tp[ok] / self.width_np[ok]
        return out
