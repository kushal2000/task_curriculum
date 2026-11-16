"""Rigid-body sheet designs: the size grid, and the plates + hinges that approximate one sheet.

Generation 0 is one hinge at the horizontal fold line: two boxes (each length/2 x width x
thickness), the -x one the articulation root, a revolute joint along -y ON THE TOP SURFACE
(z = +t/2) at x = 0. Hinged there, a 180 deg fold stacks the + plate exactly one thickness above the
- plate with no overlap -- the fold target lift (``ply_gap``). Pure numpy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

LENGTH_RANGE = (0.10, 0.30)
WIDTH_RANGE = (0.10, 0.30)
THICKNESS_RANGE = (0.002, 0.020)


def design_grid(
    n: int = 16,
    length: tuple[float, float] = LENGTH_RANGE,
    width: tuple[float, float] = WIDTH_RANGE,
    thickness: tuple[float, float] = THICKNESS_RANGE,
) -> np.ndarray:
    """``(n**3, 3)`` designs ``[length, width, thickness]`` [m]; design ``d`` is
    ``(length[d // n**2], width[d // n % n], thickness[d % n])`` on the evenly spaced levels."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    levels = [np.linspace(lo, hi, n) if n > 1 else np.array([lo]) for lo, hi in (length, width, thickness)]
    for name, (lo, hi) in zip(("length", "width", "thickness"), (length, width, thickness)):
        if not 0.0 < lo <= hi:
            raise ValueError(f"{name} range must be 0 < lo <= hi, got {(lo, hi)}")
    li, wi, ti = np.unravel_index(np.arange(n**3), (n, n, n))
    return np.stack([levels[0][li], levels[1][wi], levels[2][ti]], axis=1)


def design_of(env_ids, n_designs: int, seed: int = 0) -> np.ndarray:
    """Design index of each env: a fixed permutation of ``env_id % n_designs``.

    Every run of ``n_designs`` consecutive envs still holds each design exactly once; the
    permutation only spreads a SMALL run (a smoke test, an eval of a few hundred envs) across all
    three dimensions instead of the first lengths of the grid order."""
    perm = np.random.default_rng(int(seed)).permutation(int(n_designs))
    return perm[np.asarray(env_ids, dtype=np.int64) % int(n_designs)]


@dataclass(frozen=True)
class Box:
    """A rigid plate: a box with its centre and half extents in the sheet frame at rest."""

    name: str
    center: tuple[float, float, float]
    half_extents: tuple[float, float, float]


@dataclass(frozen=True)
class Hinge:
    """Revolute joint ``parent -> child``: a line through ``anchor`` along unit ``axis`` (sheet
    frame at rest); positive angle rotates the child about ``axis`` (right hand)."""

    parent: str
    child: str
    anchor: tuple[float, float, float]
    axis: tuple[float, float, float]
    limits_deg: tuple[float, float] = (0.0, 180.0)


@dataclass(frozen=True)
class SheetSpec:
    """Plates (the first is the articulation root) and the hinges of one sheet."""

    length: float
    width: float
    thickness: float
    plates: tuple[Box, ...]
    hinges: tuple[Hinge, ...] = field(default=())
    #: Gap between the two plies of a perfect fold: the fold target lift [m].
    ply_gap: float = 0.0

    def plate(self, name: str) -> Box:
        for p in self.plates:
            if p.name == name:
                return p
        raise KeyError(name)


def single_hinge(length: float, width: float, thickness: float) -> SheetSpec:
    """The sheet as two plates and one hinge at the horizontal fold line (module docstring)."""
    if min(length, width, thickness) <= 0.0:
        raise ValueError(f"dimensions must be positive, got {(length, width, thickness)}")
    q, hw, ht = length / 4.0, width / 2.0, thickness / 2.0
    return SheetSpec(
        length=length,
        width=width,
        thickness=thickness,
        plates=(
            Box("minus", (-q, 0.0, 0.0), (q, hw, ht)),
            Box("plus", (q, 0.0, 0.0), (q, hw, ht)),
        ),
        hinges=(Hinge("minus", "plus", anchor=(0.0, 0.0, ht), axis=(0.0, -1.0, 0.0)),),
        ply_gap=thickness,
    )


def box_mass_properties(box: Box, density: float) -> tuple[float, np.ndarray, np.ndarray]:
    """``(mass, centre of mass, principal inertia diagonal about the COM)`` of a uniform box.
    ``density`` is per AREA [kg/m^2] (as ``ClothCfg.density``), so a sheet weighs what the cloth
    weighs whatever its thickness."""
    a, b, c = (2.0 * h for h in box.half_extents)
    m = density * a * b
    inertia = m / 12.0 * np.array([b * b + c * c, a * a + c * c, a * a + b * b])
    return m, np.asarray(box.center, dtype=float), inertia


def _rot(axis, theta: float) -> np.ndarray:
    u = np.asarray(axis, dtype=float)
    u = u / np.linalg.norm(u)
    k = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)


def hinge_transform(hinge: Hinge, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """``(R, t)`` taking a child point at rest to its position with the hinge at ``angle`` [rad],
    both in the parent's (= sheet) frame: ``p' = R p + t``."""
    r = _rot(hinge.axis, angle)
    a = np.asarray(hinge.anchor, dtype=float)
    return r, a - r @ a


def plate_points(spec: SheetSpec, points: np.ndarray, angle: float) -> np.ndarray:
    """Sheet-frame rest points ``(P, 3)`` -> their positions with the single hinge at ``angle``:
    points with ``x > 0`` ride the ``+x`` plate, the rest stay. Single-hinge specs only."""
    if len(spec.hinges) != 1:
        raise ValueError("plate_points handles single-hinge specs only")
    pts = np.asarray(points, dtype=float)
    r, t = hinge_transform(spec.hinges[0], angle)
    out = pts.copy()
    moving = pts[:, 0] > 0.0
    out[moving] = pts[moving] @ r.T + t
    return out

