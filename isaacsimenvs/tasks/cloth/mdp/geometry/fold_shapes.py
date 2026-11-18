"""Stage-2 shape pool: flat sheets cut into rigid plates joined by hinges, each with named folds.

``rigid_cloth.ChainSpec`` is a 1-D chain -- every hinge parallel to one crease -- which is all the
stage-1 task (one half-fold) needed. Stage 2 wants sheets that fold along MORE than one line: a
second crease at right angles, four corners folding in. This module generalises the chain to a
**tree of convex plates** whose hinges can lie along any line in the sheet, plus optional **seams**
that close kinematic loops.

Pure ``numpy`` + arithmetic -- no torch, no Isaac, no solver -- the same contract as ``rigid_cloth``,
so geometry and folds are testable without booting Kit.

**Conventions.**

* Sheet frame: square ``[-L, L]^2`` with ``L = size / 2``, mid-plane at ``z = 0``, ``+z`` up.
* Every plate's body frame COINCIDES with the sheet frame at rest, so a plate's polygon is both its
  rest position and its body-local geometry. A child's hinge pose relative to its parent is then
  just a point and an axis in the sheet frame, with no per-link offsets to carry around.
* The first plate is the root (the stationary piece, free-floating in a simulator).
* Every hinge's axis is oriented so that a POSITIVE angle folds the child UP and over onto the
  parent's side. A full fold is ``+pi`` for every hinge of every shape.
* ``hinge="mid"`` puts the axis on the mid-plane: bidirectional (+-pi), and a 180 degree fold lands
  the child exactly IN its parent's plane (ply gap 0; see ``box2-mid`` in the rigid-cloth notes).
  ``hinge="surface"`` raises it to the top face: one-directional (0..pi), ply gap = thickness.

**Seams (loop closures).** A sheet folded along two crossing lines cannot be a tree: four quadrants
around a vertex form a loop. The tree keeps three of the four hinges and the fourth is a ``Seam`` --
an equality constraint in the simulator (MuJoCo / Newton ``CONNECT`` at the seam's two endpoints).
Without it the sheet has a slit along that line and its halves fold independently. With it the
four-crease vertex has the rigid-origami behaviour of a real map fold: it folds along ONE line at a
time (horizontal OR vertical), and a fully folded sheet can then fold along the other.

The shapes are YAML designs in ``cfg/rigid_cloth_designs/`` (schema: ``load_design``), each with a
PNG diagram beside it (``scripts/visualization/render_fold_designs.py``).

**Every shape can do the stage-1 half fold** (``+x`` half over onto ``-x``, root on the ``-x``
side); that is the membership rule for the pool. Shapes that only fold some other way (a diagonal,
thirds) were tried and dropped.

**Plate-plate contact is not modelled.** ``RigidClothEnv`` already removes every slat-slat contact
pair (its 7 mm contact margin would otherwise hold folded plies 14 mm apart); folded plates are
held only by joint limits. The pool follows the same rule.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

__all__ = [
    "Plate",
    "Hinge",
    "Seam",
    "FoldShape",
    "HALF_FOLD",
    "grid_plate_map",
    "plate_mass_properties",
    "DESIGN_DIR",
    "load_design",
    "load_pool",
    "sized",
    "POOL",
    "DEFAULT_SHAPES",
    "shape",
]

HINGE_MODES = ("mid", "surface")

#: The fold every pool shape must offer -- the stage-1 task, +x half over onto -x -- under one name,
#: so a trainer can ask any shape for it without knowing the shape.
HALF_FOLD = "half: right over left"

Point = tuple[float, float]


@dataclass(frozen=True)
class Plate:
    """A convex polygon of the sheet, CCW in the sheet's xy plane."""

    name: str
    polygon: tuple[Point, ...]

    @property
    def area(self) -> float:
        p = np.asarray(self.polygon)
        x, y = p[:, 0], p[:, 1]
        return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))

    @property
    def centroid(self) -> np.ndarray:
        p = np.asarray(self.polygon)
        x, y = p[:, 0], p[:, 1]
        cross = x * np.roll(y, -1) - np.roll(x, -1) * y
        a = 0.5 * cross.sum()
        cx = ((x + np.roll(x, -1)) * cross).sum() / (6.0 * a)
        cy = ((y + np.roll(y, -1)) * cross).sum() / (6.0 * a)
        return np.array([cx, cy])


@dataclass(frozen=True)
class Hinge:
    """Revolute joint of ``child`` on ``parent`` along the sheet line ``a -> b``."""

    child: str
    parent: str
    a: Point
    b: Point


@dataclass(frozen=True)
class Seam:
    """A hinge line between two plates that are NOT parent and child: a loop closure."""

    plate1: str
    plate2: str
    a: Point
    b: Point


#: A fold is a sequence of stages; each stage drives some hinges to new angles together.
Fold = tuple[Mapping[str, float], ...]

#: A folded start: independent parts, each a set of hinges turned up together by one angle.
StartPose = tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class FoldShape:
    name: str
    label: str
    plates: tuple[Plate, ...]
    hinges: tuple[Hinge, ...]
    seams: tuple[Seam, ...] = ()
    folds: Mapping[str, Fold] = field(default_factory=dict)
    size: float = 0.10
    thickness: float = 0.002
    hinge: str = "mid"
    #: Default pinch point for a viewer: (plate, sheet-frame xy).
    grasp: tuple[str, Point] | None = None

    def __post_init__(self) -> None:
        if self.hinge not in HINGE_MODES:
            raise ValueError(f"hinge must be one of {HINGE_MODES}, got {self.hinge!r}")
        names = [p.name for p in self.plates]
        if len(set(names)) != len(names):
            raise ValueError(f"{self.name}: duplicate plate names {names}")
        for p in self.plates:
            if p.area <= 0.0:
                raise ValueError(f"{self.name}: plate {p.name} is not CCW / has no area")
        children = [h.child for h in self.hinges]
        if len(set(children)) != len(children):
            raise ValueError(f"{self.name}: a plate has two parent hinges -- use a Seam for loops")
        if self.root in children:
            raise ValueError(f"{self.name}: the root {self.root} cannot be a hinge child")
        if set(children) != set(names[1:]):
            raise ValueError(f"{self.name}: every non-root plate needs exactly one hinge")
        self.tree_order()  # raises if disconnected
        for fold, stages in self.folds.items():
            for st in stages:
                unknown = set(st) - set(children)
                if unknown:
                    raise ValueError(f"{self.name}.{fold}: unknown hinges {sorted(unknown)}")
        if HALF_FOLD not in self.folds:
            raise ValueError(f"{self.name}: every pool shape must offer {HALF_FOLD!r}")
        total = sum(p.area for p in self.plates)
        if abs(total - self.size**2) > 1e-9:
            raise ValueError(f"{self.name}: plates cover {total:.6f} m^2, sheet is {self.size**2}")

    # --- structure

    @property
    def root(self) -> str:
        return self.plates[0].name

    def plate(self, name: str) -> Plate:
        for p in self.plates:
            if p.name == name:
                return p
        raise KeyError(name)

    def hinge_of(self, child: str) -> Hinge:
        for h in self.hinges:
            if h.child == child:
                return h
        raise KeyError(child)

    def tree_order(self) -> list[str]:
        """Plates root-first, each after its parent."""
        out = [self.root]
        pending = list(self.hinges)
        while pending:
            ready = [h for h in pending if h.parent in out]
            if not ready:
                raise ValueError(f"{self.name}: hinges {pending} do not connect to the root")
            for h in ready:
                out.append(h.child)
                pending.remove(h)
        return out

    @property
    def hinge_z(self) -> float:
        return 0.0 if self.hinge == "mid" else 0.5 * self.thickness

    def axis(self, child: str) -> tuple[np.ndarray, np.ndarray]:
        """``(point, unit direction)`` of a hinge in the sheet frame, oriented so +angle folds UP.

        With ``n`` the in-plane normal from the hinge line toward the child, the axis is ``n x z``:
        a right-handed rotation about it carries ``n`` to ``+z``.
        """
        h = self.hinge_of(child)
        a = np.array([*h.a, self.hinge_z])
        d = np.array([h.b[0] - h.a[0], h.b[1] - h.a[1], 0.0])
        d /= np.linalg.norm(d)
        c = np.array([*self.plate(child).centroid, self.hinge_z]) - a
        n = c - np.dot(c, d) * d
        n /= np.linalg.norm(n)
        return a, np.array([n[1], -n[0], 0.0])

    def joint_range(self) -> tuple[float, float]:
        return (-math.pi, math.pi) if self.hinge == "mid" else (0.0, math.pi)

    # --- kinematics

    def transforms(
        self, angles: Mapping[str, float] | None = None
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Each plate's ``(R, t)`` taking body-local (= rest sheet) coords to the sheet frame."""
        ang = dict(angles or {})
        out = {self.root: (np.eye(3), np.zeros(3))}
        for name in self.tree_order()[1:]:
            h = self.hinge_of(name)
            rp, tp = out[h.parent]
            p, u = self.axis(name)
            r = _rodrigues(u, float(ang.get(name, 0.0)))
            # x -> Tp(R (x - p) + p)
            out[name] = (rp @ r, rp @ (p - r @ p) + tp)
        return out

    def point(self, plate: str, xy: Sequence[float], angles=None, z: float = 0.0) -> np.ndarray:
        r, t = self.transforms(angles)[plate]
        return r @ np.array([xy[0], xy[1], z]) + t

    def fold_angles(self, fold: str) -> dict[str, float]:
        """Hinge angles after every stage of ``fold``, starting flat."""
        ang = {h.child: 0.0 for h in self.hinges}
        for st in self.folds[fold]:
            ang.update(st)
        return ang

    def seam_error(self, angles=None) -> float:
        """Worst gap [m] between the two sides of any seam -- 0 for a configuration the loop allows."""
        tf = self.transforms(angles)
        worst = 0.0
        for s in self.seams:
            for q in (s.a, s.b):
                x = np.array([q[0], q[1], self.hinge_z])
                r1, t1 = tf[s.plate1]
                r2, t2 = tf[s.plate2]
                worst = max(worst, float(np.linalg.norm((r1 @ x + t1) - (r2 @ x + t2))))
        return worst

    @property
    def ply_gap(self) -> float:
        """Height of a folded ply above the one it lands on [m] -- what the fold target is lifted by.

        0 for a mid-plane hinge (the folded plate lands IN its parent's plane; see ``box2-mid``), the
        slab thickness for a surface hinge (face to face). Same rule as ``rigid_cloth.chain_ply_gap``
        for the surface case; the mid case differs because a single mid hinge spends no wall slat.
        """
        return 0.0 if self.hinge == "mid" else self.thickness

    def start_folds(self) -> dict[str, dict[str, float]]:
        """The single-stage folds a partly folded episode can start from: ``{fold: {child: +-1}}``.

        The per-hinge direction of each fold, to be multiplied by the start angle. Single-stage only
        (a multi-stage fold's first stage is already a full fold), and only folds whose every hinge
        turns UP (+pi): a partial ``-pi`` from flat would push a plate into the table.
        """
        out = {}
        for name, stages in self.folds.items():
            if len(stages) != 1:
                continue
            st = stages[0]
            if all(abs(a - math.pi) < 1e-9 for a in st.values()):
                out[name] = {child: 1.0 for child in st}
        return out

    def start_poses(self, check_deg: Sequence[float] = (0.0, 60.0, 120.0, 180.0)) -> dict[str, StartPose]:
        """Folded starts that are NOT the target: ``{name: (part, ...)}``, each part a tuple of hinges
        turned up together by ONE angle, every part drawn its own angle (up to a full 180 deg fold).

        Parts are the single-stage folds of ``start_folds`` minus ``HALF_FOLD``, and a start is any
        set of them with no hinge in common that is legal at EVERY combination of ``check_deg``
        angles: seams closed and no vertex below the root's plane (a corner folded on a quarter that
        then turns over would point into the table). A named fold that is the union of other parts
        (``all four corners``, ``gate``) is left out: their combination with equal angles is it.
        ``box2`` has no start -- its only fold is the target.
        """
        parts = {n: tuple(sorted(d)) for n, d in self.start_folds().items() if n != HALF_FOLD}
        for n, hs in list(parts.items()):
            others = [set(h) for m, h in parts.items() if m != n and set(h) < set(hs)]
            if others and set().union(*others) == set(hs):
                del parts[n]
        out = {}
        names = list(parts)
        for r in range(1, len(names) + 1):
            for combo in itertools.combinations(names, r):
                flat = [h for n in combo for h in parts[n]]
                if len(set(flat)) == len(flat) and self._start_legal([parts[n] for n in combo], check_deg):
                    out[" + ".join(combo)] = tuple(parts[n] for n in combo)
        return out

    def _start_legal(self, parts: Sequence[Sequence[str]], check_deg: Sequence[float]) -> bool:
        for degs in itertools.product(check_deg, repeat=len(parts)):
            ang = {h: math.radians(d) for hs, d in zip(parts, degs) for h in hs}
            if self.seam_error(ang) > 1e-9:
                return False
            for name, (r, t) in self.transforms(ang).items():
                for v in self.plate(name).polygon:
                    if (r @ np.array([v[0], v[1], 0.0]) + t)[2] < -1e-9:
                        return False
        return True

    def half_fold_moving(self) -> list[str]:
        """Plates the half fold carries over: every plate whose centroid is on ``+x``."""
        return [p.name for p in self.plates if p.centroid[0] > 0.0]

    def describe(self) -> str:
        loops = f", {len(self.seams)} seam" if self.seams else ""
        return (
            f"{self.label}: {len(self.plates)} plates, {len(self.hinges)} hinges{loops}, "
            f"{self.thickness * 1e3:.1f} mm, hinge={self.hinge}"
        )


def _point_in_polygon(xy, polygon, tol: float) -> bool:
    """Closed convex CCW polygon test: on an edge counts as inside."""
    p = np.asarray(polygon, dtype=float)
    q = np.asarray(xy, dtype=float)
    for a, b in zip(p, np.roll(p, -1, axis=0)):
        e, r = b - a, q - a
        if e[0] * r[1] - e[1] * r[0] < -tol * np.linalg.norm(e):
            return False
    return True


def grid_plate_map(
    shape: FoldShape, resolution: int
) -> list[tuple[str, tuple[float, float, float]]]:
    """Carry the cloth's ``resolution x resolution`` grid onto ``shape``: ``[(plate, local xyz)]``.

    The pool's counterpart of ``rigid_cloth.sheet_grid_slat_map``, and for the same reason: the task
    scores a fold from that particle grid, so attaching each vertex rigidly to the plate containing
    it reconstructs the cloud from body poses and the task code never learns the sheet is plates.

    Order matches ``cloth_geometry.grid_mesh`` exactly -- ``i`` over ``x`` then ``j`` over ``y`` --
    because every index set the task computes is arithmetic on that order.

    Local coordinates ARE sheet coordinates, because every plate's body frame coincides with the
    sheet frame at rest (see the module docstring); ``z = 0`` is the mid-plane. A vertex on a crease
    or at a crease vertex belongs to several plates; the first in ``shape.plates`` wins. That choice
    cannot reach the score: every shared point of two plates lies on a hinge or seam line that both
    of them rotate about, so it has the same position whichever plate carries it.
    """
    if resolution < 2:
        raise ValueError(f"resolution must be >= 2, got {resolution}")
    L = shape.size / 2.0
    step = shape.size / (resolution - 1)
    tol = 1e-9 * shape.size
    out = []
    for i in range(resolution):
        for j in range(resolution):
            xy = (i * step - L, j * step - L)
            owner = next(
                (p.name for p in shape.plates if _point_in_polygon(xy, p.polygon, tol)), None
            )
            if owner is None:
                raise ValueError(f"{shape.name}: grid vertex {xy} is on no plate")
            out.append((owner, (xy[0], xy[1], 0.0)))
    return out


def plate_mass_properties(
    plate: Plate, thickness: float, density: float
) -> tuple[float, np.ndarray, np.ndarray]:
    """``(mass, centre of mass, inertia about the COM)`` of a uniform slab, in the plate's frame.

    ``density`` is PER AREA [kg/m^2], as ``ClothCfg.density``, so the pool's sheets weigh what the
    cloth weighs. Exact for a prism: in-plane second moments from the polygon (fan of triangles),
    plus the slab's own ``m t^2 / 12`` about the in-plane axes.
    """
    p = np.asarray(plate.polygon, dtype=float)
    area = plate.area
    m = density * area
    c = plate.centroid
    sxx = syy = sxy = 0.0
    for k in range(1, len(p) - 1):
        tri = np.array([p[0], p[k], p[k + 1]]) - c
        e1, e2 = tri[1] - tri[0], tri[2] - tri[0]
        a = 0.5 * abs(e1[0] * e2[1] - e1[1] * e2[0])
        x, y = tri[:, 0], tri[:, 1]
        sxx += a / 6.0 * (x @ x + x[0] * x[1] + x[1] * x[2] + x[2] * x[0])
        syy += a / 6.0 * (y @ y + y[0] * y[1] + y[1] * y[2] + y[2] * y[0])
        sxy += (
            a
            / 12.0
            * (
                2.0 * (x @ y)
                + x[0] * y[1]
                + x[1] * y[0]
                + x[0] * y[2]
                + x[2] * y[0]
                + x[1] * y[2]
                + x[2] * y[1]
            )
        )
    sigma = m / area
    tz = m * thickness**2 / 12.0
    inertia = np.array(
        [
            [sigma * syy + tz, -sigma * sxy, 0.0],
            [-sigma * sxy, sigma * sxx + tz, 0.0],
            [0.0, 0.0, sigma * (sxx + syy)],
        ]
    )
    return m, np.array([c[0], c[1], 0.0]), inertia


def _rodrigues(u: np.ndarray, theta: float) -> np.ndarray:
    k = np.array([[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]], [-u[1], u[0], 0.0]])
    return np.eye(3) + math.sin(theta) * k + (1.0 - math.cos(theta)) * (k @ k)


# ------------------------------------------------------------------------------------- the pool

#: The pool's designs, one YAML per shape (schema: ``load_design``).
DESIGN_DIR = Path(__file__).resolve().parents[4] / "cfg" / "rigid_cloth_designs"
#: Grasp-point inset unit [m]: a fingertip pinches this far inside a corner.
GRASP_INSET = 0.006


def load_design(path: str | Path, size: float | None = None, thickness: float | None = None) -> FoldShape:
    """A ``FoldShape`` from its YAML design, optionally at another ``size`` / ``thickness``.

    Coordinates are in units of ``L = size / 2``; fold angles in degrees. ``plates`` and ``hinges`` are
    mappings (first plate = root; a hinge is keyed by its child: ``[parent, a, b]``), ``seams`` a list of
    ``[plate1, plate2, a, b]``, ``folds`` a mapping of name -> list of stages ``{hinge child: deg}``,
    ``grasp`` ``{plate, xy, inset}`` = ``xy * L + inset * GRASP_INSET``. Mapping order is kept: the
    plate order is the articulation's body order and the fold order names the folded starts.
    """
    d = yaml.safe_load(Path(path).read_text())
    if size is not None:
        d["size"] = size
    if thickness is not None:
        d["thickness"] = thickness
    L = float(d["size"]) / 2.0

    def pt(v) -> Point:
        return (v[0] * L, v[1] * L)

    g = d.get("grasp")
    return FoldShape(
        name=d["name"],
        label=d["label"],
        plates=tuple(Plate(k, tuple(pt(v) for v in poly)) for k, poly in d["plates"].items()),
        hinges=tuple(Hinge(child, parent, pt(a), pt(b)) for child, (parent, a, b) in d["hinges"].items()),
        seams=tuple(Seam(p1, p2, pt(a), pt(b)) for p1, p2, a, b in d.get("seams") or ()),
        folds={name: tuple({k: math.radians(v) for k, v in st.items()} for st in stages)
               for name, stages in d["folds"].items()},
        size=float(d["size"]),
        thickness=float(d["thickness"]),
        hinge=d["hinge"],
        grasp=None if g is None else (
            g["plate"],
            tuple(x * L + k * GRASP_INSET for x, k in zip(g["xy"], g.get("inset", (0, 0))))),
    )


def load_pool(design_dir: str | Path = DESIGN_DIR) -> dict[str, FoldShape]:
    """Every design in ``design_dir``, keyed by name."""
    out = {}
    for path in sorted(Path(design_dir).glob("*.yaml")):
        s = load_design(path)
        out[s.name] = s
    return out


#: Every pool shape, loaded from ``DESIGN_DIR`` (the seven of the flat7 recipes).
POOL: dict[str, FoldShape] = load_pool()

#: ``FoldPoolCfg.shapes`` default (five shapes; the flat7 recipes add ``box4`` and ``star8`` by name).
DEFAULT_SHAPES: tuple[str, ...] = (
    "box2-mid", "box2diag4-mid", "box2x8-mid", "cross4-mid", "corners8-mid",
)


def sized(name: str, size: float, thickness: float) -> FoldShape:
    """Pool shape ``name`` at another sheet size and plate thickness (same design, scaled)."""
    shape(name)  # unknown names fail with the known list
    return load_design(DESIGN_DIR / f"{name}.yaml", size=size, thickness=thickness)


def shape(name: str) -> FoldShape:
    if name not in POOL:
        raise ValueError(f"unknown fold shape {name!r}; known: {sorted(POOL)}")
    return POOL[name]
