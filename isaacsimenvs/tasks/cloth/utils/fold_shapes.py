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

URDF cannot express a seam, and ``RigidClothEnv`` loads its chain from URDF, so ``box2x8``,
``cross4`` and ``corners8`` need a different import path before they can enter the env. ``box2``
and ``box2diag4`` are trees.

**Every shape can do the stage-1 half fold** (``+x`` half over onto ``-x``, root on the ``-x``
side); that is the membership rule for the pool. Shapes that only fold some other way (a diagonal,
thirds) were tried and dropped.

**Plate-plate contact is not modelled.** ``RigidClothEnv`` already removes every slat-slat contact
pair (its 7 mm contact margin would otherwise hold folded plies 14 mm apart); folded plates are
held only by joint limits. The pool follows the same rule.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "Plate",
    "Hinge",
    "Seam",
    "FoldShape",
    "HALF_FOLD",
    "grid_plate_map",
    "plate_mass_properties",
    "make_pool",
    "POOL",
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

PI = math.pi


def make_pool(size: float = 0.10, thickness: float = 0.002, hinge: str = "mid") -> dict:
    """The stage-2 shapes, keyed ``<shape><plates>-<hinge>``."""
    L = size / 2.0
    kw = {"size": size, "thickness": thickness, "hinge": hinge}
    sfx = f"-{hinge}"
    inset = 0.006  # grasp points sit this far inside a corner, where a fingertip would pinch
    out = {}

    # 1. Half fold -- the stage-1 task. +x half over onto -x.
    out["box2" + sfx] = FoldShape(
        name="box2" + sfx,
        label="half fold (2 plates)",
        plates=(
            Plate("left", ((-L, -L), (0, -L), (0, L), (-L, L))),
            Plate("right", ((0, -L), (L, -L), (L, L), (0, L))),
        ),
        hinges=(Hinge("right", "left", (0, -L), (0, L)),),
        folds={HALF_FOLD: ({"right": PI},)},
        grasp=("right", (L - inset, 0.0)),
        **kw,
    )

    # 2. Half fold + a diagonal crease in each half. The two diagonals mirror each other across the
    #    centre crease (a "V" meeting at (0, -L)), so after the half fold they lie on one line and
    #    the two-layer stack can fold along it. All three creases meet on the sheet's EDGE, so
    #    there is no loop: a chain l_out - l_in - r_in - r_out.
    tris = {
        "l_in": ((0, -L), (0, L), (-L, L)),  # left half, next to the centre crease (root)
        "l_out": ((-L, -L), (0, -L), (-L, L)),
        "r_in": ((0, -L), (L, L), (0, L)),
        "r_out": ((0, -L), (L, -L), (L, L)),
    }
    out["box2diag4" + sfx] = FoldShape(
        name="box2diag4" + sfx,
        label="half fold + diagonal crease in each half (4 triangles)",
        plates=tuple(Plate(k, v) for k, v in tris.items()),
        hinges=(
            Hinge("l_out", "l_in", (0, -L), (-L, L)),  # left diagonal
            Hinge("r_in", "l_in", (0, -L), (0, L)),  # centre crease
            Hinge("r_out", "r_in", (0, -L), (L, L)),  # right diagonal
        ),
        folds={
            HALF_FOLD: ({"r_in": PI},),
            "left diagonal": ({"l_out": PI},),
            "right diagonal": ({"r_out": PI},),
            "both diagonals": ({"l_out": PI, "r_out": PI},),
            # r_out lies UPSIDE DOWN on the stack after the half fold, so it turns -pi to move WITH
            # l_out; +pi swings it the other way round, 89 mm away from the layer it belongs on.
            "half, then diagonal (both layers)": ({"r_in": PI}, {"l_out": PI, "r_out": -PI}),
            # NOT offered: "diagonals in, then half". A 50 x 100 mm half is not a square, so its
            # diagonal fold does not land the flap on its partner triangle -- the flap overhangs the
            # centre crease by 0.6 L (30 mm), and a half fold would swing that overhang down
            # through the table.
        },
        grasp=("r_out", (L - inset, -L + 2 * inset)),
        **kw,
    )

    # 3. As (2) with BOTH diagonals in each half: an X per half, 8 triangles. Each X is a 4-crease
    #    vertex at the half's centre, i.e. a loop -- one seam per half -- so a half folds along one
    #    of its diagonals at a time, never both. "A" is box2diag4's diagonal (the V through
    #    (0, -L)), "B" the other one (a Lambda through (0, L)); each pair mirrors across the centre
    #    crease, so after the half fold either can fold through both layers.
    cl, cr = (-L / 2.0, 0.0), (L / 2.0, 0.0)
    out["box2x8" + sfx] = FoldShape(
        name="box2x8" + sfx,
        label="half fold + both diagonals in each half (8 triangles, 2 seams)",
        plates=(
            Plate("l_r", ((0, -L), (0, L), cl)),  # left half, next to the centre crease (root)
            Plate("l_t", ((0, L), (-L, L), cl)),
            Plate("l_b", ((-L, -L), (0, -L), cl)),
            Plate("l_l", ((-L, L), (-L, -L), cl)),
            Plate("r_l", ((0, L), (0, -L), cr)),
            Plate("r_b", ((0, -L), (L, -L), cr)),
            Plate("r_t", ((L, L), (0, L), cr)),
            Plate("r_r", ((L, -L), (L, L), cr)),
        ),
        hinges=(
            Hinge("l_t", "l_r", cl, (0, L)),  # left diagonal B, upper half
            Hinge("l_b", "l_r", (0, -L), cl),  # left diagonal A, lower half
            Hinge("l_l", "l_t", (-L, L), cl),  # left diagonal A, upper half
            Hinge("r_l", "l_r", (0, -L), (0, L)),  # centre crease
            Hinge("r_b", "r_l", (0, -L), cr),  # right diagonal A, lower half
            Hinge("r_t", "r_l", (0, L), cr),  # right diagonal B, upper half
            Hinge("r_r", "r_t", cr, (L, L)),  # right diagonal A, upper half
        ),
        seams=(
            Seam("l_l", "l_b", (-L, -L), cl),  # left diagonal B, lower half
            Seam("r_r", "r_b", (L, -L), cr),  # right diagonal B, lower half
        ),
        folds={
            HALF_FOLD: ({"r_l": PI},),
            "left diagonal A": ({"l_b": PI, "l_l": PI},),
            "left diagonal B": ({"l_t": PI},),
            "right diagonal A": ({"r_b": PI, "r_r": PI},),
            "right diagonal B": ({"r_t": PI},),
            # The right half lies upside down on the left after the half fold: -pi on its layer.
            "half, then diagonal A (both layers)": (
                {"r_l": PI},
                {"l_b": PI, "l_l": PI, "r_b": -PI, "r_r": -PI},
            ),
            "half, then diagonal B (both layers)": ({"r_l": PI}, {"l_t": PI, "r_t": -PI}),
        },
        grasp=("r_r", (L - inset, 0.0)),
        **kw,
    )

    # 4. Cross fold -- quadrants, creases along both axes. The fourth hinge (q3|q4, along -y of
    #    the vertical crease) is a SEAM, so the sheet folds along one full line at a time. Rooted on
    #    the -x side so "vertical" is the stage-1 half fold: +x over onto -x.
    out["cross4" + sfx] = FoldShape(
        name="cross4" + sfx,
        label="cross fold (4 quadrants, 1 seam)",
        plates=(
            Plate("q2", ((-L, 0), (0, 0), (0, L), (-L, L))),  # -x +y, root
            Plate("q1", ((0, 0), (L, 0), (L, L), (0, L))),  # +x +y
            Plate("q3", ((-L, -L), (0, -L), (0, 0), (-L, 0))),  # -x -y
            Plate("q4", ((0, -L), (L, -L), (L, 0), (0, 0))),  # +x -y
        ),
        hinges=(
            Hinge("q1", "q2", (0, 0), (0, L)),  # vertical crease, upper half
            Hinge("q3", "q2", (-L, 0), (0, 0)),  # horizontal crease, left half
            Hinge("q4", "q1", (0, 0), (L, 0)),  # horizontal crease, right half
        ),
        seams=(Seam("q4", "q3", (0, -L), (0, 0)),),  # vertical crease, lower half
        folds={
            HALF_FOLD: ({"q1": PI},),
            "half: bottom up": ({"q3": PI, "q4": PI},),
            "quarter (bottom up, then right over)": ({"q3": PI, "q4": PI}, {"q1": PI}),
        },
        grasp=("q4", (L - inset, -L + inset)),
        **kw,
    )

    # 5. Corners + cross -- the corner ("blintz") fold with the centre diamond split by the cross
    #    into four triangles, so the sheet ALSO half-folds along either axis. The diamond's quarters
    #    form the same loop as cross4 (one seam); each corner triangle hangs off its own quarter.
    #    Corners touch the creases only at points, so every half fold carries them along whole.
    quarter = {
        "c_nw": ((0, 0), (0, L), (-L, 0)),
        "c_ne": ((0, 0), (L, 0), (0, L)),
        "c_sw": ((0, 0), (-L, 0), (0, -L)),
        "c_se": ((0, 0), (0, -L), (L, 0)),
    }
    corner = {  # (polygon, the diamond quarter it hinges on); the hinge is polygon[2] -> [0]
        "ne": (((L, 0), (L, L), (0, L)), "c_ne"),
        "nw": (((0, L), (-L, L), (-L, 0)), "c_nw"),
        "sw": (((-L, 0), (-L, -L), (0, -L)), "c_sw"),
        "se": (((0, -L), (L, -L), (L, 0)), "c_se"),
    }
    right = {"c_ne": PI}
    bottom = {"c_sw": PI, "c_se": PI}
    corners_in = {k: PI for k in corner}
    out["corners8" + sfx] = FoldShape(
        name="corners8" + sfx,
        label="corners + cross (4 diamond quarters + 4 corners, 1 seam)",
        plates=(
            *(Plate(k, v) for k, v in quarter.items()),
            *(Plate(k, v[0]) for k, v in corner.items()),
        ),
        hinges=(
            Hinge("c_ne", "c_nw", (0, 0), (0, L)),  # vertical crease, upper half
            Hinge("c_sw", "c_nw", (-L, 0), (0, 0)),  # horizontal crease, left half
            Hinge("c_se", "c_ne", (0, 0), (L, 0)),  # horizontal crease, right half
            *(Hinge(k, parent, poly[2], poly[0]) for k, (poly, parent) in corner.items()),
        ),
        seams=(Seam("c_se", "c_sw", (0, -L), (0, 0)),),  # vertical crease, lower half
        folds={
            HALF_FOLD: (right,),
            "half: bottom up": (bottom,),
            "quarter (bottom up, then right over)": (bottom, right),
            **{f"corner {k}": ({k: PI},) for k in corner},
            "all four corners": (corners_in,),
            "corners in, then half": (corners_in, right),
        },
        grasp=("ne", (L - inset, L - inset)),
        **kw,
    )
    return out


POOL: dict[str, FoldShape] = make_pool()


def shape(name: str) -> FoldShape:
    if name not in POOL:
        raise ValueError(f"unknown fold shape {name!r}; known: {sorted(POOL)}")
    return POOL[name]
