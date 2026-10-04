"""Rigid approximation of the folding cloth: a chain of slats joined by hinges.

Pure ``xml.etree`` + arithmetic -- no torch, no Isaac imports, no solver -- so the geometry and the
URDF are testable without booting Kit, the same contract as ``cloth_geometry`` and
``multilink_cartpole/utils/generate_cartpole``.

**What this replaces.** ``ClothEnv`` simulates the sheet as a VBD particle cloth on a coupled
MJWarp + VBD solve. That is faithful and it is the whole cost of the task: ~96% of a step is VBD
work at 80 iterations, and even at the tuned 12 it dominates. This module expresses the same sheet
as ordinary rigid bodies in ONE articulation, so the scene is a single MJWarp solve with no
coupling, no proxy exchange and no particle self-contact.

**The topology, and why it is a 1-D chain.**

    slat_0 --hinge-- slat_1 --hinge-- ... --hinge-- slat_{N-1}

``N`` rigid slats, each spanning the sheet's full width in ``y`` and some width in ``x``, joined by
revolute joints whose axis is ``+y`` -- parallel to the crease. The articulation is free-floating
(the root has no parent) and rooted at the last slat of the STATIONARY half, so the two halves hang
off the root as two branches rather than one end dragging the other through ``N`` joints.

A full 2-D grid of plates, hinged in both directions, is the obvious richer model and it does not
work. Two independent reasons:

  * **It is not a tree.** Every 2x2 cell of plates closes a kinematic loop, which a URDF cannot
    express and an Isaac Lab ``Articulation`` cannot hold. It would need explicit loop-closure
    constraints.
  * **It would not bend anyway.** A grid of rigid quads hinged along both families of grid lines is
    rigid -- this is the standard rigid-origami result: a developable quad mesh with creases in two
    directions locks. The extra joints would buy nothing but cost.

So the honest rigid approximation of THIS task is one-dimensional: it bends about ``y`` and is rigid
across its width. That is the model's defining limitation, stated once here rather than discovered
later -- the sheet cannot drape over a fingertip, cannot wrinkle diagonally, and cannot be gathered.
It can be folded about the crease, which is what the task scores.

**Two hinge placements, and what each costs.**

``hinge="mid"`` puts every axis on the slab's mid-plane. Symmetric, bends both ways, and cannot fold
through a zero-radius turn: a single joint at 180 degrees lands the child *coincident* with its
parent, so the tightest U-turn is TWO joints at 90 degrees with one slat standing between them as
the wall. The ply gap is then the WALL's width, not the slab's thickness.

``hinge="surface"`` raises every axis to the slab's top face, so the body hangs a half-thickness
below its own axis and ONE joint at 180 degrees lays the child flat on the parent. Exact fold, ply
gap equal to ``thickness``. The price is that every hinge becomes one-directional -- the opposite
rotation drives the slab through its parent -- so the sheet acquires a distinguished "up", cannot be
folded the other way, and leans on the solver's joint-limit impedance to stay legal.

**Where the wall comes from, and why parity matters.** A mid-plane fold must spend one slat standing
vertically. If some slat straddles ``x = 0`` -- i.e. the chain has an ODD number of uniform slats,
or a narrow dedicated bar at the crease -- that slat is the wall and belongs to neither ply, so both
plies keep their full length and the fold lands on its exact mirror. If ``x = 0`` falls on a seam
(an EVEN uniform count) there is no such slat, the wall has to be taken out of the moving ply, and
the returning ply is one width short. Measured by :func:`fold_residual`: 0.00 mm at 17 slats against
6.25 mm at 16.

    This reverses what an earlier version of this module claimed. It derived the fold by bending the
    first two slats of the MOVING half, which takes the wall out of that ply for any parity, and
    concluded that odd counts were a pitch WORSE. The wall slat is a free choice, the crease slat is
    the right one whenever it exists, and the parity rule is the opposite of what was written. See
    :func:`default_num_slats` -- parity should follow the hinge mode, because a surface hinge wants
    the fold seam exactly on ``x = 0`` and therefore wants an even count.

**Non-uniform chains.** ``widths`` is a list, not a pitch, so the crease bar can be thin while the
plies are single large plates -- :func:`stacked_chain`, three bodies and two joints, whose ply gap is
the bar's width and whose cost is a rounding error. The general chain is :class:`ChainSpec`; the
uniform helpers below are thin wrappers over it.

**Colliders.** A cylinder of radius ``thickness / 2`` lying along ``y`` is the rigid analogue of the
VBD sheet's own construction -- particles of radius 8 mm on a 16.7 mm grid are a row of spheres, and
this is a row of tubes -- and it rolls smoothly through a bend. A box is flat, which is what a
folded ply has to land on, and is the right choice for a surface hinge; with a mid-plane hinge two
boxes hinged at their shared face interpenetrate at any nonzero angle (force-free, since adjacent
links are contact-filtered, but visible). The visual geometry is deliberately the same shape as the
collision geometry: the cloth task has already paid for a render that showed something other than
what was simulated (a hammer-shaped goal marker), and a flat visual sheet over a tube chain would be
the same mistake.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

__all__ = [
    "SLAT_LINK_PREFIX",
    "SLAT_JOINT_PREFIX",
    "HINGE_MODES",
    "SHAPES",
    "DEFAULT_NUM_SLATS",
    "default_num_slats",
    "slat_link_name",
    "slat_joint_name",
    "slat_pitch",
    "root_slat",
    "half_slats",
    "crease_slat",
    "slat_centres",
    "forward_kinematics",
    "point_in_sheet_frame",
    "fold_residual",
    "folded_joint_angles",
    "fold_ply_gap",
    "slat_frame_points",
    "corner_points",
    "max_bend_for_thickness",
    "plate_joint_stiffness",
    "CLOTH_REFERENCE",
    "cloth_effective_friction",
    "generate_rigid_cloth_urdf",
    "fold_direction",
    # --- general (possibly non-uniform) chains
    "ChainSpec",
    "uniform_chain",
    "stacked_chain",
    "chain_frames",
    "chain_geom_offset",
    "chain_fold_angles",
    "chain_fold_residual",
    "chain_ply_gap",
    "chain_points",
    "chain_corner_points",
    "chain_point_in_sheet_frame",
    "sheet_grid_slat_map",
    "cloth_contact_pairs",
    "write_chain_urdf",
    "VARIANTS",
    "variant",
]

SLAT_LINK_PREFIX = "slat_"
SLAT_JOINT_PREFIX = "slat_joint_"

HINGE_MODES = ("mid", "surface")
SHAPES = ("cylinder", "box")

#: Default slat count for a *surface* hinge, where the fold seam must land on ``x = 0`` and so the
#: count must be EVEN. 16 slats over the 0.10 m sheet is a 6.25 mm pitch, 2.7x finer than the VBD
#: sheet's 16.7 mm particle spacing.
#:
#: A mid-plane hinge wants the opposite parity -- see :func:`default_num_slats`, which is the
#: function to call when the hinge mode is known.
DEFAULT_NUM_SLATS = 16


def default_num_slats(hinge: str = "mid", target: int = 16) -> int:
    """Slat count of the right parity for ``hinge``, nearest to ``target``.

    **Parity follows the hinge mode, and the two modes disagree.**

    A mid-plane fold has to spend one slat standing vertically as the wall of the U. An ODD count
    puts a slat astride ``x = 0``; that slat belongs to neither ply, so it can be the wall for free
    and both plies keep their full length. An even count puts a seam there, leaving nothing to spend,
    so the wall is taken out of the moving ply and the returning ply lands one width short.

    A surface fold spends no slat at all -- one joint at 180 degrees does the whole thing -- but the
    joint it uses is a SEAM, and the seam has to be at ``x = 0`` for the fold to be centred. That
    wants an EVEN count; an odd count folds about a seam a half-width off centre.
    """
    if hinge not in HINGE_MODES:
        raise ValueError(f"hinge must be one of {HINGE_MODES}, got {hinge!r}")
    want_odd = hinge == "mid"
    n = int(target)
    if (n % 2 == 1) != want_odd:
        n += 1
    return max(3 if want_odd else 2, n)


def slat_link_name(index: int) -> str:
    return f"{SLAT_LINK_PREFIX}{index}"


def slat_joint_name(child_index: int) -> str:
    """Joints are named for their CHILD slat.

    The chain branches at the root, so "the joint before slat i" is the only unambiguous naming:
    numbering joints along the chain would give two joint 1s, one per branch.
    """
    return f"{SLAT_JOINT_PREFIX}{child_index}"


# --------------------------------------------------------------------------------- slat layout
#
# Everything below is expressed over a list of WIDTHS, so a uniform chain and a thin-bar-at-the-
# crease chain are the same code. Seams are the cumulative sums, recentred on x = 0.


def _edges(widths: Sequence[float]) -> list[float]:
    """Seam positions, ``len(widths) + 1`` of them, with the sheet centred on ``x = 0``."""
    span = float(sum(widths))
    out = [-span / 2.0]
    for w in widths:
        out.append(out[-1] + float(w))
    return out


def _centres(widths: Sequence[float]) -> list[float]:
    e = _edges(widths)
    return [0.5 * (a + b) for a, b in zip(e, e[1:])]


def _classify(widths: Sequence[float]) -> tuple[list[int], list[int], int | None]:
    """Split the chain at ``x = 0`` into ``(stationary, moving, crease_slat)``.

    A slat is stationary if it lies wholly in ``-x``, moving if wholly in ``+x``, and the crease slat
    if it straddles the origin -- which happens for an odd uniform count and for
    :func:`stacked_chain`, and not at all when ``x = 0`` falls on a seam.

    The tolerance is relative to the span because the seam case has to be exact: ``size / n``
    summed ``n / 2`` times is not bit-identical to ``size / 2``, and a slat misclassified by one
    float would move the root.
    """
    e = _edges(widths)
    tol = 1e-9 * float(sum(widths))
    stationary, moving, crease = [], [], None
    for i in range(len(widths)):
        lo, hi = e[i], e[i + 1]
        if hi <= tol:
            stationary.append(i)
        elif lo >= -tol:
            moving.append(i)
        else:
            if crease is not None:
                raise ValueError("more than one slat straddles x = 0")
            crease = i
    return stationary, moving, crease


def slat_pitch(size: float, num_slats: int) -> float:
    """Width of one slat along the fold axis [m], for a uniform chain."""
    _check(size, num_slats)
    return size / num_slats


def slat_centres(size: float, num_slats: int) -> list[float]:
    """Local ``x`` of each slat's centre at rest, sheet centred on the origin."""
    return _centres(_uniform_widths(size, num_slats))


def root_slat(num_slats: int) -> int:
    """Index of the base link: the LAST SLAT OF THE STATIONARY HALF.

    Rooting mid-sheet rather than at an edge buys three things. The kinematic depth roughly halves,
    which is what an articulation solver actually pays for. The root body's pose is a rigid frame of
    the stationary half -- the frame ``ClothEnv._stationary_frame`` fits from particles and the pose
    the adapter reports -- and it stays that frame through a fold, because the root never moves.
    And a reset that writes the root pose places the sheet symmetrically instead of hanging it off
    one edge.

    **Not the crease slat, when there is one.** An earlier version rooted an odd chain at its centre
    slat, which is the slat that stands up as the wall of the fold; the root would then be vertical
    in the folded pose, so "root pose" would no longer mean "stationary half's frame" and every
    analytic comparison against the rest pose would need an extra 90-degree rotation. Rooting one
    slat inboard of the wall keeps the stationary half, the root, and the sheet frame all aligned.
    """
    _check_slats(num_slats)
    stationary, _moving, _crease = _classify(_uniform_widths(1.0, num_slats))
    return max(stationary)


def half_slats(num_slats: int, positive: bool = True) -> list[int]:
    """Slat indices on one side of the crease, CREASE SLAT EXCLUDED when there is one.

    Mirrors ``cloth_geometry.half_indices``: for odd ``num_slats`` the crease slat straddles the
    crease and belongs to neither half. Including it would report a fold as partly done before
    anything moved, since that slat barely translates -- it only stands up.
    """
    _check_slats(num_slats)
    stationary, moving, _crease = _classify(_uniform_widths(1.0, num_slats))
    return moving if positive else stationary


def crease_slat(num_slats: int) -> int | None:
    """The slat straddling ``x = 0``, or ``None`` if a seam falls there."""
    _check_slats(num_slats)
    return _classify(_uniform_widths(1.0, num_slats))[2]


# ------------------------------------------------------------------------------------- the spec


@dataclass(frozen=True)
class ChainSpec:
    """A hinged-slat chain: widths along the fold axis, plus how it is hinged and drawn.

    ``widths`` is what makes this general. A uniform sheet is ``[size/N] * N``; a two-plate folder
    with a thin crease bar is ``[L, bar, L]``. Everything derived -- root, plies, joint limits, fold
    configuration, ply gap, URDF -- reads the widths, so the five approximations in :data:`VARIANTS`
    are five specs and one implementation.
    """

    widths: tuple[float, ...]
    size: float = 0.10
    thickness: float = 0.002
    hinge: str = "mid"
    shape: str = "cylinder"
    label: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        if len(self.widths) < 2:
            raise ValueError(f"a chain needs at least 2 slats, got {len(self.widths)}")
        if any(float(w) <= 0.0 for w in self.widths):
            raise ValueError(f"every width must be positive, got {self.widths}")
        if self.size <= 0.0:
            raise ValueError(f"size must be positive, got {self.size}")
        if self.thickness <= 0.0:
            raise ValueError(f"thickness must be positive, got {self.thickness}")
        if self.hinge not in HINGE_MODES:
            raise ValueError(f"hinge must be one of {HINGE_MODES}, got {self.hinge!r}")
        if self.shape not in SHAPES:
            raise ValueError(f"shape must be one of {SHAPES}, got {self.shape!r}")

    # --- structure

    @property
    def num_slats(self) -> int:
        return len(self.widths)

    @property
    def span(self) -> float:
        """Total extent along the fold axis [m]. Equals ``size`` for a square sheet."""
        return float(sum(self.widths))

    @property
    def edges(self) -> list[float]:
        return _edges(self.widths)

    @property
    def centres(self) -> list[float]:
        return _centres(self.widths)

    @property
    def root(self) -> int:
        return max(_classify(self.widths)[0])

    @property
    def moving(self) -> list[int]:
        return _classify(self.widths)[1]

    @property
    def stationary(self) -> list[int]:
        return _classify(self.widths)[0]

    @property
    def crease(self) -> int | None:
        """The slat straddling ``x = 0``, or ``None``."""
        return _classify(self.widths)[2]

    @property
    def wall(self) -> int | None:
        """The slat that stands vertically in the tightest fold.

        ``root + 1`` for a mid-plane hinge -- which is the crease slat when one exists, and otherwise
        the first slat of the moving ply. ``None`` for a surface hinge, which spends no slat.
        """
        return None if self.hinge == "surface" else self.root + 1

    # --- limits

    @property
    def joint_limit(self) -> float:
        """Per-joint bend ceiling [rad].

        ``pi`` (one-sided) for a surface hinge. For a mid-plane hinge, the smallest bend at which
        some slat ``i`` and slat ``i+2`` come into contact -- adjacent slats are contact-filtered by
        the articulation, so one-removed is the binding pair. See :func:`max_bend_for_thickness`.
        """
        if self.hinge == "surface":
            return math.pi
        if self.num_slats < 3:
            return math.pi / 2.0
        return min(
            _max_bend(self.widths[i], self.widths[i + 1], self.widths[i + 2], self.thickness)
            for i in range(self.num_slats - 2)
        )

    def foldable(self) -> bool:
        return self.num_slats >= (2 if self.hinge == "surface" else 3)

    def describe(self) -> str:
        w = ", ".join(f"{v * 1e3:.2f}" for v in self.widths[:4])
        if self.num_slats > 4:
            w += ", ..."
        return (
            f"{self.label or 'chain'}: {self.num_slats} slats [{w}] mm, "
            f"{self.thickness * 1e3:.2f} mm {self.shape}, hinge={self.hinge}"
        )


def _uniform_widths(size: float, num_slats: int) -> tuple[float, ...]:
    return tuple([size / num_slats] * num_slats)


def uniform_chain(
    size: float = 0.10,
    num_slats: int = DEFAULT_NUM_SLATS,
    thickness: float | None = None,
    hinge: str = "mid",
    shape: str = "cylinder",
    label: str = "",
) -> ChainSpec:
    """Equal-width slats across the sheet. ``thickness=None`` means one pitch.

    One pitch is the thickness at which a mid-plane chain's fold is held by CONTACT: the ply gap of a
    two-right-angle fold is the wall's width, so a slab thinner than its own pitch folds to a ply
    that hovers ``pitch - thickness`` above the one below it, held up by the joint limit alone. A
    surface hinge has no such coupling and wants to be thin.
    """
    _check(size, num_slats)
    pitch = size / num_slats
    return ChainSpec(
        widths=_uniform_widths(size, num_slats),
        size=size,
        thickness=pitch if thickness is None else float(thickness),
        hinge=hinge,
        shape=shape,
        label=label,
    )


def stacked_chain(
    size: float = 0.10,
    wall_width: float = 0.002,
    thickness: float = 0.002,
    hinge: str = "mid",
    shape: str = "box",
    label: str = "",
) -> ChainSpec:
    """Two full-half plates with a narrow bar between them: ``[L, wall_width, L]``.

    The minimal mid-plane folder. The bar straddles ``x = 0``, so it is the crease slat and therefore
    the wall of the fold; the two plates keep their full ``L = (size - wall_width) / 2`` and land on
    each other with a ply gap of exactly ``wall_width``. Three bodies, two joints, both hinges
    bidirectional, and a ply gap chosen independently of everything else -- which is the thing a
    uniform mid-plane chain cannot do, since there its ply gap IS its pitch.

    What it gives up is every intermediate shape: two rigid plates cannot drape, cannot roll, and
    can only be flat or folded about this one crease.
    """
    if not 0.0 < wall_width < size:
        raise ValueError(f"wall_width must be in (0, {size}), got {wall_width}")
    ply = (size - wall_width) / 2.0
    return ChainSpec(
        widths=(ply, wall_width, ply),
        size=size,
        thickness=thickness,
        hinge=hinge,
        shape=shape,
        label=label,
    )


# ---------------------------------------------------------------------------------- kinematics
#
# Link frames sit on each slat's INBOARD edge -- the edge nearer the root -- because a URDF joint
# rotates about an axis through the CHILD's frame origin. A frame at the slat centre would hinge
# the sheet about each slat's midline instead of about the seam between slats. The root is the one
# exception: its frame is its centre, since nothing hinges it.
#
# All of this is planar (x, z); the chain has no y degree of freedom by construction. A joint angle
# ``a`` rotates about ``+y``, which tips ``+x`` toward ``-z``:  R(a) . (dx, 0) = (dx cos a, -dx sin a).
# So NEGATIVE angles fold upward.


def _rot(a: float, dx: float) -> tuple[float, float]:
    return dx * math.cos(a), -dx * math.sin(a)


def chain_frames(
    spec: ChainSpec, joint_angles: Sequence[float] | None = None
) -> list[tuple[float, float, float]]:
    """Pose of every slat's link frame in the sheet frame, as ``[(x, z, angle)]``.

    ``joint_angles`` is indexed by CHILD slat (see :func:`slat_joint_name`); the root's entry is
    ignored. ``None`` means the flat rest pose.
    """
    n = spec.num_slats
    root = spec.root
    w = spec.widths
    ang = [0.0] * n if joint_angles is None else [float(v) for v in joint_angles]
    if len(ang) != n:
        raise ValueError(f"joint_angles must have {n} entries, got {len(ang)}")

    out: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)] * n
    out[root] = (spec.centres[root], 0.0, 0.0)
    # Outboard in +x, then outboard in -x. The step is the PARENT's width -- halved for the root,
    # whose frame is its centre rather than a seam.
    for i in range(root + 1, n):
        px, pz, pa = out[i - 1]
        step = w[i - 1] / 2.0 if i - 1 == root else w[i - 1]
        dx, dz = _rot(pa, step)
        out[i] = (px + dx, pz + dz, pa + ang[i])
    for i in range(root - 1, -1, -1):
        px, pz, pa = out[i + 1]
        step = -(w[i + 1] / 2.0 if i + 1 == root else w[i + 1])
        dx, dz = _rot(pa, step)
        out[i] = (px + dx, pz + dz, pa + ang[i])
    return out


def chain_geom_offset(spec: ChainSpec, index: int) -> tuple[float, float]:
    """Slat centre relative to its own link frame, as ``(dx, dz)`` in that frame.

    ``hinge="mid"`` puts the axis on the slab's mid-plane, so ``dz`` is zero. ``hinge="surface"``
    raises the axis to the slab's TOP face, so the body hangs one half-thickness BELOW its frame --
    which is what lets a single joint fold 180 degrees and land the child flat on the parent instead
    of inside it.
    """
    if not 0 <= index < spec.num_slats:
        raise ValueError(f"slat index {index} out of range for {spec.num_slats} slats")
    dz = 0.0 if spec.hinge == "mid" else -0.5 * spec.thickness
    if index == spec.root:
        return 0.0, dz
    half = spec.widths[index] / 2.0
    return (half if index > spec.root else -half), dz


def chain_point_in_sheet_frame(
    spec: ChainSpec,
    index: int,
    point: tuple[float, float, float],
    joint_angles: Sequence[float] | None = None,
) -> tuple[float, float, float]:
    """Carry a point given in slat ``index``'s link frame into the sheet frame.

    The pure-python twin of what the env will do on the GPU with ``body_pos_w`` / ``body_quat_w``.
    Having it here is what lets the fold geometry be checked without a simulator -- the same reason
    ``cloth_geometry`` exists apart from ``cloth_env``.
    """
    fx, fz, a = chain_frames(spec, joint_angles)[index]
    px, py, pz = point
    return (
        fx + px * math.cos(a) + pz * math.sin(a),
        py,
        fz - px * math.sin(a) + pz * math.cos(a),
    )


def chain_points(spec: ChainSpec, index: int, num_y: int = 2) -> list[tuple[float, float, float]]:
    """Sample points on slat ``index``, expressed in that slat's own LINK frame.

    The bridge to the existing fold metrics. ``ClothEnv`` scores a fold from a particle cloud --
    ``fold_error``, ``footprint_ratio`` and ``_stationary_frame`` all consume ``(num_envs, P, 3)``
    world positions -- and a rigid chain has bodies, not particles. Transforming these by each
    slat's ``body_pos_w`` / ``body_quat_w`` reproduces the cloud, so every one of those metrics
    carries over unchanged instead of being rewritten against body poses.

    ``num_y = 2`` gives the slat's four corners, which is the minimum that spans an area and so the
    minimum from which a rotation can be fitted.
    """
    if num_y < 2:
        raise ValueError(f"num_y must be >= 2, got {num_y}")
    cx, cz = chain_geom_offset(spec, index)
    half = spec.widths[index] / 2.0
    ys = [-spec.size / 2.0 + spec.size * k / (num_y - 1) for k in range(num_y)]
    return [(cx + sx, y, cz) for sx in (-half, half) for y in ys]


def chain_corner_points(spec: ChainSpec) -> list[tuple[int, tuple[float, float, float]]]:
    """The four corners of the MOVING half: ``[(slat_index, point_in_that_slat_frame)]``.

    The rigid counterpart of ``cloth_geometry.corner_indices``, and it answers the same question the
    same way: corners of the moving half, not points along its far edge, because four collinear
    points cannot determine a rotation -- which is how ``object_rot`` came to read identity on the
    VBD sheet.

    Both corners of the outermost slat and both of the first slat past the crease, so the set spans
    the half's whole travel. Ordered (near-left, near-right, far-left, far-right) to match
    ``corner_indices``.
    """
    moving = spec.moving
    if not moving:
        raise ValueError(f"{spec.num_slats} slats leaves no moving half")
    near, far = moving[0], moving[-1]
    y = spec.size / 2.0

    def pt(i: int, outboard: bool) -> tuple[float, float, float]:
        dx, dz = chain_geom_offset(spec, i)
        half = spec.widths[i] / 2.0
        return (dx + (half if outboard else -half), 0.0, dz)

    def at(i: int, outboard: bool, sy: float) -> tuple[float, float, float]:
        x, _y, z = pt(i, outboard)
        return (x, sy, z)

    return [
        (near, at(near, False, -y)),
        (near, at(near, False, y)),
        (far, at(far, True, -y)),
        (far, at(far, True, y)),
    ]


# ------------------------------------------------------------------------- the cloth's own grid


def sheet_grid_slat_map(
    spec: ChainSpec, resolution: int
) -> list[tuple[int, tuple[float, float, float]]]:
    """Carry the VBD sheet's particle grid onto the chain: ``[(slat, point_in_slat_frame)]``.

    This is the whole reason the rigid chain can be dropped into ``ClothEnv`` without rewriting the
    task. That env scores a fold from a particle cloud -- ``fold_error``, ``footprint_ratio``,
    ``_stationary_frame`` and the Kabsch fit in ``cloth_adapter`` all consume ``(num_envs, P, 3)``
    world positions on a ``resolution x resolution`` grid -- and a chain has bodies, not particles.
    Attaching each grid vertex rigidly to the slat that contains it reconstructs that same cloud
    from ``body_pos_w`` / ``body_quat_w``, so reward, goal, success criterion and observation layout
    are *identical* between the two manipulands. Anything that then differs between a VBD run and a
    chain run is dynamics, which is the comparison worth making.

    Order matches ``cloth_geometry.grid_mesh`` exactly -- row-major, ``i`` over ``x`` then ``j`` over
    ``y`` -- because every index set the task computes (``corner_indices``, ``half_indices``,
    ``keypoint_indices``) is arithmetic on that order. Returning a differently-ordered cloud of the
    right SHAPE would leave every one of them silently pointing at the wrong particles.

    A vertex that lands exactly on a seam belongs to the slat on its ``-x`` side. That is a real
    choice for ``box2-surface``, whose only seam sits at ``x = 0`` right where the grid has a row
    (odd resolution). It does not reach the score: the centre row is the crease and
    ``half_indices`` excludes it from both halves. It is pinned anyway so the cloud is
    reproducible rather than dependent on float comparison.

    The grid lies on each slab's **MID-PLANE**, i.e. at ``chain_geom_offset``'s ``dz`` -- zero for
    ``hinge="mid"`` and one half-thickness below the link frame for ``hinge="surface"``. That is a
    real choice and the naive one is wrong:

    A cloth is a zero-thickness surface and a slat is that surface thickened, so the sheet's material
    surface is the slab's mid-plane. Under a mid-plane hinge the two coincide and the choice is
    invisible. Under a SURFACE hinge the link frame sits on the slab's top face, and a folded child
    rotates 180 degrees about that face -- so a point placed on the frame plane lands **exactly on the
    parent's own top face**, i.e. the two plies' material surfaces coincide and the fold reads as
    zero separation. Measured: with the grid on the frame plane, ``box-surface``'s folded moving half
    settled at ``z = 1e-18`` against a ``chain_ply_gap`` of 2.00 mm, so the fold target -- which is
    lifted by exactly that gap -- carried 2 mm of error at a geometrically perfect fold. On the
    mid-plane both hinge modes separate by ``chain_ply_gap``, and one number serves both.

    The consequence is that a surface-hinge chain's rest cloud sits one half-thickness (1 mm) BELOW
    the analytic flat grid. Harmless, and it cancels: every quantity the task computes from the cloud
    is either a centred offset (``_corner_rest_offsets``) or expressed relative to the stationary
    half's own live frame (``fold_targets_w``), so a uniform translation of the whole cloud drops out.

    Raises:
        ValueError: if the sheet is not square (``span != size``), because the grid spans ``size``
            along both axes and the chain only spans ``span`` along ``x``. Silently clipping would
            put every vertex past the last seam onto the outermost slat and report a shortened
            sheet as a folded one.
    """
    if resolution < 2:
        raise ValueError(f"resolution must be >= 2, got {resolution}")
    if abs(spec.span - spec.size) > 1e-9 * spec.size:
        raise ValueError(
            f"grid mapping needs a square sheet: span {spec.span:.6f} m != size {spec.size:.6f} m"
        )

    edges = spec.edges
    frames = chain_frames(spec, None)
    step = spec.size / (resolution - 1)
    half = spec.size / 2.0
    tol = 1e-9 * spec.span

    out: list[tuple[int, tuple[float, float, float]]] = []
    for i in range(resolution):
        x = i * step - half
        # Rightmost slat whose lower edge is at or below x, i.e. seams resolve to the -x side.
        slat = 0
        for k in range(spec.num_slats):
            if x > edges[k] + tol:
                slat = k
        fx = frames[slat][0]
        dz = chain_geom_offset(spec, slat)[1]
        for j in range(resolution):
            out.append((slat, (x - fx, j * step - half, dz)))
    return out


# --------------------------------------------------------------------------------------- folding


def chain_fold_angles(spec: ChainSpec, joint_limit: float | None = None) -> list[float]:
    """Joint configuration of the tightest fold this chain can make.

    **Mid-plane hinge: two joints at 90 degrees, bracketing slat ``root + 1``.** That slat stands
    vertically as the wall of a U and everything past it lies flat on top, one wall-width up. One
    joint at 180 degrees is the intuitive answer and it is geometrically empty here -- the axis is on
    the seam at mid-depth, so a 180-degree rotation lands the slat exactly *coincident* with its
    neighbour, no lift at all. A mid-plane chain gets its ply separation from the ARC, and the
    tightest arc is two right angles.

    The wall is ``root + 1`` rather than the first slat of the moving ply, and those differ exactly
    when a slat straddles ``x = 0``: then the wall is that crease slat, which belongs to neither ply,
    and both plies keep their full length. This is the whole reason parity matters.

    **Surface hinge: ONE joint at 180 degrees.** The body hangs a half-thickness below its axis, so a
    180-degree rotation lands the child flat on the parent's top face -- touching, not overlapping.
    No slat is spent as a wall, the reflection is exact, and the ply gap is ``thickness``.

    Angles carry the sign that folds each branch up and over; see :func:`fold_direction`.
    """
    n = spec.num_slats
    root = spec.root
    ang = [0.0] * n
    if spec.hinge == "surface":
        if root + 1 >= n:
            raise ValueError(f"{n} slats leaves no moving half to fold")
        ang[root + 1] = _fold_sign(root + 1, root) * math.pi
        return ang

    if root + 2 >= n:
        raise ValueError(
            f"{n} slats leaves nothing past the wall; a mid-plane-hinge fold needs a wall slat and "
            "a returning ply, so at least 3 slats"
        )
    # Capped at 90 degrees even when the collider would allow more: the two bends must sum to 180
    # for the returning ply to come back level, so a symmetric fold cannot use more than a right
    # angle each. (A thin slab CAN fold tighter with ASYMMETRIC bends -- 180-eps and eps, giving a
    # ply gap of wall_width * sin(eps) -- but that buries the wall inside its own neighbours, which
    # is force-free only because those pairs are contact-filtered. Not offered here.)
    limit = spec.joint_limit if joint_limit is None else abs(float(joint_limit))
    limit = min(limit, math.pi / 2.0)
    ang[root + 1] = -limit
    ang[root + 2] = -limit
    return ang


def _fold_sign(index: int, root: int) -> float:
    return -1.0 if index > root else 1.0


def fold_direction(index: int, num_slats: int) -> float:
    """Sign of the joint angle that folds slat ``index`` UP and over, ``+1`` or ``-1``.

    Only meaningful for ``hinge="surface"``, where each hinge is **one-directional**: the body hangs
    below its own axis, so rotating it one way sweeps it up and clear, and the other way drives it
    straight through its parent. The clean direction mirrors across the root -- the ``+x`` branch
    folds with negative angles (a positive rotation about ``+y`` tips ``+x`` toward ``-z``), the
    ``-x`` branch with positive ones.
    """
    return _fold_sign(index, root_slat(num_slats))


def chain_ply_gap(spec: ChainSpec) -> float:
    """Height of the returning ply above the one it lands on, at a perfect fold [m].

    **The WALL's width** for a mid-plane hinge, by the two-right-angles construction -- which is the
    pitch for a uniform chain and the bar width for :func:`stacked_chain`. The slab ``thickness`` for
    a surface hinge, where the plies end up face to face.

    This is the rigid chain's analogue of the VBD sheet's measured 2.5 mm ply separation, and the
    number the fold TARGET must be lifted by. ``ClothEnv._init_fold_targets`` lifts by
    ``self_contact_radius`` for exactly this reason, having previously used ``2 * particle_radius``
    and carried ~14 mm of error at a perfect fold. The same mistake is available here.
    """
    if spec.hinge == "surface":
        return spec.thickness
    return spec.widths[spec.wall]


def chain_fold_residual(spec: ChainSpec) -> tuple[float, float]:
    """How far a PERFECT fold of this chain misses the ideal mirror, at the far edge.

    Returns ``(dx, dz)`` [m] for the moving half's far corner: ``dx`` is the in-plane shortfall,
    ``dz`` the ply height. Both are intrinsic to the discretisation, not to the controller.

    ``dx`` is zero whenever the wall of the fold is a slat that belongs to neither ply -- the crease
    slat of an odd uniform chain, or :func:`stacked_chain`'s bar -- and is one slat width otherwise,
    because the wall then has to come out of the moving ply. A surface hinge spends no wall and so
    has ``dx = 0`` whenever its fold seam is at ``x = 0``.

    **This is the number the fold target must be built from**, not an assumed cloth thickness.
    ``ClothEnv`` already made the analogous mistake on the VBD sheet -- lifting targets by
    ``2 * particle_radius`` (16 mm) where cloth-on-cloth contact settles at 2.5 mm, so a perfect
    fold scored 14 mm of error before the policy did anything.
    """
    moving = spec.moving
    if not moving:
        raise ValueError(f"{spec.num_slats} slats leaves no moving half")
    far = moving[-1]
    dx, dz = chain_geom_offset(spec, far)
    corner = (dx + spec.widths[far] / 2.0, spec.size / 2.0, dz)

    rest = chain_point_in_sheet_frame(spec, far, corner)
    folded = chain_point_in_sheet_frame(spec, far, corner, chain_fold_angles(spec))
    # Ideal: reflect across the crease at x = 0, lift onto the stationary half.
    return (folded[0] - (-rest[0]), folded[2] - rest[2])


# ------------------------------------------------------------------- thickness / limit coupling


def _max_bend(a: float, b: float, c: float, thickness: float) -> float:
    """Largest bend [rad], applied at both joints, before slats ``i`` and ``i+2`` interpenetrate.

    ``a, b, c`` are the widths of slats ``i, i+1, i+2``. With both intervening joints at ``phi``,
    the centre-to-centre vector is

        x = a/2 + b cos(phi) + (c/2) cos(2 phi)
        z =       b sin(phi) + (c/2) sin(2 phi)

    and colliders of thickness ``thickness`` touch when its length equals ``thickness``. For the
    parallel-ply geometry that a fold actually reaches, the cylinder condition (centre distance) and
    the box condition (face to face) coincide, so one formula serves both shapes.

    Found by scanning for the FIRST crossing rather than by bisecting from ``pi``. The separation is
    not monotone once the widths differ: for ``[49, 2, 49]`` it dives to 2 mm at 90 degrees and
    climbs back to 47 mm at 180, so a bisection anchored at ``pi`` reports "never touches" for a
    chain that collides squarely in the middle of its range.
    """
    if min(a, b, c) <= 0.0 or thickness <= 0.0:
        raise ValueError(f"widths and thickness must be positive, got {a}, {b}, {c}, {thickness}")

    def sep(phi: float) -> float:
        return math.hypot(
            a / 2.0 + b * math.cos(phi) + (c / 2.0) * math.cos(2.0 * phi),
            b * math.sin(phi) + (c / 2.0) * math.sin(2.0 * phi),
        )

    if sep(0.0) < thickness:
        raise ValueError(
            f"slats of width {a}, {b}, {c} already interpenetrate at rest for thickness {thickness}"
        )
    steps = 4096
    prev = 0.0
    for k in range(1, steps + 1):
        phi = math.pi * k / steps
        if sep(phi) < thickness:
            lo, hi = prev, phi
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if sep(mid) >= thickness:
                    lo = mid
                else:
                    hi = mid
            return lo
        prev = phi
    return math.pi


def max_bend_for_thickness(pitch: float, thickness: float) -> float:
    """Largest per-joint bend [rad] before slats ``i`` and ``i+2`` interpenetrate, uniform chain.

    Adjacent slats never collide -- an articulation filters the parent-child pair -- so the binding
    constraint is one-removed. At ``thickness == pitch`` this is exactly 90 degrees, which is why
    those two defaults are ONE decision rather than two guesses.

    Returns ``pi`` when no bend in ``[0, pi]`` brings them into contact.
    """
    return _max_bend(pitch, pitch, pitch, thickness)


def plate_joint_stiffness(
    youngs_modulus: float,
    fabric_thickness: float,
    width: float,
    pitch: float,
    poisson: float = 0.3,
) -> float:
    """Hinge stiffness [N m / rad] equivalent to a thin plate's bending rigidity.

    ``D = E h^3 / (12 (1 - nu^2))`` is the plate's bending rigidity per unit width; a hinge standing
    in for one ``pitch`` of continuous plate across ``width`` carries ``k = D * width / pitch``.

    **``fabric_thickness`` is the cloth's real thickness, not the collider's.** A mid-plane collider
    is ``pitch`` thick for the contact reasons above, and using that here would overstate the
    stiffness by ``(pitch / h)^3`` -- four orders of magnitude for 0.3 mm sheeting on a 6.25 mm pitch.

    Offered as a starting point, not as an equivalence to ``ClothCfg.edge_ke``. Newton's dihedral
    bending energy is not dimensionally portable to a hinge torque (the cloth cfg already flags that
    its own ``edge_ke`` was tuned on ~mm triangles and applied to a 16.7 mm grid), and the cloth's
    bending response was measured to SATURATE -- 5.0 springs a crease open in 1 step, 0.5 holds it 7,
    0.05 also holds it 7. So the number that matters is whether a crease holds, which is a drape
    measurement, not an elastic-modulus conversion.
    """
    d = youngs_modulus * fabric_thickness**3 / (12.0 * (1.0 - poisson**2))
    return d * width / pitch


# --------------------------------------------------------------------------- uniform front door
#
# The original API, kept verbatim so the env and the tests need not know about ChainSpec. Each of
# these is the general function applied to a uniform spec -- one implementation, two front doors.


def _uniform(
    size: float,
    num_slats: int,
    thickness: float | None = None,
    hinge: str = "mid",
    shape: str = "cylinder",
) -> ChainSpec:
    _check(size, num_slats)
    return uniform_chain(size, num_slats, thickness, hinge, shape)


def forward_kinematics(
    size: float, num_slats: int, joint_angles: Sequence[float] | None = None
) -> list[tuple[float, float, float]]:
    """Pose of every slat's link frame in the sheet frame; see :func:`chain_frames`."""
    return chain_frames(_uniform(size, num_slats), joint_angles)


def point_in_sheet_frame(
    size: float,
    num_slats: int,
    index: int,
    point: tuple[float, float, float],
    joint_angles: Sequence[float] | None = None,
) -> tuple[float, float, float]:
    """See :func:`chain_point_in_sheet_frame`."""
    return chain_point_in_sheet_frame(_uniform(size, num_slats), index, point, joint_angles)


def slat_frame_points(
    size: float,
    num_slats: int,
    index: int,
    num_y: int = 2,
    thickness: float | None = None,
    hinge: str = "mid",
) -> list[tuple[float, float, float]]:
    """See :func:`chain_points`."""
    return chain_points(_uniform(size, num_slats, thickness, hinge), index, num_y)


def corner_points(
    size: float, num_slats: int, thickness: float | None = None, hinge: str = "mid"
) -> list[tuple[int, tuple[float, float, float]]]:
    """See :func:`chain_corner_points`."""
    return chain_corner_points(_uniform(size, num_slats, thickness, hinge))


def folded_joint_angles(
    num_slats: int, joint_limit: float = math.pi / 2.0, hinge: str = "mid"
) -> list[float]:
    """See :func:`chain_fold_angles`. Widths do not affect the ANGLES, only where they land."""
    if hinge not in HINGE_MODES:
        raise ValueError(f"hinge must be one of {HINGE_MODES}, got {hinge!r}")
    _check_slats(num_slats)
    return chain_fold_angles(_uniform(1.0, num_slats, hinge=hinge), joint_limit)


def fold_ply_gap(
    size: float, num_slats: int, thickness: float | None = None, hinge: str = "mid"
) -> float:
    """See :func:`chain_ply_gap`."""
    return chain_ply_gap(_uniform(size, num_slats, thickness, hinge))


def fold_residual(
    size: float, num_slats: int, thickness: float | None = None, hinge: str = "mid"
) -> tuple[float, float]:
    """See :func:`chain_fold_residual`."""
    return chain_fold_residual(_uniform(size, num_slats, thickness, hinge))


def _geom_offset(
    index: int,
    num_slats: int,
    pitch: float,
    thickness: float | None = None,
    hinge: str = "mid",
) -> tuple[float, float]:
    """See :func:`chain_geom_offset`. Kept for the renderers, which work slat by slat."""
    spec = ChainSpec(
        widths=_uniform_widths(pitch * num_slats, num_slats),
        size=pitch * num_slats,
        thickness=pitch if thickness is None else float(thickness),
        hinge=hinge,
    )
    return chain_geom_offset(spec, index)


# ------------------------------------------------------------------------------------- the URDF


def _check_slats(num_slats: int) -> None:
    if num_slats < 2:
        raise ValueError(f"num_slats must be >= 2, got {num_slats}")


def _check(size: float, num_slats: int) -> None:
    if size <= 0.0:
        raise ValueError(f"size must be positive, got {size}")
    _check_slats(num_slats)


def _inertial(
    link: ET.Element,
    mass: float,
    dims: tuple[float, float, float],
    com_x: float,
    com_z: float = 0.0,
) -> None:
    """Thin-plate (solid cuboid) inertia about the slat's own centre.

    The cuboid, not the cylinder that may be drawn: mass is distributed like the piece of cloth the
    slat stands for, and the cylinder is a collision hull whose radius was chosen by the contact
    argument above rather than by where the fabric's mass is.
    """
    lx, ly, lz = dims
    node = ET.SubElement(link, "inertial")
    ET.SubElement(node, "origin", xyz=f"{com_x} 0 {com_z}", rpy="0 0 0")
    ET.SubElement(node, "mass", value=f"{mass:.9g}")
    ET.SubElement(
        node,
        "inertia",
        ixx=f"{mass * (ly * ly + lz * lz) / 12.0:.9g}",
        ixy="0",
        ixz="0",
        iyy=f"{mass * (lx * lx + lz * lz) / 12.0:.9g}",
        iyz="0",
        izz=f"{mass * (lx * lx + ly * ly) / 12.0:.9g}",
    )


def _slat_link(
    robot: ET.Element,
    name: str,
    *,
    width_x: float,
    width: float,
    thickness: float,
    mass: float,
    offset_x: float,
    offset_z: float,
    shape: str,
    rgba: str,
) -> None:
    link = ET.SubElement(robot, "link", name=name)
    for tag in ("visual", "collision"):
        node = ET.SubElement(link, tag)
        geom = ET.SubElement(node, "geometry")
        if shape == "cylinder":
            # URDF cylinders run along their own +z, so roll by 90 degrees to lay it along y --
            # parallel to the hinge axis, which is what makes the chain a smooth roll rather than a
            # stack of wedges.
            ET.SubElement(
                node, "origin", xyz=f"{offset_x} 0 {offset_z}", rpy=f"{math.pi / 2.0:.9g} 0 0"
            )
            ET.SubElement(geom, "cylinder", radius=f"{thickness / 2.0:.9g}", length=f"{width:.9g}")
        else:
            ET.SubElement(node, "origin", xyz=f"{offset_x} 0 {offset_z}", rpy="0 0 0")
            ET.SubElement(geom, "box", size=f"{width_x:.9g} {width:.9g} {thickness:.9g}")
        if tag == "visual":
            material = ET.SubElement(node, "material", name=f"{name}_color")
            ET.SubElement(material, "color", rgba=rgba)
    _inertial(link, mass, (width_x, width, thickness), offset_x, offset_z)


def write_chain_urdf(
    spec: ChainSpec,
    out_path: str | Path,
    *,
    density: float = 2.0,
    joint_limit: float | None = None,
    joint_damping: float = 1.0e-5,
    joint_friction: float = 0.0,
    effort_limit: float = 1.0,
    velocity_limit: float = 50.0,
    color: tuple[float, float, float] = (0.85, 0.35, 0.55),
) -> Path:
    """Write ``spec`` as a URDF.

    Args:
        density: Surface density [kg/m^2], as in ``ClothCfg.density`` -- PER AREA. Total mass is
            ``density * size * span``, split across the slats **in proportion to their widths**, so
            :func:`stacked_chain`'s thin bar is light and its plates are heavy.
        joint_limit: Per-joint bend limit [rad]. ``None`` takes ``spec.joint_limit``.
        joint_damping: Hinge damping [N m s / rad]. The cloth's whole bending response lives here
            and in whatever stiffness the actuator config applies -- URDF has no spring element, so
            **stiffness is not expressible in this file**. Set it on the Isaac Lab actuator
            (``stiffness`` with a zero target), sized with :func:`plate_joint_stiffness`.
        effort_limit: Torque ceiling [N m]. These joints are passive; this caps only an actuator, and
            it is deliberately small so a mis-wired drive cannot flick the sheet across the table.
    """
    if density <= 0.0:
        raise ValueError(f"density must be positive, got {density}")
    if not spec.foldable():
        raise ValueError(
            f"{spec.num_slats} slats cannot fold with a {spec.hinge!r} hinge: a mid-plane fold needs "
            "a wall slat and a returning ply (3+), a surface fold needs 2+"
        )
    limit = spec.joint_limit if joint_limit is None else abs(float(joint_limit))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rgba = f"{color[0]} {color[1]} {color[2]} 1.0"
    root = spec.root
    n = spec.num_slats

    robot = ET.Element("robot", name="rigid_cloth")
    for i in range(n):
        dx, dz = chain_geom_offset(spec, i)
        _slat_link(
            robot,
            slat_link_name(i),
            width_x=spec.widths[i],
            width=spec.size,
            thickness=spec.thickness,
            mass=density * spec.size * spec.widths[i],
            offset_x=dx,
            offset_z=dz,
            shape=spec.shape,
            rgba=rgba,
        )

    # Emitted root-outward on both branches. A URDF is order-independent, but a parser that walks it
    # in document order then sees each parent before its child, which keeps the tree obvious to read.
    def _joint(child: int, parent: int, dx: float) -> None:
        joint = ET.SubElement(robot, "joint", name=slat_joint_name(child), type="revolute")
        ET.SubElement(joint, "parent", link=slat_link_name(parent))
        ET.SubElement(joint, "child", link=slat_link_name(child))
        ET.SubElement(joint, "origin", xyz=f"{dx:.9g} 0 0", rpy="0 0 0")
        ET.SubElement(joint, "axis", xyz="0 1 0")
        # A mid-plane hinge is symmetric. A surface hinge is NOT: the slab hangs a half-thickness
        # below its own axis, so only one sense of rotation sweeps it clear of its parent and the
        # other drives it straight through. The permitted sense mirrors across the root.
        if spec.hinge == "surface":
            lower, upper = (-limit, 0.0) if _fold_sign(child, root) < 0 else (0.0, limit)
        else:
            lower, upper = -limit, limit
        ET.SubElement(
            joint,
            "limit",
            lower=f"{lower:.9g}",
            upper=f"{upper:.9g}",
            effort=f"{effort_limit:.9g}",
            velocity=f"{velocity_limit:.9g}",
        )
        ET.SubElement(
            joint, "dynamics", damping=f"{joint_damping:.9g}", friction=f"{joint_friction:.9g}"
        )

    for i in range(root + 1, n):
        _joint(i, i - 1, spec.widths[i - 1] / 2.0 if i - 1 == root else spec.widths[i - 1])
    for i in range(root - 1, -1, -1):
        _joint(i, i + 1, -(spec.widths[i + 1] / 2.0 if i + 1 == root else spec.widths[i + 1]))

    ET.indent(robot, space="  ")
    ET.ElementTree(robot).write(out_path, encoding="unicode", xml_declaration=True)
    return out_path


def generate_rigid_cloth_urdf(
    out_path: str | Path,
    *,
    size: float = 0.10,
    num_slats: int = DEFAULT_NUM_SLATS,
    thickness: float | None = None,
    density: float = 2.0,
    shape: str = "cylinder",
    hinge: str = "mid",
    joint_limit: float | None = None,
    joint_damping: float = 1.0e-5,
    joint_friction: float = 0.0,
    effort_limit: float = 1.0,
    velocity_limit: float = 50.0,
    color: tuple[float, float, float] = (0.85, 0.35, 0.55),
) -> Path:
    """Write a UNIFORM hinged-slat sheet as a URDF; see :func:`write_chain_urdf`.

    Args:
        size: Side length of the square sheet [m]. Matches ``ClothCfg.size``.
        num_slats: Slats along the fold axis. Parity matters and depends on the hinge mode --
            :func:`default_num_slats` is the function that knows which.
        thickness: Collider thickness [m]. ``None`` means ``pitch``, which for a mid-plane hinge is
            the value that lets the chain bend to 90 degrees with slats ``i`` and ``i+2`` exactly
            touching AND makes the folded plies rest on each other rather than hover.
        shape: ``"cylinder"`` (default) or ``"box"``.
        hinge: ``"mid"`` or ``"surface"``; see the module docstring for the trade.
    """
    _check(size, num_slats)
    if shape not in SHAPES:
        raise ValueError(f"shape must be one of {SHAPES}, got {shape!r}")
    if hinge not in HINGE_MODES:
        raise ValueError(f"hinge must be one of {HINGE_MODES}, got {hinge!r}")
    return write_chain_urdf(
        uniform_chain(size, num_slats, thickness, hinge, shape),
        out_path,
        density=density,
        joint_limit=joint_limit,
        joint_damping=joint_damping,
        joint_friction=joint_friction,
        effort_limit=effort_limit,
        velocity_limit=velocity_limit,
        color=color,
    )


# ------------------------------------------------------------------------------------- variants
#
# Five approximations of the same 100 mm sheet, each isolating one decision. They are meant to be
# compared, not ranked in the abstract: the first three trade discretisation against exactness, the
# last two trade away every shape but "flat" and "folded" in exchange for near-zero cost.

_SLAB = 0.002  # 2 mm, a plausible folded-fabric ply

#: What the VBD sheet this model replaces actually is, from ``ClothCfg`` / ``Cloth.yaml``.
#:
#: Kept here because two of these are traps. **The cloth has two thicknesses**: particles present a
#: radius of 8 mm to rigid bodies and a self-contact radius of 2 mm to each other, and those are
#: independent rest offsets that never meet. A rigid slab has ONE shape, so it satisfies
#: ``ply_gap >= 2 * standoff`` by construction, while the cloth runs ``2.5 mm`` against ``8 mm`` --
#: the opposite order. **No rigid model can match both**; :func:`chain_ply_gap` matches the ply gap,
#: which is what the fold is scored on.
#:
#: And **the two solvers mix friction by different rules** -- Newton's VBD takes the geometric mean
#: of the two materials (``rigid_vbd_kernels.py``: ``mixed_mu = sqrt(friction_mu * shape_mu)``)
#: while MJWarp takes the element-wise maximum (``collision_core.py``: ``wp.max(...)``). A cloth with
#: ``mu = 0.25`` is therefore SLIPPERIER than everything it touches, and a rigid slat with the same
#: number is as grippy as the grippiest thing it touches. Copying the coefficient does not copy the
#: behaviour; see :func:`cloth_effective_friction`.
CLOTH_REFERENCE = {
    "size": 0.10,
    "density": 2.0,  # kg/m^2, per area -- the rigid chain uses the same, so total mass matches
    "resolution": 7,  # 7 x 7 = 49 particles, 72 triangles
    "particle_radius": 0.008,  # cloth-vs-RIGID standoff
    "self_contact_radius": 0.002,  # cloth-vs-CLOTH; measured to settle at 2.5 mm
    "measured_ply_gap": 0.0025,
    "soft_contact_mu": 0.25,
    "shape_mu": {"finger_tip": 1.5, "table": 0.5, "robot": 0.5, "object": 0.5},
}


def cloth_effective_friction(shape_mu: float, cloth_mu: float | None = None) -> float:
    """Friction the VBD cloth actually gets against a shape of coefficient ``shape_mu``.

    Newton's VBD mixes as the **geometric mean**, so the cloth's 0.25 against the 1.5 fingertips is
    ``sqrt(0.25 * 1.5) = 0.61``, not 0.25 and not 1.5.

    This is the number a rigid slat has to reproduce, and under MJWarp's element-wise **maximum** it
    cannot: ``max(anything, 1.5) >= 1.5``. The only contact where the two rules agree is slat on
    slat, where both materials are the cloth's own and ``max(0.25, 0.25) == sqrt(0.25 * 0.25)``.
    """
    mu = CLOTH_REFERENCE["soft_contact_mu"] if cloth_mu is None else float(cloth_mu)
    if mu < 0.0 or shape_mu < 0.0:
        raise ValueError(f"friction coefficients must be non-negative, got {mu}, {shape_mu}")
    return math.sqrt(mu * float(shape_mu))


def cloth_contact_pairs(
    spec: "ChainSpec",
    externals: dict[str, float],
    cloth_mu: float | None = None,
) -> list[tuple[str, str, float]]:
    """Explicit contact pairs that make a rigid chain rub like the cloth it replaces.

    Returns ``[(slat_link, external_geom, friction)]``, one entry per slat per external surface,
    with ``friction = sqrt(cloth_mu * external_mu)`` -- Newton's VBD mixing rule, imposed on a
    solver that would otherwise take the element-wise maximum.

    **Every external surface needs one, not just the fingers.** The maximum rule over-grips against
    anything the sheet can touch, and the error is simply larger where the other surface is grippier:
    against the 1.5 fingertips it gives 1.5 instead of 0.61 (2.5x), against the 0.5 table 0.5
    instead of 0.35 (1.4x). The one contact that needs NO pair is slat on slat, where both materials
    are the cloth's own and ``max(mu, mu) == sqrt(mu * mu)`` exactly.

    ``externals`` maps a geom name to that surface's RAW material coefficient -- ``table_friction``
     0.5, ``finger_tip_friction`` 1.5 and so on from ``Cloth.yaml``'s ``assets`` block -- not to the
    value you want out. The mixing is done here so the call site cannot forget it.

    Neither URDF nor the Isaac Lab asset config can express this, the same gap as joint stiffness
    and friction itself, so it has to be applied to the compiled model. MJWarp does honour it
    (``collision_core.py`` reads ``pair_friction`` in preference to the per-geom maximum), which is
    what makes this route viable at all -- unlike the ``solimp`` knob the surface hinge needs, whose
    reachability is still unverified.

    A cheaper alternative that is NOT this: raising the slats' ``geom_priority`` above everything
    else makes their own coefficient win outright for every contact. One knob instead of ``N x M``
    pairs, but it yields a single number against all surfaces (0.25 everywhere) where the cloth's
    effective value varies with what it touches (0.61 vs fingers, 0.35 vs table). Use it only if
    pair count becomes a problem.
    """
    if not externals:
        raise ValueError("externals must name at least one surface to pair against")
    out: list[tuple[str, str, float]] = []
    for i in range(spec.num_slats):
        for geom, mu in externals.items():
            out.append((slat_link_name(i), geom, cloth_effective_friction(mu, cloth_mu)))
    return out

VARIANTS: dict[str, ChainSpec] = {
    # 1. Odd uniform boxes, mid-plane hinge. The crease slat is the wall, so both plies keep full
    #    length. Thickness = pitch so the folded plies actually touch.
    "box-mid-odd": uniform_chain(
        num_slats=17, hinge="mid", shape="box", label="17 boxes, mid hinge"
    ),
    # 2. Even uniform boxes, surface hinge. Exact fold with a thin slab, at the price of
    #    one-directional hinges.
    "box-surface": uniform_chain(
        num_slats=16, thickness=_SLAB, hinge="surface", shape="box",
        label="16 boxes, surface hinge",
    ),
    # 3. As (1) but cylinders: the same kinematics with a collider that rolls through a bend instead
    #    of wedging at its corners.
    "cyl-mid-odd": uniform_chain(
        num_slats=17, hinge="mid", shape="cylinder", label="17 cylinders, mid hinge"
    ),
    # 4. The floor: two plates, one surface hinge. Nothing but flat and folded.
    "box2-surface": uniform_chain(
        num_slats=2, thickness=_SLAB, hinge="surface", shape="box",
        label="2 boxes, surface hinge",
    ),
    # 5. Two plates with a thin bar between them, mid-plane hinges. Bidirectional, exact fold, ply
    #    gap chosen freely -- the bar's width -- for three bodies.
    "box3-mid": stacked_chain(
        wall_width=_SLAB, thickness=_SLAB, label="2 plates + crease bar, mid hinge"
    ),
}


def variant(name: str) -> ChainSpec:
    if name not in VARIANTS:
        raise ValueError(f"unknown variant {name!r}; known: {sorted(VARIANTS)}")
    return VARIANTS[name]
