"""Grid mesh authoring and the fold definition.

Kept apart from the env so both are testable without a simulator: the fold targets are pure
geometry, and getting them wrong would look exactly like a policy that cannot fold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "grid_mesh",
    "irregular_mesh",
    "sub_mesh",
    "SheetMesh",
    "sheet_mesh",
    "half_indices",
    "half_mesh",
    "keypoint_indices",
    "folded_targets",
]


def grid_mesh(size: float, resolution: int, diagonals: str = "alternate") -> tuple[list, list]:
    """Vertices and triangle indices for a flat square sheet in the local XY plane.

    Vertices are row-major in ``x`` then ``y``, centred on the origin, at ``z = 0``. Row-major
    order is what makes :func:`half_indices` and :func:`folded_targets` simple index arithmetic
    rather than a spatial search.

    Returns:
        ``(vertices, indices)`` -- ``resolution**2`` points and ``2*(resolution-1)**2`` triangles
        as a flat index list, the form ``add_cloth_mesh`` expects.

    ``diagonals`` picks how each cell is cut into two triangles:

    * ``alternate`` (default): the two cuts in a checkerboard, so the mesh has no handedness.
    * ``uniform``: every cell cut (i, j+1)-(i+1, j), the mesh before 2026-11-09.
    * ``flipped``: every cell cut (i, j)-(i+1, j+1), the mirror image of ``uniform``.

    A uniform cut couples bending to twist: a released fold springs back as a twist with one end of
    the crease lifted, and flipping the cut mirrors it exactly (debug_outputs/train_logs/results/cloth_fold_lift,
    {base,flip,alt}: denim skew +7.8 / -7.8 / 0.0 mm, bulge 11.2 / 11.2 / 5.9 mm). The checkerboard
    leaves the cantilever droop within 2 deg of the uniform mesh.
    """
    if diagonals not in ("alternate", "uniform", "flipped"):
        raise ValueError(f"diagonals must be alternate, uniform or flipped, got {diagonals!r}")
    if resolution < 2:
        raise ValueError(f"resolution must be >= 2, got {resolution}")

    step = size / (resolution - 1)
    half = size / 2.0
    verts = [
        (i * step - half, j * step - half, 0.0)
        for i in range(resolution)
        for j in range(resolution)
    ]

    idx: list[int] = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            a = i * resolution + j
            b = a + 1
            c = (i + 1) * resolution + j
            d = c + 1
            # Consistent winding (+z normals either way): a degenerate or inconsistently wound
            # triangle gives the solver a zero or inverted normal, which shows up as a sheet that
            # self-intersects at rest.
            flip = diagonals == "flipped" or (diagonals == "alternate" and (i + j) % 2 == 1)
            idx += [a, c, d, a, d, b] if flip else [a, c, b, b, c, d]
    return verts, idx


def half_indices(resolution: int, axis: str = "x", positive: bool = True) -> list[int]:
    """Vertex indices on one side of the centre line.

    The centre row/column itself is excluded: those particles are the hinge and belong to neither
    half, so tracking them would report the fold as half-complete before anything moved.
    """
    if axis not in ("x", "y"):
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
    mid = (resolution - 1) / 2.0
    out = []
    for i in range(resolution):
        for j in range(resolution):
            coord = i if axis == "x" else j
            if (coord > mid) if positive else (coord < mid):
                out.append(i * resolution + j)
    return out


def corner_indices(resolution: int, axis: str = "x") -> list[int]:
    """The four corners of the MOVING half, ordered crease-near then crease-far.

    Corners rather than points along one edge: four points spanning an area determine a rotation,
    whereas four collinear points on the far edge are degenerate about their own axis and cannot.
    That degeneracy is why the orientation was previously reported as identity.

    Order is fixed -- (near-left, near-right, far-left, far-right) -- so a rotation fit against the
    rest corners compares like with like.
    """
    if axis not in ("x", "y"):
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
    # `resolution // 2` is the HINGE row for odd resolutions, not the first row past it:
    # `half_indices` splits on `mid = (resolution - 1) / 2`, so for resolution 7 the halves are
    # rows 0-2 and 4-6 and row 3 belongs to NEITHER. Picking row 3 put two of the four tracked
    # keypoints ON the fold axis, where they barely move during a fold while the target rule
    # (reflect + lift) still asks them to rise one lift. Measured consequences at resolution 7,
    # thickness 16 mm:
    #   * a geometrically perfect fold scored fold_error 0.016 m, not 0 -- 40% of the 0.04
    #     tolerance spent before the policy does anything;
    #   * the corner set is not rigid under a fold (hinge->far distance 50.00 -> 52.50 mm), so the
    #     Kabsch fit returned 162.3 deg instead of 180 and `object_rot` never reads a completed
    #     fold.
    # With every keypoint on the moving half both go away exactly: the set is rigid, the fit is
    # 180.0 deg, and a perfect fold scores 0.
    mid = resolution // 2 + (resolution % 2)   # first row/column strictly past the crease
    far = resolution - 1
    lo, hi = 0, resolution - 1

    def vid(i: int, j: int) -> int:
        return i * resolution + j

    if axis == "x":
        return [vid(mid, lo), vid(mid, hi), vid(far, lo), vid(far, hi)]
    return [vid(lo, mid), vid(hi, mid), vid(lo, far), vid(hi, far)]


def half_mesh(
    size: float, resolution: int, axis: str = "x"
) -> tuple[list, list]:
    """The MOVING half as its own flat mesh, centred on its own centroid.

    For drawing the goal: the fold target is a rigid transform of the moving half (a reflection
    about the crease, which for a rigid body is a 180-degree rotation), so a rigid mesh of the
    half's rest shape, posed at the folded pose, is exactly the configuration being scored.

    A box primitive is the obvious shortcut and it is wrong twice over -- it reads as a solid brick
    rather than a sheet, and it has a thickness the cloth does not have (a VBD cloth is a *surface*;
    its thickness is a collision radius the renderer never draws). Reusing the sheet's own grid and
    triangulation means the marker looks like what it is standing in for.

    Returns:
        ``(vertices, indices)`` in the half's local frame, ``z = 0``.
    """
    verts, _ = grid_mesh(size, resolution)
    keep = sorted(set(half_indices(resolution, axis, positive=True)))
    remap = {v: i for i, v in enumerate(keep)}

    # Re-centre on the half's own centroid: the marker's pose is the folded *centroid*, so the mesh
    # must be expressed about its centroid or it draws offset by half its own width.
    pts = [list(verts[v]) for v in keep]
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    out_verts = [(p[0] - cx, p[1] - cy, 0.0) for p in pts]

    # Keep only triangles fully inside the half, so the mesh has no edges dangling across the
    # crease into vertices that are not part of it.
    _, full_idx = grid_mesh(size, resolution)
    out_idx: list[int] = []
    for t in range(0, len(full_idx), 3):
        tri = full_idx[t : t + 3]
        if all(v in remap for v in tri):
            out_idx += [remap[v] for v in tri]
    return out_verts, out_idx


def keypoint_indices(resolution: int, axis: str = "x", count: int = 4) -> list[int]:
    """``count`` vertex indices spread over the moving half.

    Picks the far edge's corners first, then fills along that edge. The far edge is what has to
    travel furthest in a fold, so it is both the most informative to track and the hardest to
    satisfy -- keypoints near the hinge would report success for a fold that never lifted.
    """
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    far = resolution - 1  # index of the far row/column, the edge that travels

    def vid(i: int, j: int) -> int:
        return i * resolution + j

    # Spread `count` samples across the far edge, endpoints included.
    picks = []
    for k in range(count):
        t = 0.0 if count == 1 else k / (count - 1)
        pos = int(round(t * (resolution - 1)))
        picks.append(vid(far, pos) if axis == "x" else vid(pos, far))
    return picks


def folded_targets(
    verts: list, indices: list[int], axis: str = "x", lift: float = 0.0
) -> list[tuple[float, float, float]]:
    """Where the given vertices end up once the moving half is folded over.

    A fold about the centre line is a reflection: the coordinate normal to the line negates, the
    others are unchanged. ``lift`` raises the target by the sheet's own thickness, since the folded
    flap rests *on top of* the stationary half rather than inside it.

    These are cloth-local coordinates; the env adds the world spawn transform.
    """
    ax = 0 if axis == "x" else 1
    out = []
    for i in indices:
        v = list(verts[i])
        v[ax] = -v[ax]
        v[2] += lift
        out.append(tuple(v))
    return out


# --------------------------------------------------------------------------------------------
# Irregular sheet: no mesh line the fold has to follow
# --------------------------------------------------------------------------------------------


def irregular_mesh(
    size: float,
    edge_length: float,
    seed: int = 0,
    jitter: float = 0.3,
    rotate: bool = True,
) -> tuple[list, list]:
    """A square sheet triangulated without a preferred fold line.

    A regular grid can only bend along its own rows, columns and diagonals, and with an odd
    resolution one of those rows lies exactly on the midline -- so every fold lands there whatever
    the hand does. Real cloth folds about equally easily along any line near the centre. This mesh
    gets as close to that as a triangle mesh can:

    * interior points on a near-equilateral (hexagonal) lattice, so every edge is about
      ``edge_length`` and no direction is short-cut by a diagonal,
    * the lattice turned by a random angle (``rotate``), so its three edge directions are not
      aligned with the sheet's sides or the fold axis,
    * each interior point jittered by up to ``jitter * edge_length``, which breaks the remaining
      straight lines of vertices into zig-zags,
    * Delaunay triangulation of those points plus evenly spaced boundary points.

    The four corners are always vertices (the fold keypoints are corners). Deterministic for a
    given ``seed``; one mesh is shared by every env.

    Returns ``(vertices, indices)`` in the same form as :func:`grid_mesh`: centred, ``z = 0``,
    flat index list, every triangle wound with a +z normal.
    """
    import numpy as np
    from scipy.spatial import Delaunay

    if edge_length <= 0.0 or edge_length >= size:
        raise ValueError(f"edge_length must be in (0, size), got {edge_length} for size {size}")
    rng = np.random.default_rng(seed)
    half = 0.5 * size
    n = max(2, int(round(size / edge_length)))  # boundary segments per side
    h = size / n

    # Boundary: evenly spaced, corners included, counter-clockwise without repeats.
    t = np.linspace(-half, half, n + 1)
    bottom = np.stack([t, np.full_like(t, -half)], 1)
    right = np.stack([np.full_like(t, half), t], 1)[1:]
    top = np.stack([t[::-1], np.full_like(t, half)], 1)[1:]
    left = np.stack([np.full_like(t, -half), t[::-1]], 1)[1:-1]
    boundary = np.concatenate([bottom, right, top, left])

    # Interior: hexagonal lattice, rotated and jittered, kept 0.6 h clear of the boundary.
    ang = float(rng.uniform(0.0, math.pi / 3.0)) if rotate else 0.0
    dy = h * math.sqrt(3.0) / 2.0
    m = int(math.ceil(size / min(h, dy))) + 2
    ii, jj = np.meshgrid(np.arange(-m, m + 1), np.arange(-m, m + 1), indexing="ij")
    lat = np.stack([(ii + 0.5 * (jj % 2)) * h, jj * dy], -1).reshape(-1, 2)
    c, s = math.cos(ang), math.sin(ang)
    lat = lat @ np.array([[c, s], [-s, c]])
    if jitter > 0.0:
        r = jitter * h * np.sqrt(rng.uniform(0.0, 1.0, len(lat)))
        phi = rng.uniform(0.0, 2.0 * math.pi, len(lat))
        lat = lat + np.stack([r * np.cos(phi), r * np.sin(phi)], 1)
    lim = half - 0.6 * h
    interior = lat[(np.abs(lat[:, 0]) <= lim) & (np.abs(lat[:, 1]) <= lim)]

    pts = np.concatenate([boundary, interior])
    # Bulge the boundary out by a hair so Qhull sees a strictly convex hull and keeps every
    # collinear side point as a vertex (it would otherwise drop them as "coplanar").
    bulge = pts.copy()
    nb = len(boundary)
    bulge[:nb] *= 1.0 + 1e-7 * (1.0 + rng.uniform(0.0, 1.0, (nb, 1)))
    tri = Delaunay(bulge).simplices

    # Wind every triangle +z and drop slivers.
    a, b, cc = pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]]
    cross = (b[:, 0] - a[:, 0]) * (cc[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (cc[:, 0] - a[:, 0])
    tri = np.where((cross < 0)[:, None], tri[:, [0, 2, 1]], tri)
    tri = tri[np.abs(cross) > 1e-6 * h * h]
    used = np.unique(tri)
    if len(used) != len(pts):
        raise RuntimeError(f"irregular_mesh left {len(pts) - len(used)} vertices out of the triangulation")

    verts = [(float(x), float(y), 0.0) for x, y in pts]
    return verts, [int(i) for i in tri.reshape(-1)]


def sub_mesh(verts: list, indices: list[int], keep: list[int]) -> tuple[list, list]:
    """The triangles of ``(verts, indices)`` whose three vertices are all in ``keep``.

    Vertices come back in ``sorted(keep)`` order, re-centred on their own XY centroid (the goal
    marker is posed at the folded centroid), triangles re-indexed into that order.
    """
    keep = sorted(set(keep))
    remap = {v: i for i, v in enumerate(keep)}
    pts = [list(verts[v]) for v in keep]
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    out_verts = [(p[0] - cx, p[1] - cy, 0.0) for p in pts]
    out_idx: list[int] = []
    for t in range(0, len(indices), 3):
        tri = indices[t : t + 3]
        if all(v in remap for v in tri):
            out_idx += [remap[v] for v in tri]
    return out_verts, out_idx


@dataclass(frozen=True)
class SheetMesh:
    """One sheet's mesh and the fold bookkeeping derived from it.

    The env and task read the sheet only through this, so a grid and an irregular mesh are
    interchangeable: the policy sees four corner keypoints either way.
    """

    verts: list
    indices: list
    size: float
    axis: str
    kind: str
    edge_length: float
    #: Moving-half corners, (near-left, near-right, far-left, far-right) as :func:`corner_indices`.
    corner_idx: list = field(default_factory=list)
    #: Vertices strictly on the + (moving) and - (stationary) side of the crease band.
    moving_idx: list = field(default_factory=list)
    stationary_idx: list = field(default_factory=list)

    @property
    def num_particles(self) -> int:
        return len(self.verts)

    @property
    def num_triangles(self) -> int:
        return len(self.indices) // 3

    def half_mesh(self) -> tuple[list, list]:
        """The moving half as its own mesh, vertices in ``moving_idx`` order (see :func:`half_mesh`)."""
        return sub_mesh(self.verts, self.indices, self.moving_idx)


def sheet_mesh(
    size: float,
    axis: str = "x",
    kind: str = "grid",
    resolution: int = 7,
    edge_length: float = 0.0,
    seed: int = 0,
) -> SheetMesh:
    """Build the sheet: ``kind="grid"`` is the legacy :func:`grid_mesh` (``resolution`` per side),
    ``kind="irregular"`` is :func:`irregular_mesh` at ``edge_length``.

    For the irregular mesh the halves exclude a band of half an edge around the midline (the grid
    excludes its hinge row the same way), and the near corners are the first boundary vertices past
    that band on the two side edges.
    """
    if axis not in ("x", "y"):
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
    if kind == "grid":
        verts, idx = grid_mesh(size, resolution)
        return SheetMesh(
            verts=verts, indices=idx, size=size, axis=axis, kind=kind,
            edge_length=size / (resolution - 1),
            corner_idx=corner_indices(resolution, axis),
            moving_idx=sorted(set(half_indices(resolution, axis, positive=True))),
            stationary_idx=sorted(set(half_indices(resolution, axis, positive=False))),
        )
    if kind != "irregular":
        raise ValueError(f"mesh kind must be grid or irregular, got {kind!r}")
    if edge_length <= 0.0:
        raise ValueError("an irregular sheet needs edge_length > 0")
    verts, idx = irregular_mesh(size, edge_length, seed=seed)
    ax, lat = (0, 1) if axis == "x" else (1, 0)
    half = 0.5 * size
    band = 0.25 * edge_length
    eps = 1e-9 * size
    moving = [i for i, v in enumerate(verts) if v[ax] > band]
    stationary = [i for i, v in enumerate(verts) if v[ax] < -band]

    def on_side(i: int, sign: float) -> bool:
        return abs(verts[i][lat] - sign * half) < eps

    def near(sign: float) -> int:
        cand = [i for i in moving if on_side(i, sign)]
        return min(cand, key=lambda i: verts[i][ax])

    def far(sign: float) -> int:
        cand = [i for i in moving if on_side(i, sign) and abs(verts[i][ax] - half) < eps]
        if len(cand) != 1:
            raise RuntimeError("irregular sheet lost a corner vertex")
        return cand[0]

    corners = [near(-1.0), near(1.0), far(-1.0), far(1.0)]
    return SheetMesh(
        verts=verts, indices=idx, size=size, axis=axis, kind=kind, edge_length=edge_length,
        corner_idx=corners, moving_idx=moving, stationary_idx=stationary,
    )
