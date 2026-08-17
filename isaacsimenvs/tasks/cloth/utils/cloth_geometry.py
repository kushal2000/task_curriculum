"""Grid mesh authoring and the fold definition.

Kept apart from the env so both are testable without a simulator: the fold targets are pure
geometry, and getting them wrong would look exactly like a policy that cannot fold.
"""

from __future__ import annotations

__all__ = ["grid_mesh", "half_indices", "keypoint_indices", "folded_targets"]


def grid_mesh(size: float, resolution: int) -> tuple[list, list]:
    """Vertices and triangle indices for a flat square sheet in the local XY plane.

    Vertices are row-major in ``x`` then ``y``, centred on the origin, at ``z = 0``. Row-major
    order is what makes :func:`half_indices` and :func:`folded_targets` simple index arithmetic
    rather than a spatial search.

    Returns:
        ``(vertices, indices)`` -- ``resolution**2`` points and ``2*(resolution-1)**2`` triangles
        as a flat index list, the form ``add_cloth_mesh`` expects.
    """
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
            # Consistent winding: a degenerate or inconsistently wound triangle gives the solver a
            # zero or inverted normal, which shows up as a sheet that self-intersects at rest.
            idx += [a, c, b, b, c, d]
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
