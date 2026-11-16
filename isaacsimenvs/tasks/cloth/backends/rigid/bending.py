"""Convert sheet bending coefficients into geometry-dependent hinge coefficients.

For sheet rigidity B [N m] and bending viscosity C [N m s], each hinge uses
k = B ℓ/s and c = C ℓ/s. Here ℓ is the crease length and s is the distance between
neighboring plate centroids measured perpendicular to the crease in the sheet
plane. These are lumped finite-volume coefficients, not separate hinge materials.
Collision thickness does not change this in-plane discretization.
"""
from __future__ import annotations

import numpy as np

from .designs import SheetSpec


def hinge_geometry_factors(spec: SheetSpec) -> np.ndarray:
    """Return dimensionless ℓ/s in ``spec.hinges`` order for arbitrary convex plates.

    The dual-cell span is the sum of the two COM distances to the crease, projected
    into the sheet plane. Diagonal creases and unequal plate widths therefore use
    their actual scaled geometry. One B and one C per cloth multiply these factors;
    sampling independent per-hinge coefficients would break material consistency.
    """
    centers = {plate.name: np.asarray(plate.center, dtype=float) for plate in spec.plates}
    result = []
    for hinge in spec.hinges:
        if hinge.edge is None:
            raise ValueError('bending geometry requires the physical crease endpoints')
        edge = np.asarray(hinge.edge, dtype=float)
        if edge.shape != (2, 3) or not np.isfinite(edge).all():
            raise ValueError('crease endpoints must be a finite (2, 3) array')
        direction = edge[1]-edge[0]
        length = np.linalg.norm(direction)
        normal = np.array([-direction[1], direction[0], 0.])
        transverse_length = np.linalg.norm(normal)
        if length <= 0 or transverse_length <= 0:
            raise ValueError('crease must have nonzero length in the sheet plane')
        normal /= transverse_length
        parent, child = centers[hinge.parent], centers[hinge.child]
        span = abs((parent-edge[0]) @ normal) + abs((child-edge[0]) @ normal)
        if not np.isfinite(span) or span <= 0:
            raise ValueError('adjacent plate centroids must have a positive transverse span')
        result.append(float(length/span))
    return np.asarray(result, dtype=float)
