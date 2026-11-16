"""Physical hinge inertia of a locked child subtree in the cloth's flat rest pose.

This reference has units kg m². A fraction of it can be used as numerical joint
armature; armature is added inertia for stabilization, not a cloth material property
or the articulation's configuration-dependent effective inertia during motion.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .designs import SheetSpec, polygon_mass_properties


def hinge_inertias(spec: SheetSpec, density: float) -> np.ndarray:
    """Return one exact locked-subtree inertia per hinge, in ``spec.hinges`` order.

    ``density`` is areal density [kg/m²]. Every plate has the same rest body frame.
    Reconstruct each full COM inertia tensor from its principal axes, project onto
    the unit hinge axis, then add the parallel-axis term for all child descendants.
    Hinge axes may be diagonal and plates need not be rectangular.
    """
    properties = {}
    children = {plate.name: [] for plate in spec.plates}
    if len(children) != len(spec.plates):
        raise ValueError('plate names must be unique')
    for plate in spec.plates:
        mass, center, diagonal, quaternion = polygon_mass_properties(plate, density)
        rotation = Rotation.from_quat(np.asarray(quaternion)[[1, 2, 3, 0]]).as_matrix()
        properties[plate.name] = mass, center, (rotation * diagonal) @ rotation.T
    parents = set()
    for hinge in spec.hinges:
        if hinge.parent not in children or hinge.child not in children:
            raise ValueError('hinge references an unknown plate')
        if hinge.child in parents:
            raise ValueError('a child plate may have only one parent hinge')
        parents.add(hinge.child)
        children[hinge.parent].append(hinge.child)

    result = []
    for hinge in spec.hinges:
        axis, anchor = np.asarray(hinge.axis, dtype=float), np.asarray(hinge.anchor, dtype=float)
        norm = np.linalg.norm(axis)
        if not np.isfinite(norm) or norm <= 0 or not np.isfinite(anchor).all():
            raise ValueError('hinge axis and anchor must be finite, with a nonzero axis')
        axis = axis / norm
        stack, visited, inertia = [hinge.child], set(), 0.
        while stack:
            name = stack.pop()
            if name in visited:
                raise ValueError('hinge graph contains a cycle')
            visited.add(name)
            mass, center, tensor = properties[name]
            distance = np.cross(axis, center-anchor)
            inertia += float(axis @ tensor @ axis + mass * (distance @ distance))
            stack.extend(children[name])
        result.append(inertia)
    return np.asarray(result, dtype=float)
