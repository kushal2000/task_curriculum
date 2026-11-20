"""ClothCfg -- the sheet the fold task is defined on: its geometry and the fold.

The ``cloth`` block of ``env.ClothEnvCfg``, for both backends: the rigid plate pool stands in for
this sheet (same nominal size, the same thickness to the hand, the same fold and keypoints), so a
policy sees one task on either simulator. The VBD backend's sheet adds its material, contact and
coupling (``vbd_cloth.VbdClothCfg``). Values: ``config/task.yaml``. It imports nothing but
``configclass``, so both stacks read the same definition of the task.
"""

from __future__ import annotations

from dataclasses import MISSING

# The function, not the package attribute: under Isaac Lab 3.0 `isaaclab.utils.configclass` can be
# shadowed by the same-named submodule (see `newton/compat.py`), and this module must import
# before the compat shim has run.
from isaaclab.utils.configclass import configclass

__all__ = ["ClothCfg"]


@configclass
class ClothCfg:
    """The sheet's geometry and the fold definition, shared by both backends."""

    # --- geometry -------------------------------------------------------------------------
    size: float = MISSING
    """Side length of the square sheet [m]."""

    thickness: float = MISSING
    """Cloth thickness [m], as ``2 * particle_radius``. On the rigid pool: the thickness the hand
    meets at the nominal plate (slab + 2 x contact margin), observed in ``object_scales``.

    **A VBD cloth is a surface: thickness IS the particle collision radius**, there is no
    volumetric dimension. That makes it bounded by grid spacing -- particles closer together than
    their diameter self-collide at rest and the sheet crumples before anything touches it. The
    constraint is ``thickness <= size / (resolution - 1)``, checked in ``VbdClothCfg``.

    Thicker is genuinely better for *contact*: the cable was ungraspable at 10 mm and liftable at
    30 mm, and thin geometry tunnels through fingers. But it is worse for *folding*, which needs
    bending. And 30 mm on a 100 mm sheet forces a 4x4 grid (spacing 33 mm), which is a stiff mat
    rather than cloth -- genuine 30 mm material is a volumetric soft body (``add_soft_mesh``, a
    tet mesh), not a cloth.

    **16 mm, adopted from SIM1 Table 2 (particle radius 0.008 m).** This is NOT a measurement on
    this task -- it is their calibrated value, taken wholesale along with the rest of their cloth
    material. It forces ``resolution=7``: spacing must be >= thickness, and 16 mm needs
    ``size / thickness + 1 = 7`` (spacing 16.7 mm). ``max_resolution_for_thickness`` returns
    exactly 7, so this pair sits on the guard boundary and any finer grid will raise.

    The previous value was 12 mm at ``resolution=9`` (spacing 12.5 mm), chosen as "thick enough for
    the hand to feel, thin enough to fold". That reasoning still stands and nothing measured
    displaced it; it was replaced to match SIM1, not to improve on it. If the fold regresses, this
    and ``resolution`` are the first pair to revert."""

    resolution: int = MISSING
    """Particles per side, so the sheet is ``resolution**2`` particles and
    ``2*(resolution-1)**2`` triangles. 7 gives 49 particles / 72 triangles. The rigid pool reads
    the same grid off its plates (keypoints, fold targets).

    **Driven by ``thickness``, not chosen independently.** 16 mm particles cannot sit on a grid
    finer than 16.7 mm spacing without overlapping at rest, so matching SIM1's particle radius
    forces this down from 9 (81 particles / 128 triangles). The fold geometry survives the change
    unchanged -- ``corner_indices`` still returns the four far-half corners at the same
    coordinates, ``folded_targets`` is identical, and the odd resolution keeps a hinge row -- but
    the sheet is meaningfully coarser, and a 72-triangle sheet is close to the floor at which
    "cloth" stops being a useful description."""

    start_height: float = MISSING
    """Spawn height above ``reset.table_reset_z`` [m].

    **The table's collision surface is at ``table_reset_z + 0.150``, not + 0.045.** See
    ``table_half_thickness``. Everything below is measured with zero actions, so the robot never
    approaches the sheet and anything it does is spawn behaviour:

        0.150  -> spawn 0.5300, surface 0.5298: in contact. EJECTED upward at ~1.7 m/s, peaks at
                  0.68 by step 10, falls back, settles by step 30 with a 17 mm spread across a
                  12 mm sheet -- i.e. visibly wrinkled.
        0.300  -> spawn 0.6800, a genuine 15 cm drop. Falls at exactly g (predicted 0.0340 m over
                  the first 5 steps, measured 0.0349), lands at 0.5360 and settles PERFECTLY flat
                  (min = mean = max to four decimals).

    An earlier version of this docstring reported 0.150 as "settles dead flat", which was wrong: it
    was read after settling and never checked the spread. Every episode was therefore starting with
    the sheet thrown into the air and landing creased, and that was mis-attributed in turn to
    thickness, to grid resolution, and to cloth physics.

    Values below the surface are worse still, and were correctly diagnosed at the time: at 0.010 the
    sheet starts inside the table and reaches 51 m by step 80."""

    table_half_thickness: float = MISSING
    """Distance from ``reset.table_reset_z`` up to the table's COLLISION surface [m].

    **Measured, not read off the USD.** The visual box is 0.045 half-thick, and that number sat here
    for a long time -- but the sheet settles with its lowest particles at 0.5360 against a table root
    of 0.3800, putting the contact surface at 0.5298, i.e. 0.1498 above the root. Confirmed twice
    from different drop heights, and `Table/box` is the only non-robot collider in the scene, so
    that is what the cloth is resting on.

    Getting it wrong is not a cosmetic error. At 0.045 the guard below accepted
    ``start_height = 0.15``, which looked like a 10 cm drop and actually spawned the sheet 0.2 mm
    above the surface -- effectively in contact. Contact resolution then ejected it upward at
    ~1.7 m/s at the start of EVERY episode: it flew to 0.68, fell back, and settled visibly
    wrinkled (17 mm spread across a 12 mm sheet). With a real drop it falls at exactly g and lands
    perfectly flat (spread 0.0000). The crumpled spawn had been attributed to thickness, to grid
    resolution, and to cloth physics; it was this."""

    # --- the fold -------------------------------------------------------------------------
    fold_axis: str = MISSING
    """Axis normal to the fold line, in cloth-local coordinates: the ``+`` half folds onto ``-``."""

    num_keypoints: int = MISSING
    """Tracked keypoints on the moving half. Four matches the task's existing keypoint machinery
    and is the minimum that pins a folded flap's position *and* orientation."""

    max_folded_footprint: float = MISSING
    """Largest sheet extent along the fold axis, as a fraction of ``size``, that still counts as
    folded.

    A second condition on success, independent of keypoint distance. A perfect fold gives ~0.5; an
    unfolded sheet gives 1.0. Keypoint proximity alone proved insufficient: a rigid slide scored
    "folds" because the targets were world-fixed, and after that was corrected a *crumpled* sheet
    still scored transiently by putting four keypoints near their targets. 0.65 leaves room for a
    real fold's drape while excluding anything close to flat."""

    keypoint_tolerance: float = MISSING
    """Per-keypoint distance for success [m]. Deliberately looser than the cable's 1.5 cm: a
    folded flap settles wherever the drape puts it, and demanding rigid-body precision from a
    sheet would measure the tolerance rather than the fold."""

    nan_policy: str = "raise"
    """What to do when any observed quantity goes non-finite: ``"raise"`` or ``"reset"``.

    ``raise`` is right for probes and evaluation -- a NaN there is a bug to diagnose, and the guard
    names which quantity went bad first (it is how the intermittent failure was localised to
    ``robot_joint_pos`` rather than to the cloth).

    ``reset`` is right for training. The offending envs have their observation zeroed so nothing
    non-finite can reach the network or the gradient, and are terminated so they reset through the
    normal path. Over a multi-hour run on 4 GPUs a single diverged env would otherwise kill
    everything; the failure has been seen roughly once per three 900-step evals."""

    def check(self) -> None:
        """Reject a sheet that would spawn inside the table: it presents as violent instability rather
        than as a configuration error, which is exactly how it was first diagnosed here -- as a
        solver problem."""
        if self.start_height <= self.table_half_thickness:
            raise ValueError(
                f"cloth start_height {self.start_height:.3f} m does not clear the table: "
                f"`reset.table_reset_z` is the table CENTRE and its surface is "
                f"{self.table_half_thickness:.3f} m higher. The sheet would spawn inside the "
                f"table and be ejected in one step. Use start_height > "
                f"{self.table_half_thickness:.3f}."
            )

