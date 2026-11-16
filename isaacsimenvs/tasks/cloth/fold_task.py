"""The fold task, shared by the rigid-body env (PhysX) and the deformable env (Newton VBD).

Simulator-free: nothing here starts Kit or imports a physics backend at module level (the scene
helpers import ``pxr`` / ``isaaclab.sim`` inside the functions that need them), so both envs and the
fast tests load it. Sections:

* ``ClothCfg`` -- the task block of both env configs (sheet grid, fold axis, success test, and the
  VBD material/contact parameters the rigid env ignores);
* sheet geometry -- the particle grid, its halves, the four moving-half corners, the fold targets;
* rigid sheet designs -- the 4096-design size grid and the single-hinge plates + hinge;
* ``ClothAsRigidObject`` -- the Play task's rigid-object surface over a sheet's points;
* ``ClothTaskMixin`` -- targets, reward, success, dones, goal marker: the task itself.
"""

from __future__ import annotations

import contextlib
import math
import sys
from dataclasses import dataclass, field

import numpy as np
import torch
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_from_matrix

__all__ = ['Box', 'ClothAsRigidObject', 'ClothCfg', 'ClothTaskMixin', 'Hinge', 'SheetSpec', 'box_mass_properties', 'corner_indices', 'design_grid', 'design_of', 'fold_rest', 'folded_targets', 'grid_mesh', 'half_indices', 'half_mesh', 'hinge_transform', 'keypoint_indices', 'plate_points', 'single_hinge', 'task_yaw']


# ==================================================================================================
# ClothCfg -- ClothCfg -- the cloth task's own block: sheet geometry, material, the fold, and the coupling.
# (was cloth_cfg.py)
# ==================================================================================================

# The function, not the package attribute: under Isaac Lab 3.0 `isaaclab.utils.configclass` can be
# shadowed by the same-named submodule (see `newton/compat.py`), and this module must import
# before the compat shim has run.



@configclass
class ClothCfg:
    """Cloth geometry, material, the fold definition, and the coupling that lets the hand feel it."""

    # --- geometry -------------------------------------------------------------------------
    size: float = 0.10
    """Side length of the square sheet [m]."""

    thickness: float = 0.016
    """Cloth thickness [m], as ``2 * particle_radius``.

    **A VBD cloth is a surface: thickness IS the particle collision radius**, there is no
    volumetric dimension. That makes it bounded by grid spacing -- particles closer together than
    their diameter self-collide at rest and the sheet crumples before anything touches it. The
    constraint is ``thickness <= size / (resolution - 1)``, checked in ``__post_init__``.

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

    resolution: int = 7
    """Particles per side, so the sheet is ``resolution**2`` particles and
    ``2*(resolution-1)**2`` triangles. 7 gives 49 particles / 72 triangles.

    **Driven by ``thickness``, not chosen independently.** 16 mm particles cannot sit on a grid
    finer than 16.7 mm spacing without overlapping at rest, so matching SIM1's particle radius
    forces this down from 9 (81 particles / 128 triangles). The fold geometry survives the change
    unchanged -- ``corner_indices`` still returns the four far-half corners at the same
    coordinates, ``folded_targets`` is identical, and the odd resolution keeps a hinge row -- but
    the sheet is meaningfully coarser, and a 72-triangle sheet is close to the floor at which
    "cloth" stops being a useful description."""

    reset_offset_xy: list[float] = [0.0, 0.0]
    """Env-frame shift of the sheet's reset centre [m], added to every reset (pinned or random).

    For sheets larger than the 10 cm default: the arm reaches only ~18 cm past the table centre on
    the far side, so a big sheet is moved toward the robot (+y), as the fold pool does with its
    ``reach_shift`` -- by the growth of the half-diagonal, ``(size - 0.10) * sqrt(2) / 2``, which
    keeps the farthest point at any yaw where a 10 cm sheet's would be (0.0707 m for 20 cm)."""

    start_height: float = 0.30
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

    table_half_thickness: float = 0.150
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

    # --- material -------------------------------------------------------------------------
    density: float = 2.0
    """Surface density [kg/m^2]. **Per AREA, not per volume.**

    This reaches ``ModelBuilder.add_cloth_mesh(density=...)`` unchanged (via the ``newton:density``
    USD attribute, read in ``isaaclab_contrib.deformable.deformable_object``), and Newton documents
    that argument as "the density per-area of the mesh". The old 100.0 was written as though it
    were volumetric: it made the 0.1 x 0.1 m sheet weigh **1 kg**, a handkerchief with the mass of
    a litre of water, and it disagreed by 83x with the ``area_density = density * thickness``
    that ``cloth_env`` computes for the ``ClothAsRigidObject`` mass proxy.

    2.0 kg/m^2 is SIM1 Table 2. Still heavy for cloth -- real cotton sheeting is 0.15-0.3 kg/m^2 --
    but it is their calibrated value and it is 50x lighter than what this ran before."""
    tri_ke: float = 1.0e2
    """In-plane stretch stiffness -- the Lame ``mu`` (shear modulus) of the membrane model.

    **1.0e2 is SIM1 Table 2, and also Newton's own ``default_tri_ke``.** Their "calibrated"
    elasticity is the library default; what they actually tuned away from default is bending.

    This was 1.0e4. The justification was the cable, where softening stretch was catastrophic
    (x0.01 peaked at 94.9 m/s) -- but a cable is a 1-D chain whose only in-plane mode IS stretch,
    and the sheet has an area term (``tri_ka``) carrying load the cable had no analogue for. The
    cable result is not evidence about a membrane. If the sheet oscillates or blows up, this is the
    first suspect and 1.0e3 is the obvious intermediate."""
    tri_ka: float = 1.0e2
    """In-plane shear/area stiffness -- the Lame ``lambda`` (area modulus). SIM1 Table 2.

    Note their own solver comment recommends ``lambda: 1000.0+`` to maintain area against
    penetration, and then their env ships 1e2. Table 2 agrees with the env, so 1e2 it is."""
    tri_kd: float = 1.5e-6
    """In-plane damping. SIM1 Table 2 (was 1.0e-5 here).

    Newton's ``default_tri_kd`` is 10.0; both this and the previous value are far below it, which
    is right for a membrane that has to drape rather than settle."""
    edge_ke: float = 8.0e-4
    """Bending stiffness. Low -- a cloth that will not bend cannot be folded. This is the opposite
    of the cable, where a stiff bend response was the thing ringing.

    **8.0e-4 is SIM1 Table 2.** Note their released env ships ``bending_ke = 1e-4``; Table 2 and
    the code disagree, and Table 2 is the stated calibration, so this follows Table 2.

    The previous 0.5 was measured here with a teleported fold: at 5.0 the crease springs open after
    **1** step; at 0.5 it holds **7**; a further 10x softer (0.05) gives the same 7. That measured
    saturation is the reason this change is expected to be inert -- 8.0e-4 is well past the knee,
    so it should behave like 0.05, which behaved like 0.5. It is adopted for alignment, not for an
    expected gain.

    One caveat on transferring it at all: the dihedral bending energy scales with edge length, and
    SIM1 tuned this on a scanned garment with ~mm edges against our 16.7 mm grid. The number is not
    dimensionally portable even though the saturation argument says it should not matter."""
    edge_kd: float = 1.0e-3
    """Bending damping. SIM1 Table 2 (was 1.0e-2 here)."""

    plastic_yield_curvature: float = 0.0
    """Bending plasticity: yield curvature [1/m], 0 = off (purely elastic, as before).

    A hinge bent past ``kappa x hinge width`` (16.7 mm along the grid, 11.8 mm on the diagonals)
    keeps the excess as a new rest angle, so a pressed fold stays folded instead of springing back
    (``utils/plasticity.py``, after Narain et al. 2013). Reset to flat with the sheet.
    It is a mesh-level parameter: in the scripted fold-press-release test
    (``scripts/analysis/cloth_plasticity_probe.py``) 100 (95 deg per grid cell) holds a pressed
    towel or cotton fold without changing the free drape; napkin and denim need 50, chiffon and silk 25."""

    plastic_set_curvature: float = 0.0
    """Two-stage plasticity [1/m], 0 = off (single-stage, every result before 2026-11-10). A hinge
    gains set only while bent past ``plastic_set_curvature x hinge width`` (16.7 mm along the grid);
    gentler bends -- a light touch, a hanging drape -- stay elastic, and set can always shrink, so
    unfolding still irons a crease out (``utils/plasticity.py``)."""

    # --- the fold -------------------------------------------------------------------------
    fold_axis: str = "x"
    """Axis normal to the fold line, in cloth-local coordinates: the ``+`` half folds onto ``-``."""

    max_folded_footprint: float = 0.65
    """Largest sheet extent along the fold axis, as a fraction of ``size``, that still counts as
    folded.

    A second condition on success, independent of keypoint distance. A perfect fold gives ~0.5; an
    unfolded sheet gives 1.0. Keypoint proximity alone proved insufficient: a rigid slide scored
    "folds" because the targets were world-fixed, and after that was corrected a *crumpled* sheet
    still scored transiently by putting four keypoints near their targets. 0.65 leaves room for a
    real fold's drape while excluding anything close to flat."""

    keypoint_tolerance: float = 0.04
    """Per-keypoint distance for success [m]. Deliberately looser than the cable's 1.5 cm: a
    folded flap settles wherever the drape puts it, and demanding rigid-body precision from a
    sheet would measure the tolerance rather than the fold."""

    task: str = "fold"
    """``fold``: start flat, fold the ``+`` half onto ``-``. ``unfold``: start with the ``+`` half
    folded over, open it flat.

    Unfold keeps everything else: the keypoints, the stationary-half frame and the reward. Only the
    target (the flat rest pose instead of its reflection), the goal marker's orientation and the
    footprint condition (``min_unfolded_footprint``) change. The folded start comes from
    ``unfold_start_deg`` on the VBD sheet and the single chain, and from
    ``fold_pool.init_fold_starts: target`` on the pool.

    ``mixed`` (PhysX pool and chain only): every episode draws its own task, unfold with
    ``unfold_prob``, fold otherwise. Target, goal attitude, success test and start follow the
    episode's task."""

    unfold_prob: float = 0.5
    """``task: mixed`` only: the share of episodes that are unfolds."""

    min_unfolded_footprint: float = 0.9
    """``task: unfold`` only: smallest sheet extent along the fold axis, as a fraction of ``size``,
    that counts as unfolded (flat = 1.0, a half fold = 0.5)."""

    unfold_on_table: bool = False
    """``task: unfold`` (and the unfold episodes of ``mixed``): success also needs the sheet to lie
    flat ON THE TABLE (``ClothTaskMixin.flat_on_table``). Without it the flat target rides on the
    stationary half's live frame, tilt and height included, so a sheet held open in the air or
    tilted against the fingers passes. Off by default, so earlier results keep their meaning."""

    on_table_max_spread: float = 0.015
    """``unfold_on_table``: largest height range over all particles (max z - min z) [m] that
    counts as flat. A level sheet at rest has ~0; 1.5 cm is about one VBD sheet thickness, a 9 deg
    tilt on a 10 cm sheet. Measured on old unfold successes (debug_outputs/train_logs/results/flat_on_table): rigid
    sheets still tilted span 6-14 cm, VBD wrinkles pass 49 / 78 / 91 % at 1 / 1.5 / 2 cm."""

    on_table_max_lift: float = 0.015
    """``unfold_on_table``: largest height of the LOWEST particle above the table's collision
    surface [m]. At rest the particles sit above it by the contact offset: the VBD sheet ~6 mm
    (particle radius), the plates half their thickness."""

    unfold_start_deg: list[float] = [180.0, 180.0]
    """``task: unfold``, VBD sheet and single chain: the angle the ``+`` half starts turned up about
    the crease [deg], drawn uniformly per episode. 180 = folded flat onto the other half: one
    ``self_contact_radius`` above it (VBD), the chain's tightest fold (``chain_fold_angles``; on
    ``box3-mid`` both bar hinges at 90, plies 2 mm apart). The pool reads ``fold_pool.init_fold_deg``."""

    # --- release-and-persist (ClothTaskMixin, fold task) ----------------------------------
    release_persist: bool = False
    """Score a fold only once it SURVIVES THE HAND LEAVING. Off: the episode ends and
    ``reward.reach_goal_bonus`` pays on ``termination.success_steps`` of the fold criterion, hand on
    the cloth (every result before 2026-11-08). On:

        r += I_fold * [ (1 - I_rel) * r_release + I_rel * I_nc * r_persist ]

    ``I_nc`` = hand clear of the cloth (``hand_cloth_distance() > release_clearance``); ``I_rel``
    latches once folded AND clear for ``release_latch_steps``. ``r_release`` = the one-shot
    ``release_bonus`` on the latch plus ``release_dist_rew_scale * max(d - d*, 0)``, a ratchet on the
    clearance ``d`` (capped at ``release_clearance``) against its running max ``d*``, tracked only
    while folded. ``r_persist`` = ``persist_rew_per_step`` while folded and clear, and
    ``persist_bonus`` once that has held ``persist_window_steps`` (success: episode ends). Losing
    the fold after the latch pays ``persist_fail_penalty`` and ends the episode. The fold entry
    (``success_steps`` held) pays ``fold_entry_bonus`` once and no longer ends the episode."""

    release_clearance: float = 0.03
    """Hand-to-cloth distance [m] that counts as no contact: nearest fingertip/palm BODY ORIGIN to
    the nearest particle. The origins sit ~1 cm inside the finger surface, so 3 cm is a clear gap."""
    release_latch_steps: int = 3
    release_fold_tolerance: float = 0.0
    """Keypoint error [m] the fold must be under for the release to LATCH (with the hand clear for
    ``release_latch_steps``). 0 = the fold criterion's own ``keypoint_tolerance``. Tighter (e.g.
    0.010 against 0.015) makes the policy let go of a cleaner fold, leaving margin for the drift
    after release; the persist window still only needs ``keypoint_tolerance``."""
    fold_entry_bonus: float = 1000.0
    """Fold entry (the `success_steps` hold), paid once. 1000 = the old `reach_goal_bonus`: at 300
    the denim pilot (job 28395821) stopped folding, entry 100% -> ~47% by epoch 60."""
    release_bonus: float = 300.0
    release_dist_rew_scale: float = 3000.0
    """Per metre of new clearance; at most ``release_clearance * scale`` = 90 per episode."""
    persist_rew_per_step: float = 2.0
    persist_bonus: float = 1000.0
    persist_window_steps: int = 180
    """Policy steps (3 s at 60 Hz) the fold must stay folded with the hand clear. 180 since 2026-11-10
    (60 = 1 s before; release runs passed 180 explicitly)."""
    persist_fail_penalty: float = -300.0
    persist_fail_terminates: bool = True
    """End the episode when the fold is lost after release. False (renders) keeps it running to the
    timeout, so what happens after a spring-back (re-press, further opening) stays on screen."""
    persist_consecutive: bool = True
    """True: the persist window must be ``persist_window_steps`` IN A ROW of folded & hand-clear;
    any re-touch or fold loss resets the count. False: re-touching only pauses it, so a policy can
    collect the window across several presses."""
    release_touch: str = "contact"
    """What counts as the hand TOUCHING the cloth for the release latch and the persist window.
    Default ``contact`` since 2026-11-10; ``origin`` reproduces every result before then: a ``release_contact_bodies`` origin within
    ``release_clearance`` of a particle -- blind to a finger segment or the palm side lying on the
    sheet away from its origin. ``contact``: any hand-cloth contact record the cloth solver holds this
    step (particle, edge or face -- the ``full_surface_contact`` records VBD consumes) with its gap,
    computed as VBD does, under ``release_touch_gap``. Every robot shape that collides with the cloth
    counts, so no part of the hand can be on the sheet. The clearance ratchet reward still uses the
    origin distance."""
    release_touch_gap: float = 0.002
    """``release_touch=contact``: a contact record counts as touching while ``gap = n . (x - x_body)
    - radius - shape_margin`` [m] is below this. VBD applies force at gap < 0; the 2 mm slack also
    catches a hand resting on the sheet between solver iterations."""
    release_contact_bodies: str = ""
    """Regex of robot bodies whose origins define the hand-cloth distance. Empty = the 5 fingertip
    DP links + the palm (iiwa14_link_7, the wrist flange), which misses the PP/MP finger segments
    and the finger bases. ``left_.*|iiwa14_link_7`` covers every hand link: the palm meshes belong to
    iiwa14_link_7 (the hand is mounted on the flange), so ``left_.*`` alone misses the palm."""

    # --- coupling (defaults carried from the cable result) --------------------------------
    proxy_mode: str = "staggered"
    """See `docs/phase4_cable_env.md`. `lagged` syncs the hand's *begin* pose with its *end*
    velocities; the mismatch scales with hand speed and was the single change that took the cable
    from ~0 to 5 goals in an episode."""

    substeps: int = 6
    """Cloth substeps inside one coupled step, i.e. the VBD timestep.

    **6 makes the cloth dt 1/1440 s, matching SIM1.** Their loop is 60 Hz with 24 substeps; ours is
    ``sim.dt`` 1/120 / ``newton.num_substeps`` 2 / this, so 6 lands on the same 0.694 ms. It was 2
    (1/480 s, 3x larger).

    **This directly contradicts a cable measurement**: on the cable there was a sharp optimum at 2,
    with 1 peaking at 37 m/s and 4/8/16 degrading monotonically, on the theory that fewer, larger
    substeps mean fewer proxy exchanges per step and the exchange is where the energy enters. That
    was never re-measured for the sheet, and the cloth is a different solver entry, but it is a
    direct conflict and not a gap -- if the sheet gains energy at reset or under contact, revert
    this first.

    Costs 3x the VBD work per policy step, on top of the iteration increase."""

    vbd_iterations: int = 12
    """VBD solver iterations per substep.

    **12 is SIM1's ``lift2`` value** (their ``acone`` task uses 16). Everything below was measured
    at 10, which is Newton's own default and was 8x cheaper than the 80 this inherited from the
    cable; 12 is a 20% increase on that and the cost model below predicts it directly. Measured at
    256 envs on pinned hardware, per-step cost is almost entirely this term:

        ms/step ~ 45.5 + 11.52 x iterations

    which fits every point (10 -> 160.7, 20 -> 275.9, 40 -> 489.0, 80 -> 933.7 measured). At 80
    iterations ~96% of the step is VBD. Dropping to 10 gives 5.80x throughput at 1024 envs
    (1077 -> 6253 env sps) with memory unchanged at 44.85 GB, and removes the ~20% per-step
    step-up previously seen above 128 envs -- that overhead was itself iteration work, so the scene
    now scales flat from 1 to 1024 envs.

    Quality is unchanged, on five seeds / 160 episodes each side:

        80 iters: 2/160 goals, best_fold_err 0.0775 +- 0.0028, falls 2-5
        10 iters: 3/160 goals, best_fold_err 0.0792 +- 0.0012, falls 2-4

    The error bars overlap heavily and nothing is monotonic in iteration count, so the claim is "10
    is not worse", not "10 is better". 80 was never validated for the cloth; 10 is the documented
    default of both `SolverVBD` (`solver_vbd.py:233`) and `isaaclab_contrib`'s VBD config, and
    Newton's own coupled-solver example and ADMM test use 10 and 5.

    That best_fold_err is flat across an 8x range also rules something out: the ~0.077 stopping
    point is not a solver-convergence limit.

    **10 is a COUPLING floor, not a cloth-physics one.** Newton's standalone cloth examples run 4-10
    (example_cloth_franka.py uses 5, example_cloth_twist.py 4), and iterations here serve two jobs at
    once: converging the sheet's internal dynamics, and resolving contact against the rigid proxy.
    What degrades at 5 is only the second -- the hand stops gripping, footprint stays near 1.0, the
    sheet is untouched rather than unstable -- so the extra iterations are buying contact, not cloth.

    A previous version of this note proposed `SolverVBD(rigid_contact_history=True)` as the targeted
    fix, and recorded it as tried-and-reverted because contact warm-starting packs contact ids into
    20 bits, capping buffered contacts at 2**20 = 1,048,576 globally while this scene budgets
    7,962,624 triangle pairs at 32 envs. **That blocker is real but it was aimed at the wrong
    lever.** In Newton 1.5 `rigid_contact_history` is body-BODY only -- it drives
    `snapshot_body_body_contact_history` and never touches body-particle contacts, so it could not
    have helped the sheet's grip even if it had fit.

    The knob that does reach cloth-vs-hand contact is `rigid_avbd_beta`; see that field."""
    coupler_iterations: int = 1
    """Raising this made the cable *worse* at every setting tried."""

    mass_scale: float = 1.0
    """KEEP AT 1.0. 0.05 was ~500x worse on the cable and NaN'd the robot."""

    collide_interval: int = 1

    use_cuda_graph: bool = False
    """Capture the coupled MJWarp + VBD step as a CUDA graph. Off by default because every result so
    far was produced without it (inherited from the cable, whose reason -- entries stepping at
    different rates -- was never measured: the substep counts are fixed, so a graph records them).
    At small env counts the step is kernel-LAUNCH bound (decimation 2 x substeps 6 x 12 VBD
    iterations per policy step), which is what makes the live viewer's cloth ~4 steps/s."""

    soft_contact_ke: float = 5.0e2
    """Body-particle contact stiffness [N/m]. SIM1 Table 2 (was 8.0e3 here).

    The per-contact value is the AVERAGE of this and the rigid shape's own material stiffness
    (`NewtonShapeCfg.ke`, 2.5e3), so the effective ceiling moves 5250 -> 1500."""

    soft_contact_kd: float = 5.0e-3
    """Body-particle contact damping [N*s/m]. SIM1 Table 2.

    NEW FIELD. `NewtonModelCfg` defaults it to 1.0e-2 and `build_physics` was passing only `ke` and
    `mu`, so this ran at the library default and was never stated."""

    soft_contact_mu: float = 0.25
    """Body-particle friction coefficient. SIM1 Table 2 (`self_contact_friction`; was 10.0 here).

    Effective per-contact friction is `sqrt(soft_contact_mu * shape_mu)`, so against the fingertips
    (`assets.finger_tip_friction` 1.5) this moves from `sqrt(10 x 1.5)` = 3.9 to `sqrt(0.25 x 1.5)`
    = 0.61. SIM1's own effective value is 0.5 -- they set every shape's mu to 1.0.

    Table 2's `robot_friction` 1.5 / `table_friction` 0.0 are NOT copied: their env overwrites
    every shape's mu to 1.0 immediately after building the table, so those two rows describe a
    configuration that does not run. Our per-shape frictions stay as the rigid-tool task sets
    them."""

    self_contact_mu: float = 0.0
    """Cloth-on-cloth friction coefficient (ply on ply). 0 = ``soft_contact_mu`` (previous behaviour).

    Newton's VBD self-contact uses the raw ``model.soft_contact_mu``, while every cloth-vs-shape
    contact uses ``sqrt(soft_contact_mu * shape_mu)``. The fabric presets calibrate
    ``soft_contact_mu`` for cloth-on-table, which left ply-on-ply as the LOWEST friction in the
    scene (big napkin 0.265 against 0.36 table, 0.63 fingertips), so a pressed fold slid apart.

    When > 0: ``model.soft_contact_mu`` becomes this value and ``ClothEnv`` rescales every
    shape's friction IN THE CLOTH SOLVER'S VIEW ONLY by ``soft_contact_mu / self_contact_mu``, so
    each cloth-vs-shape coefficient stays ``sqrt(soft_contact_mu * shape_mu)``. The rigid solver
    (robot vs table) keeps the original shape frictions."""

    table_contact_mu: float = 0.0
    """Effective cloth-on-table friction coefficient. 0 = keep ``sqrt(soft_contact_mu *
    table_friction)`` (0.36 for the measured presets). Applied in the cloth solver's view only, on
    its BOX shapes (the table); the rigid solver's table friction is unchanged."""

    friction_epsilon: float = 0.0
    """``SolverVBD(friction_epsilon=...)`` [m/s]: relative tangential speed below which friction is
    smoothed into a stiff spring (stick). 0 = Newton's default (1e-2). Smaller = less creep under a
    sustained tangential load, and a stiffer friction Hessian."""

    rigid_contact_k_start: float = 1.0e2
    """Body-particle contact penalty SEED for AVBD ramping [N/m].

    **Inert unless `rigid_avbd_beta` > 0**, which is why it did nothing until now. Newton computes
    `rigid_contact_k_start_value = -1.0 if linear_beta == 0.0 else k_start` (solver_vbd.py:677) and
    the kernel reads `k_floor = avg_ke if k_start < 0.0 else min(k_start, avg_ke)`
    (rigid_vbd_kernels.py:3395) -- so with beta 0 the seed is discarded and every contact starts at
    full material stiffness."""

    rigid_avbd_beta: float = 5.0e5
    """AVBD penalty ramp rate for body-particle contacts [N/m per m of penetration].

    NEW FIELD, and **the one change here that is not from SIM1**: their solver (a fork of Newton
    0.1.3) has no contact ramping at all, so matching them exactly would mean leaving this at 0.
    Newton 1.5 grew AVBD natively -- `SolverVBD` is documented as "VBD for particles and Augmented
    VBD (AVBD) for rigid bodies", citing Giles et al. 2025 -- and
    `update_body_particle_contact_penalty` (rigid_vbd_kernels.py:4953) ramps exactly the
    cloth-vs-hand contacts this task depends on. It ships disabled (`rigid_avbd_beta = 0.0`).

    Enabling it is the Newton-native answer to the same failure SIM1's strain limit addresses:
    cloth going soft under fast gripper motion.

    Sizing, which is arithmetic and NOT a measurement. The kernel does
    `k += beta * penetration`, clamped to the per-contact ceiling (~1500, see `soft_contact_ke`),
    from a seed of `rigid_contact_k_start` = 100. To close that 1400 gap inside one substep's 12
    iterations needs `beta * penetration` ~ 117, i.e. beta ~ 1.2e6 at 0.1 mm penetration and
    ~1.2e5 at 1 mm. 5.0e5 sits mid-range: it reaches the ceiling within a substep for penetrations
    above ~0.24 mm and ramps more gradually below that.

    **This is the item most likely to go the wrong way.** With only 12 iterations, starting soft
    and ramping can leave contacts SOFTER than the current fixed-k behaviour, which would weaken
    the grip rather than strengthen it. A/B it against 0.0 before trusting it."""

    # --- self-contact ---------------------------------------------------------------------
    enable_self_contact: bool = True
    """Whether the sheet collides with ITSELF.

    NEW FIELD, and the largest behavioural change in this config. `VBDSolverCfg` defaults
    `particle_enable_self_contact` to False and `build_physics` never set it, so until now the
    folded flap passed straight THROUGH the half it was supposed to land on. For a folding task
    that is not a tuning knob, it is a missing mechanic. SIM1 runs `handle_self_contact=True`."""

    self_contact_radius: float = 0.002
    """Distance at which cloth primitives start to repel each other [m]. SIM1 Table 2.

    **This is a geometric distance between mesh primitives, NOT scaled by `particle_radius`** --
    see `evaluate_self_contact_force_norm` (particle_vbd_kernels.py:821), which is driven by
    `collision_radius` alone.

    So copying SIM1's number does not reproduce SIM1's behaviour on our geometry. Their cloth is a
    scanned garment with ~mm triangles, where 2 mm is comparable to the mesh scale; our sheet is
    16 mm thick on a 16.7 mm grid, so a folded flap will settle 2 mm above the layer beneath it
    rather than ~16 mm -- the two layers visually interpenetrate by most of their thickness.

    That interacts with the fold targets: `cloth_env` builds them with `lift = 2 * particle_radius`
    = 16 mm, so the targets sit ~14 mm above where the flap can actually rest. That is inside the
    0.04 m `keypoint_tolerance`, so it biases the reward rather than making the fold impossible.
    Scaling this to ~`thickness` (with margin ~1.5x) is the consistent alternative and costs more
    in collision detection. Left at SIM1's value; flagged, not hidden."""

    self_contact_margin: float = 0.003
    """Detection margin for self-contact [m]. SIM1 Table 2, and 1.5x the radius, which is the ratio
    Newton's own error message asks for. Must be >= `self_contact_radius` or the solver raises."""

    self_contact_detection_interval: int = -1
    """How often self-contact detection runs. ``-1`` = once before initialisation, which is both
    the `VBDSolverCfg` default and what SIM1 uses."""

    conservative_bound_relaxation: float = 0.42
    """Relaxation factor for the penetration-free displacement bound (SIM1's beta, Table 2).

    Newton 1.5 defaults this to 0.85; 0.42 was the default in the Newton 0.1.3 that SIM1 forked, so
    their Table 2 row is arguably just that default restated. Kept anyway -- it is the value their
    calibration ran against, and a tighter bound truncates more aggressively, which is the safe
    direction on a sheet that has never had self-contact enabled before.

    Only has any effect when `enable_self_contact` is True."""

    proxy_links: str = "hand"
    """Which hand links the cloth can feel. `tips` (five distal phalanges) makes an enclosing grasp
    geometrically impossible; a pinch grasp is arguably enough to drag a cloth, but the asymmetry
    against the rigid baseline is not worth reintroducing."""

    proxy_table: bool = True
    """The table must be in the proxy set or the sheet falls through it."""

    full_surface_contact: bool = False
    """Cloth EDGES and TRIANGLE INTERIORS collide with the proxy shapes, not only its particles.

    Newton 1.5.2 ``CollisionPipeline(enable_rigid_soft_full_surface_contact=True)``: SDF-based
    edge/face soft contacts (Macklin et al. 2020 local optimisation), consumed by ``SolverVBD``.
    Targets the finger pierce (docs/reports/4_physx/cloth_finger_pierce.md): per-particle contact
    lets a fingertip push the particle under it aside and reach the table between particles.

    Every participating mesh shape (hand proxy links, a mesh table) needs a volume SDF, which
    ``ClothEnv`` provisions before finalize (``shape_force_sdf``). Rigid-rigid contacts run in
    Newton's pipeline too (``use_mujoco_contacts: false``), so those SDFs can change hand-table
    contact as well. Off by default, so earlier results keep their meaning."""

    full_surface_sdf_resolution: int = 0
    """``max_resolution`` of the provisioned SDFs (multiple of 8); 0 = Newton's default."""

    def proxy_body_patterns(self, robot_spec=None) -> list[str]:
        """Body-path regexes of the rigid shapes the cloth feels (the proxy set).

        ``robot_spec`` (a ``robots.RobotSpec``, the Kuka + Sharpa when omitted) names the hand bodies.
        """
        from isaacsimenvs.tasks.play.robots import KUKA_SHARPA

        robot_spec = robot_spec or KUKA_SHARPA
        robot = r"/World/envs/env_.*/Robot/"
        segments = ("distal",) if self.proxy_links == "tips" else ("distal", "middle", "proximal")
        out = [robot + robot_spec.finger_links_regex(*segments)]
        if self.proxy_links == "hand":
            palm = (robot_spec.palm_body_name, *robot_spec.palm_side_link_names)
            out.append(robot + "(" + "|".join(palm) + ")")
        if self.proxy_table:
            out.append(r"/World/envs/env_.*/Table")
        return out

    color: tuple[float, float, float] = (0.85, 0.35, 0.55)

    nan_policy: str = "raise"
    """What to do when any observed quantity goes non-finite: ``"raise"`` or ``"reset"``.

    ``raise`` is right for probes and evaluation -- a NaN there is a bug to diagnose, and the guard
    names which quantity went bad first (it is how the intermittent failure was localised to
    ``robot_joint_pos`` rather than to the cloth).

    ``reset`` is right for training. The offending envs have their observation zeroed so nothing
    non-finite can reach the network or the gradient, and are terminated so they reset through the
    normal path. Over a multi-hour run on 4 GPUs a single diverged env would otherwise kill
    everything; the failure has been seen roughly once per three 900-step evals."""

    # --- buffers ---------------------------------------------------------------------------
    rigid_body_particle_contact_buffer_size: int = 65536
    """Body-to-particle soft-contact capacity **PER RIGID BODY**, not global.

    Newton's own comment on the argument is explicit: "Per-body soft-contact list capacity"
    (`newton/_src/solvers/vbd/solver_vbd.py:266`, default 256). So it must NOT scale with env count:
    the allocation is this value x body count, and body count already scales with envs. Multiplying
    it by `num_envs` as well makes the allocation quadratic -- at 512 envs that is 1,048,576 x 2,304
    bodies = 2.4e9 elements, which overflows Warp's signed-int32 array dimension and aborts before a
    single step.

    1024, inherited from the cable where the manipuland was bodies rather than particles, was far
    too small: a 32-env run died with a CUDA device-side assert, which is what an out-of-bounds
    write into a contact buffer looks like from the host."""
    rigid_body_contact_buffer_size: int = 16384
    """Body-to-body contact capacity per rigid body (Newton default 64). Also per-body: the failure
    message is literally "Per-body rigid contact buffer overflowed 87 > 64"."""
    per_particle_triangle_pairs: int = 2048
    """Triangle-pair budget per cloth particle. The budget is GLOBAL and must scale with env count
    *and* geometry: on the cable a correctly env-scaled 4.19M still overflowed once segments
    doubled.

    Raised from 512 after a 32-env run overflowed twice. That run then died with
    ``normal expects all elements of std >= 0.0`` from the policy's sampler -- dropped contacts let
    a sheet penetrate, which produced NaN in that env's observation. The policy error was the
    symptom; the overflow was the cause, and it is worth remembering that this class of failure
    surfaces far from its origin."""

    @property
    def particle_radius(self) -> float:
        """Half the thickness: what the solver actually consumes."""
        return 0.5 * self.thickness

    @property
    def num_particles(self) -> int:
        return self.resolution**2

    @property
    def spacing(self) -> float:
        return self.size / (self.resolution - 1)

    @property
    def max_resolution_for_thickness(self) -> int:
        """Finest grid whose spacing still admits this thickness without self-overlap."""
        return max(2, int(self.size / self.thickness) + 1)

    def __post_init__(self) -> None:
        """Reject a sheet that would spawn inside the table, or a thickness the grid cannot hold.

        Both failures present as violent instability rather than as configuration errors, which is
        exactly how they were first diagnosed here -- as a solver problem.
        """
        if self.start_height <= self.table_half_thickness:
            raise ValueError(
                f"cloth start_height {self.start_height:.3f} m does not clear the table: "
                f"`reset.table_reset_z` is the table CENTRE and its surface is "
                f"{self.table_half_thickness:.3f} m higher. The sheet would spawn inside the "
                f"table and be ejected in one step. Use start_height > "
                f"{self.table_half_thickness:.3f}."
            )
        self._check_thickness()
        self.check_task()

    def check_task(self) -> None:
        """Validate the fold/unfold switch. Called again at env build: Hydra assigns CLI overrides
        AFTER ``__post_init__`` (see ``cloth_env._install_cloth``)."""
        if self.task not in ("fold", "unfold", "mixed"):
            raise ValueError(f"cloth.task must be fold, unfold or mixed, got {self.task!r}")
        if self.task == "mixed" and not 0.0 < float(self.unfold_prob) < 1.0:
            raise ValueError(f"cloth.unfold_prob must be in (0, 1) for task mixed, got {self.unfold_prob}")
        lo, hi = (float(v) for v in self.unfold_start_deg)
        if not 0.0 < lo <= hi <= 180.0:
            raise ValueError(f"cloth.unfold_start_deg must be 0 < lo <= hi <= 180, got {[lo, hi]}")
        if not 0.5 < float(self.min_unfolded_footprint) <= 1.0:
            raise ValueError(
                f"cloth.min_unfolded_footprint must be in (0.5, 1], got {self.min_unfolded_footprint}"
            )
        if not (float(self.on_table_max_spread) > 0.0 and float(self.on_table_max_lift) > 0.0):
            raise ValueError(
                f"cloth.on_table_max_spread / on_table_max_lift must be > 0, got "
                f"{self.on_table_max_spread} / {self.on_table_max_lift}"
            )

    def _check_thickness(self) -> None:
        """Warn when particle DIAMETER exceeds grid spacing. No longer an error.

        **The original premise was wrong.** This raised on `thickness > spacing`, on the grounds
        that closer particles "self-collide at rest and the sheet crumples before anything
        touches it". They cannot: self-collision is driven by `self_contact_radius` via
        `evaluate_self_contact_force_norm`, and `particle_radius` appears in exactly three
        kernels, all of them body-contact. The two never meet.

        The crumpling this guard was written for is documented in `start_height` above as really
        being the spawn-height bug ("attributed to thickness, to grid resolution, and to cloth
        physics; it was this").

        Independent evidence: SIM1's shipping garment violates the old rule by 40% and simulates
        fine. Measured from their `short-shirt.usdc` -- 7,021 vertices, 13,767 triangles, median
        edge 11.40 mm -- against an 8 mm particle radius, i.e. a 16 mm diameter at 1.40x the
        median edge. Newton's own `example_cloth_franka.py` ships the same 8 mm radius.

        Kept as a WARNING because the ratio is still worth knowing: a diameter far above spacing
        means neighbouring particles present overlapping collision spheres to a finger, which
        changes grip in ways nobody here has measured. But it is not a correctness constraint, and
        as an error it forced this task from a 9x9 sheet to 7x7 for no reason.
        """
        if self.thickness > self.spacing + 1e-9:
            import warnings

            warnings.warn(
                f"cloth thickness {self.thickness:.4f} m (2 x particle_radius) exceeds grid "
                f"spacing {self.spacing:.4f} m, ratio {self.thickness / self.spacing:.2f}x. This "
                f"is NOT an error -- particle_radius is a cloth-RIGID rest offset and cannot "
                f"cause self-collision, which uses self_contact_radius "
                f"({self.self_contact_radius:.4f} m, {self.self_contact_radius / self.spacing:.2f}x "
                f"spacing). For reference SIM1 ships 1.40x and Newton's example_cloth_franka.py "
                f"the same 8 mm radius. Flagged because a diameter well above spacing gives "
                f"neighbouring particles overlapping collision spheres, which changes grip.",
                stacklevel=2,
            )


    def build_physics(self, newton_cfg, num_envs: int, robot_spec=None):
        """``NewtonCfg`` for the coupled MJWarp (robot) + VBD (cloth) scene.

        The one structural difference from the cable: the VBD entry owns **particles**, not bodies.
        A cloth is a particle system, so ``all_particles=True`` is what assigns the sheet to the
        deformable solver -- a body selector would match nothing and the cloth would be integrated
        by nobody, presenting as a sheet frozen in mid-air.

        ``robot_spec`` (a ``robots.RobotSpec``, the Kuka + Sharpa when omitted) names the hand
        bodies the cloth couples to.
        """
        from isaaclab_contrib.coupling import (
            CouplerEntryCfg,
            CouplerProxyCfg,
            CouplerProxyMappingCfg,
        )
        from isaaclab_contrib.deformable import NewtonModelCfg, VBDSolverCfg
        from isaaclab_newton.physics import (
            MJWarpSolverCfg,
            NewtonCfg,
            NewtonCollisionPipelineCfg,
            NewtonShapeCfg,
        )

        from isaacsimenvs.tasks.play.robots import KUKA_SHARPA

        robot_spec = robot_spec or KUKA_SHARPA
        rigid_bodies = [r"/World/envs/env_.*/Robot"]
        proxy_bodies = self.proxy_body_patterns(robot_spec)
        if self.proxy_table:
            rigid_bodies.append(r"/World/envs/env_.*/Table")

        budget = self.triangle_pair_budget(newton_cfg, num_envs)

        # No VBDSolverCfg subclass here, deliberately. `class_type` carries
        # "{DIR}.vbd_manager:NewtonVBDManager" and `{DIR}` is substituted against the *defining*
        # module, so subclassing inside this package makes it hunt for
        # `isaacsimenvs.tasks.cloth.vbd_manager`. The cable subclasses it only to add
        # `rigid_body_contact_buffer_size`, which is a body-BODY buffer; a cloth is particles, so
        # the buffer that matters (`rigid_body_particle_contact_buffer_size`) is already a field.
        vbd = VBDSolverCfg(
            iterations=self.vbd_iterations,
            rigid_body_particle_contact_buffer_size=self.rigid_body_particle_contact_buffer_size,
            rigid_contact_k_start=self.rigid_contact_k_start,
            # Self-contact. Without these the folded flap passes through the half it lands on --
            # `particle_enable_self_contact` defaults to False and this call never set it.
            particle_enable_self_contact=self.enable_self_contact,
            particle_self_contact_radius=self.self_contact_radius,
            particle_self_contact_margin=self.self_contact_margin,
            particle_collision_detection_interval=self.self_contact_detection_interval,
        )
        # `rigid_body_contact_buffer_size` is NOT a field on VBDSolverCfg (the cable subclasses it
        # for exactly this reason), but it IS in SolverVBD's signature, and the manager passes
        # through any cfg attribute the signature accepts. Setting it on the instance therefore
        # reaches the solver without a subclass -- and without a subclass, `{DIR}` in `class_type`
        # still resolves against isaaclab_contrib rather than this package.
        #
        # It defaults to 64. A 32-env cloth run hit "Per-body rigid contact buffer overflowed
        # 87 > 64" and then died with a CUDA device-side assert. Declaring the field in this config
        # without wiring it here left it inert: it read as configured and did nothing.
        vbd.rigid_body_contact_buffer_size = self.rigid_body_contact_buffer_size

        # Same pass-through for two more `SolverVBD` arguments that `VBDSolverCfg` does not
        # declare. `_filter_solver_kwargs` filters `solver_cfg.to_dict()` against the solver
        # signature, and isaaclab's `class_to_dict` walks `obj.__dict__`, so instance attributes
        # set here do reach the solver.
        #
        #   * `particle_conservative_bound_relaxation`: Newton 1.5 defaults 0.85; SIM1 ran 0.42.
        #   * `rigid_avbd_beta`: ships 0.0, which DISABLES AVBD contact ramping and makes
        #     `rigid_contact_k_start` above a no-op. See the field docstring -- this is the one
        #     setting here that could plausibly weaken the grip rather than strengthen it.
        vbd.particle_conservative_bound_relaxation = self.conservative_bound_relaxation
        vbd.rigid_avbd_beta = self.rigid_avbd_beta
        if self.friction_epsilon > 0.0:
            vbd.friction_epsilon = self.friction_epsilon

        solver = CouplerProxyCfg(
            entries=[
                CouplerEntryCfg(
                    name="rigid",
                    solver_cfg=MJWarpSolverCfg(
                        solver=newton_cfg.solver,
                        integrator=newton_cfg.integrator,
                        njmax=newton_cfg.njmax,
                        nconmax=newton_cfg.nconmax,
                        impratio=newton_cfg.impratio,
                        cone=newton_cfg.cone,
                        iterations=newton_cfg.iterations,
                        ls_iterations=newton_cfg.ls_iterations,
                        use_mujoco_contacts=newton_cfg.use_mujoco_contacts,
                        ccd_iterations=newton_cfg.ccd_iterations,
                        disable_sensors=newton_cfg.disable_sensors,
                    ),
                    bodies=rigid_bodies,
                ),
                CouplerEntryCfg(
                    name="cloth",
                    solver_cfg=vbd,
                    # Particles, not bodies -- see the docstring.
                    all_particles=True,
                    include_static_shapes=True,
                    substeps=self.substeps,
                ),
            ],
            proxies=[
                CouplerProxyMappingCfg(
                    source="rigid",
                    destination="cloth",
                    bodies=proxy_bodies,
                    mass_scale=self.mass_scale,
                    mode=self.proxy_mode,
                    collide_interval=self.collide_interval,
                    # The proxy builds its OWN pipeline and defaults max_triangle_pairs to a
                    # GLOBAL 1e6; the outer collision_cfg never reaches it. On the cable that
                    # silently dropped contacts above ~16 envs for hours.
                    collision_pipeline=NewtonCollisionPipelineCfg(
                        rigid_contact_max=newton_cfg.rigid_contact_max,
                        max_triangle_pairs=budget,
                        # Edge/face soft contacts -- see `full_surface_contact`.
                        enable_rigid_soft_full_surface_contact=self.full_surface_contact,
                    ),
                )
            ],
            iterations=self.coupler_iterations,
            model_cfg=NewtonModelCfg(
                soft_contact_ke=self.soft_contact_ke,
                # `soft_contact_kd` was previously left at the `NewtonModelCfg` default (1.0e-2)
                # because this call only passed `ke` and `mu`.
                soft_contact_kd=self.soft_contact_kd,
                # Self-contact reads this raw; cloth-vs-shape pairs are rescaled in the cloth view
                # (`ClothEnv._install_cloth_friction`) so they keep using `soft_contact_mu`.
                soft_contact_mu=self.self_contact_mu if self.self_contact_mu > 0.0 else self.soft_contact_mu,
            ),
        )

        return NewtonCfg(
            solver_cfg=solver,
            collision_cfg=NewtonCollisionPipelineCfg(
                rigid_contact_max=newton_cfg.rigid_contact_max,
                max_triangle_pairs=budget,
            ),
            default_shape_cfg=NewtonShapeCfg(),
            num_substeps=newton_cfg.num_substeps,
            collision_decimation=newton_cfg.collision_decimation,
            use_cuda_graph=bool(self.use_cuda_graph),
        )

    def triangle_pair_budget(self, newton_cfg, num_envs: int) -> int:
        """Global triangle-pair budget for this sheet at ``num_envs``.

        Scales with particle count as well as env count. The cable overflowed a correctly
        env-scaled 4.19M as soon as its segment count doubled, and a sheet has far more triangles
        than a cable -- so geometry has to be in this product, not just ``num_envs``.
        """
        base = newton_cfg.resolve_max_triangle_pairs(num_envs)
        per_env = self.num_particles * self.per_particle_triangle_pairs
        return int(max(base, num_envs * per_env) * 1.5)


# ==================================================================================================
# Sheet geometry -- Grid mesh authoring and the fold definition.
# (was utils/cloth_geometry.py)
# ==================================================================================================

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


# ==================================================================================================
# Rigid sheet designs -- Rigid-body sheet designs: the size grid and the plates + hinges that approximate one sheet.
# (was utils/sheet_designs.py)
# ==================================================================================================

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


# ==================================================================================================
# Sheet as a rigid object -- Present a cloth ``DeformableObject`` through the rigid-object surface the Play task reads.
# (was utils/cloth_adapter.py)
# ==================================================================================================

def _mat_to_wxyz(r: torch.Tensor) -> torch.Tensor:
    """Rotation matrices ``(N, 3, 3)`` to wxyz quaternions, via the trace formulation.

    Uses the largest-diagonal branch rather than the naive trace form, which loses precision (and
    can divide by ~0) when the rotation approaches 180 degrees -- exactly the case a completed fold
    produces.
    """
    n = r.shape[0]
    q = torch.zeros((n, 4), device=r.device, dtype=r.dtype)
    trace = r[:, 0, 0] + r[:, 1, 1] + r[:, 2, 2]

    big = trace > 0
    if bool(big.any()):
        sq = torch.sqrt(trace[big] + 1.0) * 2.0
        q[big, 0] = 0.25 * sq
        q[big, 1] = (r[big, 2, 1] - r[big, 1, 2]) / sq
        q[big, 2] = (r[big, 0, 2] - r[big, 2, 0]) / sq
        q[big, 3] = (r[big, 1, 0] - r[big, 0, 1]) / sq
    rest = ~big
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        sel = rest & (r[:, i, i] >= r[:, j, j]) & (r[:, i, i] >= r[:, k, k])
        if not bool(sel.any()):
            continue
        sq = torch.sqrt(1.0 + r[sel, i, i] - r[sel, j, j] - r[sel, k, k]) * 2.0
        q[sel, 0] = (r[sel, k, j] - r[sel, j, k]) / sq
        q[sel, 1 + i] = 0.25 * sq
        q[sel, 1 + j] = (r[sel, j, i] + r[sel, i, j]) / sq
        q[sel, 1 + k] = (r[sel, k, i] + r[sel, i, k]) / sq
    # eps guards a zero-norm quaternion, which a degenerate fit can produce.
    return torch.nn.functional.normalize(q, dim=-1, eps=1e-12)


def _to_torch(value):
    """Isaac Lab 3.0 hands back warp-backed ProxyArrays; the task code wants torch."""
    return value.torch if hasattr(value, "torch") else value


def task_yaw(root_pose: torch.Tensor) -> torch.Tensor:
    """The heading a reset pose asks for, as BOTH manipuland adapters decode it. ``(n, 7) -> (n,)``.

    Shared on purpose: whatever this returns is where the sheet starts, and a chain standing in for
    the cloth has to start exactly where the cloth would for the same reset draw.

    **The decode reads ``root_pose[:, 3:7]`` as (w, x, y, z), but it does not receive that.**
    ``patches.install_pose_write_conversion`` wraps ``env.object.write_root_pose_to_sim`` and
    reorders the task's wxyz quaternion to xyzw before either adapter sees it. For the random draws
    of training that is harmless -- ``random_orientation`` is an isotropic Gaussian, so any
    relabelling of its components is still Haar-uniform and the decoded heading is still uniform on
    [0, 2 pi). For a SPECIFIC requested orientation it is not: every pure yaw arrives as
    (0, 0, sin, cos) and decodes to 180 deg, so the pinned evaluation start (identity) puts the
    sheet at 180 deg. It is kept, not fixed, because every cloth result so far -- training and
    evaluation -- was produced under it; fixing it would move the cloth's pinned start.
    """
    qw, qx, qy, qz = root_pose[:, 3], root_pose[:, 4], root_pose[:, 5], root_pose[:, 6]
    return torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def fold_rest(rest: torch.Tensor, angle: torch.Tensor, axis: int, lift: float) -> torch.Tensor:
    """The rest sheet with its ``+`` half turned up about the centre line by ``angle`` [rad].

    ``rest`` ``(P, 3)`` local, crease at coordinate 0 along ``axis``; ``angle`` ``(n,)``; returns
    ``(n, P, 3)``. The ``+`` half turns rigidly about an axis ``lift / 2`` above the crease, so at
    180 deg it lies reflected onto the ``-`` half exactly ``lift`` above it (no ply interpenetration),
    and at 0 deg it is the rest sheet. Particles on the crease line itself stay put.
    """
    n = angle.shape[0]
    out = rest.unsqueeze(0).expand(n, -1, -1).clone()
    moving = rest[:, axis] > 1e-9
    r = 0.5 * float(lift)
    a = rest[moving, axis].unsqueeze(0)                    # (1, M) distance past the crease
    c, s = torch.cos(angle).unsqueeze(1), torch.sin(angle).unsqueeze(1)
    # (a, -r) about the pivot (0, r), turned by `angle` from +axis toward +z.
    out[:, moving, axis] = a * c + r * s
    out[:, moving, 2] = rest[moving, 2].unsqueeze(0) + a * s + r * (1.0 - c)
    return out


class ClothAsRigidObject:
    """Adapt a cloth ``DeformableObject`` to the rigid-object reads the Play task performs."""

    def __init__(
        self,
        cloth,
        *,
        num_envs: int,
        rest_local: list,
        keypoint_idx: list[int],
        corner_idx: list[int],
        corner_rest: list,
        spawn_z: float,
        mass: float,
        device,
        reset_offset_xy=(0.0, 0.0),
        fold_start=None,
    ) -> None:
        """``fold_start``: ``None`` (reset flat) or ``(axis, (lo_deg, hi_deg), lift)`` -- reset with
        the ``+`` half turned up by an angle drawn uniformly per reset (``fold_rest``)."""
        self._cloth = cloth
        self._num_envs = num_envs
        self._device = device
        self._rest_local = torch.tensor(rest_local, device=device, dtype=torch.float32)
        self._kp_idx = torch.tensor(keypoint_idx, device=device, dtype=torch.long)
        self._corner_idx = torch.tensor(corner_idx, device=device, dtype=torch.long)
        self._corner_rest = torch.tensor(corner_rest, device=device, dtype=torch.float32)
        self._spawn_z = float(spawn_z)
        self._mass = float(mass)
        #: Env-frame XY added to every reset centre (``cloth.reset_offset_xy``).
        self._reset_offset = torch.tensor(
            [float(reset_offset_xy[0]), float(reset_offset_xy[1])], device=device, dtype=torch.float32
        )
        self._fold_start = fold_start
        self.data = _ClothData(self)

    # ---------------------------------------------------------------- particle access

    def _particles(self) -> torch.Tensor:
        """``(num_envs, P, 3)`` particle positions in world coordinates."""
        return _to_torch(self._cloth.data.nodal_pos_w).view(self._num_envs, -1, 3)

    def _velocities(self) -> torch.Tensor:
        return _to_torch(self._cloth.data.nodal_vel_w).view(self._num_envs, -1, 3)

    def fit_rotation_wxyz(self) -> torch.Tensor:
        """Least-squares rotation from rest corners to current corners, as wxyz. ``(N, 4)``."""
        cur = self._particles()[:, self._corner_idx, :]                 # (N, 4, 3)
        q = cur - cur.mean(dim=1, keepdim=True)                         # centred current
        p = self._corner_rest - self._corner_rest.mean(dim=0, keepdim=True)  # centred rest (4, 3)

        # H = P^T Q, then R = V diag(1, 1, det(VU^T)) U^T. The determinant term forbids a
        # reflection: without it a badly deformed patch can fit an improper "rotation", which is a
        # mirror -- and a mirrored frame fed to a policy is worse than a wrong one.
        h = torch.einsum("ki,nkj->nij", p, q)
        # A crumpled patch can drive the corners toward collinear or coincident, leaving H rank
        # deficient. The SVD then returns an arbitrary rotation and the quaternion can normalise
        # to NaN, which reaches the policy as a non-finite observation and surfaces far away as
        # "normal expects all elements of std >= 0.0" from the action sampler. Ridge the diagonal
        # so a degenerate patch degrades to identity instead of to NaN.
        h = h + 1e-8 * torch.eye(3, device=h.device, dtype=h.dtype).unsqueeze(0)
        u, _, vh = torch.linalg.svd(h)
        v = vh.transpose(-2, -1)
        det = torch.linalg.det(v @ u.transpose(-2, -1))
        d = torch.ones((cur.shape[0], 3), device=cur.device)
        d[:, 2] = det
        r = v @ torch.diag_embed(d) @ u.transpose(-2, -1)               # (N, 3, 3)
        quat = _mat_to_wxyz(r)

        # Last line of defence: never hand a non-finite orientation to the policy. An identity
        # fallback is wrong but bounded; a NaN poisons the whole observation vector and kills the
        # run several layers away from the cause.
        bad = ~torch.isfinite(quat).all(dim=-1)
        if bool(bad.any()):
            quat = quat.clone()
            quat[bad] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=quat.device, dtype=quat.dtype)
        return quat

    def keypoints_w(self) -> torch.Tensor:
        """``(num_envs, K, 3)`` tracked keypoints on the moving half."""
        return self._particles()[:, self._kp_idx, :]

    # ---------------------------------------------------------------- writes

    def write_root_pose_to_sim(self, root_pose: torch.Tensor, env_ids=None) -> None:
        """Restore the sheet flat, centred on the requested position, at the requested YAW.

        XY and yaw of ``root_pose`` are honoured; the height is the configured spawn height, and
        roll/pitch are dropped. A cloth has no rigid pose to teleport, and the task only ever calls
        this at reset.

        **Yaw only, not the full orientation.** `reset_utils` draws a Haar-uniform SO(3) quaternion,
        which is right for a rigid tool that may land in any attitude and wrong for a sheet: roll or
        pitch would stand it on edge or bury it in the table, and the task is defined on a sheet
        lying flat. Taking the yaw of a uniform SO(3) draw is itself uniform on [0, 2*pi), so this
        gets a completely random heading without a second RNG stream and without disturbing the
        seed sequence. A `fixed_start_pose` quaternion is honoured the same way, by its yaw.

        Previously the quaternion was dropped entirely, so every episode presented the sheet
        axis-aligned and the policy never saw a fold in any other heading.
        """
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)
        env_ids = torch.as_tensor(env_ids, device=self._device, dtype=torch.long)

        # `root_pose` already has ONE ROW PER RESETTING ENV, aligned with `env_ids` -- it is not
        # indexed by absolute env id. Writing `centre[env_ids]` therefore indexes a subset tensor
        # with absolute ids and reads out of bounds as soon as a partial reset occurs: at 4 envs
        # they usually reset together so ids 0..3 happen to be valid, but at 8+ a reset of
        # e.g. envs [5, 7] indexes a 2-row tensor with 5 and 7. That surfaced as a CUDA
        # device-side assert (IndexKernel.cu "index out of bounds"), which reads like a physics
        # buffer problem and is not one.
        centre = root_pose[:, :3].clone()
        centre[:, 2] = self._spawn_z
        centre[:, :2] += self._reset_offset

        # Roll and pitch are discarded; see `task_yaw` for what this decode actually receives.
        yaw = task_yaw(root_pose)
        cos_y = torch.cos(yaw).unsqueeze(1)   # (n, 1), broadcast over particles
        sin_y = torch.sin(yaw).unsqueeze(1)

        if self._fold_start is None:
            local = self._rest_local.unsqueeze(0)      # (1, P, 3)
        else:  # task: unfold -- the + half starts turned up about the crease
            axis, (lo, hi), lift = self._fold_start
            ang = torch.empty(len(env_ids), device=self._device).uniform_(
                math.radians(float(lo)), math.radians(float(hi))
            )
            local = fold_rest(self._rest_local, ang, axis, lift)   # (n, P, 3)
        rest_x, rest_y, rest_z = local[..., 0], local[..., 1], local[..., 2]
        rotated = torch.stack(
            (
                rest_x * cos_y - rest_y * sin_y,
                rest_x * sin_y + rest_y * cos_y,
                rest_z.expand(cos_y.shape[0], -1),
            ),
            dim=-1,
        )                                              # (n, P, 3)

        # (len(env_ids), P, 3): the (rest or folded) sheet yawed, then translated to each centre.
        target = rotated + centre.unsqueeze(1)
        self._cloth.write_nodal_pos_to_sim_index(target, env_ids)
        self._cloth.write_nodal_velocity_to_sim_index(torch.zeros_like(target), env_ids)

    def write_root_velocity_to_sim(self, root_velocity: torch.Tensor, env_ids=None) -> None:
        """Zero the sheet's particle velocities.

        The task writes a root velocity at reset; for a sheet the meaningful equivalent is that no
        particle is moving. Applying the requested velocity to every particle would launch the
        whole sheet as a rigid body.
        """
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)
        env_ids = torch.as_tensor(env_ids, device=self._device, dtype=torch.long)
        n_particles = self._rest_local.shape[0]
        zeros = torch.zeros((len(env_ids), n_particles, 3), device=self._device)
        self._cloth.write_nodal_velocity_to_sim_index(zeros, env_ids)

    def set_external_force_and_torque(self, *args, **kwargs) -> None:
        """No-op.

        The task applies external forces to the manipuland for domain randomisation. Spreading a
        body force over a particle sheet is not equivalent to applying it to a rigid body, and
        randomisation is off for every measurement taken here -- so a silent no-op is safer than a
        wrong implementation.
        """
        return None

    def __getattr__(self, name):
        """Defer anything unlisted to the underlying cloth, so a missed read is loud."""
        return getattr(self._cloth, name)


class _ClothData:
    """The ``.data`` surface: rigid-object reads derived from the moving half."""

    def __init__(self, owner: ClothAsRigidObject) -> None:
        self._owner = owner

    @property
    def root_pos_w(self) -> torch.Tensor:
        return self._owner.keypoints_w().mean(dim=1)

    @property
    def root_quat_w(self) -> torch.Tensor:
        """Orientation of the moving half, fitted from its four corners (wxyz).

        Kabsch: the least-squares rotation taking the corners' REST offsets to their CURRENT
        offsets. Four corners span an area, so the fit is well-posed -- four points along one edge
        would be collinear and leave rotation about that edge undetermined, which is why this
        previously returned identity.

        A fold is then a rigid transform of the half (180 degrees about the crease), so the task's
        own keypoint machinery -- ``centroid + R * rest_offsets`` -- produces meaningful goal
        keypoints without any special-casing.
        """
        return self._owner.fit_rotation_wxyz()

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        return self._owner._velocities()[:, self._owner._kp_idx, :].mean(dim=1)

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        return torch.zeros((self._owner._num_envs, 3), device=self._owner._device)

    @property
    def default_mass(self) -> torch.Tensor:
        return torch.full(
            (self._owner._num_envs, 1), self._owner._mass, device=self._owner._device
        )

    def __getattr__(self, name):
        return getattr(self._owner._cloth.data, name)


# ==================================================================================================
# Bulk USD edits -- Bulk per-env USD edits without Kit's per-edit cost.
# (was utils/usd_bulk.py)
# ==================================================================================================

#: Layers inserted into a stage are only referenced by identifier there; keep them alive.
_LIVE_LAYERS: list = []


def _under_kit() -> bool:
    return "omni.kit.app" in sys.modules or "omni.usd" in sys.modules


@contextlib.contextmanager
def bulk_edit_layer(stage, tag: str):
    """Yield the ``Sdf.Layer`` to author per-env specs in; they are on the stage when this exits."""
    from pxr import Sdf

    if not _under_kit():
        with Sdf.ChangeBlock():
            yield stage.GetRootLayer()
        return
    layer = Sdf.Layer.CreateAnonymous(f"isaacsimenvs_{tag}")
    yield layer
    _LIVE_LAYERS.append(layer)
    stage.GetRootLayer().subLayerPaths.insert(0, layer.identifier)


# ==================================================================================================
# The fold task -- ClothTaskMixin -- the fold task itself, on whichever simulator the env class is built on.
# (was cloth_task.py)
# ==================================================================================================

def _join(root: str, rel: str) -> str:
    return f"{root}/{rel}" if rel else root


def _set_attr_spec(layer, prim_path, name: str, vtype, value) -> None:
    """Author ``name = value`` on ``prim_path`` in ``layer`` (an ``over`` if the prim is not there)."""
    from pxr import Sdf

    spec = Sdf.CreatePrimInLayer(layer, Sdf.Path(str(prim_path)))
    attr = spec.attributes.get(name)
    if attr is None:
        attr = Sdf.AttributeSpec(spec, name, vtype)
    attr.default = value


def _author_in_every_env(asset: str, layout, author, check, samples: int = 4):
    """Apply one env's edit to ``/World/envs/env_<i>/<asset>`` in every env that has it.

    ``layout(prim)`` reads what to edit from the first env's composed prim; ``author(layer, root,
    layout)`` writes it for one env as Sdf specs into ``usd_bulk.bulk_edit_layer``, which reaches the
    stage in one step (per-env edits on Kit's root layer cost O(N^2)). Every env holds
    the same asset (one variant), so one env's layout is every env's; ``check(prim, layout)`` --
    ``None`` or a message -- confirms that on the first, last and a few envs between, through the
    composed stage. Returns ``(envs edited, layout)``.
    """
    from isaaclab.sim.utils import get_current_stage
    from pxr import Sdf

    stage = get_current_stage()
    envs = stage.GetPrimAtPath("/World/envs")
    roots = []
    if envs.IsValid():
        for env in envs.GetChildren():
            if env.GetName().startswith("env_") and env.GetChild(asset).IsValid():
                roots.append(f"{env.GetPath()}/{asset}")
    if not roots:
        return 0, None
    roots.sort(key=lambda p: int(p.split("/")[3][4:]))
    lay = layout(stage.GetPrimAtPath(roots[0]))
    with bulk_edit_layer(stage, asset) as layer:
        for root in roots:
            author(layer, root, lay)
    picks = sorted({0, len(roots) - 1, *range(0, len(roots), max(1, len(roots) // samples))})
    for i in picks:
        bad = check(stage.GetPrimAtPath(roots[i]), lay)
        if bad:
            raise RuntimeError(f"[cloth] {asset} edit did not land in every env: {bad}")
    return len(roots), lay



class ClothTaskMixin:
    """The fold task, independent of the simulator the Play env underneath runs on."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        cfg.cloth.check_task()  # CLI overrides land after ClothCfg.__post_init__
        super().__init__(cfg, render_mode, **kwargs)
        self._init_fold_targets()
        self._fold_hold = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Envs whose state went non-finite this step; consumed by `_get_dones` to force a reset.
        self._nonfinite_envs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._nonfinite_count = 0
        # Envs whose reward left the plausible range; counted separately from the non-finite ones
        # because they pass every isfinite check.
        self._insane_count = 0
        # Best fold error seen so far this episode, for the dense progress ratchet. Kept separate
        # from the inherited `_closest_keypoint_max_dist`, which `compute_intermediate_values`
        # resolves against the APPROXIMATE metric inside `super()._get_dones()` -- reusing it would
        # ratchet against a different quantity than the one being rewarded. -1 is the sentinel the
        # task uses for "not yet observed".
        self._closest_fold_err = torch.full(
            (self.num_envs,), -1.0, dtype=torch.float32, device=self.device
        )
        # Set in `_get_dones`, consumed by `_get_rewards` in the same step: which envs earned the
        # sparse bonus this step, under the fold criterion rather than the inherited one.
        self._fold_bonus_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Release-and-persist (`cloth.release_persist`): per-episode state, written in `_get_dones`,
        # consumed by `_get_rewards` in the same step, cleared in `_reset_idx`.
        self._release_mode = bool(cfg.cloth.release_persist)
        if self._release_mode and (self._unfold or self._mixed):
            raise ValueError("cloth.release_persist is defined for task=fold only")
        n, dev = self.num_envs, self.device
        self._release_contact_ids = None
        self._touch_cache = None  # release_touch=contact: robot-shape mask etc., built on first use
        if str(cfg.cloth.release_touch) not in ("origin", "contact"):
            raise ValueError(f"cloth.release_touch must be origin or contact, got {cfg.cloth.release_touch!r}")
        self._entry_paid = torch.zeros(n, dtype=torch.bool, device=dev)    # fold entry bonus paid
        self._released = torch.zeros(n, dtype=torch.bool, device=dev)      # I_rel, latched
        self._rel_streak = torch.zeros(n, dtype=torch.long, device=dev)    # folded & clear, in a row
        self._persist_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self._best_clear = torch.full((n,), -1.0, dtype=torch.float32, device=dev)  # d*, -1 = unset
        self._release_rew_step = torch.zeros(n, dtype=torch.float32, device=dev)
        self._release_latch_mask = torch.zeros(n, dtype=torch.bool, device=dev)
        self._persist_step_mask = torch.zeros(n, dtype=torch.bool, device=dev)
        self._persist_done = torch.zeros(n, dtype=torch.bool, device=dev)
        self._persist_fail = torch.zeros(n, dtype=torch.bool, device=dev)
        # Fold error of a perfectly flat sheet -- the worst legitimate value. Used as the stand-in
        # when an env's fold error is non-finite, so the substitution can never look like progress.
        # Unfold: the worst start is folded flat, its far corners one sheet side from their targets.
        if self._unfold:
            self._flat_fold_err = float(torch.as_tensor(self._task_span()).max())
        else:
            self._flat_fold_err = float(self.fold_error().nan_to_num(nan=0.0).max().item())
            if self._mixed:  # the worse of the two tasks' worst values
                self._flat_fold_err = max(self._flat_fold_err, float(torch.as_tensor(self._task_span()).max()))
        self._drive_goal_marker()

    # ------------------------------------------------------------------ construction

    def _write_object_scale(self) -> None:
        """Tell the task the sheet's real dimensions.

        ``_object_scale_per_env`` is filled by ``setup_scene`` from the *procedural rigid tool*
        pool and nothing updates it when the manipuland is replaced. Left alone, the policy is
        handed a hammer's bounding box while facing a cloth, and the keypoint offsets that size the
        reward and the success test are derived from it. The cable env had exactly this bug.

        Convention: bounding box normalised by ``reward.object_base_size``.
        """
        c = self.cfg.cloth
        base = float(self.cfg.reward.object_base_size)
        bbox = torch.tensor(
            [c.size, c.size, c.thickness], device=self.device, dtype=torch.float32
        )
        self._object_scale_per_env[:] = bbox / base
        print(
            f"[cloth] object_scale = {[round(float(v), 2) for v in bbox / base]} "
            f"(bbox {[round(float(v), 3) for v in bbox]} m / base {base})",
            flush=True,
        )

    def _neutralise_rigid_object(self) -> None:
        """Disable the inherited rigid tool's colliders so it cannot touch the cloth.

        Walks the whole subtree: the colliders sit at ``/Object/<mesh>/collisions``, two levels
        down, and a one-level loop silently disables nothing.

        The layout is read from ONE env and authored into all of them at the Sdf level, in one
        change block (``_author_in_every_env``): a Usd-API edit per env re-notifies the whole stage
        under Kit, which made scene construction quadratic in the env count -- 24576 envs never
        finished building.
        """
        from pxr import Sdf, Usd, UsdPhysics

        def layout(prim):
            n = len(str(prim.GetPath())) + 1
            colliders, bodies = [], []
            for target in Usd.PrimRange(prim):
                rel = str(target.GetPath())[n:]
                if target.HasAPI(UsdPhysics.CollisionAPI):
                    colliders.append(rel)
                if target.HasAPI(UsdPhysics.RigidBodyAPI):
                    bodies.append(rel)
            return colliders, bodies

        def author(layer, root, lay):
            colliders, bodies = lay
            for rel in colliders:
                _set_attr_spec(layer, _join(root, rel), "physics:collisionEnabled", Sdf.ValueTypeNames.Bool, False)
            for rel in bodies:
                _set_attr_spec(layer, _join(root, rel), "physics:kinematicEnabled", Sdf.ValueTypeNames.Bool, True)

        def check(prim, lay):
            colliders, _ = lay
            for rel in colliders:
                api = UsdPhysics.CollisionAPI(prim.GetPrimAtPath(rel) if rel else prim)
                if api.GetCollisionEnabledAttr().Get() is not False:
                    return f"{prim.GetPath()}/{rel} still collides"
            return None

        prims, lay = _author_in_every_env("Object", layout, author, check)
        colliders = prims * len(lay[0]) if lay else 0
        print(f"[cloth] neutralised rigid tool ({prims} prims, {colliders} colliders off)", flush=True)
        if prims and not colliders:
            raise RuntimeError(
                "found the inherited rigid tool but disabled none of its colliders -- it would "
                "stay live and corrupt every measurement. Check the prim layout."
            )

    def _reshape_goal_marker(self) -> None:
        """Give the goal marker the folded half's shape instead of the tool's.

        ``goal_viz`` spawns from the same procedural handle+head USD as the manipuland, so the cloth
        task renders a hammer-shaped ghost for a goal that is a folded sheet. The marker *pose* is
        already correct and success is scored on keypoints, not on this mesh -- but the render is
        the only thing a person actually looks at, and a hammer standing in for a fold invites
        exactly the wrong reading. The same defect on the cable task cost hours.

        Swaps the tool geometry for a thin slab the size of the moving half. ``collisionEnabled``
        stays false so the marker can never touch the sheet;
        ``render_newton._reveal_goal_viz`` draws it by setting the VISIBLE flag, which does not
        require a collider. Authored in one Sdf change block, as ``_neutralise_rigid_object``.
        """
        from pxr import Gf, Sdf, Usd, UsdPhysics, Vt

        c = self.cfg.cloth
        # The half's own grid, not a box: the marker stands in for a folded SHEET, and a solid slab
        # both reads as a brick and shows a thickness the cloth does not have (a VBD cloth is a
        # surface -- its thickness is a collision radius the renderer never draws, so the sheet
        # itself always renders flat).
        gv, gi = half_mesh(float(c.size), int(c.resolution), c.fold_axis)
        points = Vt.Vec3fArray([Gf.Vec3f(*[float(x) for x in v]) for v in gv])
        indices = Vt.IntArray([int(i) for i in gi])
        counts = Vt.IntArray([3] * (len(gi) // 3))

        def layout(prim):
            # The rigid body sits on a child (`.../GoalViz/object_root`), not the GoalViz root.
            # Deactivating that child would remove the body the physics view resolves, so keep the
            # body and replace only the geometry beneath it.
            body = next((d for d in Usd.PrimRange(prim) if d.HasAPI(UsdPhysics.RigidBodyAPI)), prim)
            rel = str(body.GetPath())[len(str(prim.GetPath())) + 1:]
            return rel, [child.GetName() for child in body.GetChildren()]

        def author(layer, root, lay):
            rel, children = lay
            body = _join(root, rel)
            for name in children:
                Sdf.CreatePrimInLayer(layer, f"{body}/{name}").active = False
            mesh = Sdf.CreatePrimInLayer(layer, f"{body}/cloth_marker")
            mesh.specifier = Sdf.SpecifierDef
            mesh.typeName = "Mesh"
            mesh.SetInfo("apiSchemas", Sdf.TokenListOp.Create(prependedItems=["PhysicsCollisionAPI"]))
            _set_attr_spec(layer, mesh.path, "points", Sdf.ValueTypeNames.Point3fArray, points)
            _set_attr_spec(layer, mesh.path, "faceVertexIndices", Sdf.ValueTypeNames.IntArray, indices)
            _set_attr_spec(layer, mesh.path, "faceVertexCounts", Sdf.ValueTypeNames.IntArray, counts)
            _set_attr_spec(layer, mesh.path, "physics:collisionEnabled", Sdf.ValueTypeNames.Bool, False)

        def check(prim, lay):
            rel, children = lay
            body = prim.GetPrimAtPath(rel) if rel else prim
            marker = body.GetChild("cloth_marker")
            if not marker.IsValid() or len(marker.GetAttribute("points").Get() or []) != len(gv):
                return f"{body.GetPath()} has no cloth_marker mesh"
            if any(body.GetChild(n).IsActive() for n in children):
                return f"{body.GetPath()} still shows the tool geometry"
            return None

        marked, _ = _author_in_every_env("GoalViz", layout, author, check)
        print(
            f"[cloth] goal marker reshaped on {marked} prims "
            f"(half-sheet mesh, {len(gv)} verts / {len(gi) // 3} tris)",
            flush=True,
        )

    # ------------------------------------------------------------------ the fold

    @property
    def _unfold(self) -> bool:
        """``cloth.task: unfold`` -- start folded, target the flat sheet."""
        return self.cfg.cloth.task == "unfold"

    @property
    def _mixed(self) -> bool:
        """``cloth.task: mixed`` -- each episode draws fold or unfold (``_unfold_envs``)."""
        return self.cfg.cloth.task == "mixed"

    def _unfold_envs(self) -> torch.Tensor:
        """``(num_envs,)`` bool: which envs' current episode is an unfold."""
        m = getattr(self, "_unfold_env", None)
        if m is None:
            m = torch.full((self.num_envs,), self._unfold, dtype=torch.bool, device=self.device)
            self._unfold_env = m
        return m

    def _draw_task(self, env_ids: torch.Tensor) -> None:
        """``task: mixed``: draw the next episode's task for ``env_ids``. Before the reset writes the
        start (which follows the task)."""
        if self._mixed:
            self._unfold_envs()[env_ids] = torch.rand(env_ids.numel(), device=self.device) < float(self.cfg.cloth.unfold_prob)

    def _by_task(self, unfold: torch.Tensor, fold: torch.Tensor) -> torch.Tensor:
        """Per env, ``unfold`` where the episode is an unfold and ``fold`` elsewhere (leading axis N)."""
        m = self._unfold_envs().view(-1, *([1] * (fold.dim() - 1)))
        return torch.where(m, unfold, fold)

    def _task_span(self):
        """Sheet side [m] (a float, or ``(num_envs,)`` where sizes differ per env): how far a far
        corner travels between the flat sheet and the half fold."""
        return float(self.cfg.cloth.size)

    def _init_fold_targets(self) -> None:
        """World-frame targets for the tracked keypoints, per env."""
        c = self.cfg.cloth
        # PLY SEPARATION IS SET BY `self_contact_radius`, NOT BY `particle_radius`.
        # These are two independent rest offsets that never meet: `particle_radius` is the
        # cloth<->RIGID offset (three body-contact kernels), `self_contact_radius` is the
        # cloth<->CLOTH one (`evaluate_self_contact_force_norm`). A fold is cloth on cloth, so the
        # rigid offset has no say in where the flap comes to rest.
        #
        # This used `2 * particle_radius` = 16 mm while the solver settles the plies at 2.5 mm,
        # so every keypoint on the moving half carried ~14 mm of error at a PERFECT fold. Measured
        # with `scripts/analysis/cloth_fold_probe.py`, teleport-and-release, 4 envs, stable 180
        # steps: settled fold_error 0.0160 -> 0.0020 from this line alone, with the 8 mm grip
        # radius untouched. See docs/reports/1_vbd_cloth/cloth_sim1_physics_and_fold_criterion.md.
        lift = c.self_contact_radius  # where cloth-on-cloth contact actually settles
        local = folded_targets(self._cloth_rest_local, self._cloth_kp_idx, c.fold_axis, lift)

        t = torch.tensor(local, device=self.device, dtype=torch.float32)  # (K, 3)
        # Rest geometry only. `fold_targets_w` combines these with the sheet's CURRENT crease, so
        # the criterion is translation-invariant.
        self._kp_idx_t = torch.tensor(self._cloth_kp_idx, device=self.device, dtype=torch.long)
        rest = torch.tensor(self._cloth_rest_local, device=self.device, dtype=torch.float32)
        corners = rest[self._kp_idx_t]
        self._corner_rest_offsets = corners - corners.mean(dim=0, keepdim=True)
        ax = 0 if c.fold_axis == "x" else 1
        kp_rest = rest[self._kp_idx_t]                       # (K, 3) local
        crease_local = rest[:, ax].max() - c.size / 2.0      # crease at the sheet's midline (=0)
        self._kp_rest_offset = kp_rest[:, ax] - crease_local  # distance past the crease
        self._kp_rest_lateral = kp_rest[:, 1 - ax]

        # Same two quantities for EVERY particle of the moving half, so the folded configuration
        # can be drawn as a ghost copy of the sheet rather than approximated by a rigid marker.
        moving = sorted(set(half_indices(c.resolution, c.fold_axis, positive=True)))
        self._moving_idx = torch.tensor(moving, device=self.device, dtype=torch.long)
        mv_rest = rest[self._moving_idx]
        self._mv_rest_offset = mv_rest[:, ax] - crease_local
        self._mv_rest_lateral = mv_rest[:, 1 - ax]

        # Where each particle belongs after the fold, expressed once in the sheet's REST frame:
        # reflect across the crease, then lift by one SELF-CONTACT RADIUS onto the stationary
        # half -- that is the distance cloth-on-cloth contact actually settles at. At
        # runtime this is carried onto the sheet's live pose by the stationary half's own rigid
        # frame, which is what makes the target follow the cloth when it is rotated as well as
        # when it is translated.
        def _folded_rest(sel: torch.Tensor) -> torch.Tensor:
            f = rest[sel].clone()
            if self._unfold:  # the target is the flat sheet itself, on the same live frame
                return f
            f[:, ax] = 2.0 * crease_local - f[:, ax]
            f[:, 2] = f[:, 2] + c.self_contact_radius
            return f

        self._mv_folded_rest = _folded_rest(self._moving_idx)
        self._kp_folded_rest = _folded_rest(self._kp_idx_t)
        # task: mixed -- the unfold episodes' targets, the flat rest itself.
        self._mv_flat_rest = rest[self._moving_idx].clone()
        self._kp_flat_rest = rest[self._kp_idx_t].clone()
        st_rest = rest[torch.tensor(
            half_indices(c.resolution, c.fold_axis, positive=False),
            device=self.device, dtype=torch.long,
        )]
        self._st_rest_centroid = st_rest.mean(dim=0)
        self._st_rest_centred = st_rest - self._st_rest_centroid
        self._fold_targets_xy = t.unsqueeze(0) + self.scene.env_origins.unsqueeze(1)
        self._stationary_idx = torch.tensor(
            half_indices(c.resolution, c.fold_axis, positive=False),
            device=self.device, dtype=torch.long,
        )
        self._kp_idx_t = torch.tensor(self._cloth_kp_idx, device=self.device, dtype=torch.long)

    def fold_targets_w(self) -> torch.Tensor:
        """``(num_envs, K, 3)`` fold targets, defined RELATIVE TO THE SHEET, not to the world.

        **A fixed world target cannot tell a fold from a slide.** With targets pinned at the rest
        sheet's mirrored position, simply pushing the whole sheet half its width in ``-x`` puts the
        moving half exactly where the targets are, and a flat sheet's height matches too. Measured:
        the pretrained tool policy -- which has never seen cloth -- scored 19 "folds" in 31 envs
        that way, with zero falls. It was sliding the sheet, not folding it.

        So the fold is defined in the sheet's REST frame -- each keypoint reflected across the
        crease and lifted one self-contact radius onto the stationary half -- and then carried onto the
        stationary half's CURRENT rigid frame, fitted from its particles.

        That frame is the whole point. Translating or rotating the cloth moves the target with it,
        so neither can reduce the error; only actually folding the flap can. An earlier version got
        this half right -- the fold-axis coordinate tracked the live crease, but the lateral one
        stayed pinned to the spawn position, so a rotated sheet was scored against an axis-aligned
        target and a genuine fold could read as a failure.
        """

        # Carried onto the stationary half's CURRENT rigid frame, so the target follows the sheet
        # through rotation as well as translation. The rest-frame definition already encodes both
        # the reflection across the crease and the lift onto the stationary half. The lift is
        # `self_contact_radius`, MEASURED: two plies settle 2.5 mm apart at scr = 2 mm, while the
        # sheet floats 8.0 mm above the TABLE at particle_radius = 8 mm. Two different offsets for
        # two different contact pairs; only the cloth-cloth one governs a fold.
        if self._mixed:
            return self._by_task(self._folded_w(self._kp_flat_rest), self._folded_w(self._kp_folded_rest))
        return self._folded_w(self._kp_folded_rest)

    def _stationary_frame(self, parts: torch.Tensor):
        """Rigid transform taking the stationary half's REST pose to its CURRENT pose.

        Returns ``(R, centroid)`` with ``R`` of shape ``(N, 3, 3)``.

        This is what makes the fold target follow the sheet through a *rotation*, not just a
        translation. The previous construction set the fold-axis coordinate from the live crease but
        pinned the lateral coordinate to ``rest_lateral + env_origin`` -- the spawn position -- so
        once the hand turned the cloth the target stayed axis-aligned while the sheet rotated under
        it. On screen the goal appeared beside the stationary half and turned ninety degrees away
        from it; in the metric it meant a genuinely folded but rotated sheet scored as unfolded.
        """
        cur = parts[:, self._stationary_idx, :]
        centroid = cur.mean(dim=1)
        q = cur - centroid.unsqueeze(1)
        h = torch.einsum("ki,nkj->nij", self._st_rest_centred, q)
        # Ridge, determinant correction and an identity fallback, for the same reasons as the
        # corner fit in the adapter: a crumpled half can drive this rank deficient, and a NaN or a
        # mirrored frame here would silently corrupt every target.
        h = h + 1e-8 * torch.eye(3, device=h.device, dtype=h.dtype).unsqueeze(0)
        u, _, vh = torch.linalg.svd(h)
        v = vh.transpose(-2, -1)
        d = torch.ones((cur.shape[0], 3), device=cur.device, dtype=cur.dtype)
        d[:, 2] = torch.linalg.det(v @ u.transpose(-2, -1))
        r = v @ torch.diag_embed(d) @ u.transpose(-2, -1)
        bad = ~torch.isfinite(r).all(dim=-1).all(dim=-1)
        if bool(bad.any()):
            r = r.clone()
            r[bad] = torch.eye(3, device=r.device, dtype=r.dtype)
        return r, centroid

    def _folded_w(self, folded_rest: torch.Tensor) -> torch.Tensor:
        """Carry rest-frame folded positions onto the sheet's live pose. ``(N, M, 3)``."""
        r, centroid = self._stationary_frame(self._particles_w())
        local = folded_rest - self._st_rest_centroid
        return torch.einsum("nij,mj->nmi", r, local) + centroid.unsqueeze(1)

    def _crease_w(self, parts: torch.Tensor, ax: int) -> torch.Tensor:
        """``(num_envs,)`` current position of the hinge line along the fold axis.

        **Midway between the two halves' inboard edges, not the stationary half's edge.**
        ``half_indices`` deliberately excludes the hinge row from both halves, so
        ``stationary[..., ax].amax()`` is one grid spacing *inboard* of the actual hinge. Using it
        as the crease reflected every target one spacing too far: measured at ``size=0.10``,
        ``resolution=9``, targets spanned x in [-0.0625, -0.025] where a true reflection about the
        hinge gives [-0.05, -0.0125] -- a systematic 12.5 mm error, a third of the 40 mm tolerance,
        placing the goal beyond the stationary half's outer edge instead of on top of it.

        The midpoint is used rather than the hinge row itself so this stays correct for an even
        ``resolution``, where there is no hinge row at all.
        """
        st = parts[:, self._stationary_idx, ax].amax(dim=1)
        mv = parts[:, self._moving_idx, ax].amin(dim=1)
        return 0.5 * (st + mv)

    def folded_half_w(self) -> torch.Tensor:
        """``(num_envs, M, 3)`` -- where EVERY particle of the moving half belongs when folded.

        The same construction as :meth:`fold_targets_w`, over the whole half instead of the four
        scored keypoints. This exists so the goal can be drawn as a ghost copy of the sheet: a
        folded cloth is a *surface*, and no rigid marker can depict one -- a hammer, a box and a
        flat mesh all read as some kind of slab, because ``goal_viz`` is a rigid body with a single
        pose and a single shape. The renderer can log an arbitrary mesh, so it does.

        Shares :meth:`_folded_w` with the scored targets, so the drawing cannot drift from the
        criterion -- the ghost is the goal, over every particle instead of over four keypoints.
        """
        if self._mixed:
            return self._by_task(self._folded_w(self._mv_flat_rest), self._folded_w(self._mv_folded_rest))
        out = self._folded_w(self._mv_folded_rest)
        return out

    def folded_half_topology(self) -> list[int]:
        """Triangle indices for :meth:`folded_half_w`, in that array's own vertex ordering.

        **Winding is reversed**, because a fold is a reflection and a reflection inverts triangle
        orientation. Keeping the rest sheet's winding leaves every folded triangle facing downward,
        so the goal replica renders as an unlit back face -- a black shape with a lit rim, which is
        exactly how it first appeared.
        """
        _, idx = half_mesh(
            float(self.cfg.cloth.size), int(self.cfg.cloth.resolution), self.cfg.cloth.fold_axis
        )
        if self._unfold:  # the flat goal is not reflected, so its winding is the rest sheet's
            return list(idx)
        return [v for t in range(0, len(idx), 3) for v in reversed(idx[t : t + 3])]

    def _particles_w(self) -> torch.Tensor:
        return self.object._particles()

    def hand_cloth_distance(self) -> torch.Tensor:
        """``(num_envs,)``: nearest distance [m] from any fingertip or palm body ORIGIN to any cloth
        particle. The release test (``cloth.release_clearance``) uses this, not
        ``_curr_fingertip_distances``, which measures to the sheet's fitted frame origin (its
        centre): a fingertip pinning a corner is far from the centre while still touching."""
        if self._release_contact_ids is None:  # resolved lazily: the robot's bodies exist after setup
            rx = str(self.cfg.cloth.release_contact_bodies)
            if rx:
                ids, names = self.robot.find_bodies(rx)
                if not ids:
                    raise ValueError(f"cloth.release_contact_bodies={rx!r} matched no robot body")
                print(f"[cloth] release contact bodies ({len(ids)}): {names}", flush=True)
                self._release_contact_ids = list(ids)
            else:
                self._release_contact_ids = list(self._fingertip_body_ids) + [self._palm_body_id]
        hand = self.robot.data.body_state_w[:, self._release_contact_ids, 0:3]
        hand = hand.torch if hasattr(hand, "torch") else hand
        d = torch.cdist(hand.float(), self._particles_w().float()).amin(dim=(1, 2))
        return torch.nan_to_num(d, nan=0.0)  # a diverged env reads as touching: never released

    def _cloth_contact_buffer(self):
        """``(contacts, entry)``: the hand-cloth contacts the cloth (VBD) entry steps on, and that
        entry. Under ``SolverCoupledProxy`` they come from the proxy pipeline built on the cloth view
        (``get_proxy_contacts(<rigid entry>, "cloth")``; particle / edge / face records, shape and
        particle ids global, body ids LOCAL to the cloth view). ``(None, entry)`` before the first
        step or without a proxy pipeline."""
        from isaaclab_newton.physics import NewtonManager

        solver = NewtonManager._solver
        cloth = solver.solver("cloth")
        name = next((k for k, e in getattr(solver, "_entries", {}).items() if e.solver is cloth), None)
        if name is None:
            return None, None
        entry = solver._entries[name]
        for src, dst in getattr(solver, "_proxy_collision_configs", {}):
            if dst == name:
                return solver.get_proxy_contacts(src, dst), entry
        return getattr(solver, "_entry_contact_buffers", {}).get(name), entry

    def hand_cloth_touching(self) -> torch.Tensor:
        """``(num_envs,)`` bool: any robot shape is in contact with the env's cloth this step
        (``cloth.release_touch=contact``). Reads the cloth solver's own contact records and evaluates
        each one's gap as VBD's body-particle / edge / face contact does (``-penetration_depth``:
        ``n . (x - bx) - radius - shape_margin``, ``x`` the barycentric point over the record's
        particles, ``bx`` the body-side point at the current body pose); touching = gap <
        ``release_touch_gap``. A non-finite gap reads as touching."""
        import warp as wp
        from isaaclab_newton.physics import NewtonManager

        out = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        buf, entry = self._cloth_contact_buffer()
        if buf is None or buf.soft_contact_max == 0:
            return out
        model, state = NewtonManager._model, NewtonManager._state_0
        if self._touch_cache is None:
            import re

            view, dev = entry.view, self.device
            labels = list(model.body_label)
            sb_local = view.shape_body.numpy()  # body ids local to the cloth view
            l2g = entry.body_local_to_global.numpy() if entry.body_local_to_global is not None else None
            sb = [int(l2g[b]) if (l2g is not None and b >= 0) else int(b) for b in sb_local]
            robot = [b >= 0 and re.search(r"/Robot/", str(labels[b])) is not None for b in sb]
            if not any(robot):
                raise RuntimeError("cloth.release_touch=contact: no robot shape found in the cloth view")
            if int(view.particle_count) != int(model.particle_count):
                raise RuntimeError("cloth.release_touch=contact assumes the cloth view keeps the model's particle ids")
            margin = view.shape_margin
            margin = wp.to_torch(margin).float() if margin is not None and margin.shape[0] > 0 \
                else torch.zeros(len(sb), device=dev)
            self._touch_cache = {
                "robot": torch.tensor(robot, dtype=torch.bool, device=dev),
                "shape_body": torch.tensor(sb, dtype=torch.long, device=dev),  # GLOBAL body ids
                "margin": margin,
                "world": wp.to_torch(model.particle_world).long(),
                "radius": wp.to_torch(model.particle_radius).float(),
            }
            print(f"[cloth] release_touch=contact: {sum(robot)} robot shapes in the cloth view, gap < "
                  f"{float(self.cfg.cloth.release_touch_gap) * 1e3:.1f} mm counts as touching", flush=True)
        tc = self._touch_cache
        m = int(buf.soft_contact_max)
        count = wp.to_torch(buf.soft_contact_count).long()  # (1,), stays on device
        shape = wp.to_torch(buf.soft_contact_shape).long()
        sel = (torch.arange(m, device=self.device) < count) & (shape >= 0)
        sel &= tc["robot"][shape.clamp(min=0)]
        ids = sel.nonzero(as_tuple=True)[0]
        if ids.numel() == 0:
            return out
        shape = shape[ids]
        corners = wp.to_torch(buf.soft_contact_indices).view(m, 3)[ids].long()
        bary = wp.to_torch(buf.soft_contact_barycentric).view(m, 3)[ids].float()
        normal = wp.to_torch(buf.soft_contact_normal).view(m, 3)[ids].float()
        body_pos = wp.to_torch(buf.soft_contact_body_pos).view(m, 3)[ids].float()
        pq = wp.to_torch(state.particle_q).view(-1, 3).float()
        ok = corners >= 0
        cc = corners.clamp(min=0)
        x = (bary.unsqueeze(-1) * pq[cc] * ok.unsqueeze(-1)).sum(1)
        radius = torch.where(ok, tc["radius"][cc], torch.zeros_like(bary)).amax(1)
        body = tc["shape_body"][shape]
        bq = wp.to_torch(state.body_q).view(-1, 7).float()[body.clamp(min=0)]
        p, qv, qw = bq[:, 0:3], bq[:, 3:6], bq[:, 6:7]  # warp transform: p, then quat xyzw
        t = 2.0 * torch.cross(qv, body_pos, dim=-1)
        bx = body_pos + qw * t + torch.cross(qv, t, dim=-1) + p
        gap = (normal * (x - bx)).sum(-1) - radius - tc["margin"][shape]
        self._touch_last_gap = gap  # diagnostics (cloth_touch_probe)
        self._touch_last_env = tc["world"][cc[:, 0]]
        touch = ~(gap >= float(self.cfg.cloth.release_touch_gap))  # NaN -> touching
        env = self._touch_last_env
        env = env[touch & (env >= 0) & (env < self.num_envs)]
        out[env] = True
        return out

    def cloth_keypoints_w(self) -> torch.Tensor:
        """Tracked keypoint positions, ``(num_envs, K, 3)`` in world coordinates."""
        nodal = self._cloth.data.nodal_pos_w
        nodal = nodal.torch if hasattr(nodal, "torch") else nodal
        return nodal.view(self.num_envs, -1, 3)[:, self._kp_idx_t, :]

    def _tracked_points_w(self) -> torch.Tensor:
        """The points ``point_dr_mode`` perturbs (obs_utils): the moving-half corners themselves,
        in keypoint order -- what a point tracker sees -- rather than the rigid reconstruction."""
        return self.cloth_keypoints_w()

    def fold_error(self) -> torch.Tensor:
        """Per-env max keypoint distance from its fold target, ``(num_envs,)``.

        The *max* rather than the mean: a fold with one corner still flat on the table is not a
        fold, and averaging would call it two-thirds done.
        """
        return (self.cloth_keypoints_w() - self.fold_targets_w()).norm(dim=-1).amax(dim=-1)

    def _sync_observed_keypoints(self) -> None:
        """Pin the keypoint offsets to the moving half's REST corners.

        With a fitted rotation available, the task's own machinery does the right thing:
        ``obj_kp = centroid + R * rest_offsets`` tracks the real corners, and
        ``goal_kp = goal_centroid + R_goal * rest_offsets`` is the *folded* configuration.

        The previous version set the offsets to the sheet's live deviation, which made the object
        keypoints exact but silently destroyed the goal: with one shared offsets array,
        ``obj_kp - goal_kp`` collapses to ``obj_centroid - goal_centroid`` repeated four times, so
        the policy saw where the goal centroid was and nothing about its shape. Rest offsets plus a
        real rotation keep both halves of the comparison meaningful.
        """
        rest = self._corner_rest_offsets.unsqueeze(0).expand(self.num_envs, -1, -1)
        self._keypoint_offsets[:] = rest
        self._keypoint_offsets_fixed[:] = rest

    def _drive_goal_marker(self) -> None:
        """Point `goal_viz` at the fold target, so the OBSERVED goal is the fold.

        Without this the policy is shown the inherited tool-pose goal while success is measured as
        a fold: it would be optimising a different objective from the one being scored. The task
        reads the goal exclusively through `goal_viz`, so moving the marker moves the goal.
        """
        centre = self.fold_targets_w().mean(dim=1)  # (num_envs, 3)
        pose = torch.zeros((self.num_envs, 7), device=self.device)
        pose[:, :3] = centre
        # Folded ORIENTATION, not just position: a fold rotates the half 180 degrees about the
        # crease line. With the manipuland's orientation now fitted from its corners, the task's
        # keypoint machinery turns this into the folded corner positions -- so `keypoints_rel_goal`
        # finally describes a fold rather than a translation.
        #
        # The crease runs along the axis perpendicular to `fold_axis`, so a fold about an x-normal
        # crease is a 180-degree rotation about y (and vice versa) IN THE SHEET'S REST FRAME.
        #
        # Composed with the stationary half's live rotation rather than written as a fixed world
        # quaternion. The position above already tracks rotation (it comes from `fold_targets_w`),
        # but a constant orientation does not, so a yawed sheet was shown a goal whose position was
        # right and whose attitude was the rest one. That reaches the POLICY, not just the render:
        # the task turns `goal_viz`'s pose into `keypoints_rel_goal`. Required before init yaw
        # could be randomised.
        r, _ = self._stationary_frame(self._particles_w())        # (N, 3, 3) rest -> current
        fold_rest = torch.eye(3, device=self.device).repeat(self.num_envs, 1, 1)
        # 180 degrees about rest y (fold_axis == "x") or about rest x. Unfold: the rest attitude.
        if self._unfold:
            pass
        elif self.cfg.cloth.fold_axis == "x":
            fold_rest[:, 0, 0] = -1.0
            fold_rest[:, 2, 2] = -1.0
        else:
            fold_rest[:, 1, 1] = -1.0
            fold_rest[:, 2, 2] = -1.0
        if self._mixed:
            fold_rest = self._by_task(torch.eye(3, device=self.device).expand_as(fold_rest), fold_rest)
        pose[:, 3:7] = self._goal_quat_from_matrix(torch.bmm(r, fold_rest))
        self.goal_viz.write_root_pose_to_sim(pose)

    def _goal_quat_from_matrix(self, rot: torch.Tensor) -> torch.Tensor:
        """The quaternion ``_drive_goal_marker`` writes for the fold rotation ``rot``.

        ``isaaclab.utils.math.quat_from_matrix`` as imported by this module, unpatched. Under Isaac
        Lab 3.0 that returns (x, y, z, w), which the pose-write patch then treats as (w, x, y, z), so
        the goal orientation the task reads back is the true one's components relabelled -- and every
        Newton policy was trained on that. A backend whose helper is wxyz overrides this to hand the
        task the same relabelled quaternion (see ``rigid_body_env``).
        """
        return quat_from_matrix(rot)

    def _get_rewards(self) -> torch.Tensor:
        """Re-point the inherited Play reward at the fold.

        Follows `PreciseAssemblyEnv._get_rewards` (play2perfect): keep `compute_rewards`, then
        subtract the terms that fight this task and add fold-based replacements. Written as
        `reward - term + replacement` so every inherited term stays visible in `_reward_terms` and
        in the wandb/tensorboard breakdown.

        Three substitutions:

        **Lift terms out.** `lifting_rew` (x20) and `lifting_bonus` (300) pay for raising the
        manipuland 0.15 m above its spawn height. A fold keeps the sheet on the table, so these
        reward the opposite of the task -- exactly why play2perfect masks them for insertion, where
        the goal pulls the peg *down*.

        **The `lifted` gate off the keypoint term.** `keypoint_reward` is multiplied by
        `lifted.float()` (`reward_utils.py:50`), and `_lifted_object` latches only on a 0.15 m rise.
        A folding sheet never rises that far, so the inherited keypoint reward pays **exactly zero
        for the entire episode**. Masking the lift terms without also removing this gate would leave
        the main shaping signal dead -- it already is.

        **The keypoint metric.** The inherited term ratchets on `_keypoints_max_dist`, which is
        `max |obj_kp - goal_kp|` built from the fitted rigid frame plus rest offsets -- an
        approximation of the fold. `fold_error()` is the exact quantity the success criterion and
        the eval both use. Ratcheting on that makes the dense signal *fold progress*.

        The bonus is likewise re-keyed: the inherited `reach_goal_bonus` fires on `_near_goal` /
        `_is_success`, which are computed from the approximate metric against
        `_current_success_tolerance`. `_fold_bonus_mask` is set in `_get_dones` from the same
        `is_folded()` the criterion uses, so the +1000 pays for a fold and nothing else.

        Kept unchanged: `fingertip_delta_rew` (approach shaping, helps the hand find the sheet) and
        both joint-velocity penalties, which are play2perfect's `r_smooth` and the regularisation
        the pretrained policy was trained under.
        """
        from isaacsimenvs.tasks.play.utils.logging_utils import log_step_metrics
        from isaacsimenvs.tasks.play.utils.reward_utils import compute_rewards

        reward = compute_rewards(self)
        terms = self._reward_terms
        rew_cfg = self.cfg.reward

        # --- out: lift progress, lift bonus, and the approximate keypoint term ---------------
        reward = reward - terms["lifting_rew"] - terms["lift_bonus_rew"] - terms["keypoint_rew"]
        zero = torch.zeros_like(terms["lifting_rew"])
        terms["lifting_rew"] = zero
        terms["lift_bonus_rew"] = zero

        # --- in: dense fold progress, ungated by `lifted` -------------------------------------
        # Same ratchet as `reward_utils.keypoint_reward`: pay only for improvement on the best
        # fold error seen so far this goal, clamped at zero so regression is free rather than
        # punished, and monotone so it cannot be farmed by oscillating.
        # A non-finite fold error would poison the ratchet permanently (torch.minimum with NaN
        # propagates) and then every subsequent reward. Envs whose state has diverged are being
        # reset this step anyway -- `_guard_finite` flagged them and `_get_dones` terminated them --
        # so substituting the flat-sheet value costs nothing and keeps the ratchet finite.
        fold_err = self.fold_error()
        bad = ~torch.isfinite(fold_err)
        if bool(bad.any()):
            fold_err = torch.where(bad, torch.full_like(fold_err, self._flat_fold_err), fold_err)
        sentinel = self._closest_fold_err < 0.0
        self._closest_fold_err = torch.where(sentinel, fold_err, self._closest_fold_err)
        delta = torch.clamp(self._closest_fold_err - fold_err, 0.0, 100.0)
        self._closest_fold_err = torch.minimum(self._closest_fold_err, fold_err)
        fold_kp_rew = delta * float(rew_cfg.keypoint_rew_scale)
        reward = reward + fold_kp_rew
        terms["keypoint_rew"] = fold_kp_rew

        # --- the sparse bonus, keyed on the fold criterion ------------------------------------
        reward = reward - terms["bonus_rew"]
        if self._release_mode:
            c = self.cfg.cloth
            fold_bonus = self._fold_bonus_mask.float() * float(c.fold_entry_bonus)
            release_rew = self._release_rew_step + self._release_latch_mask.float() * float(c.release_bonus)
            persist_rew = (
                self._persist_step_mask.float() * float(c.persist_rew_per_step)
                + self._persist_done.float() * float(c.persist_bonus)
                + self._persist_fail.float() * float(c.persist_fail_penalty)
            )
            reward = reward + release_rew + persist_rew
            terms["release_rew"] = release_rew
            terms["persist_rew"] = persist_rew
        else:
            fold_bonus = self._fold_bonus_mask.float() * float(rew_cfg.reach_goal_bonus)
        reward = reward + fold_bonus
        terms["bonus_rew"] = fold_bonus

        # LAST LINE OF DEFENCE. `compute_rewards` reads robot state, fingertip distances and the
        # inherited keypoint metric, any of which can be non-finite when an articulation diverges --
        # and a NaN reward becomes a NaN advantage and then a NaN gradient, which destroys the
        # policy rather than just losing one env. Observed in a smoke run as
        # `shaped_rewards/iter -> nan` while `rewards/iter` still read finite.
        nonfinite_rew = ~torch.isfinite(reward)
        if bool(nonfinite_rew.any()):
            n = int(nonfinite_rew.sum())
            reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
            self._nonfinite_envs |= nonfinite_rew
            print(
                f"[cloth] non-finite REWARD in {n}/{self.num_envs} envs -- zeroed and flagged "
                f"for reset",
                flush=True,
            )
        self.extras["nonfinite_reward"] = float(nonfinite_rew.float().mean())

        # MAGNITUDE, not just finiteness. Checking `isfinite` alone leaves a hole that a diverging
        # articulation drives straight through: joint velocities and actions grow enormous BEFORE
        # they become non-finite, the action penalties are sums of squares, and the reward reaches
        # -1e5 to -1e11 while staying perfectly finite. `nan_to_num` does not touch it, the
        # nonfinite counter stays at 0.0, and the value lands in the advantage and the gradient.
        #
        # Measured on job 197715: kuka_actions_penalty reached -784,354 for a single episode
        # against a typical -13 to -36, total_reward reached -1.1e11, and `rewards/iter` was still
        # reading -1.8 MILLION five hours in, with 56% of samples below -1000 from 44% of the way
        # through the run. 199350 showed the same from 71.5% in. Both runs were compromised while
        # every non-finite guard reported clean.
        #
        # The legitimate per-step maximum is the reach_goal_bonus (1000) plus shaping of order 100,
        # so 1e4 is ~10x headroom over anything the task can pay. An env outside it is diverging by
        # definition, so it gets the same treatment as a non-finite one: contribute nothing, and
        # reset through the normal path.
        insane_rew = reward.abs() > self.REWARD_SANITY_LIMIT
        if bool(insane_rew.any()):
            n = int(insane_rew.sum())
            worst = float(reward[insane_rew].abs().max())
            reward = torch.where(insane_rew, torch.zeros_like(reward), reward)
            self._nonfinite_envs |= insane_rew
            print(
                f"[cloth] INSANE REWARD in {n}/{self.num_envs} envs (worst |r|={worst:.3e}, "
                f"limit {self.REWARD_SANITY_LIMIT:.0e}) -- zeroed and flagged for reset",
                flush=True,
            )
        self.extras["insane_reward"] = float(insane_rew.float().mean())
        self._insane_count += int(insane_rew.sum())
        self.extras["insane_total"] = float(self._insane_count)

        terms["total_reward"] = reward
        self.extras["fold_err_mean"] = float(fold_err.mean())
        self.extras["fold_rate"] = float(self._fold_bonus_mask.float().mean())
        log_step_metrics(self)
        return reward

    def _get_observations(self):
        # Offsets must be current *before* the task reads them, or the observation lags the sheet
        # by one step.
        self._sync_observed_keypoints()
        # The marker has to be re-driven EVERY step, not just at reset. `fold_targets_w` is defined
        # against the sheet's *current* crease and stationary height -- that is what makes the
        # criterion translation-invariant -- so a marker written once at reset drifts away from the
        # target actually being scored as soon as the sheet moves. It then shows the policy (and the
        # viewer) a goal that is not the goal.
        self._drive_goal_marker()
        obs = super()._get_observations()
        self._guard_finite(obs)
        return obs

    def _guard_finite(self, obs) -> None:
        """Keep non-finite values out of the policy, and flag the env for reset.

        A NaN anywhere in the 140-dim vector reaches the policy, propagates through the network and
        surfaces as ``normal expects all elements of std >= 0.0`` from the action sampler -- a
        message that names the sampler and says nothing about the cloth. Guarding the *rotation fit*
        was not enough, because that only covered one of several ways a blown-up sheet produces
        NaN. This reports which quantity went bad first, so the next fix is aimed rather than
        guessed.
        """
        # Ordered most-upstream first, so the guard names the ROOT quantity rather than the
        # observation it poisons.
        #
        # **The robot is checked, not just the cloth.** 80 of the 140 policy columns are robot
        # state (joint_pos 0-28, joint_vel 29-57, palm_pos 87-89, palm_rot 90-93,
        # fingertip_pos_rel_palm 98-112) against 20 that are cloth-derived. The first version of
        # this guard watched only cloth quantities and duly reported "obs bad, cloth fine" --
        # true, and useless, because it never looked where most of the vector comes from. In a
        # coupled solve the MJWarp articulation can diverge from a bad contact while the VBD sheet
        # stays perfectly well behaved.
        #
        # `reward_buf` is deliberately NOT here: the task does feed the reward into an observation
        # ("reward": reward_buf * 0.01, `obs_utils.py:326`) but only into `state_list`, the
        # critic's 162-dim state. It is absent from `obs_list`, so it cannot reach `obs["policy"]`.
        robot = self.robot.data
        offenders = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        parts = {
            "particles": self._particles_w(),
            "velocities": self.object._velocities(),
            "quat": self.object.data.root_quat_w,
            "fold_targets": self.fold_targets_w(),
            "robot_joint_pos": robot.joint_pos,
            "robot_joint_vel": robot.joint_vel,
            "robot_body_pos_w": robot.body_pos_w,
            "robot_body_quat_w": robot.body_quat_w,
            "obs": obs["policy"],
        }
        for name, tensor in parts.items():
            if tensor is None:
                continue
            flat = tensor.reshape(tensor.shape[0], -1)
            bad = ~torch.isfinite(flat).all(dim=-1)
            if not bool(bad.any()):
                continue
            offenders |= bad
            ids = torch.nonzero(bad).flatten().tolist()
            where = ""
            if name == "obs":
                # Which columns, so the term can be identified against `cfg.obs.obs_list`.
                cols = torch.nonzero(~torch.isfinite(flat[ids[0]])).flatten().tolist()
                where = f" cols {cols[:12]}{'...' if len(cols) > 12 else ''} of {flat.shape[1]}"
            # The OFFENDING envs' own progress, not `episode_length_buf.max()`. The max is taken
            # over all envs, so it is only equal to the offender's step while the envs are still
            # synchronised -- i.e. during the first episode. After that it saturates near
            # `episode_length` and every report reads "at step 599", which looks like the failure
            # clusters at the end of an episode when it does not. Per-env progress says when the
            # divergence actually happened.
            steps = self.episode_length_buf[bad].tolist()
            msg = (
                f"cloth: non-finite {name} in envs {ids[:8]}"
                f"{'...' if len(ids) > 8 else ''} ({len(ids)}/{self.num_envs}) "
                f"at env-step {steps[:8]}{'...' if len(steps) > 8 else ''}{where}"
            )
            if self.cfg.cloth.nan_policy == "raise":
                raise RuntimeError(msg)
            # `reset`: report once per occurrence and carry on. Printing every step would drown a
            # training log, so this is deliberately not rate-limited beyond being per-detection.
            print(f"[cloth] {msg} -- resetting those envs", flush=True)

        if not bool(offenders.any()):
            return

        # NEVER let a non-finite value reach the policy. The env is flagged for reset below, but the
        # reset only takes effect on the NEXT step, so this step's observation would still be handed
        # to the network -- and a single NaN propagates through the LSTM into the action sampler,
        # then into the gradient. Zeroing is not physically meaningful, but the env is being thrown
        # away anyway; what matters is that the number entering the optimizer is finite.
        for tensor in obs.values():
            if torch.is_tensor(tensor):
                torch.nan_to_num_(tensor, nan=0.0, posinf=0.0, neginf=0.0)

        # Latched, and consumed by `_get_dones` on the same step so the env terminates and resets
        # through the normal path rather than being patched in place.
        self._nonfinite_envs |= offenders
        self._nonfinite_count += int(offenders.sum())
        self.extras["nonfinite_envs"] = float(offenders.float().mean())
        self.extras["nonfinite_total"] = float(self._nonfinite_count)

    def footprint_ratio(self) -> torch.Tensor:
        """Sheet extent along the fold axis, as a fraction of its unfolded size.

        A fold halves the footprint; a slide or a crumple does not necessarily. Keypoint proximity
        alone cannot tell those apart -- a dragged, crumpled sheet put keypoints near their targets
        and scored, which is why this exists as a second, independent condition.

        **Measured along the sheet's own fold axis, not the world's.** This used to take the extent
        of `parts[:, :, ax]` -- a world axis -- which is only the fold axis while the sheet is
        axis-aligned. A yawed sheet then reports an inflated footprint for purely geometric
        reasons: an unfolded 0.10 m sheet at 45 degrees spans 0.141 m in world x, a ratio of 1.41
        with no deformation at all. That is the likely source of the 1.03-1.41 values seen in the
        render HUD, which were previously read as the sheet stretching under the hand.

        Since `footprint_ratio < max_folded_footprint` is half of `is_folded`, the world-axis
        version made a correctly folded but rotated sheet score as unfolded. It also had to be
        fixed before init yaw could be randomised at all.

        `r[:, :, ax]` is the image of the rest basis vector under the stationary half's fitted
        rotation, i.e. the fold axis carried onto the sheet's current pose. Reduces exactly to the
        old behaviour when the sheet is axis-aligned (r = I).
        """
        ax = 0 if self.cfg.cloth.fold_axis == "x" else 1
        parts = self._particles_w()
        r, _ = self._stationary_frame(parts)
        axis_dir = r[:, :, ax]                                   # (N, 3), unit
        coord = torch.einsum("npi,ni->np", parts, axis_dir)       # (N, P)
        return (coord.amax(dim=1) - coord.amin(dim=1)) / self.cfg.cloth.size

    def footprint_ok(self, footprint: torch.Tensor) -> torch.Tensor:
        """The footprint half of success: halved for a fold, near full for an unfold."""
        if self._unfold:
            return footprint > self.cfg.cloth.min_unfolded_footprint
        if self._mixed:
            return self._by_task(footprint > self.cfg.cloth.min_unfolded_footprint,
                                 footprint < self.cfg.cloth.max_folded_footprint)
        return footprint < self.cfg.cloth.max_folded_footprint

    def flat_on_table(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(spread, lift, ok)``, each ``(num_envs,)``: does the sheet lie flat on the table?

        ``spread`` is the height range over all particles (max z - min z), ``lift`` the lowest
        particle's height above the table's collision surface. Both are needed: spread alone passes
        a level sheet held in the air, lift alone a sheet with one edge down and the rest raised.
        The surface is taken per env, because the table height is drawn per reset
        (``reset.table_reset_z_range``)."""
        c = self.cfg.cloth
        z = self._particles_w()[..., 2]
        low = z.amin(dim=1)
        spread = z.amax(dim=1) - low
        surface = self.scene.env_origins[:, 2] + self._table_z_per_env + c.table_half_thickness
        lift = low - surface
        self._flat_spread, self._flat_lift = spread, lift  # read by the eval at episode end
        return spread, lift, (spread < c.on_table_max_spread) & (lift < c.on_table_max_lift)

    def is_folded(self) -> torch.Tensor:
        """Both conditions: keypoints on target AND the footprint halved (unfold: back to full).
        With ``cloth.unfold_on_table`` an unfold must also lie flat on the table.

        Named for the fold; with ``cloth.task: unfold`` it is the unfold's success test."""
        ok = (self.fold_error() < self.cfg.cloth.keypoint_tolerance) & self.footprint_ok(
            self.footprint_ratio()
        )
        if self.cfg.cloth.unfold_on_table and (self._unfold or self._mixed):
            flat = self.flat_on_table()[2]
            ok = ok & (flat | ~self._unfold_envs()) if self._mixed else ok & flat
        return ok

    #: Any |reward| above this is a diverging env, not a task outcome: the largest the task can
    #: legitimately pay in one step is `reach_goal_bonus` (1000) plus shaping of order 100.
    REWARD_SANITY_LIMIT = 1.0e4

    #: Steps the fold criterion must hold to count as a *held* fold. Independent of
    #: `termination.success_steps`, which the finetune recipe drives to 1 so the reward bonus fires
    #: on first entry -- the strict criterion still has to be reported.
    HELD_FOLD_STEPS = 10

    def _get_dones(self):
        """Terminate on a completed fold, on the sheet leaving the table, or on timeout.

        The inherited test scores a *tool pose* against `goal_viz`, which is meaningless here: the
        manipuland is a sheet and the goal is a fold. This REPLACES the success half of it while
        keeping the task's fall/hand-far conditions.

        **The inherited increment is undone, not merely added to.** `super()._get_dones()` runs
        Play's own keypoint-success path in full, which increments `_successes`
        (`termination_utils.py:46`) whenever `_keypoints_max_dist <= _current_success_tolerance *
        keypoint_scale`. With `eval_success_tolerance` unset that fires at 0.075 * 1.5 = 11.25 cm,
        against the fold criterion's 4 cm -- so `_successes` had two independent writers and every
        "goals" number on this task was ambiguous between a fold and a loose tool-pose hit. Worse,
        a reward keyed on `_successes` would have been farmable at 11.25 cm without ever folding.
        Snapshot and restore so the fold is the only writer.
        """
        successes_before = self._successes.clone()
        terminated, truncated = super()._get_dones()
        self._successes.copy_(successes_before)

        folded = self.is_folded()

        # Two counters, deliberately. `entered` is what the reward bonus and termination key on --
        # `termination.success_steps` is 1 under the finetune recipe, matching Play2Perfect's
        # `success_steps=1` + `force_consecutive_near_goal_steps=True` lump sum. `held` is the
        # strict criterion every baseline number so far used, kept so training and evaluation can be
        # compared against the same history. The gap between them is a diagnostic: a flap that
        # enters the target and springs back scores `entered` but not `held`.
        self._fold_hold = torch.where(folded, self._fold_hold + 1, torch.zeros_like(self._fold_hold))
        entered = self._fold_hold >= int(self.cfg.termination.success_steps)
        held = self._fold_hold >= int(self.HELD_FOLD_STEPS)

        # `_get_rewards` runs after `_get_dones` in the same step, so this is the mask the sparse
        # bonus pays on. Keyed on the fold criterion, never on the inherited `_is_success`.
        if self._release_mode:
            success = self._update_release(folded, entered)
        else:
            self._fold_bonus_mask.copy_(entered)
            success = entered

        if bool(success.any()):
            ids = success.nonzero(as_tuple=True)[0]
            self._successes[ids] += 1

        # NOTE: `episode_length_buf` is deliberately NOT reset here. The inherited task chains
        # goals within an episode, but a folded sheet stays folded, so re-arming would let the
        # criterion pay repeatedly for one fold. The episode ends on the fold instead (below),
        # which is the analogue of Play2Perfect ending the episode on a successful retract.
        # Terminate any env whose state went non-finite, so it resets through the normal path.
        # `_guard_finite` has already zeroed the observation for these envs, so nothing non-finite
        # reaches the policy or the optimizer; this is what stops one diverged env from poisoning a
        # multi-hour run. Cleared after consumption -- the reset itself restores a valid state.
        nonfinite = self._nonfinite_envs.clone()
        self._nonfinite_envs.zero_()

        self._termination_reasons["fold"] = entered
        self._termination_reasons["fold_held"] = held
        if self._unfold or self._mixed:
            # Whether a success lay flat on the table, published with or without
            # `cloth.unfold_on_table`, so one eval of an old policy shows both success rates.
            self._termination_reasons["on_table"] = entered & self.flat_on_table()[2]
        if self._mixed:
            # Averaged over finished episodes (logged as done_<key>): task_unfold = their unfold share;
            # unfold_ok / fold_ok over the matching share are the per-task success rates.
            u = self._unfold_envs()
            self._termination_reasons["task_unfold"] = u.clone()
            self._termination_reasons["unfold_ok"] = entered & u
            self._termination_reasons["fold_ok"] = entered & ~u
        self._termination_reasons["nonfinite"] = nonfinite
        if self._release_mode:
            self._termination_reasons["released"] = self._released.clone()
            self._termination_reasons["persist"] = self._persist_done.clone()
            self._termination_reasons["persist_fail"] = self._persist_fail.clone()
            fail_end = self._persist_fail if self.cfg.cloth.persist_fail_terminates else torch.zeros_like(self._persist_fail)
            return terminated | self._persist_done | fail_end | nonfinite, truncated
        return terminated | entered | nonfinite, truncated

    def _update_release(self, folded: torch.Tensor, entered: torch.Tensor) -> torch.Tensor:
        """``cloth.release_persist``: advance the release / persist state for this step and return
        the success mask (the fold held ``persist_window_steps`` with the hand clear).

        Writes the per-step masks `_get_rewards` pays on. The fold must first be ENTERED (the
        ``success_steps`` hold, which pays ``fold_entry_bonus`` once) before a release can latch, so
        a hand that never touched the sheet cannot release anything."""
        c = self.cfg.cloth
        self._fold_bonus_mask.copy_(entered & ~self._entry_paid)  # fold entry, paid once
        self._entry_paid |= entered

        d = self.hand_cloth_distance()
        if str(c.release_touch) == "contact":  # no robot shape in contact with the sheet at all
            clear = ~self.hand_cloth_touching()
            self.extras["hand_cloth_touch"] = float((~clear).float().mean())
        else:
            clear = d > float(c.release_clearance)

        # r_release ratchet: d* = running max of the capped clearance while folded, not yet released.
        track = folded & self._entry_paid & ~self._released
        d_cap = d.clamp(max=float(c.release_clearance))
        unset = track & (self._best_clear < 0.0)
        self._best_clear = torch.where(unset, d_cap, self._best_clear)
        gain = torch.where(track, (d_cap - self._best_clear).clamp(min=0.0), torch.zeros_like(d))
        self._best_clear = torch.where(track, torch.maximum(self._best_clear, d_cap), self._best_clear)
        self._release_rew_step.copy_(gain * float(c.release_dist_rew_scale))

        # I_rel latches on `release_latch_steps` in a row of folded & clear (and, with
        # `release_fold_tolerance`, fold error under it), after the fold entry.
        latch_ok = folded & clear & self._entry_paid
        if float(c.release_fold_tolerance) > 0.0:  # release only a fold cleaner than the criterion's
            latch_ok = latch_ok & (self.fold_error() < float(c.release_fold_tolerance))
        self._rel_streak = torch.where(latch_ok, self._rel_streak + 1, torch.zeros_like(self._rel_streak))
        newly = ~self._released & (self._rel_streak >= int(c.release_latch_steps))
        self._release_latch_mask.copy_(newly)
        self._released |= newly

        # Persist: count folded & clear steps after the latch (re-touching pauses, does not reset).
        ok = self._released & folded & clear
        self._persist_step_mask.copy_(ok)
        if c.persist_consecutive:  # a re-touch or fold loss restarts the window
            self._persist_steps = torch.where(ok, self._persist_steps + 1, torch.zeros_like(self._persist_steps))
        else:
            self._persist_steps += ok.long()
        self._persist_done.copy_(self._released & (self._persist_steps >= int(c.persist_window_steps)))
        self._persist_fail.copy_(self._released & ~folded)
        self.extras["hand_cloth_dist"] = float(d.mean())
        return self._persist_done

    def _reset_idx(self, env_ids):
        if self._mixed and not hasattr(self.cfg, "fold_pool") and hasattr(self.object, "set_start_deg"):
            ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else \
                torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
            self._draw_task(ids)
            lo, hi = (float(v) for v in self.cfg.cloth.unfold_start_deg)
            deg = torch.empty(ids.numel(), device=self.device).uniform_(lo, hi)
            deg = deg * self._unfold_envs()[ids]  # the fold episodes start flat
            self.object.set_start_deg(ids, deg)
        super()._reset_idx(env_ids)
        if hasattr(self, "_fold_hold"):
            self._fold_hold[env_ids] = 0
        if hasattr(self, "_closest_fold_err"):
            # Back to the sentinel: the ratchet is per-episode, so a new episode must not inherit
            # the previous one's best fold error (which would make all progress unrewarded).
            self._closest_fold_err[env_ids] = -1.0
            self._fold_bonus_mask[env_ids] = False
        if hasattr(self, "_released"):
            for buf in (self._entry_paid, self._released, self._release_latch_mask,
                        self._persist_step_mask, self._persist_done, self._persist_fail):
                buf[env_ids] = False
            self._rel_streak[env_ids] = 0
            self._persist_steps[env_ids] = 0
            self._release_rew_step[env_ids] = 0.0
            self._best_clear[env_ids] = -1.0
        # The task re-samples `goal_viz` on reset; put it back on the fold target.
        if hasattr(self, "_fold_targets_xy"):
            self._drive_goal_marker()

    def fold_fraction(self) -> torch.Tensor:
        """Progress in ``[0, 1]``: 0 at the rest pose, 1 at the target.

        Reported rather than used for reward, so a run that moves the cloth without folding it is
        visibly distinct from one that does not touch it -- the distinction that took far too long
        to establish on the cable.
        """
        unfold_frac = (1.0 - self.fold_error() / torch.as_tensor(self._task_span())).clamp(0.0, 1.0)
        if self._unfold:  # from folded flat (far corners one sheet side away) to flat
            return unfold_frac
        start = (
            torch.tensor(
                [self._cloth_rest_local[i] for i in self._cloth_kp_idx],
                device=self.device,
                dtype=torch.float32,
            )
            .unsqueeze(0)
            + torch.tensor(
                [0.0, 0.0, float(self.cfg.reset.table_reset_z) + self.cfg.cloth.start_height],
                device=self.device,
            )
            + self.scene.env_origins.unsqueeze(1)
        )
        span = (start - self.fold_targets_w()).norm(dim=-1).amax(dim=-1).clamp(min=1e-6)
        frac = (1.0 - self.fold_error() / span).clamp(0.0, 1.0)
        return self._by_task(unfold_frac, frac) if self._mixed else frac
