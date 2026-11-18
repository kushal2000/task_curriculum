"""ClothTrackerCfg -- observation DR that replays the real CoTracker pipeline in sim.

Kit-free (``configclass`` only), like ``cloth_cfg``. The pipeline itself is
``utils/cloth_tracker.py``; this block only holds its switches and ranges. Every range is drawn once
per episode unless its doc says otherwise. Lengths in metres, times in policy steps (1/60 s).
"""

from __future__ import annotations

from isaaclab.utils.configclass import configclass

__all__ = ["ClothTrackerCfg"]


@configclass
class ClothTrackerCfg:
    """The real pipeline: a stereo pair tracks a 7x7 grid inset from the sheet's edges; a point is
    fresh when both cameras see it; each half is fitted rigidly from its fresh points; if either
    half has too few (or too collinear) fresh points the whole sheet's estimate -- object and goal --
    holds its last value. The object is the moving half's corners, the goal the stationary half's
    fit mirrored across the crease (the sim's own fold target, from the estimate)."""

    enabled: bool = False

    # --- DR-0: the pipeline ------------------------------------------------------------------
    inset: float = 0.1
    """Tracked grid inset from each edge, as a fraction of the side: the 7x7 grid spans
    ``1 - 2 * inset`` of the sheet. Positions are bilinear in the particle grid."""
    n_min: int = 4
    """A half is fitted only with at least this many fresh points."""
    min_spread: float = 0.02
    """... and only when the fresh points' extent along their minor in-plane axis (rest frame) is
    at least this [m]: nearly collinear points leave the rotation about their line free."""

    # --- DR-1: cameras and hand occlusion ------------------------------------------------------
    cam_a_base: tuple[float, float, float] = (0.645, -0.963, 0.856)
    """Camera A in the robot base frame (left_camera_nov7). Env frame = base + ``robot_root_pos``."""
    cam_b_base: tuple[float, float, float] = (-0.525, -0.936, 0.876)
    """Camera B in the robot base frame (right_camera_nov14)."""
    cam_pos_noise: tuple[float, float, float] = (0.05, 0.05, 0.05)
    """Per-camera position offset, uniform in +-this per axis. Only the optical centre matters for
    occlusion (no field-of-view test), so orientation is not drawn."""
    hand_occlusion: bool = False
    """DR-1: a point is hidden from a camera when the ray to it passes within ``radius + margin`` of
    one of the hand / wrist / forearm capsules."""
    margin_range: tuple[float, float] = (0.0, 0.015)
    """Capsule inflation per episode (CoTracker also drops points next to an occluder's edge)."""
    finger_radius: float = 0.009
    palm_radius: float = 0.02
    wrist_radius: float = 0.04
    arm_radius: float = 0.05
    end_pullback: float = 0.005
    """The ray stops this short of the point, so a finger touching the sheet from BELOW, at the
    point, does not count as covering it."""

    # --- DR-1b: re-acquisition failure (applies to every occlusion source) ----------------------
    reacquire: bool = False
    reacquire_delay_s: tuple[float, float] = (0.0, 0.5)
    """After a point stops being covered it stays not-fresh for a uniform draw in this range [s]."""
    lost_prob_range: tuple[float, float] = (0.0, 0.5)
    """Per episode, the probability that a point uncovered again is lost for a long time instead
    (``lost_duration_s``)."""
    lost_duration_s: tuple[float, float] = (1.0, 3.0)
    """How long such a point stays lost [s], uniform. (Lost until the episode end left 85-90% of the
    points not fresh on a folding policy, while the real data has only 3-6% of the not-fresh points
    uncovered.)"""

    # --- DR-2: cloth self-occlusion ----------------------------------------------------------
    self_occlusion: bool = False
    """A point is hidden when the ray to it crosses the OTHER half's surface (the 7x7 particle
    grid's triangles), e.g. the stationary half under a folded flap."""
    self_occlusion_gap: float = 0.001
    """A hit counts only if it is at least this far before the point [m]."""

    # --- DR-3: per-point tracking error --------------------------------------------------------
    noise_std_range: tuple[float, float] = (0.0002, 0.001)
    """Isotropic white noise std per episode [m], drawn per point per camera FRAME (held frames
    repeat it)."""
    reacquire_bias_range: tuple[float, float] = (0.003, 0.005)
    """Per episode half-width b [m]; each time a point becomes fresh again its offset is redrawn
    uniform in +-b per axis (kept until it is next lost). Zero at the episode start."""

    # --- DR-4: calibration ---------------------------------------------------------------------
    calib_xyz: tuple[float, float, float] = (0.015, 0.015, 0.01)
    calib_rot_deg: float = 2.0
    """Per-episode rigid transform of every observed point (object and goal alike): rotation
    uniform in +-this about x, y and z, about the sheet's centre at the episode start."""

    # --- DR-5: frame rate and latency ----------------------------------------------------------
    frame_period_choices: tuple[int, ...] = (2, 3, 4)
    """A camera frame every k policy steps, k per episode, random phase. Between frames the
    estimate holds."""
    latency_choices: tuple[int, ...] = (2, 3)
    """The policy reads the estimate this many steps after its frame was captured."""

    # --- DR-6: size ----------------------------------------------------------------------------
    size_scale_range: tuple[float, float] = (0.95, 1.05)
    """The pipeline's assumed sheet size / the true one, per episode: scales the rest model the
    halves are fitted to, the observed keypoint offsets and ``object_scales`` (x, y)."""
