"""FoldPoolCfg -- the stage-2 pool's block: which shapes, how often, how big.

Split out of ``fold_pool_env_cfg`` so the Newton and PhysX envs share one definition (see
``cloth_cfg``).
"""

from __future__ import annotations

from isaaclab.utils.configclass import configclass

from isaacsimenvs.tasks.cloth.utils import fold_shapes as fs

__all__ = ["FoldPoolCfg"]


@configclass
class FoldPoolCfg:
    shapes: list[str] = list(fs.DEFAULT_SHAPES)
    """Pool members, by ``fold_shapes.POOL`` name. Each env carries ALL of them every episode and
    uses one; the rest are parked. Order is the order of the per-shape logs."""

    weights: list[float] | None = None
    """Relative draw probabilities, one per shape. ``None`` is uniform."""

    fold_lift_from_ply_gap: bool = True
    """Lift the fold target by the POOL's ply gap (0 mm for mid-plane hinges) instead of the cloth's
    ``self_contact_radius`` (2 mm). Same switch as ``rigid_cloth.fold_lift_from_ply_gap``, for the
    pool; it must agree across the pool, since the target is built once for every env."""

    park_distance: float = 3.0
    """How far (+y, from the env origin) the parked shapes hover [m]. Only has to clear this env's
    own robot and table: worlds never collide, so another env's shapes are never in the way."""

    park_spacing: float = 0.3
    """Gap between consecutive parked shapes along +y [m]; a 100 mm sheet needs more than 0.1."""

    park_height: float = 2.0
    """Height of the parked shapes above the env origin [m]."""

    size_range: list[float] = [0.10, 0.10]
    """Sheet side length [m], drawn ONCE PER ENV at startup and fixed for the whole run.

    Not per episode: Newton bakes collision meshes at build time and ignores a later change of
    their scale, so each env's sheets are built at their own size by a per-world replication hook
    (``fold_shape_assets.scale_builder_world``). All five shapes in an env share its size. The task
    -- fold targets, keypoints, footprint, ``object_scales`` -- follows each env's size."""

    thickness_range: list[float] = [0.002, 0.002]
    """Plate (slab) thickness [m], drawn once per env like ``size_range``. NOT the thickness the
    hand feels: that is slab + 2 x ``rigid_cloth.contact_margin`` (2 + 2 x 7 = 16 mm by default),
    and the margin is already randomised PER EPISODE by ``rigid_cloth.randomization``."""

    size_seed: int = 0
    """Seed for the per-env size / thickness draws, so a run's sizes are reproducible."""

    reach_shift: list[float] = [0.0, 1.0]
    """Unit direction (env-frame x, y) toward the robot base. A sheet larger than ``cloth.size``
    has its reset centre moved this way by the growth of its half-diagonal, so its farthest point
    (at any yaw) stays where a ``cloth.size`` sheet's would be: the arm reaches only ~18 cm past the
    table centre on the far side. ``[0, 0]`` disables the shift."""

    one_shape_per_env: bool = False
    """PhysX only: each env carries ONE pool shape, fixed for the run, instead of all of them.

    Assigned round-robin (env ``i`` -> shape ``i % S``; ``weights`` -> a seeded draw), so every SAPG
    block sees every shape. The shape distribution is then per env rather than per episode, which
    with thousands of envs is the same distribution. 1.6-1.8x faster end to end at 24576 envs: the
    four parked articulations per env are gone (PhysX steps every articulation, parked or not; a
    single-shape pool runs 107-117k env-steps/s against 62-66k for all five). Newton cannot do this
    (its worlds must be identical) and refuses it."""

    init_fold_prob: float = 0.0
    """Share of episodes that start partly folded instead of flat; 0 = always flat.

    A folded start picks one of the active shape's SINGLE-STAGE named folds uniformly (the half fold
    included; ``box2-mid`` has only that one) and turns its hinges up by an angle drawn uniformly from
    ``init_fold_deg``. A single-stage fold is a rigid rotation about one crease line, so a shape with
    seams stays closed at any angle (checked at startup). Multi-stage folds are excluded: their first
    stage is a full 180 deg fold."""

    init_fold_deg: list[float] = [15.0, 120.0]
    """Range of the folded start's angle [deg], drawn uniformly per episode (per PART with
    ``init_fold_starts: other``). Positive = the moving plates lift off the table, so no start can
    press a plate into it. Up to 180 (fully folded) with ``other``; below 180 with ``single``,
    whose starts include the target fold itself."""

    init_fold_starts: str = "single"
    """Which folded starts.

    * ``single``: every single-stage named fold, the target half fold included, all its hinges at
      one angle (the 2026-10-06 runs).
    * ``other``: ``FoldShape.start_poses`` -- every legal combination of the NON-target single-stage
      folds (corners, strips, diagonals, bottom-up), each part at its own angle. A shape whose only
      fold is the target (``box2``) starts flat."""

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.init_fold_prob) <= 1.0:
            raise ValueError(f"fold_pool.init_fold_prob must be in [0, 1], got {self.init_fold_prob}")
        if self.init_fold_starts not in ("single", "other"):
            raise ValueError(f"fold_pool.init_fold_starts must be single or other, got {self.init_fold_starts!r}")
        lo, hi = (float(v) for v in self.init_fold_deg)
        top = 180.0 if self.init_fold_starts == "other" else 180.0 - 1e-6
        if not 0.0 <= lo <= hi <= top:
            raise ValueError(
                f"fold_pool.init_fold_deg must be 0 <= lo <= hi {'<=' if top == 180.0 else '<'} 180 "
                f"with init_fold_starts={self.init_fold_starts}, got {[lo, hi]}"
            )
        unknown = [n for n in self.shapes if n not in fs.POOL]
        if unknown:
            raise ValueError(f"unknown fold_pool.shapes {unknown}; known: {sorted(fs.POOL)}")
        if not self.shapes:
            raise ValueError("fold_pool.shapes is empty")
        if self.weights is not None:
            if len(self.weights) != len(self.shapes):
                raise ValueError(
                    f"fold_pool.weights has {len(self.weights)} entries for {len(self.shapes)} shapes"
                )
            if any(w < 0.0 for w in self.weights) or sum(self.weights) <= 0.0:
                raise ValueError("fold_pool.weights must be non-negative with a positive sum")
        for name in ("size_range", "thickness_range"):
            lo, hi = (float(v) for v in getattr(self, name))
            if not 0.0 < lo <= hi:
                raise ValueError(f"fold_pool.{name} must be 0 < lo <= hi, got {[lo, hi]}")
        if self.park_spacing <= 1.5 * float(self.size_range[1]):
            raise ValueError(
                f"park_spacing {self.park_spacing} m would let parked sheets of up to "
                f"{self.size_range[1]} m overlap"
            )
