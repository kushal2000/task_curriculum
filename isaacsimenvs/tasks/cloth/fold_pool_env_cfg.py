"""FoldPoolEnvCfg -- stage 2: one policy, a pool of rigid fold shapes, a shape drawn per episode.

Subclasses :class:`RigidClothEnvCfg` and keeps all of it -- the task (``cloth:`` grid, fold axis,
tolerances), the chain's physical parameters (``rigid_cloth:`` density, damping, armature, friction,
contact margin, physics DR) -- and adds only which shapes are in the pool and how often each is
drawn. ``rigid_cloth.variant`` is inert here: the manipuland is the pool, not a chain.

The task is unchanged: fold the ``+x`` half over onto ``-x``, which every pool shape can do (that is
the pool's membership rule, ``fold_shapes.HALF_FOLD``). What changes is the fold TARGET height:
every pool shape has a mid-plane hinge and therefore a ply gap of 0, so with
``fold_lift_from_ply_gap`` (default) the target sits IN the stationary half's plane rather than the
cloth's 2 mm above it, and a geometrically perfect fold scores zero.
"""

from __future__ import annotations

# IMPORT ORDER IS LOAD-BEARING; see the note in `cloth_env_cfg`.
from isaacsimenvs.tasks.cloth.rigid_cloth_env_cfg import RigidClothEnvCfg  # isort: skip

from isaaclab.utils import configclass  # noqa: E402

from isaacsimenvs.tasks.cloth.utils import fold_shapes as fs  # noqa: E402

__all__ = ["FoldPoolCfg", "FoldPoolEnvCfg"]


@configclass
class FoldPoolCfg:
    shapes: list[str] = list(fs.POOL)
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

    def __post_init__(self) -> None:
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


@configclass
class FoldPoolEnvCfg(RigidClothEnvCfg):
    """Rigid-cloth task with a per-episode random shape from the stage-2 pool."""

    fold_pool: FoldPoolCfg = FoldPoolCfg()
