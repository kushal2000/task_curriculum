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

from isaacsimenvs.tasks.cloth.fold_pool_cfg import FoldPoolCfg  # noqa: E402

__all__ = ["FoldPoolCfg", "FoldPoolEnvCfg"]


@configclass
class FoldPoolEnvCfg(RigidClothEnvCfg):
    """Rigid-cloth task with a per-episode random shape from the stage-2 pool."""

    fold_pool: FoldPoolCfg = FoldPoolCfg()
