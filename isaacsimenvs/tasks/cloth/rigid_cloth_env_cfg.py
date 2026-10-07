"""RigidClothEnvCfg — the cloth-folding task with a hinged rigid chain in place of the VBD sheet.

Subclasses :class:`~isaacsimenvs.tasks.cloth.cloth_env_cfg.ClothEnvCfg` and keeps its ``cloth:``
block, because that block is where the *task* is defined: ``size``, ``resolution`` and ``fold_axis``
fix the particle grid the fold is scored on, and ``start_height`` / ``table_half_thickness`` fix
where the sheet spawns. Only the manipuland's **dynamics** change, so only the solver and the
manipuland's own parameters are new.

**The solver topology collapses.** ``ClothEnvCfg`` builds a coupled MJWarp + VBD solve with a proxy
mapping, because a particle sheet and a rigid robot cannot be integrated by one solver. A chain of
hinged boxes is rigid, so it is MJWarp's to begin with: :meth:`RigidClothEnv.build_physics_cfg`
falls back to ``PlayNewtonEnv``'s plain ``cfg.newton.build(...)``. Everything in ``cloth:`` from
``proxy_mode`` down -- substeps, VBD iterations, self-contact radii, AVBD ramping, contact buffers --
is **inert here**, and is left in place rather than deleted so the two configs stay diffable and so
``ClothCfg``'s calibration notes remain the reference for what is being approximated.

**What is NOT inherited and has to be re-derived.** Three of the cloth's physical parameters do not
transfer, and each has a field below rather than a silent reuse:

  * *friction*. Newton's VBD mixes the two materials as ``sqrt(mu_a * mu_b)``; MJWarp takes
    ``max(mu_a, mu_b)``. Copying ``soft_contact_mu = 0.25`` onto a rigid slat therefore makes it as
    grippy as the grippiest thing it touches instead of slipperier than everything -- 1.5 against a
    fingertip where the cloth feels 0.61. See :func:`rigid_cloth.cloth_effective_friction` and
    ``slat_friction`` below.
  * *bending stiffness*. The cloth's bending lives in ``edge_ke`` / ``edge_kd`` on every mesh edge; a
    chain's lives in the hinges. URDF cannot express a spring, so stiffness is an actuator setting
    (``joint_stiffness``) and damping is in the URDF (``joint_damping``).
  * *where the plies come to rest*. The cloth settles folded plies at ``self_contact_radius`` = 2 mm;
    a chain settles them at its own ply gap, which is 5.88 mm for the mid-hinge uniform chains. The
    fold target is built from that number, so ``fold_lift_from_ply_gap`` decides whether a perfect
    fold scores zero for this variant or carries the cloth's offset as a constant bias.
"""

from __future__ import annotations

# IMPORT ORDER IS LOAD-BEARING; see the note in `cloth_env_cfg`.
from isaacsimenvs.tasks.cloth.cloth_env_cfg import ClothEnvCfg  # isort: skip

from isaaclab.utils import configclass  # noqa: E402

from isaacsimenvs.tasks.cloth.rigid_cloth_cfg import (  # noqa: E402
    RigidClothCfg,
    RigidClothRandomizationCfg,
)

__all__ = ["RigidClothCfg", "RigidClothEnvCfg", "RigidClothRandomizationCfg"]


@configclass
class RigidClothEnvCfg(ClothEnvCfg):
    """Play task with a hinged rigid chain approximating the folding cloth."""

    rigid_cloth: RigidClothCfg = RigidClothCfg()
