"""PhysX configs for the rigid-cloth (stage 1) and fold-pool (stage 2) tasks.

Isaac Sim 5.1 / Isaac Lab 2.3 (``.venv_isaacsim``), on top of ``PlayEnvCfg`` -- the Play task's own
PhysX config -- rather than ``PlayNewtonEnvCfg``. The task blocks (``cloth``, ``rigid_cloth``,
``fold_pool``) are the SAME classes the Newton envs use, so a hydra override means the same thing on
both simulators. The Newton solver block (``newton``, ``use_gravcomp``, ``meshify``) has no
counterpart here; ``physx_chain`` holds the few knobs that exist only on this side.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from isaacsimenvs.tasks.cloth.cloth_cfg import ClothCfg
from isaacsimenvs.tasks.cloth.fold_pool_cfg import FoldPoolCfg
from isaacsimenvs.tasks.cloth.rigid_cloth_cfg import RigidClothCfg
from isaacsimenvs.tasks.play.play_env_cfg import PlayEnvCfg

__all__ = ["FoldPoolPhysxEnvCfg", "PhysxChainCfg", "RigidClothPhysxEnvCfg"]


@configclass
class PhysxChainCfg:
    """How the chain's contacts are expressed in PhysX terms."""

    slat_friction_combine: str = "min"
    """Friction combine mode on every slat (plate) collider.

    Newton reproduces the cloth's per-pair friction with MJWarp geom priority (slat 1, fingertip 2).
    PhysX has no priority; it has combine modes, and when two differ the higher one wins
    (average < min < multiply < max). Slats at ``min`` with ``rigid_cloth.friction`` (0.354) feel
    min(0.354, 0.5) = 0.354 against the table and the robot links; fingertips at ``max`` with
    ``rigid_cloth.fingertip_friction`` (0.612) win against the slats (0.612) and against the table
    (max(0.612, 0.5) = 0.612) -- the same four pairs Newton's priority rule produces."""

    fingertip_friction_combine: str = "max"
    """Friction combine mode on the fingertip colliders; see ``slat_friction_combine``."""

    contact_offset_margin: float = 0.002
    """A slat's PhysX contact offset is its rest offset (the 7 mm ``rigid_cloth.contact_margin``) plus
    this [m]. PhysX starts generating contacts at the contact offset and holds them at the rest
    offset, so this is how early a contact is seen, not where the surface is."""

    solver_position_iterations: int = 8
    """The chain articulation's TGS position iterations. 8, as the robot's."""

    solver_velocity_iterations: int = 0
    """The chain articulation's TGS velocity iterations. 0, as the robot's."""

    hinge_armature: float = 1.0e-3
    """Hinge armature on PhysX [kg m^2], replacing ``rigid_cloth.joint_armature`` (1e-5); 0 keeps that.

    At 1e-5 -- about the plate's own inertia about its hinge -- PhysX's reduced-coordinate solve
    leaks a fingertip's squeeze into the hinge: pressing one plate flat on the table makes the OTHER
    plate jump 5-6 cm (Newton: no jump; the cloth: no lift). Policies trained from scratch on PhysX
    learned to fold with that jump and folded the cloth 3-6% of the time. At 1e-3 the untouched plate
    stays within 3 mm, passive physics (rest height, friction, hinge release, half fold) is
    unchanged, and stage 1 trained here folds the cloth as often as Newton-trained stage 1 (55% / 45%
    vs 52% / 43%). 3e-4 still leaks enough to halve that (24% pinned). See docs/results/physx_port.md."""

    slat_compliant_stiffness: float = 0.0
    """PhysX compliant-contact spring on every slat contact (table, robot, fingertips); 0 = rigid.

    Per unit mass when ``slat_compliant_acceleration`` [1/s^2], else [N/m]. MJWarp's contacts are
    soft constraints (solref/solimp) while PhysX's are rigid; this is the PhysX knob that softens a
    slat's contacts the same way, for calibrating how a fingertip pressing a plate behaves."""

    slat_compliant_damping: float = 0.0
    """Damping of the compliant contact above [1/s per unit mass, or N s/m]."""

    slat_compliant_acceleration: bool = True
    """Read ``slat_compliant_stiffness`` / ``_damping`` per unit mass (MuJoCo's solref is too)."""

    slat_max_depenetration_velocity: float = 0.0
    """Cap on how fast PhysX pushes a slat out of a penetration [m/s]; 0 = leave the converter's.

    A fingertip driven down onto a plate lying on the table squeezes it, and an uncapped
    depenetration shoots the plate out of the sandwich -- the "flick" a PhysX-trained policy learns
    to fold with, which neither a cloth nor the Newton chain allows."""

    gpu_patches_per_env: int = 48
    """PhysX GPU contact-patch buffer per env (``sim.physx.gpu_max_rigid_patch_count`` is raised to at
    least this times ``num_envs``). The pool needed ~16/env at 12288 envs before the hand touched
    anything; the default buffer (163840) overflowed there."""

    gpu_contacts_per_env: int = 256
    """Likewise for ``gpu_max_rigid_contact_count``."""

    gpu_pairs_per_env: int = 128
    """Likewise for ``gpu_found_lost_pairs_capacity`` and ``gpu_total_aggregate_pairs_capacity``."""

    size_quantum: float = 0.001
    """Fold pool only: per-env sheet sizes are quantised to this [m], and one USD per distinct size and
    shape is written. PhysX parses every env on its own (``replicate_physics: false``), so envs can
    simply reference differently sized sheets -- no per-world builder hook as on Newton."""


@configclass
class RigidClothPhysxEnvCfg(PlayEnvCfg):
    """The rigid-chain fold task on PhysX."""

    cloth: ClothCfg = ClothCfg()
    rigid_cloth: RigidClothCfg = RigidClothCfg()
    physx_chain: PhysxChainCfg = PhysxChainCfg()


@configclass
class FoldPoolPhysxEnvCfg(RigidClothPhysxEnvCfg):
    """The stage-2 fold-shape pool on PhysX."""

    fold_pool: FoldPoolCfg = FoldPoolCfg()
