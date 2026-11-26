"""The fold-shape pool on kit-less Isaac Lab 3 + ovphysx: the PhysX pool's configuration, with its
``sim.physx`` block moved to ``physx_scene`` (Isaac Lab 3's ``SimulationCfg`` has no PhysX block;
``OvPhysxCfg`` carries only GPU buffers and two flags and reads the rest off the USD PhysicsScene)."""

from __future__ import annotations

from typing import Literal

from isaaclab.utils import configclass

from .rigid_cloth import RigidClothEnvCfg


@configclass
class Il2PhysxSceneCfg:
    """Isaac Lab 2.3's ``PhysxCfg`` fields WITH ITS DEFAULTS (isaaclab 2.3.2 sim/simulation_cfg.py).

    Not Isaac Lab 3's ``isaaclab_physx.PhysxCfg``: its defaults differ (e.g.
    ``enable_external_forces_every_iteration`` True there, False in 2.3), and every policy trained
    under 2.3's. task.yaml's ``rigid.sim.physx`` is applied on top; the backend authors the result
    as ``physxScene:*`` on the PhysicsScene prim, as 2.3's SimulationContext / Isaac Sim's
    PhysicsContext did."""

    solver_type: Literal[0, 1] = 1
    solve_articulation_contact_last: bool = False
    min_position_iteration_count: int = 1
    max_position_iteration_count: int = 255
    min_velocity_iteration_count: int = 0
    max_velocity_iteration_count: int = 255
    enable_ccd: bool = False
    enable_stabilization: bool = False
    enable_external_forces_every_iteration: bool = False
    enable_enhanced_determinism: bool = False
    bounce_threshold_velocity: float = 0.5
    friction_offset_threshold: float = 0.04
    friction_correlation_distance: float = 0.025
    gpu_max_rigid_contact_count: int = 2**23
    gpu_max_rigid_patch_count: int = 5 * 2**15
    gpu_found_lost_pairs_capacity: int = 2**21
    gpu_found_lost_aggregate_pairs_capacity: int = 2**25
    gpu_total_aggregate_pairs_capacity: int = 2**21
    gpu_collision_stack_size: int = 2**26
    gpu_heap_capacity: int = 2**26
    gpu_temp_buffer_capacity: int = 2**24
    gpu_max_num_partitions: int = 8
    gpu_max_soft_body_contacts: int = 2**20
    gpu_max_particle_contacts: int = 2**20


@configclass
class RigidOvPhysxClothEnvCfg(RigidClothEnvCfg):
    """The fold task on the fold-shape pool, on PhysX through Isaac Lab 3 + ovphysx (kit-less)."""

    physx_scene: Il2PhysxSceneCfg = Il2PhysxSceneCfg()
