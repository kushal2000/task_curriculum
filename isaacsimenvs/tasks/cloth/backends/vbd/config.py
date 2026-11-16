"""Newton configuration for the deformable cloth task (Isaac Lab 3).

ClothCfg supplies the shared task settings; NewtonSolverCfg configures the robot,
and DeformableClothCfg configures the cloth and their coupling. Fabric presets live in
fabrics.py. Historical tuning rationale is preserved in physics_notes.md.
"""

from __future__ import annotations

# IMPORT ORDER IS LOAD-BEARING: the compat shim re-asserts `isaaclab.utils.configclass` as a function
# after `isaaclab_physx` shadows it with its same-named submodule.
from isaacsimenvs.tasks.cloth.backends.vbd.newton_support import compat  # isort: skip

compat.install_isaaclab3_compat()

from isaaclab.sim import SimulationCfg  # noqa: E402
from isaaclab.utils.configclass import configclass  # noqa: E402

from isaacsimenvs.tasks.cloth.backends.robot import ROBOT  # noqa: E402
from isaacsimenvs.tasks.cloth.cloth_cfg import ClothCfg  # noqa: E402

__all__ = ["DeformableClothCfg", "DeformableClothEnvCfg", "NewtonSolverCfg"]


@configclass
class NewtonSolverCfg:
    """MJWarp / Newton settings tuned for the reference robot task. See physics_notes.md for measurements."""

    # --- The decisive one -------------------------------------------------------------------
    max_triangle_pairs: int | None = None
    """Global narrow-phase triangle-pair capacity; None sizes it from the env count.
    Undersizing silently drops contacts."""

    per_env_triangle_pairs: int = 65_536
    """Automatic triangle-pair capacity per env, including headroom above measured demand."""

    # --- Constraint budgets, same failure mode ------------------------------------------------
    njmax: int = 8192
    """Constraint rows per world. Undersizing drops constraints."""

    nconmax: int = 2048
    """Max contact points per world; same failure mode. The reference ran 800 while the solver's
    own message asked for 1081."""

    # --- Contact model, from the task authors' own MuJoCo scene -------------------------------
    cone: str = "elliptic"
    """Friction cone. MuJoCo's default is ``pyramidal``, which approximates the friction limit
    surface with a box and under combined slip lets a held tool rotate more readily. The
    SimToolReal authors' own MuJoCo model of this robot uses elliptic."""

    impratio: float = 10.0
    """Frictional-to-normal constraint impedance. The authors use 10, not MuJoCo's default 1; a
    higher ratio stiffens frictional constraints and stops a grasped tool creeping."""

    condim: int = 6
    """Contact dimensions: 6 enables torsional and rolling friction. Set before solver compilation."""

    # --- Pipeline -----------------------------------------------------------------------------
    use_mujoco_contacts: bool = False
    """Use MuJoCo contacts. False keeps Newton per-world mesh collision data."""

    solver: str = "newton"
    integrator: str = "implicitfast"
    iterations: int = 100
    ls_iterations: int = 15
    ccd_iterations: int = 35
    update_data_interval: int = 2
    rigid_contact_max: int = 4_000_000
    num_substeps: int = 2
    collision_decimation: int = 0
    """Re-collide every N substeps; 0 = once per tick. Coupled to :attr:`num_substeps` -- raising
    substeps without also re-colliding more often leaves the solver integrating several steps
    against one stale collision update."""

    use_cuda_graph: bool = True
    disable_sensors: bool = True
    """Required for ``deterministic_mode="run_to_run"``, and safe here: every observation field is
    read through Isaac Lab's articulation and rigid-body state, never from MuJoCo sensors."""

    deterministic_mode: str = "not_guaranteed"

    def resolve_max_triangle_pairs(self, num_envs: int) -> int:
        """Triangle-pair budget for ``num_envs``, honouring an explicit override."""
        if self.max_triangle_pairs is not None:
            return int(self.max_triangle_pairs)
        return max(1_000_000, int(num_envs) * self.per_env_triangle_pairs)



@configclass
class DeformableClothCfg:
    """The VBD cloth: geometry, material, plasticity, and the coupled MJWarp + VBD solve."""

    # --- geometry -------------------------------------------------------------------------
    size: float = 0.10
    """Side length of the square cloth [m]."""

    thickness: float = 0.016
    """Cloth-rigid collision diameter [m], equal to twice particle_radius.
    Self-contact uses self_contact_radius independently; exceeding grid spacing only warns."""

    resolution: int = 7
    """Particles per side: resolution**2 particles and 2*(resolution-1)**2 triangles.
    An odd resolution places a particle row on the fold crease."""

    start_height: float = 0.30
    """Spawn height above the episode table centre [m].
    The collision surface is 0.150 m above that centre; include particle radius and clearance."""

    # --- material -------------------------------------------------------------------------
    density: float = 2.0
    """Surface mass density [kg/m²], passed unchanged to Newton add_cloth_mesh."""
    tri_ke: float = 1.0e2
    """Membrane shear modulus (in-plane stretch stiffness)."""
    tri_ka: float = 1.0e2
    """In-plane shear/area stiffness -- the Lame ``lambda`` (area modulus). SIM1 Table 2.

    Note their own solver comment recommends ``lambda: 1000.0+`` to maintain area against
    penetration, and then their env ships 1e2. Table 2 agrees with the env, so 1e2 it is."""
    tri_kd: float = 1.5e-6
    """In-plane damping. SIM1 Table 2 (was 1.0e-5 here).

    Newton's ``default_tri_kd`` is 10.0; both this and the previous value are far below it, which
    is right for a membrane that has to drape rather than settle."""
    edge_ke: float = 8.0e-4
    """Dihedral bending stiffness; its effect depends on mesh edge length."""
    edge_kd: float = 1.0e-3
    """Bending damping. SIM1 Table 2 (was 1.0e-2 here)."""

    plastic_yield_curvature: float = 0.0
    """Yield curvature [1/m]; 0 disables plasticity.
    Bends past curvature * hinge width update rest angles; episode resets restore flat angles."""

    plastic_set_curvature: float = 0.0
    """Two-stage plasticity [1/m], 0 = off (single-stage, every result before 2026-11-10). A hinge
    gains set only while bent past ``plastic_set_curvature x hinge width`` (16.7 mm along the grid);
    gentler bends -- a light touch, a hanging drape -- stay elastic, and set can always shrink, so
    unfolding still irons a crease out (``utils/plasticity.py``)."""

    # --- coupling (defaults carried from the cable result) --------------------------------
    proxy_mode: str = "staggered"
    """See `docs/phase4_cable_env.md`. `lagged` syncs the hand's *begin* pose with its *end*
    velocities; the mismatch scales with hand speed and was the single change that took the cable
    from ~0 to 5 goals in an episode."""

    substeps: int = 6
    """VBD substeps per coupled step. Cloth dt = sim.dt / newton.num_substeps / substeps."""

    vbd_iterations: int = 12
    """VBD iterations per cloth substep, affecting contact quality and runtime.
    Reference measurements and the contact-convergence tradeoff are in physics_notes.md."""
    coupler_iterations: int = 1
    """Raising this made the cable *worse* at every setting tried."""

    mass_scale: float = 1.0
    """KEEP AT 1.0. 0.05 was ~500x worse on the cable and NaN'd the robot."""

    collide_interval: int = 1

    use_cuda_graph: bool = False
    """Capture the coupled solver step as a CUDA graph; disabled for baseline reproducibility."""

    soft_contact_ke: float = 5.0e2
    """Body-particle contact stiffness [N/m]. SIM1 Table 2 (was 8.0e3 here).

    The per-contact value is the AVERAGE of this and the rigid shape's own material stiffness
    (`NewtonShapeCfg.ke`, 2.5e3), so the effective ceiling moves 5250 -> 1500."""

    soft_contact_kd: float = 5.0e-3
    """Body-particle contact damping [N*s/m]. SIM1 Table 2.

    NEW FIELD. `NewtonModelCfg` defaults it to 1.0e-2 and `build_physics` was passing only `ke` and
    `mu`, so this ran at the library default and was never stated."""

    soft_contact_mu: float = 0.25
    """Body-particle friction parameter; effective friction is sqrt(soft_contact_mu * shape_mu)."""

    self_contact_mu: float = 0.0
    """Cloth-on-cloth friction; 0 falls back to soft_contact_mu.
    A positive override rescales cloth-view shape friction to preserve cloth-rigid friction."""

    table_contact_mu: float = 0.0
    """Effective cloth-on-table friction coefficient. 0 = keep ``sqrt(soft_contact_mu *
    table_friction)`` (0.36 for the measured presets). Applied in the cloth solver's view only, on
    its BOX shapes (the table); the rigid solver's table friction is unchanged."""

    friction_epsilon: float = 0.0
    """``SolverVBD(friction_epsilon=...)`` [m/s]: relative tangential speed below which friction is
    smoothed into a stiff spring (stick). 0 = Newton's default (1e-2). Smaller = less creep under a
    sustained tangential load, and a stiffer friction Hessian."""

    rigid_contact_k_start: float = 1.0e2
    """Initial AVBD body-particle penalty [N/m]; ignored unless rigid_avbd_beta > 0."""

    rigid_avbd_beta: float = 5.0e5
    """AVBD body-particle penalty ramp rate [N/m per m of penetration]; 0 disables ramping.
    Finite iteration counts can leave ramped contacts softer than fixed-stiffness contacts."""

    # --- self-contact ---------------------------------------------------------------------
    enable_self_contact: bool = True
    """Enable cloth self-collision so a folded flap contacts the stationary half."""

    self_contact_radius: float = 0.002
    """Self-contact distance between mesh primitives [m], independent of particle_radius.
    This can differ from the ply gap used by fold targets."""

    self_contact_margin: float = 0.003
    """Detection margin for self-contact [m]. SIM1 Table 2, and 1.5x the radius, which is the ratio
    Newton's own error message asks for. Must be >= `self_contact_radius` or the solver raises."""

    self_contact_detection_interval: int = -1
    """How often self-contact detection runs. ``-1`` = once before initialisation, which is both
    the `VBDSolverCfg` default and what SIM1 uses."""

    conservative_bound_relaxation: float = 0.42
    """Penetration-free displacement bound relaxation; used only with self-contact enabled."""

    proxy_links: str = "hand"
    """Which hand links the cloth can feel. `tips` (five distal phalanges) makes an enclosing grasp
    geometrically impossible; a pinch grasp is arguably enough to drag a cloth, but the asymmetry
    against the rigid baseline is not worth reintroducing."""

    proxy_table: bool = True
    """The table must be in the proxy set or the cloth falls through it."""

    full_surface_contact: bool = False
    """Enable proxy contacts with cloth edges and triangle interiors as well as particles.
    Participating mesh shapes need volume SDFs, provisioned by DeformableClothEnv."""

    full_surface_sdf_resolution: int = 0
    """``max_resolution`` of the provisioned SDFs (multiple of 8); 0 = Newton's default."""

    def proxy_body_patterns(self, robot_spec=None) -> list[str]:
        """Body-path regexes of the rigid shapes the cloth feels (the proxy set).

        ``robot_spec`` names the hand bodies (``robot.py``'s robot when omitted).
        """
        robot_spec = robot_spec or ROBOT
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

    # --- buffers ---------------------------------------------------------------------------
    rigid_body_particle_contact_buffer_size: int = 65536
    """Body-particle contact capacity PER RIGID BODY; do not multiply by env count."""
    rigid_body_contact_buffer_size: int = 16384
    """Body-to-body contact capacity per rigid body (Newton default 64). Also per-body: the failure
    message is literally "Per-body rigid contact buffer overflowed 87 > 64"."""
    per_particle_triangle_pairs: int = 2048
    """Triangle-pair capacity per cloth particle; total budget scales with particles and envs."""

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
        """Finest grid whose particle collision spheres do not overlap at rest."""
        return max(2, int(self.size / self.thickness) + 1)

    def check_thickness(self) -> None:
        """Warn about overlapping cloth-rigid collision spheres when thickness exceeds spacing.
        This is not a self-collision constraint: self-contact uses its own radius."""
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
        """Build the coupled robot/cloth solver configuration."""
        from .solver import build_physics

        return build_physics(self, newton_cfg, num_envs, robot_spec)

    def triangle_pair_budget(self, newton_cfg, num_envs: int) -> int:
        """Global triangle-pair capacity, scaled by both env count and particle count with 50% headroom."""
        base = newton_cfg.resolve_max_triangle_pairs(num_envs)
        per_env = self.num_particles * self.per_particle_triangle_pairs
        return int(max(base, num_envs * per_env) * 1.5)


@configclass
class DeformableClothEnvCfg(ClothCfg):
    """Fold a VBD cloth with the Flexiv + right Wuji, on a coupled MJWarp + VBD solve."""

    backend: str = "vbd"
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 120.0, render_interval=2, gravity=(0.0, 0.0, -9.81))
    """``sim.physics`` is set at construction from ``newton`` + ``deformable_cloth`` (``DeformableClothCfg.build_physics``)."""
    newton: NewtonSolverCfg = NewtonSolverCfg()
    use_gravcomp: bool = True
    """The robot's gravity-free bodies as MuJoCo gravity compensation (global gravity stays -9.81)."""
    deformable_cloth: DeformableClothCfg = DeformableClothCfg()
