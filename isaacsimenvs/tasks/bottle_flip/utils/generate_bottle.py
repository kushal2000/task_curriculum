"""Procedural bottle URDF generation and the fill-level inertia model.

Pure `xml.etree` + arithmetic — no torch, no Isaac imports — so
`tests/test_generate_bottle.py` runs without Kit.

`bottle_inertial_properties` is written with nothing but `+ - * /` and `pi`, which means
the *same* function serves two callers: the URDF writer (Python floats) and
`reset_utils.apply_fill_level` (batched torch tensors). One implementation, so the
baked-in geometry and the runtime fill level can never drift apart.

The bottle is one rigid link — a full cylinder for visual and collision — whose mass,
centre of mass and inertia are assembled from three components:

    wall     hollow tube,   centred at z = 0
    base     solid disc,    at the bottom
    liquid   solid cylinder, resting on the base, height = fill x usable height

Centre-of-mass height is therefore **not** monotonic in fill level. An empty bottle's CoM
sits near its geometric centre (the wall dominates); adding a little liquid drags the CoM
down; filling it further pushes the CoM back up. For the default 0.22 m x 0.035 m bottle
the CoM bottoms out around 24% full — which is the familiar result that a bottle about
one-third full is the easiest to land, recovered here from the mass distribution alone.

That means difficulty must be mapped through the CoM, not through fill directly:
`reset_utils.apply_difficulty` interpolates fill over ``[fill_easy, fill_hard]`` with
``fill_easy`` at the low-CoM end, a range over which CoM height *is* monotonic.

What is not modelled is sloshing — the liquid is rigid, so in-flight redistribution and
the damping it provides on landing are both absent. Bottles here are harder to land than
real ones, uniformly, which is a level shift rather than a reordering of difficulty.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

__all__ = [
    "BOTTLE_ROOT_LINK",
    "bottle_inertial_properties",
    "generate_bottle_urdf",
    "generate_bottle_urdfs",
]

_PI = 3.141592653589793

# All procedural bottle URDFs share this link name so that MultiUsdFileCfg-spawned prims
# have an identical internal structure across envs — see the note on `_OBJECT_ROOT_LINK`
# in `tasks/play/utils/generate_objects.py`: RigidObject derives its view regex from
# env_0 and applies it everywhere, so a varying link name silently drops envs.
BOTTLE_ROOT_LINK = "object_root"


def bottle_inertial_properties(
    height,
    radius,
    wall_thickness,
    fill_fraction,
    shell_density: float = 950.0,
    liquid_density: float = 1000.0,
):
    """Mass, centre-of-mass height, and inertia about the CoM for a partly filled bottle.

    Works elementwise, so `height`/`radius`/`fill_fraction` may be floats or torch
    tensors of a common shape.

    The link frame is the bottle's geometric centre: z runs from ``-height/2`` (base) to
    ``+height/2`` (mouth), so ``com_z`` is negative for any real fill level.

    Returns:
        ``(mass, com_z, ixx, iyy, izz)``. ``ixx == iyy`` by symmetry; both are taken
        about the CoM, which is what URDF `<inertial>` expects alongside an offset
        `<origin>`.
    """
    r_in = radius - wall_thickness

    # --- wall: hollow tube, centred on the link origin ---
    m_wall = shell_density * _PI * (radius * radius - r_in * r_in) * height
    z_wall = 0.0 * height  # keeps the expression tensor-shaped when inputs are tensors
    izz_wall = 0.5 * m_wall * (radius * radius + r_in * r_in)
    ixx_wall = 0.25 * m_wall * (radius * radius + r_in * r_in) + m_wall * height * height / 12.0

    # --- base: solid disc at the bottom ---
    m_base = shell_density * _PI * radius * radius * wall_thickness
    z_base = -height / 2.0 + wall_thickness / 2.0
    izz_base = 0.5 * m_base * radius * radius
    ixx_base = (
        0.25 * m_base * radius * radius + m_base * wall_thickness * wall_thickness / 12.0
    )

    # --- liquid: solid cylinder resting on the base ---
    fill_height = fill_fraction * (height - wall_thickness)
    m_liq = liquid_density * _PI * r_in * r_in * fill_height
    z_liq = -height / 2.0 + wall_thickness + fill_height / 2.0
    izz_liq = 0.5 * m_liq * r_in * r_in
    ixx_liq = 0.25 * m_liq * r_in * r_in + m_liq * fill_height * fill_height / 12.0

    mass = m_wall + m_base + m_liq
    com_z = (m_wall * z_wall + m_base * z_base + m_liq * z_liq) / mass

    # Parallel-axis onto the combined CoM for the transverse axes; the z axis is shared
    # by all three components, so izz just adds.
    d_wall = z_wall - com_z
    d_base = z_base - com_z
    d_liq = z_liq - com_z
    ixx = (
        ixx_wall + m_wall * d_wall * d_wall
        + ixx_base + m_base * d_base * d_base
        + ixx_liq + m_liq * d_liq * d_liq
    )
    izz = izz_wall + izz_base + izz_liq
    return mass, com_z, ixx, ixx, izz


def generate_bottle_urdf(
    out_path: str | Path,
    *,
    height: float,
    radius: float,
    wall_thickness: float = 0.002,
    fill_fraction: float = 0.35,
    shell_density: float = 950.0,
    liquid_density: float = 1000.0,
) -> Path:
    """Write a single-link bottle URDF with the given nominal fill level.

    The fill baked in here is only the starting point: when
    `cfg.assets.runtime_fill` is on, `reset_utils.apply_fill_level` overwrites mass, CoM
    and inertia per-env at every reset, so the same USD serves the whole fill range.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mass, com_z, ixx, iyy, izz = bottle_inertial_properties(
        height, radius, wall_thickness, fill_fraction, shell_density, liquid_density
    )

    robot = ET.Element("robot", name="bottle")
    link = ET.SubElement(robot, "link", name=BOTTLE_ROOT_LINK)

    for tag in ("visual", "collision"):
        node = ET.SubElement(link, tag)
        ET.SubElement(node, "origin", xyz="0 0 0", rpy="0 0 0")
        geom = ET.SubElement(node, "geometry")
        ET.SubElement(geom, "cylinder", length=f"{height}", radius=f"{radius}")
        if tag == "visual":
            material = ET.SubElement(node, "material", name="bottle_color")
            ET.SubElement(material, "color", rgba="0.25 0.55 0.85 0.85")

    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", xyz=f"0 0 {com_z}", rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=f"{mass}")
    ET.SubElement(
        inertial, "inertia",
        ixx=f"{ixx}", ixy="0", ixz="0", iyy=f"{iyy}", iyz="0", izz=f"{izz}",
    )

    ET.ElementTree(robot).write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path


def generate_bottle_urdfs(
    out_dir: str | Path,
    *,
    num_variants: int,
    height_range: tuple[float, float],
    aspect_ratio_range: tuple[float, float],
    wall_thickness: float = 0.002,
    nominal_fill_fraction: float = 0.35,
    shell_density: float = 950.0,
    liquid_density: float = 1000.0,
) -> tuple[list[Path], list[tuple[float, float]]]:
    """Emit a pool of bottle URDFs spanning the height and aspect-ratio ranges.

    Isaac Lab's `MultiUsdFileCfg` hands env *i* variant ``i % len(pool)``, so a pool
    swept deterministically across the range gives every env a different bottle while
    keeping the assignment reproducible across runs. Shape is fixed once the USD is
    baked — it is a per-env, per-*stage* knob. Fill level is the axis that varies per
    *episode*; see `bottle_inertial_properties`.

    Returns:
        ``(urdf_paths, shapes)`` where each shape is ``(height, radius)``.
    """
    if num_variants < 1:
        raise ValueError(f"num_variants must be >= 1, got {num_variants}.")

    out_dir = Path(out_dir)
    paths: list[Path] = []
    shapes: list[tuple[float, float]] = []
    for i in range(num_variants):
        # Deterministic sweep rather than random sampling: the pool is reproducible, and
        # `experiments/run_curriculum.sh` compares arms that must see the same bottles.
        t = 0.0 if num_variants == 1 else i / (num_variants - 1)
        height = height_range[0] + t * (height_range[1] - height_range[0])
        aspect = aspect_ratio_range[0] + t * (aspect_ratio_range[1] - aspect_ratio_range[0])
        radius = height / (2.0 * aspect)
        if radius <= wall_thickness:
            raise ValueError(
                f"Variant {i} has radius {radius:.4f} m, which is not thicker than the "
                f"{wall_thickness} m wall. Lower aspect_ratio_range or raise height_range."
            )
        path = generate_bottle_urdf(
            out_dir / f"bottle_{i:03d}.urdf",
            height=height,
            radius=radius,
            wall_thickness=wall_thickness,
            fill_fraction=nominal_fill_fraction,
            shell_density=shell_density,
            liquid_density=liquid_density,
        )
        paths.append(path)
        shapes.append((height, radius))
    return paths, shapes
