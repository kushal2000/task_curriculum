"""Procedural multi-link cartpole URDF generation.

Pure `xml.etree` — no torch, no Isaac imports — so `tests/test_generate_cartpole.py`
exercises it without booting Kit.

The emitted articulation is always built with the **full** `n_max` pole links:

    rail (fixed base)
      └─ slider_to_cart      prismatic, axis +x
           └─ cart           box
                └─ pole_joint_0   revolute, axis +y
                     └─ pole_0    cylinder, length L[0], +z
                          └─ pole_joint_1 …
                               └─ pole_{n_max-1}

Every env spawns this same articulation, so PhysX sees one uniform DOF count and Isaac
Lab's single `Articulation` view is valid. Difficulty is *not* expressed by generating
different geometries — it is expressed at runtime by locking a subset of the pole joints
(see `reset_utils.apply_difficulty`), which fuses consecutive links into one rigid
segment. Locking joints 1..n_max-1 yields a single rigid pendulum of length `sum(L)`;
unlocking them all yields a full `n_max`-link chain of the *same total length*.

That is what makes the difficulty ladder clean: total length, total mass and total
inertia are invariant across the curriculum, so the only thing that varies is the number
of unactuated degrees of freedom — the actual source of difficulty.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence

__all__ = [
    "CART_JOINT_NAME",
    "POLE_JOINT_PREFIX",
    "pole_joint_name",
    "generate_cartpole_urdf",
]

CART_JOINT_NAME = "slider_to_cart"
POLE_JOINT_PREFIX = "pole_joint_"
_RAIL_LINK = "rail"
_CART_LINK = "cart"
_POLE_LINK_PREFIX = "pole_"


def pole_joint_name(index: int) -> str:
    return f"{POLE_JOINT_PREFIX}{index}"


def _inertial(parent: ET.Element, mass: float, ixx: float, iyy: float, izz: float, com_z: float) -> None:
    inertial = ET.SubElement(parent, "inertial")
    ET.SubElement(inertial, "origin", xyz=f"0 0 {com_z}", rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=f"{mass}")
    ET.SubElement(
        inertial,
        "inertia",
        ixx=f"{ixx}", ixy="0", ixz="0", iyy=f"{iyy}", iyz="0", izz=f"{izz}",
    )


def _box_link(
    robot: ET.Element, name: str, size: tuple[float, float, float], mass: float, rgba: str
) -> None:
    lx, ly, lz = size
    link = ET.SubElement(robot, "link", name=name)
    for tag in ("visual", "collision"):
        node = ET.SubElement(link, tag)
        ET.SubElement(node, "origin", xyz="0 0 0", rpy="0 0 0")
        geom = ET.SubElement(node, "geometry")
        ET.SubElement(geom, "box", size=f"{lx} {ly} {lz}")
        if tag == "visual":
            material = ET.SubElement(node, "material", name=f"{name}_color")
            ET.SubElement(material, "color", rgba=rgba)
    # Solid cuboid about its own centre.
    _inertial(
        link,
        mass,
        ixx=mass * (ly * ly + lz * lz) / 12.0,
        iyy=mass * (lx * lx + lz * lz) / 12.0,
        izz=mass * (lx * lx + ly * ly) / 12.0,
        com_z=0.0,
    )


def _cylinder_pole_link(
    robot: ET.Element, name: str, length: float, radius: float, mass: float, rgba: str
) -> None:
    """A pole link whose base sits at the link origin and extends along +z."""
    link = ET.SubElement(robot, "link", name=name)
    for tag in ("visual", "collision"):
        node = ET.SubElement(link, tag)
        # Cylinder geometry is centred on its own origin, so shift it up by half.
        ET.SubElement(node, "origin", xyz=f"0 0 {length / 2.0}", rpy="0 0 0")
        geom = ET.SubElement(node, "geometry")
        ET.SubElement(geom, "cylinder", length=f"{length}", radius=f"{radius}")
        if tag == "visual":
            material = ET.SubElement(node, "material", name=f"{name}_color")
            ET.SubElement(material, "color", rgba=rgba)
    # Solid cylinder about its own centre of mass (which is at length/2).
    ixx = mass * (3.0 * radius * radius + length * length) / 12.0
    _inertial(link, mass, ixx=ixx, iyy=ixx, izz=0.5 * mass * radius * radius, com_z=length / 2.0)


def generate_cartpole_urdf(
    out_path: str | Path,
    link_lengths: Sequence[float],
    *,
    rail_length: float = 6.0,
    cart_size: tuple[float, float, float] = (0.3, 0.2, 0.15),
    cart_mass: float = 1.0,
    pole_radius: float = 0.02,
    pole_density: float = 800.0,
    pole_joint_limit: float = 3.14159265,
    pole_joint_damping: float = 0.0,
    cart_effort_limit: float = 400.0,
    cart_velocity_limit: float = 20.0,
) -> Path:
    """Write a cartpole URDF with ``len(link_lengths)`` serially-jointed pole links.

    Args:
        out_path: Destination ``.urdf``. Parent directories are created.
        link_lengths: Length of each pole link, base → tip. Its length sets ``n_max``.
        pole_density: Used with ``pole_radius`` to derive each link's mass from its
            volume, so a long link is correctly heavier than a short one.

    Returns:
        The path written.
    """
    lengths = [float(v) for v in link_lengths]
    if not lengths:
        raise ValueError("link_lengths must contain at least one link.")
    if any(v <= 0.0 for v in lengths):
        raise ValueError(f"All link lengths must be positive, got {lengths}.")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    robot = ET.Element("robot", name="multilink_cartpole")

    # --- Rail: the fixed base the cart slides along. -------------------------
    rail = ET.SubElement(robot, "link", name=_RAIL_LINK)
    visual = ET.SubElement(rail, "visual")
    ET.SubElement(visual, "origin", xyz="0 0 0", rpy="0 0 0")
    geom = ET.SubElement(visual, "geometry")
    ET.SubElement(geom, "box", size=f"{rail_length} 0.05 0.05")
    material = ET.SubElement(visual, "material", name="rail_color")
    ET.SubElement(material, "color", rgba="0.3 0.3 0.35 1.0")
    # No collision on the rail — the cart is constrained by its prismatic joint, and a
    # rail collider would just fight that constraint.
    _inertial(rail, mass=1.0, ixx=1e-3, iyy=1e-3, izz=1e-3, com_z=0.0)

    # --- Cart ---------------------------------------------------------------
    _box_link(robot, _CART_LINK, cart_size, cart_mass, rgba="0.2 0.45 0.8 1.0")
    slider = ET.SubElement(robot, "joint", name=CART_JOINT_NAME, type="prismatic")
    ET.SubElement(slider, "parent", link=_RAIL_LINK)
    ET.SubElement(slider, "child", link=_CART_LINK)
    ET.SubElement(slider, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(slider, "axis", xyz="1 0 0")
    ET.SubElement(
        slider,
        "limit",
        lower=f"{-rail_length / 2.0}",
        upper=f"{rail_length / 2.0}",
        effort=f"{cart_effort_limit}",
        velocity=f"{cart_velocity_limit}",
    )
    ET.SubElement(slider, "dynamics", damping="0.0", friction="0.0")

    # --- Pole chain ---------------------------------------------------------
    parent_link = _CART_LINK
    # First pole joint sits on top of the cart; later ones sit at the tip of the
    # preceding link.
    joint_offset_z = cart_size[2] / 2.0
    for idx, length in enumerate(lengths):
        link_name = f"{_POLE_LINK_PREFIX}{idx}"
        mass = pole_density * 3.14159265 * pole_radius * pole_radius * length
        # Shade the chain base → tip so the segment structure is readable on video.
        shade = 0.25 + 0.6 * (idx / max(1, len(lengths) - 1)) if len(lengths) > 1 else 0.55
        _cylinder_pole_link(
            robot, link_name, length, pole_radius, mass, rgba=f"0.85 {shade:.3f} 0.2 1.0"
        )

        joint = ET.SubElement(robot, "joint", name=pole_joint_name(idx), type="revolute")
        ET.SubElement(joint, "parent", link=parent_link)
        ET.SubElement(joint, "child", link=link_name)
        ET.SubElement(joint, "origin", xyz=f"0 0 {joint_offset_z}", rpy="0 0 0")
        ET.SubElement(joint, "axis", xyz="0 1 0")
        # Explicit (rather than `continuous`) so the DOF has position limits defined in
        # USD — `reset_utils` narrows them per-env to lock a joint, and PhysX can only
        # narrow a limit that already exists.
        ET.SubElement(
            joint,
            "limit",
            lower=f"{-pole_joint_limit}",
            upper=f"{pole_joint_limit}",
            effort="0.0",
            velocity="100.0",
        )
        ET.SubElement(joint, "dynamics", damping=f"{pole_joint_damping}", friction="0.0")

        parent_link = link_name
        joint_offset_z = length

    ET.ElementTree(robot).write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path
