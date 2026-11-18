"""Read rigid-cloth configs saved before the thickness curriculum and the size grid were removed.

Old ``.hydra/config.yaml`` files (training runs up to 2026-11-18) have ``thickness_curriculum``,
``thickness_range`` and ``design_levels``; the code now has one ``thickness`` and
``num_geometries``. :func:`upgrade_rigid_cloth` rewrites such a ``rigid_cloth`` dict in place so
old checkpoints can still be evaluated and viewed. No simulator imports.
"""
from __future__ import annotations


def upgrade_rigid_cloth(rc: dict, checkpoint_thickness: float | None = None) -> list[str]:
    """Map removed keys onto the current ones (in place); returns a note per change.

    * ``thickness_curriculum`` (enabled) -> ``thickness`` = the checkpoint's curriculum thickness
      if given, else the curriculum start (what the run was training on);
    * ``thickness_range`` -> ``thickness`` = its first value (all old recipes fixed it);
    * ``design_levels`` L -> ``num_geometries`` = L * L (the old grid's distinct length x width sizes).
      The sizes themselves are re-drawn, not the old grid points.
    """
    notes = []
    curriculum = rc.pop("thickness_curriculum", None)
    thickness_range = rc.pop("thickness_range", None)
    if "thickness" not in rc:
        thickness = None
        if curriculum and curriculum.get("enabled"):
            thickness = checkpoint_thickness if checkpoint_thickness is not None else curriculum.get("start")
            notes.append(f"thickness_curriculum -> thickness {float(thickness):g} m "
                         f"({'checkpoint' if checkpoint_thickness is not None else 'curriculum start'})")
        elif thickness_range:
            thickness = thickness_range[0]
            notes.append(f"thickness_range {list(thickness_range)} -> thickness {float(thickness):g} m")
        if thickness is not None:
            rc["thickness"] = float(thickness)
    levels = rc.pop("design_levels", None)
    if levels is not None and "num_geometries" not in rc:
        rc["num_geometries"] = int(levels) ** 2
        notes.append(f"design_levels {levels} -> num_geometries {rc['num_geometries']} (sizes re-drawn)")
    return notes
