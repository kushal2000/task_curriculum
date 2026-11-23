"""Cloth friction set per contact: ``cloth.cloth_{cloth,table,fingertip,robot}_friction``.

VBD has one rule: a cloth contact's friction is ``sqrt(cloth's mu x the other side's mu)``. The cloth's
mu is ONE number (``cloth.soft_contact_mu``), so on its own it moves every cloth contact together; the
other side's mu is the shape's (``assets.*_friction``), which MJWarp also uses for the hand against
the table. So without these settings the cloth's frictions and the robot's cannot be set apart.

To set a cloth contact directly, the cloth solver gets its own copy of the shapes' mu (MJWarp keeps the
model's), and the value to write is solved from the same rule:

* ``cloth_cloth_friction`` x: the cloth's mu becomes x (cloth-cloth = sqrt(x * x) = x). To keep the
  cloth's other contacts where they were, every shape mu in the copy is multiplied by
  ``soft_contact_mu / x``.
* ``cloth_table_friction`` / ``cloth_fingertip_friction`` / ``cloth_robot_friction`` y: those shapes'
  mu in the copy is ``y**2 / cloth's mu``, so sqrt(cloth's mu x shape's mu) = y. The robot is every
  robot shape that is not a fingertip (in practice the palm and phalanges: ``cloth.proxy_links``).

``null`` leaves a contact to the rule; with all four ``null`` nothing here runs.

When: the copy must exist before ``SolverVBD`` is built, so :func:`construction_hooks` is active during
``DirectRLEnv.__init__``; its values need the final shape mu (``apply_newton_materials``), and every
model refresh copies the model's values back over it, so :func:`install` writes them and re-writes
them after each refresh. Extends the scheme of kushal/2026-11-17_compare_with_yibo's
``install_cloth_friction`` (cloth-cloth and cloth-table only) to the hand.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from unittest.mock import patch

__all__ = ["CONTACTS", "cloth_mu", "construction_hooks", "install", "report", "used"]

#: The cloth-rigid contacts that can be set directly: (cfg field, shape group).
CONTACTS = (
    ("cloth_table_friction", "table"),
    ("cloth_fingertip_friction", "fingertip"),
    ("cloth_robot_friction", "robot"),
)

#: The table body, as the proxy set matches it (``physics.py``).
_TABLE_BODY = re.compile(r"/World/envs/env_[^/]+/Table(?:/|$)")
_ROBOT_BODY = re.compile(r"/World/envs/env_[^/]+/Robot/")


def used(cloth) -> bool:
    """Whether any cloth contact is set directly."""
    return cloth.cloth_cloth_friction is not None or any(getattr(cloth, f) is not None for f, _ in CONTACTS)


def cloth_mu(cloth) -> float:
    """The cloth's mu in every VBD contact: ``cloth_cloth_friction`` if set, else ``soft_contact_mu``."""
    if cloth.cloth_cloth_friction is not None:
        return float(cloth.cloth_cloth_friction)
    return float(cloth.soft_contact_mu)


@contextmanager
def construction_hooks(cloth):
    """Give the cloth solver its own copy of the shapes' mu before ``SolverVBD`` is built."""
    if not used(cloth):
        yield
        return
    from isaaclab_contrib.deformable.vbd_manager import NewtonVBDManager

    original_create = NewtonVBDManager.__dict__["_create_solver"]

    def create_with_own_copy(cls, model_view, solver_cfg):
        if not hasattr(model_view, "_cow_array"):
            raise RuntimeError("per-contact cloth friction needs the coupled solver (one model view per solver)")
        model_view._cow_array("shape_material_mu")  # copy-on-write: from here on the view has its own
        return original_create.__func__(cls, model_view, solver_cfg)

    with patch.object(NewtonVBDManager, "_create_solver", classmethod(create_with_own_copy)):
        yield


def _cloth_view(coupled_solver):
    import newton

    views = [e.view for e in coupled_solver._entries.values() if isinstance(e.solver, newton.solvers.SolverVBD)]
    if len(views) != 1:
        raise RuntimeError(f"expected one VBD solver in the coupled solver, found {len(views)}")
    return views[0]


def _shape_groups(model, fingertip_link_names) -> dict:
    """Boolean masks over the model's shapes: ``table``, ``fingertip`` (the distal links, matched as
    ``apply_newton_materials`` matches them) and ``robot`` (every other robot shape)."""
    import numpy as np

    labels = [str(label) for label in model.body_label]
    shape_body = model.shape_body.numpy()

    def mask(test):
        return np.array([b >= 0 and test(labels[b]) for b in shape_body], dtype=bool)

    table = mask(lambda label: _TABLE_BODY.search(label) is not None)
    fingertip = mask(lambda label: any(name in label.rsplit("/", 1)[-1] for name in fingertip_link_names))
    robot = mask(lambda label: _ROBOT_BODY.search(label) is not None) & ~fingertip
    return {"table": table, "fingertip": fingertip, "robot": robot}


def install(coupled_solver, cloth, fingertip_link_names) -> None:
    """Write the cloth's copy of the shapes' mu now and after every model refresh. Call before the
    ``notify_solver`` that follows ``apply_newton_materials``."""
    if not used(cloth):
        return
    import numpy as np
    import warp as wp

    view = _cloth_view(coupled_solver)
    model = view.parent
    if "shape_material_mu" not in view.overrides:
        raise RuntimeError("the cloth solver has no own copy of the shapes' mu (construction_hooks not active?)")
    if view.shape_material_mu.shape[0] != model.shape_material_mu.shape[0]:
        raise RuntimeError("the cloth solver's shapes are not the model's; cannot map frictions")

    groups = _shape_groups(model, fingertip_link_names)
    set_contacts = [(float(getattr(cloth, f)), groups[g]) for f, g in CONTACTS if getattr(cloth, f) is not None]
    for field, group in CONTACTS:
        if getattr(cloth, field) is not None and not groups[group].any():
            raise RuntimeError(f"cloth.{field} is set but no {group} shape matched")
    keep_unset_contacts = float(cloth.soft_contact_mu) / cloth_mu(cloth)

    def write_copy() -> None:
        shape_mu = model.shape_material_mu.numpy().astype(np.float64) * keep_unset_contacts
        for friction, shapes in set_contacts:
            shape_mu[shapes] = friction**2 / cloth_mu(cloth)
        view.shape_material_mu.assign(wp.array(shape_mu.astype(np.float32), dtype=float, device=view.shape_material_mu.device))

    original_refresh = coupled_solver._refresh_model_view_overrides

    def refresh_then_write_copy(flags: int) -> None:
        original_refresh(flags)
        write_copy()

    coupled_solver._refresh_model_view_overrides = refresh_then_write_copy
    write_copy()


def report(coupled_solver, fingertip_link_names) -> str:
    """The friction of each cloth contact, as the cloth solver computes it."""
    import numpy as np

    view = _cloth_view(coupled_solver)
    mu = float(view.soft_contact_mu)
    contact = np.sqrt(mu * view.shape_material_mu.numpy().astype(np.float64))
    groups = _shape_groups(view.parent, fingertip_link_names)

    def values(shapes):
        return sorted({round(float(v), 3) for v in contact[shapes]})

    return f"cloth-cloth {mu:.3f}, " + ", ".join(f"cloth-{g} {values(m)}" for g, m in groups.items())
