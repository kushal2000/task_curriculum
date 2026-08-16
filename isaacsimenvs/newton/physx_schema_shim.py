"""A kit-less stand-in for ``pxr.PhysxSchema``.

``scene_utils._bake_usd`` authors PhysX-namespaced USD attributes through ``pxr.PhysxSchema``,
which ships as an Isaac Sim plugin rather than with ``usd-core``. The Newton venv has no Isaac
Sim, so the import fails outright and no asset can be baked.

This is not a workaround invented here -- it mirrors Isaac Lab's own kit-less pattern in
``isaaclab/sim/schemas/schemas.py``, which calls ``prim.AddAppliedSchema("PhysxRigidBodyAPI")``
and writes the namespaced attributes directly rather than going through the plugin.

The equivalence matters and is not incidental: Isaac Lab documents that **Newton's USD importer
reads the same** ``physxCollision:contactOffset`` and ``restOffset`` attributes PhysX does. The
PhysX-namespaced names are the shared authoring convention, not a PhysX-only detail. Applying the
API schema names too is kept purely for fidelity with the Kit bake -- Newton ignores API schemas
it does not implement, and it keeps the two backends' USD as close as possible.

Adapted from ``simtoolreal_newton.upstream`` in github.com/kushal2000/isaac_newton @ beb9efb.
"""

from __future__ import annotations

import sys
import types

_CONTACT_OFFSET_ATTR = "physxCollision:contactOffset"
_REST_OFFSET_ATTR = "physxCollision:restOffset"


class _AttrHandle:
    """Minimal stand-in for a USD attribute handle supporting ``.Set(value)``."""

    def __init__(self, prim, name: str) -> None:
        self._prim = prim
        self._name = name

    def Set(self, value) -> None:  # noqa: N802  (USD naming)
        from pxr import Sdf

        attr = self._prim.GetAttribute(self._name)
        if not attr:
            attr = self._prim.CreateAttribute(self._name, Sdf.ValueTypeNames.Float)
        attr.Set(float(value))


class _AppliedAPI:
    """Base for the ``Apply``/constructor pattern of USD applied-API schemas."""

    _schema_name: str = ""

    def __init__(self, prim) -> None:
        self._prim = prim

    @classmethod
    def Apply(cls, prim):  # noqa: N802
        if cls._schema_name and cls._schema_name not in prim.GetAppliedSchemas():
            prim.AddAppliedSchema(cls._schema_name)
        return cls(prim)


class PhysxRigidBodyAPI(_AppliedAPI):
    _schema_name = "PhysxRigidBodyAPI"


class PhysxArticulationAPI(_AppliedAPI):
    _schema_name = "PhysxArticulationAPI"


class PhysxCollisionAPI(_AppliedAPI):
    """Supports the ``PhysxCollisionAPI(prim) or PhysxCollisionAPI.Apply(prim)`` idiom.

    The task code relies on the constructor being *falsy* when the schema is not yet applied, so
    ``__bool__`` reports whether the prim already carries it.
    """

    _schema_name = "PhysxCollisionAPI"

    def __bool__(self) -> bool:
        return self._schema_name in self._prim.GetAppliedSchemas()

    def CreateContactOffsetAttr(self, *_args, **_kwargs) -> _AttrHandle:  # noqa: N802
        return _AttrHandle(self._prim, _CONTACT_OFFSET_ATTR)

    def CreateRestOffsetAttr(self, *_args, **_kwargs) -> _AttrHandle:  # noqa: N802
        return _AttrHandle(self._prim, _REST_OFFSET_ATTR)


def install() -> bool:
    """Expose this module as ``pxr.PhysxSchema`` when the real plugin is unavailable.

    Returns True if the shim was installed, False if the genuine plugin is present -- which is
    the case inside the Isaac Sim venv, where it must not be shadowed.
    """
    import pxr

    if hasattr(pxr, "PhysxSchema"):
        return False

    module = types.ModuleType("pxr.PhysxSchema")
    module.PhysxRigidBodyAPI = PhysxRigidBodyAPI
    module.PhysxArticulationAPI = PhysxArticulationAPI
    module.PhysxCollisionAPI = PhysxCollisionAPI

    sys.modules["pxr.PhysxSchema"] = module
    pxr.PhysxSchema = module  # type: ignore[attr-defined]
    return True


__all__ = [
    "PhysxArticulationAPI",
    "PhysxCollisionAPI",
    "PhysxRigidBodyAPI",
    "install",
]
