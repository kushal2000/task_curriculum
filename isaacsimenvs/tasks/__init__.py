"""Task registry for isaacsimenvs.

Each task subpackage registers itself with gymnasium on import (side effect
in its ``__init__.py``). Importing ``isaacsimenvs`` (or any child) is enough
to expose all task ids to ``gym.make`` / ``gym.spec``.

Three env families: ``play`` (rigid-body manipulation, PhysX), the rigid-plate
cloth fold envs and the VBD cloth fold env (both in ``cloth``).
"""

from . import cloth              # gym.register("Isaacsimenvs-Cloth-Direct-v0", ...)
from . import play               # gym.register("Isaacsimenvs-Play-Direct-v0", ...)

__all__ = ["cloth", "play"]
