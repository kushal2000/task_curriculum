"""Task registry for isaacsimenvs.

Each task subpackage registers itself with gymnasium on import (side effect in its
``__init__.py``). Importing ``isaacsimenvs`` (or any child) is enough to expose all task ids to
``gym.make`` / ``gym.spec``.

``play`` is the play2perfect rigid-object pretraining env (PhysX); ``cloth`` the fold task on a VBD
sheet (Newton) or on the fold-shape pool of hinged plates (PhysX).
"""

from . import cloth  # gym.register("Isaacsimenvs-Cloth-Direct-v0" / "Isaacsimenvs-RigidCloth-Direct-v0")
from . import play   # gym.register("Isaacsimenvs-PlayFlexivWujiRightHumanLike-Direct-v0")

__all__ = ["cloth", "play"]
