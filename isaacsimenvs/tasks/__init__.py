"""Task registry for isaacsimenvs.

Each task subpackage registers itself with gymnasium on import (side effect
in its ``__init__.py``). Importing ``isaacsimenvs`` (or any child) is enough
to expose all task ids to ``gym.make`` / ``gym.spec``.

``play`` is the play2perfect rigid-object pretraining env (PhysX); ``play_newton`` is its
Newton port, the base class of the VBD cloth env.
"""

from . import cloth              # gym.register("Isaacsimenvs-Cloth-Direct-v0", ...)
from . import play               # gym.register("Isaacsimenvs-Play-Direct-v0" / "-PlayFlexivWuji-" / ...)
from . import play_newton        # gym.register("Isaacsimenvs-PlayNewton-Direct-v0", ...)

__all__ = ["cloth", "play", "play_newton"]
