"""Task registry for isaacsimenvs.

Each task subpackage registers itself with gymnasium on import (side effect
in its ``__init__.py``). Importing ``isaacsimenvs`` (or any child) is enough
to expose all task ids to ``gym.make`` / ``gym.spec``.

``play`` is not a curriculum task — it is the play2perfect Kuka + Sharpa base
env, kept verbatim so ``bottle_flip`` can subclass it and so play2perfect
checkpoints stay weight-compatible.
"""

from . import bottle_flip        # gym.register("Isaacsimenvs-BottleFlip-Direct-v0", ...)
from . import cable              # gym.register("Isaacsimenvs-Cable-Direct-v0", ...)
from . import multilink_cartpole  # gym.register("Isaacsimenvs-MultiLinkCartpole-Direct-v0", ...)
from . import play               # gym.register("Isaacsimenvs-Play-Direct-v0", ...)
from . import play_newton        # gym.register("Isaacsimenvs-PlayNewton-Direct-v0", ...)

__all__ = ["bottle_flip", "cable", "multilink_cartpole", "play", "play_newton"]
