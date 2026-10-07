"""Bulk per-env USD edits without Kit's per-edit cost.

Under Kit, every spec edit on a stage's ROOT layer carries a cost that grows with the layer, so N
per-env edits are O(N^2) even inside one ``Sdf.ChangeBlock``. Measured on the PhysX chain: binding the
fingertip / slat materials took 13 s at 3072 envs and 46-55 s at 6144 (the identical loop in plain
usd-core: 0.2 s / 0.5 s), and at 24576 envs scene construction took ~19 min. Switching off Kit's USD
notice listeners (as Isaac Sim's cloner does) made no difference (A/B, 3 runs each).

So the edits go into a fresh anonymous layer -- plain Sdf, nothing attached to it -- which is then
inserted ONCE as the strongest sublayer of the root layer. Its opinions are weaker than the root
layer's own but stronger than the referenced assets', which is all these edits override. Without Kit
(Newton) the root layer is edited directly, as before: it is fast there, and Newton's cloner copies
root-layer specs from env_0, which it would not do for a sublayer.
"""

from __future__ import annotations

import contextlib
import sys

#: Layers inserted into a stage are only referenced by identifier there; keep them alive.
_LIVE_LAYERS: list = []


def _under_kit() -> bool:
    return "omni.kit.app" in sys.modules or "omni.usd" in sys.modules


@contextlib.contextmanager
def bulk_edit_layer(stage, tag: str):
    """Yield the ``Sdf.Layer`` to author per-env specs in; they are on the stage when this exits."""
    from pxr import Sdf

    if not _under_kit():
        with Sdf.ChangeBlock():
            yield stage.GetRootLayer()
        return
    layer = Sdf.Layer.CreateAnonymous(f"isaacsimenvs_{tag}")
    yield layer
    _LIVE_LAYERS.append(layer)
    stage.GetRootLayer().subLayerPaths.insert(0, layer.identifier)
