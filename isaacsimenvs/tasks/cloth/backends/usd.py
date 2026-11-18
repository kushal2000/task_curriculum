"""USD edits shared by both backends: bulk per-env authoring, and the two edits that turn the Play
scene's rigid tool and goal marker into a cloth task's.

**Bulk per-env edits without Kit's per-edit cost.**

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


def _join(root: str, rel: str) -> str:
    return f"{root}/{rel}" if rel else root


def _set_attr_spec(layer, prim_path, name: str, vtype, value) -> None:
    """Author ``name = value`` on ``prim_path`` in ``layer`` (an ``over`` if the prim is not there)."""
    from pxr import Sdf

    spec = Sdf.CreatePrimInLayer(layer, Sdf.Path(str(prim_path)))
    attr = spec.attributes.get(name)
    if attr is None:
        attr = Sdf.AttributeSpec(spec, name, vtype)
    attr.default = value


def _author_in_every_env(asset: str, layout, author, check, samples: int = 4):
    """Apply one env's edit to ``/World/envs/env_<i>/<asset>`` in every env that has it.

    ``layout(prim)`` reads what to edit from the first env's composed prim; ``author(layer, root,
    layout)`` writes it for one env as Sdf specs into ``usd_bulk.bulk_edit_layer``, which reaches the
    stage in one step (per-env edits on Kit's root layer cost O(N^2)). Every env holds
    the same asset (one variant), so one env's layout is every env's; ``check(prim, layout)`` --
    ``None`` or a message -- confirms that on the first, last and a few envs between, through the
    composed stage. Returns ``(envs edited, layout)``.
    """
    from isaaclab.sim.utils import get_current_stage
    from pxr import Sdf

    stage = get_current_stage()
    envs = stage.GetPrimAtPath("/World/envs")
    roots = []
    if envs.IsValid():
        for env in envs.GetChildren():
            if env.GetName().startswith("env_") and env.GetChild(asset).IsValid():
                roots.append(f"{env.GetPath()}/{asset}")
    if not roots:
        return 0, None
    roots.sort(key=lambda p: int(p.split("/")[3][4:]))
    lay = layout(stage.GetPrimAtPath(roots[0]))
    with bulk_edit_layer(stage, asset) as layer:
        for root in roots:
            author(layer, root, lay)
    picks = sorted({0, len(roots) - 1, *range(0, len(roots), max(1, len(roots) // samples))})
    for i in picks:
        bad = check(stage.GetPrimAtPath(roots[i]), lay)
        if bad:
            raise RuntimeError(f"[cloth] {asset} edit did not land in every env: {bad}")
    return len(roots), lay


def neutralise_rigid_object(env) -> None:
    """Disable the inherited rigid tool's colliders so it cannot touch the cloth.

    Walks the whole subtree: the colliders sit at ``/Object/<mesh>/collisions``, two levels
    down, and a one-level loop silently disables nothing.

    The layout is read from ONE env and authored into all of them at the Sdf level, in one
    change block (``_author_in_every_env``): a Usd-API edit per env re-notifies the whole stage
    under Kit, which made scene construction quadratic in the env count -- 24576 envs never
    finished building.
    """
    from pxr import Sdf, Usd, UsdPhysics

    def layout(prim):
        n = len(str(prim.GetPath())) + 1
        colliders, bodies = [], []
        for target in Usd.PrimRange(prim):
            rel = str(target.GetPath())[n:]
            if target.HasAPI(UsdPhysics.CollisionAPI):
                colliders.append(rel)
            if target.HasAPI(UsdPhysics.RigidBodyAPI):
                bodies.append(rel)
        return colliders, bodies

    def author(layer, root, lay):
        colliders, bodies = lay
        for rel in colliders:
            _set_attr_spec(layer, _join(root, rel), "physics:collisionEnabled", Sdf.ValueTypeNames.Bool, False)
        for rel in bodies:
            _set_attr_spec(layer, _join(root, rel), "physics:kinematicEnabled", Sdf.ValueTypeNames.Bool, True)

    def check(prim, lay):
        colliders, _ = lay
        for rel in colliders:
            api = UsdPhysics.CollisionAPI(prim.GetPrimAtPath(rel) if rel else prim)
            if api.GetCollisionEnabledAttr().Get() is not False:
                return f"{prim.GetPath()}/{rel} still collides"
        return None

    prims, lay = _author_in_every_env("Object", layout, author, check)
    colliders = prims * len(lay[0]) if lay else 0
    print(f"[cloth] neutralised rigid tool ({prims} prims, {colliders} colliders off)", flush=True)
    if prims and not colliders:
        raise RuntimeError(
            "found the inherited rigid tool but disabled none of its colliders -- it would "
            "stay live and corrupt every measurement. Check the prim layout."
        )


def reshape_goal_marker(env) -> None:
    """Give the goal marker the folded half's shape instead of the tool's.

    ``goal_viz`` spawns from the same procedural handle+head USD as the manipuland, so the cloth
    task renders a hammer-shaped ghost for a goal that is a folded sheet. The marker *pose* is
    already correct and success is scored on keypoints, not on this mesh -- but the render is
    the only thing a person actually looks at, and a hammer standing in for a fold invites
    exactly the wrong reading. The same defect on the cable task cost hours.

    Swaps the tool geometry for a thin slab the size of the moving half. ``collisionEnabled``
    stays false so the marker can never touch the sheet;
    ``render_newton._reveal_goal_viz`` draws it by setting the VISIBLE flag, which does not
    require a collider. Authored in one Sdf change block, as ``_neutralise_rigid_object``.
    """
    from pxr import Gf, Sdf, Usd, UsdPhysics, Vt

    from isaacsimenvs.tasks.cloth.mdp.geometry.grid import half_mesh

    c = env.cfg.cloth
    # The half's own grid, not a box: the marker stands in for a folded SHEET, and a solid slab
    # both reads as a brick and shows a thickness the cloth does not have (a VBD cloth is a
    # surface -- its thickness is a collision radius the renderer never draws, so the sheet
    # itself always renders flat).
    gv, gi = half_mesh(float(c.size), int(c.resolution), c.fold_axis)
    points = Vt.Vec3fArray([Gf.Vec3f(*[float(x) for x in v]) for v in gv])
    indices = Vt.IntArray([int(i) for i in gi])
    counts = Vt.IntArray([3] * (len(gi) // 3))

    def layout(prim):
        # The rigid body sits on a child (`.../GoalViz/object_root`), not the GoalViz root.
        # Deactivating that child would remove the body the physics view resolves, so keep the
        # body and replace only the geometry beneath it.
        body = next((d for d in Usd.PrimRange(prim) if d.HasAPI(UsdPhysics.RigidBodyAPI)), prim)
        rel = str(body.GetPath())[len(str(prim.GetPath())) + 1:]
        return rel, [child.GetName() for child in body.GetChildren()]

    def author(layer, root, lay):
        rel, children = lay
        body = _join(root, rel)
        for name in children:
            Sdf.CreatePrimInLayer(layer, f"{body}/{name}").active = False
        mesh = Sdf.CreatePrimInLayer(layer, f"{body}/cloth_marker")
        mesh.specifier = Sdf.SpecifierDef
        mesh.typeName = "Mesh"
        mesh.SetInfo("apiSchemas", Sdf.TokenListOp.Create(prependedItems=["PhysicsCollisionAPI"]))
        _set_attr_spec(layer, mesh.path, "points", Sdf.ValueTypeNames.Point3fArray, points)
        _set_attr_spec(layer, mesh.path, "faceVertexIndices", Sdf.ValueTypeNames.IntArray, indices)
        _set_attr_spec(layer, mesh.path, "faceVertexCounts", Sdf.ValueTypeNames.IntArray, counts)
        _set_attr_spec(layer, mesh.path, "physics:collisionEnabled", Sdf.ValueTypeNames.Bool, False)

    def check(prim, lay):
        rel, children = lay
        body = prim.GetPrimAtPath(rel) if rel else prim
        marker = body.GetChild("cloth_marker")
        if not marker.IsValid() or len(marker.GetAttribute("points").Get() or []) != len(gv):
            return f"{body.GetPath()} has no cloth_marker mesh"
        if any(body.GetChild(n).IsActive() for n in children):
            return f"{body.GetPath()} still shows the tool geometry"
        return None

    marked, _ = _author_in_every_env("GoalViz", layout, author, check)
    print(
        f"[cloth] goal marker reshaped on {marked} prims "
        f"(half-sheet mesh, {len(gv)} verts / {len(gi) // 3} tris)",
        flush=True,
    )
