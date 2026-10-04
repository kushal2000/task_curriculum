"""The three MuJoCo scenes the rigid-cloth figures are built from, and the camera that shoots them.

`rigid_cloth_render.py` draws one variant in four panels; `rigid_cloth_variants_render.py` draws all
five in a grid. They are different figures, but they settle, fold and drape the same chain the same
way, and they had drifted into two copies of this code -- one of which had already lost the `spec`
generalisation and still read the sheet's width off the CLI namespace.

Everything a figure legitimately chooses is an argument: the camera, the palette, and how long to
run. Everything that must not differ between the picture and the measurement -- model construction,
placement, the settle -- comes from `rigid_cloth_probe`, so the thing drawn is the thing measured.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _rigid_cloth_common import load_module  # noqa: E402  (needs the path insert above)

rc = load_module("isaacsimenvs/tasks/cloth/utils/rigid_cloth.py")
probe = load_module("scripts/analysis/rigid_cloth_probe.py")

#: Role colours for `paint_by_role`. The variants figure's caption names them, so they are fixed:
#: pink stationary half, orange wall of the fold, blue returning ply.
STATIONARY = (0.85, 0.35, 0.55, 1.0)
WALL = (0.95, 0.68, 0.22, 1.0)
MOVING = (0.29, 0.56, 0.76, 1.0)


def _slat_geoms(model, spec):
    """(slat index, geom index) for every geom of every slat.

    MuJoCo's URDF importer discards visual geoms and keeps collision ones, so what these colour --
    and therefore what every figure draws -- IS the collision hull. That is the honest thing to show
    and the reason the URDF gives both the same shape.
    """
    import mujoco

    for i in range(spec.num_slats):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, rc.slat_link_name(i))
        for g in range(model.body_geomadr[bid], model.body_geomadr[bid] + model.body_geomnum[bid]):
            yield i, g


def paint_by_parity(model, spec) -> None:
    """Alternate the shade slightly, so individual slats are countable in the render."""
    for i, g in _slat_geoms(model, spec):
        shade = 1.0 if i % 2 else 0.82
        model.geom_rgba[g] = [0.85 * shade, 0.35 * shade, 0.55 * shade, 1.0]


def paint_by_role(model, spec) -> None:
    """Colour by role, so a fold is visible as a fold rather than as a row of tubes."""
    for i, g in _slat_geoms(model, spec):
        model.geom_rgba[g] = (
            WALL if i == spec.wall else MOVING if i in spec.moving else STATIONARY
        )


def shot(model, data, spec, args, paint, *, lookat, distance, azimuth, elevation):
    """One offscreen frame of an already-settled model."""
    import mujoco

    paint(model, spec)
    # The offscreen framebuffer defaults to 640x480 and `Renderer` refuses anything larger, so it
    # has to be widened on the model before the renderer is built.
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
    # A URDF carries no lights, so the scene is lit by the headlight alone and renders near-black.
    model.vis.headlight.ambient[:] = [0.45, 0.45, 0.45]
    model.vis.headlight.diffuse[:] = [0.8, 0.8, 0.8]
    model.vis.headlight.specular[:] = [0.2, 0.2, 0.2]

    # A camera closer than ~0.6 of the sheet's width sits INSIDE it and renders the inside of a
    # slat. That is what the first close-up panel did.
    min_distance = 0.6 * spec.size + 0.02
    if distance < min_distance:
        raise ValueError(
            f"camera distance {distance:.3f} m would sit inside a {spec.size:.3f} m sheet; "
            f"use at least {min_distance:.3f} m"
        )
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    renderer = mujoco.Renderer(model, args.height, args.width)
    renderer.update_scene(data, cam)
    px = renderer.render()
    renderer.close()
    return px


def flat(args, spec, paint, **cam):
    """Dropped from `--drop_height` and settled: does it lie like a sheet?"""
    model, data = probe._build(args, spec)
    probe._place(model, data, spec, [0.0, 0.0, args.drop_height])
    probe._run(model, data, args.settle_steps)
    return shot(model, data, spec, args, paint, **cam)


def folded(args, spec, paint, **cam):
    """Placed at its own fold angles and settled: is the fold an arc with a wall?"""
    model, data = probe._build(args, spec)
    probe._place(
        model, data, spec, [0.0, 0.0, probe._rest_height(spec)], rc.chain_fold_angles(spec)
    )
    probe._run(model, data, args.settle_steps)
    return shot(model, data, spec, args, paint, **cam)


def drape(args, spec, paint, steps: int | None = None, **cam):
    """Clamped at the root over a table edge: how far from limp does it hang?"""
    model, data = probe._build(
        args, spec, extra_table=True, free_base=False,
        root_pos=[spec.centres[spec.root], 0.0, probe._rest_height(spec)],
    )
    probe._place(model, data, spec, None)
    probe._run(model, data, args.settle_steps if steps is None else steps)
    return shot(model, data, spec, args, paint, **cam)
