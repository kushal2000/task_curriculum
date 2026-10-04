"""Schematic side view: where the hinges are, and why a folded chain is shorter on top.

Two questions a 3D render answers badly, because a side-on view collapses every slat to a circle
and the hinges are invisible inside the seams:

  * **Where are the hinges?** One per seam, ``num_slats - 1`` of them, axis ``+y``. There is no
    designated "crease" joint -- the crease is wherever the chain happens to bend, and
    ``folded_joint_angles`` merely picks the two that make the tightest fold.
  * **Why are the two plies different lengths?** Because a rigid chain turning 180 degrees has to
    SPEND length on the turn. One slat stands up as the wall of the U and is no longer part of
    either flat ply, so the returning ply is exactly one slat shorter. That is not drift or a
    settling artefact -- it is ``fold_residual``, it is intrinsic to the discretisation, and it
    shrinks linearly as ``pitch`` does.

Drawn from ``forward_kinematics`` alone -- no simulator -- so the picture is the model's own
geometry rather than a particular run of it.

    .venv_isaaclab3/bin/python -m scripts.analysis.rigid_cloth_diagram
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

# The shared by-path loader; `_rigid_cloth_common` explains why these scripts must not import
# `isaacsimenvs` as a package. `sys.path` first, so the import works both as
# `-m scripts.analysis.<name>` (sys.path[0] is the repo root) and as a plain script path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _rigid_cloth_common import load_module  # noqa: E402  (needs the path insert above)

rc = load_module("isaacsimenvs/tasks/cloth/utils/rigid_cloth.py")

SLAT_FILL = "#d95f9a"
SLAT_EDGE = "#8c2f60"
MOVE_FILL = "#4a90c2"
MOVE_EDGE = "#22536f"
HINGE = "#1b2a3a"
BENT = "#e8a33d"


def _draw(
    ax, size, num_slats, angles, *, bent=(), title="", mm=1e3, thickness=None, hinge="mid"
):
    import matplotlib.patches as mpatches

    pitch = rc.slat_pitch(size, num_slats)
    frames = rc.forward_kinematics(size, num_slats, angles)
    root = rc.root_slat(num_slats)

    # Slats as circles: a side view along +y is exactly what the collider looks like.
    t = pitch if thickness is None else thickness
    # The moving half is drawn in a second colour. Without it the folded plies are the same shade
    # touching face to face, and a correct 2 mm fold is indistinguishable from an unfolded sheet.
    moving = set(rc.half_slats(num_slats, positive=True))
    for i, (fx, fz, a) in enumerate(frames):
        # (x, y, z) -- unpacking this as (x, z, _) silently drew every slat at z = 0, which looked
        # like the folded ply had simply failed to render.
        dx, dz = rc._geom_offset(i, num_slats, pitch, thickness, hinge)
        cx, _cy, cz = rc.point_in_sheet_frame(size, num_slats, i, (dx, 0.0, dz), angles)
        deg = -a * 180.0 / 3.141592653589793
        ax.add_patch(
            mpatches.Rectangle(
                (-pitch / 2 * mm, -t / 2 * mm),
                pitch * mm,
                t * mm,
                facecolor=MOVE_FILL if i in moving else SLAT_FILL,
                edgecolor=MOVE_EDGE if i in moving else SLAT_EDGE,
                linewidth=0.8,
                zorder=2,
                alpha=0.95,
                transform=matplotlib.transforms.Affine2D()
                .rotate_deg(deg)
                .translate(cx * mm, cz * mm)
                + ax.transData,
            )
        )

    # Hinges sit at each non-root slat's frame origin -- i.e. on the seam it shares with its parent.
    for i, (fx, fz, _a) in enumerate(frames):
        if i == root:
            continue
        ax.plot(
            fx * mm,
            fz * mm,
            marker="o",
            markersize=4.2 if i in bent else 2.8,
            color=BENT if i in bent else HINGE,
            zorder=4,
        )
    ax.plot(
        frames[root][0] * mm,
        frames[root][1] * mm,
        marker="s",
        markersize=5.5,
        color="#111111",
        zorder=5,
    )
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]", fontsize=9)
    ax.set_ylabel("z [mm]", fontsize=9)
    ax.grid(alpha=0.18, linewidth=0.6)
    return frames, pitch


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size", type=float, default=0.10)
    p.add_argument("--num_slats", type=int, default=rc.DEFAULT_NUM_SLATS)
    p.add_argument("--thickness", type=float, default=0.002, help="slab thickness [m]")
    p.add_argument("--out", default="docs/results/rigid_cloth_hinges.png")
    args = p.parse_args()

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    size, n, t = args.size, args.num_slats, args.thickness
    pitch = rc.slat_pitch(size, n)
    mm = 1e3
    moving = rc.half_slats(n, positive=True)

    fig, axes = plt.subplots(3, 1, figsize=(11, 10.4))

    # --- rest -------------------------------------------------------------------------------
    _draw(
        axes[0],
        size,
        n,
        None,
        title=f"rest: {n} slats, {n - 1} hinges -- one per seam, axis +y",
        mm=mm,
        thickness=t,
        hinge="mid",
    )
    axes[0].annotate(
        f"a hinge on every seam ({n - 1} of them)\nblack square = articulation root (slat {rc.root_slat(n)})",
        xy=(0, 0),
        xytext=(-50, 15),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color=HINGE, lw=1.0),
    )
    axes[0].set_ylim(-12, 26)

    # --- mid-plane hinge --------------------------------------------------------------------
    ang_mid = rc.folded_joint_angles(n, hinge="mid")
    _draw(
        axes[1],
        size,
        n,
        ang_mid,
        bent={moving[0], moving[1]},
        title='hinge="mid" (axis on the slab mid-plane): TWO joints at 90 deg, one slat spent as the wall',
        mm=mm,
        thickness=t,
        hinge="mid",
    )
    dx_m, _ = rc.fold_residual(size, n, t, "mid")
    axes[1].annotate(
        f"returning ply is one slat short:\n{dx_m * mm:.2f} mm of in-plane residual",
        xy=((-size / 2 + 2 * pitch) * mm, pitch * mm),
        xytext=(-54, 18),
        fontsize=9,
        color="#a3521b",
        arrowprops=dict(arrowstyle="->", color=BENT, lw=1.3),
    )
    axes[1].set_ylim(-12, 26)

    # --- surface hinge ----------------------------------------------------------------------
    ang_surf = rc.folded_joint_angles(n, hinge="surface")
    _draw(
        axes[2],
        size,
        n,
        ang_surf,
        bent={moving[0]},
        title='hinge="surface" (axis on the slab top face): ONE joint at 180 deg, exact mirror',
        mm=mm,
        thickness=t,
        hinge="surface",
    )
    dx_s, dz_s = rc.fold_residual(size, n, t, "surface")
    axes[2].annotate(
        f"lands on its exact mirror:\nin-plane residual {dx_s * mm:.2f} mm, ply gap {dz_s * mm:.2f} mm\n"
        f"COST: every hinge is now one-directional",
        xy=(-size / 2 * mm, t * mm),
        xytext=(-54, 16),
        fontsize=9,
        color="#1d6b3a",
        arrowprops=dict(arrowstyle="->", color="#2e8b57", lw=1.3),
    )
    axes[2].set_ylim(-12, 26)

    # The whole point of the surface hinge is that the folded half lands ON its mirror, so at a
    # shared vertical scale the result looks like an unfolded sheet. An inset at the crease is the
    # only honest way to show two plies 2 mm apart next to a 100 mm sheet.
    inset = axes[2].inset_axes([0.60, 0.42, 0.37, 0.52])
    _draw(inset, size, n, ang_surf, bent={moving[0]}, mm=mm, thickness=t, hinge="surface")
    inset.set_xlim(-16, 5)
    inset.set_ylim(-3.2, 5.2)
    inset.set_xlabel("")
    inset.set_ylabel("")
    inset.set_title("")
    inset.tick_params(labelsize=7)
    inset.text(
        -15,
        4.0,
        f"crease, zoomed: two plies {dz_s * mm:.2f} mm apart",
        fontsize=7.5,
        color="#1d6b3a",
    )
    axes[2].indicate_inset_zoom(inset, edgecolor="#2e8b57")

    for ax in axes:
        ax.axvline(0, color="#777", ls="--", lw=0.9)
        ax.set_xlim(-58, 58)

    fig.suptitle(
        f"Mid-plane vs surface hinge: {n} slats, pitch {pitch * mm:.2f} mm, slab {t * mm:.1f} mm "
        f"-- in-plane fold residual {dx_m * mm:.2f} mm vs {dx_s * mm:.2f} mm",
        fontsize=12.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=135)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
