"""Side-view schematic of all five approximations: where each hinge sits, and what the fold costs.

A 3D render answers "does it fold" and answers "where is the axis" badly -- side on, every slat
collapses to a line and the hinge is an invisible point inside a seam. This is drawn from
``chain_frames`` alone, no simulator, so the picture is the model's own geometry rather than one run
of it, and the hinge markers can be put exactly on the axes.

Each row is one variant: rest pose on the left with every hinge marked, folded pose on the right with
the bent hinges highlighted. The two are drawn at the same scale, so the 5.9 mm ply gap of a
mid-plane chain and the 2 mm gap of a surface or bar-crease chain are directly comparable.

    .venv_isaaclab3/bin/python -m scripts.analysis.rigid_cloth_variants_diagram
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

# The shared by-path loader; `_rigid_cloth_common` explains why these scripts must not import
# `isaacsimenvs` as a package. `sys.path` first, so the import works both as
# `-m scripts.analysis.<name>` (sys.path[0] is the repo root) and as a plain script path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _rigid_cloth_common import load_module  # noqa: E402  (needs the path insert above)

rc = load_module("isaacsimenvs/tasks/cloth/utils/rigid_cloth.py")

STATIONARY = ("#d95f9a", "#8c2f60")
WALL = ("#f0a83c", "#a3651b")
MOVING = ("#4a90c2", "#22536f")
HINGE = "#1b2a3a"
BENT = "#d33"
MM = 1e3


def _draw(ax, spec, angles, *, bent=(), title=""):
    """One pose of one chain, slats as rectangles and hinges as dots on the actual axes."""
    import matplotlib.patches as mpatches

    frames = rc.chain_frames(spec, angles)
    for i, (fx, fz, a) in enumerate(frames):
        dx, dz = rc.chain_geom_offset(spec, i)
        # point_in_sheet_frame returns (x, y, z); unpacking it as (x, z, _) silently draws every
        # slat at z = 0, which looks exactly like the folded ply having failed to render.
        cx, _cy, cz = rc.chain_point_in_sheet_frame(spec, i, (dx, 0.0, dz), angles)
        fill, edge = WALL if i == spec.wall else (MOVING if i in spec.moving else STATIONARY)
        if spec.shape == "cylinder":
            # Side on, a cylinder lying along y IS a circle of diameter `thickness` -- and drawing it
            # as a rectangle would be the same class of mistake the module docstring warns about,
            # showing something other than what is simulated. It is also the point of the shape: a
            # circle is rotation-invariant, so the chain rolls through a bend instead of wedging.
            # Any gap between neighbours here is real, and is why a cylinder chain wants
            # thickness == pitch.
            ax.add_patch(
                mpatches.Circle(
                    (cx * MM, cz * MM),
                    spec.thickness / 2 * MM,
                    facecolor=fill,
                    edgecolor=edge,
                    linewidth=0.7,
                    zorder=2,
                    alpha=0.95,
                )
            )
        else:
            ax.add_patch(
                mpatches.Rectangle(
                    (-spec.widths[i] / 2 * MM, -spec.thickness / 2 * MM),
                    spec.widths[i] * MM,
                    spec.thickness * MM,
                    facecolor=fill,
                    edgecolor=edge,
                    linewidth=0.7,
                    zorder=2,
                    alpha=0.95,
                    transform=matplotlib.transforms.Affine2D()
                    .rotate_deg(-a * 180.0 / math.pi)
                    .translate(cx * MM, cz * MM)
                    + ax.transData,
                )
            )

    # Hinges sit at each non-root slat's frame origin -- i.e. ON the seam it shares with its parent,
    # at mid-depth or at the top face depending on the mode. That is the whole distinction.
    for i, (fx, fz, _a) in enumerate(frames):
        if i == spec.root:
            continue
        ax.plot(
            fx * MM,
            fz * MM,
            marker="o",
            markersize=5.0 if i in bent else 3.0,
            color=BENT if i in bent else HINGE,
            zorder=5,
        )
    ax.plot(
        frames[spec.root][0] * MM,
        frames[spec.root][1] * MM,
        marker="s",
        markersize=5.0,
        color="#111",
        zorder=6,
    )
    ax.set_aspect("equal")
    ax.grid(alpha=0.18, linewidth=0.5)
    ax.axvline(0, color="#888", ls="--", lw=0.8)
    if title:
        ax.set_title(title, fontsize=9.5)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="docs/results/rigid_cloth_variants_hinges.png")
    args = p.parse_args()

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(rc.VARIANTS)
    fig, axes = plt.subplots(len(names), 2, figsize=(13, 2.15 * len(names)))

    for r, name in enumerate(names):
        spec = rc.variant(name)
        angles = rc.chain_fold_angles(spec)
        bent = {i for i, a in enumerate(angles) if abs(a) > 1e-12}
        dx, dz = rc.chain_fold_residual(spec)

        _draw(axes[r][0], spec, None, title=f"{name} -- rest, {spec.num_slats - 1} hinges")
        _draw(
            axes[r][1],
            spec,
            angles,
            bent=bent,
            title=(
                f"folded: {len(bent)} joint(s) at "
                f"{', '.join(f'{abs(math.degrees(angles[i])):.0f}' for i in sorted(bent))} deg"
            ),
        )
        axes[r][0].set_ylabel(
            f"{spec.hinge} hinge\n{spec.shape}", fontsize=9, fontweight="bold"
        )
        axes[r][1].set_xlabel(
            f"in-plane residual {dx * MM:.2f} mm, ply gap {dz * MM:.2f} mm", fontsize=8.5
        )
        for ax in axes[r]:
            ax.set_xlim(-58, 58)
            ax.set_ylim(-6, 11)
            ax.tick_params(labelsize=7)

    fig.suptitle(
        "Where each hinge sits, and what the fold costs\n"
        "black square: articulation root      dots: hinge axes      red: the joints that bend\n"
        "pink: stationary half      orange: the wall of the fold      blue: returning ply",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=135)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
