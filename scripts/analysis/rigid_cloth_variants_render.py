"""Render all five rigid-cloth approximations in the three states that distinguish them.

The comparison table in ``rigid_cloth_probe --all`` says WHAT each variant does; this says what each
one looks like while doing it, which is the only way to see three things numbers hide:

  * **where the hinges are** -- a mid-plane chain bends about each seam's centre-line and a surface
    chain about each seam's top edge, and only the folded crease shows the difference;
  * **what the wall of the fold costs** -- ``box-mid-odd`` spends its crease slat standing vertically
    and ``box3-mid`` spends a dedicated 2 mm bar, which is the same mechanism at two scales;
  * **that ``box2-surface``'s plies interpenetrate** -- its two plates are a parent-child pair, so the
    contact is filtered and nothing stops them, which reads as a single plate in the render.

Slats are coloured by ROLE rather than by index: the stationary half, the wall, and the returning
ply. A single-colour sheet folded onto itself is indistinguishable from an unfolded one, which is how
an earlier version of the uniform render managed to show a correct 2 mm fold as a flat sheet.

Offscreen via OSMesa, so it runs headless:

    MUJOCO_GL=osmesa .venv_isaaclab3/bin/python -m scripts.analysis.rigid_cloth_variants_render
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

# The shared by-path loader; `_rigid_cloth_common` explains why these scripts must not import
# `isaacsimenvs` as a package. `sys.path` first, so the import works both as
# `-m scripts.analysis.<name>` (sys.path[0] is the repo root) and as a plain script path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _rigid_cloth_scenes as sc  # noqa: E402  (needs the path insert above)
from _rigid_cloth_common import load_module  # noqa: E402

rc = load_module("isaacsimenvs/tasks/cloth/utils/rigid_cloth.py")

#: (column title, scene, camera). The cameras are this figure's own: a grid of small panels wants a
#: lower, closer view than the four-panel figure does.
COLUMNS = (
    ("flat", sc.flat, dict(lookat=[0, 0, 0], distance=0.26, azimuth=135, elevation=-26)),
    # Three-quarter view from low down: a pure side view collapses every slat to a line and the two
    # plies overlap into one, while a top view cannot show the ply gap at all.
    ("folded", sc.folded, dict(lookat=[-0.012, 0, 0.004], distance=0.19, azimuth=118, elevation=-14)),
    ("cantilever", sc.drape, dict(lookat=[0.0, 0, -0.02], distance=0.20, azimuth=90, elevation=-8)),
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--density", type=float, default=2.0)
    p.add_argument("--damping", type=float, default=1.0e-5)
    # Match the cloth: slats at its own soft_contact_mu, ground at what the cloth EFFECTIVELY
    # feels against the 0.5 table under Newton's geometric-mean mixing. See `rigid_cloth_probe`.
    p.add_argument("--friction", type=float, default=rc.CLOTH_REFERENCE["soft_contact_mu"])
    p.add_argument("--no-pairs", dest="pairs", action="store_false")
    p.set_defaults(pairs=True)
    p.add_argument(
        "--ground_friction",
        type=float,
        default=rc.CLOTH_REFERENCE["shape_mu"]["table"],
    )
    p.add_argument("--stiffness", type=float, default=0.0)
    p.add_argument("--dt", type=float, default=1.0 / 240.0)
    p.add_argument("--drop_height", type=float, default=0.05)
    p.add_argument("--settle_steps", type=int, default=480)
    p.add_argument("--width", type=int, default=560)
    p.add_argument("--height", type=int, default=420)
    p.add_argument("--out_dir", default="/tmp/rigid_cloth_probe")
    p.add_argument("--out", default="docs/results/rigid_cloth_variants.png")
    args = p.parse_args()
    # `probe._build` reads these off the same namespace; the geometry comes from the spec instead.
    args.variant = None
    args.size = 0.10
    args.num_slats = rc.DEFAULT_NUM_SLATS
    args.thickness = None
    args.shape = "cylinder"
    args.hinge = "mid"

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(rc.VARIANTS)
    columns = COLUMNS
    fig, axes = plt.subplots(len(names), len(columns), figsize=(3.5 * len(columns), 2.8 * len(names)))

    for r, name in enumerate(names):
        spec = rc.variant(name)
        args.variant = name
        gap = rc.chain_ply_gap(spec)
        dx, _dz = rc.chain_fold_residual(spec)
        for c, (title, scene, cam) in enumerate(columns):
            ax = axes[r][c]
            ax.imshow(scene(args, spec, sc.paint_by_role, **cam))
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=12, fontweight="bold", pad=6)
            if c == 0:
                ax.set_ylabel(
                    f"{name}\n{spec.num_slats} bodies, {spec.hinge}",
                    fontsize=9.5,
                    fontweight="bold",
                )
            for s in ax.spines.values():
                s.set_edgecolor("#bbbbbb")
        axes[r][1].set_xlabel(
            f"ply gap {gap * 1e3:.2f} mm, in-plane residual {dx * 1e3:.2f} mm", fontsize=9
        )

    fig.suptitle(
        "Five rigid approximations of the same 100 mm folding sheet\n"
        "pink: stationary half      orange: the wall of the fold      blue: returning ply",
        fontsize=12.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=125)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
