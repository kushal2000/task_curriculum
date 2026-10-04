"""Render the hinged-slat rigid cloth in its three characteristic states.

A picture is the fastest way to check the two claims that matter and are hard to read off numbers:
that the chain is a *sheet* rather than a row of loose tubes, and that its fold is an ARC with one
slat standing as the wall of a U -- not the reflection a cloth makes.

Shares its model construction with ``rigid_cloth_probe`` so the thing drawn is the thing measured.
Offscreen via OSMesa, so it runs headless:

    MUJOCO_GL=osmesa .venv_isaaclab3/bin/python -m scripts.analysis.rigid_cloth_render

Writes a four-panel PNG: settled flat, folded (side on, where the U is visible), the crease in
close-up, and the clamped cantilever that measures drape.
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
probe = load_module("scripts/analysis/rigid_cloth_probe.py")


def _spec(args):
    return rc.uniform_chain(
        args.size, args.num_slats, args.thickness, hinge=args.hinge, shape=args.shape
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size", type=float, default=0.10)
    p.add_argument("--num_slats", type=int, default=rc.DEFAULT_NUM_SLATS)
    p.add_argument("--thickness", type=float, default=None)
    p.add_argument("--density", type=float, default=2.0)
    p.add_argument("--shape", default="cylinder", choices=("cylinder", "box"))
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
    p.add_argument("--drape_steps", type=int, default=1920)
    p.add_argument("--hinge", default="mid", choices=rc.HINGE_MODES)
    p.add_argument("--variant", default=None, help="unused here; `_build` reads it for a filename")
    p.add_argument("--width", type=int, default=760)
    p.add_argument("--height", type=int, default=560)
    p.add_argument("--out_dir", default="/tmp/rigid_cloth_probe")
    p.add_argument("--out", default="docs/results/rigid_cloth_states.png")
    args = p.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pitch = rc.slat_pitch(args.size, args.num_slats)
    dx, dz = rc.fold_residual(args.size, args.num_slats)
    spec = _spec(args)
    paint = sc.paint_by_parity
    panels = [
        (
            sc.flat(args, spec, paint, lookat=[0, 0, 0], distance=0.26, azimuth=135, elevation=-28),
            "settled flat",
            f"{args.num_slats} slats, pitch {pitch * 1e3:.2f} mm",
        ),
        (
            # Three-quarter view: a pure side view collapses every slat to a circle, so the U reads
            # as two rows of dots and the fold is invisible.
            sc.folded(args, spec, paint,
                      lookat=[-0.012, 0, 0.004], distance=0.20, azimuth=125, elevation=-22),
            "folded (side on)",
            f"ply gap = pitch = {pitch * 1e3:.2f} mm",
        ),
        (
            # Side on at the crease, just outside the sheet's own half-width: the wall slat and the
            # two plies have to be countable, which is the whole claim of `chain_fold_angles`.
            sc.folded(args, spec, paint,
                      lookat=[-0.014, 0, 1.0 * pitch], distance=0.105, azimuth=97, elevation=-9),
            "the crease, close up",
            "two 90 deg joints, one slat as the wall",
        ),
        (
            sc.drape(args, spec, paint, steps=args.drape_steps,
                     lookat=[0.0, 0, -0.02], distance=0.20, azimuth=90, elevation=-8),
            "clamped cantilever",
            f"drape {probe.probe_drape(args, spec)['drape_fraction']:.3f} of limp",
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2))
    for ax, (px, title, sub) in zip(axes.ravel(), panels):
        ax.imshow(px)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=6)
        ax.set_xlabel(sub, fontsize=9.5, labelpad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#bbbbbb")
    fig.suptitle(
        f"Rigid cloth approximation: {args.num_slats} hinged slats, "
        f"{args.size * 1e3:.0f} mm sheet, fold residual {dx * 1e3:.2f} / {dz * 1e3:.2f} mm",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
