"""Render the fold-pool YAML designs (``isaacsimenvs/tasks/cloth/config/designs/``) as PNG schematics.

    .venv_isaaclab3/bin/python isaacsimenvs/scripts/visualization/rigid_cloth/render_designs.py [--only box2-mid ...]

One PNG next to each YAML: plates (moving half of the target fold orange, the rest blue, root marked),
the target crease solid, other hinges dashed, seams dotted, the viewer grasp point, and the named folds.
Reads the designs through ``fold_shapes.load_design``, so the picture is what the env builds.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({"hatch.linewidth": 0.5, "hatch.color": "#8aa0b4"})
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
NAVY, BLUE, ORANGE, INK, MUTED, RED = "#203c59", "#c6dbea", "#f4c692", "#26384b", "#617181", "#b8423a"


def _fold_shapes():
    # By path: importing the package would register every gym task.
    path = REPO / "isaacsimenvs/tasks/cloth/mdp/geometry/fold_shapes.py"
    spec = importlib.util.spec_from_file_location("fold_shapes", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fold_shapes"] = mod  # dataclasses look their module up there
    spec.loader.exec_module(mod)
    return mod


def _xy(p):
    """Sheet (x, y) -> plot: +x right, +y up, so the plate names' compass directions read as drawn."""
    return (p[0], p[1])


def render(fs, path: Path) -> Path:
    s = fs.load_design(path)
    L = s.size / 2.0
    moving = set(s.half_fold_moving())
    target = set(s.folds[fs.HALF_FOLD][0])
    fig = plt.figure(figsize=(13.5, 6.6), dpi=130)
    fig.text(0.06, 0.92, s.name, fontsize=20, weight="bold", color=INK)
    fig.text(0.06, 0.875, "YAML DESIGN  ·  FOLD POOL  ·  GEOMETRY SCHEMATIC", fontsize=8, weight="bold", color=NAVY)
    fig.text(0.06, 0.83, s.label, fontsize=10, color=MUTED)
    ax = fig.add_axes((0.06, 0.1, 0.5, 0.68))
    ax.set_aspect("equal")
    ax.axis("off")
    pad = 0.35 * L
    ax.set_xlim(-L - pad, L + pad)
    ax.set_ylim(-L - pad, L + pad)

    for p in s.plates:
        poly = [_xy(v) for v in p.polygon]
        ax.add_patch(Polygon(poly, closed=True, facecolor=ORANGE if p.name in moving else BLUE,
                             edgecolor=NAVY, lw=0.8, hatch="//" if p.name == s.root else None, alpha=0.95))
        c = _xy(p.centroid)
        ax.text(*c, p.name + ("\n(root)" if p.name == s.root else ""), ha="center", va="center",
                fontsize=8, color=INK)
    for h in s.hinges:
        (x0, y0), (x1, y1) = _xy(h.a), _xy(h.b)
        if h.child in target:
            ax.plot([x0, x1], [y0, y1], color=NAVY, lw=2.6, solid_capstyle="butt")
        else:
            ax.plot([x0, x1], [y0, y1], color=MUTED, lw=1.4, ls=(0, (4, 3)))
    for sm in s.seams:
        (x0, y0), (x1, y1) = _xy(sm.a), _xy(sm.b)
        ax.plot([x0, x1], [y0, y1], color=RED, lw=1.6, ls=(0, (1, 2)))
    if s.grasp is not None:
        ax.plot(*_xy(s.grasp[1]), marker="o", ms=6, mfc="white", mec=RED, mew=1.6)
    ax.text(L + 0.12 * L, 0, "MOVING HALF  +x", rotation=-90, ha="center", va="center", fontsize=8,
            weight="bold", color="#8a5a2b")
    ax.text(-L - 0.12 * L, 0, "SUPPORT HALF  −x", rotation=90, ha="center", va="center", fontsize=8,
            weight="bold", color="#3d6585")
    o = (-L - 0.25 * L, -L - 0.25 * L)
    for d, lab in (((0.25 * L, 0.0), "+x"), ((0.0, 0.25 * L), "+y")):
        ax.annotate("", xy=(o[0] + d[0], o[1] + d[1]), xytext=o, arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
        ax.text(o[0] + 1.3 * d[0], o[1] + 1.3 * d[1], lab, fontsize=7, color=MUTED, ha="center", va="center")
    x = 0.57
    fig.text(x, 0.72, str(len(s.plates)), fontsize=30, weight="bold", color=NAVY)
    fig.text(x + 0.05, 0.735, f"plates  ·  {len(s.hinges)} hinges  ·  {len(s.seams)} seam{'' if len(s.seams) == 1 else 's'}", fontsize=10, color=MUTED)
    handles = [Line2D([], [], color=NAVY, lw=2.6, label="target crease (half fold)"),
               Line2D([], [], color=MUTED, lw=1.4, ls=(0, (4, 3)), label="other hinge"),
               Line2D([], [], color=RED, lw=1.6, ls=(0, (1, 2)), label="seam (loop closure)"),
               Line2D([], [], color=RED, marker="o", mfc="white", ls="", label="viewer grasp point")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(x - 0.005, 0.68), frameon=False, fontsize=8.5)
    y = 0.47
    fig.text(x, y, "FOLDS  (stages separated by →)", fontsize=8, weight="bold", color=NAVY)
    step = min(0.04, 0.38 / max(len(s.folds), 1))
    for name, stages in s.folds.items():
        y -= step
        seq = "  →  ".join(", ".join(f"{k} {math.degrees(v):+.0f}°" for k, v in st.items()) for st in stages)
        fig.text(x, y, name, fontsize=8, color=INK, weight="bold" if name == fs.HALF_FOLD else None)
        fig.text(x + 0.17, y, seq, fontsize=7.5, color=MUTED)
    fig.text(0.06, 0.03, f"Source: {path.name}  ·  {s.size * 100:.0f} cm sheet at nominal size, "
             f"{s.thickness * 1e3:.0f} mm plates, {s.hinge}-plane hinges. Flat sheet; not a simulation render.",
             fontsize=7.5, color=MUTED)
    out = path.with_suffix(".png")
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    fs = _fold_shapes()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", nargs="*", default=None, help="design names (default: all)")
    args = p.parse_args()
    for path in sorted(fs.DESIGN_DIR.glob("*.yaml")):
        if args.only and path.stem not in args.only:
            continue
        print("wrote", render(fs, path))


if __name__ == "__main__":
    main()
