"""Render normalized YAML cloth designs as reproducible PNG schematics.

    .venv/isaaclab3/bin/python -m isaacsimenvs.scripts.visualization.rigid_cloth.render_designs

Only the catalog is read: these images illustrate crease geometry, not simulated cloth.
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import FancyArrowPatch, Polygon

from isaacsimenvs import CONFIG_DIR
from isaacsimenvs.tasks.cloth.backends.rigid import designs

NAVY = "#203c59"
BLUE = "#c6dbea"
ORANGE = "#f4c692"
INK = "#26384b"
MUTED = "#617181"
TOL = 1e-9


def render(path):
    data = yaml.safe_load(path.read_text())
    if data.get("shape") != "rectangle":
        raise ValueError(f"{path}: only rectangle designs are supported")
    creases = data["creases"]
    panels = [np.asarray(plate.vertices) for plate in designs.load_design(path).plates]
    centers = [name for name, pts in creases.items() if np.allclose(np.asarray(pts)[:, 0], 0.0, atol=TOL)]
    if len(centers) != 1:
        raise ValueError(f"{path}: expected exactly one target crease at x=0")
    image_name = Path(data["image"])
    if image_name.name != str(image_name) or image_name.suffix != ".png":
        raise ValueError(f"{path}: image must be a local PNG filename")

    fig = plt.figure(figsize=(14, 9), facecolor="white")
    fig.text(.065, .94, data["title"], color=INK, fontsize=26, weight="bold", va="top")
    fig.text(.067, .879, "YAML DESIGN  ·  GEOMETRY SCHEMATIC", fontsize=10, weight="bold", color=NAVY)
    fig.text(.067, .842, textwrap.fill(data["purpose"], 108), fontsize=12, color=MUTED, va="top", linespacing=1.4)

    ax = fig.add_axes([.055, .115, .69, .68])
    ax.set_aspect("equal")
    ax.set_xlim(-.79, .79)
    ax.set_ylim(-.69, .69)
    ax.axis("off")
    for panel in panels:
        # Screen coordinates are (y, x): the x=0 target crease is horizontal.
        color = ORANGE if panel[:, 0].mean() > TOL else BLUE
        ax.add_patch(Polygon(panel[:, ::-1], facecolor=color, edgecolor="white", linewidth=2))
    ax.add_patch(Polygon([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]],
                         fill=False, edgecolor=NAVY, linewidth=1.5))
    accessory = 0
    for name, endpoints in creases.items():
        points = np.asarray(endpoints)
        central = np.allclose(points[:, 0], 0.0, atol=TOL)
        ax.plot(points[:, 1], points[:, 0], color=NAVY if central else "#6a7680",
                linewidth=3 if central else 1.7, linestyle="-" if central else (0, (4, 3)),
                solid_capstyle="round", zorder=3)
        if not central:
            accessory += 1
    ax.text(0, .555, "MOVING HALF  +x", ha="center", color="#8f5e2f", fontsize=11, weight="bold")
    ax.text(0, -.56, "SUPPORT HALF  −x", ha="center", va="top", color="#426781", fontsize=11, weight="bold")
    ax.annotate("target fold", xy=(-.49, 0), xytext=(-.56, 0), ha="right", va="center",
                fontsize=10, color=NAVY, arrowprops={"arrowstyle": "-", "color": NAVY})
    ax.add_patch(FancyArrowPatch((.57, .31), (.57, -.31), connectionstyle="arc3,rad=-.6",
                                arrowstyle="-|>", mutation_scale=18, linewidth=1.8, color=NAVY))
    ax.text(.86, 0, "fold +x onto −x", rotation=90, ha="center", va="center", color=NAVY, fontsize=10)
    for end, label, offset in (((-.58, -.64), "+y", (.015, 0)), ((-.72, -.50), "+x", (0, .015))):
        ax.annotate("", xy=end, xytext=(-.72, -.64), arrowprops={"arrowstyle": "->", "color": MUTED})
        ax.text(end[0] + offset[0], end[1] + offset[1], label, color=MUTED, fontsize=9, va="center")

    fig.text(.785, .695, str(len(panels)), fontsize=42, weight="bold", color=NAVY)
    fig.text(.787, .658, "panels", fontsize=13, color=MUTED)
    fig.text(.787, .596, "1 target crease", fontsize=12, color=INK)
    fig.text(.787, .56, f"{accessory} accessory crease{'s' if accessory != 1 else ''}", fontsize=12, color=INK)
    legend = fig.add_axes([.785, .39, .16, .10])
    legend.axis("off")
    legend.plot([0, .22], [.75, .75], color=NAVY, linewidth=3)
    legend.plot([0, .22], [.25, .25], color="#6a7680", linewidth=1.7, linestyle=(0, (4, 3)))
    legend.text(.29, .75, "target fold", va="center", fontsize=10, color=INK)
    legend.text(.29, .25, "accessory", va="center", fontsize=10, color=INK)
    legend.set(xlim=(0, 1), ylim=(0, 1))
    fig.text(.787, .315, "Normalized coordinates\nx, y ∈ [−½, ½]", fontsize=10, color=MUTED, linespacing=1.5)
    fig.text(.787, .235, "Loaded by the viewer\nand cloth environment",
             fontsize=11, color=NAVY, linespacing=1.5)
    fig.text(.067, .045, f"Source: {path.name}   ·   Flat sheet; creases define panel boundaries. Not a simulation render.",
             fontsize=10, color=MUTED)
    png = path.with_name(image_name.name)
    fig.savefig(png, dpi=100, metadata={"Software": "render_cloth_designs.py"})
    plt.close(fig)
    return png


def overview(paths):
    entries = [(path, yaml.safe_load(path.read_text())) for path in paths]
    preferred = {name: i for i, name in enumerate(("single_hinge", "strip_4", "strip_8", "hinge_band", "flexible_corners"))}
    entries.sort(key=lambda item: (preferred.get(item[0].stem, len(preferred)), item[0].stem))
    rows = (len(entries) + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(18, 5 * rows + 1), squeeze=False, facecolor="white")
    fig.suptitle("Cloth design catalog", x=.055, ha="left", color=INK, fontsize=28, weight="bold")
    for ax, (path, data) in zip(axes.flat, entries):
        panels = [np.asarray(plate.vertices) for plate in designs.load_design(path).plates]
        for panel in panels:
            ax.add_patch(Polygon(panel[:, ::-1], facecolor=ORANGE if panel[:, 0].mean() > TOL else BLUE,
                                 edgecolor="white", linewidth=2))
        ax.add_patch(Polygon([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]],
                             fill=False, edgecolor=NAVY, linewidth=1.3))
        for endpoints in data["creases"].values():
            points = np.asarray(endpoints)
            central = np.allclose(points[:, 0], 0.0, atol=TOL)
            ax.plot(points[:, 1], points[:, 0], color=NAVY if central else "#6a7680",
                    linewidth=3 if central else 1.5, linestyle="-" if central else (0, (4, 3)))
        ax.set_title(data["title"], fontsize=16, color=INK, weight="bold", pad=12)
        ax.text(0, -.58, f"{len(panels)} panels · YAML design", fontsize=11, ha="center", color=MUTED)
        ax.set(xlim=(-.58, .58), ylim=(-.66, .58), aspect="equal")
        ax.axis("off")
    for ax in list(axes.flat)[len(entries):]:
        ax.axis("off")
    if len(entries) < axes.size:
        legend = axes.flat[-1]
        legend.text(.12, .75, "One target fold", fontsize=22, color=NAVY, weight="bold", transform=legend.transAxes)
        legend.text(.12, .61, "Orange +x half folds onto blue −x half.", fontsize=12, color=INK, transform=legend.transAxes)
        legend.plot([.12, .32], [.48, .48], color=NAVY, linewidth=3, transform=legend.transAxes)
        legend.text(.37, .48, "target crease", va="center", fontsize=12, color=INK, transform=legend.transAxes)
        legend.plot([.12, .32], [.36, .36], color="#6a7680", linewidth=1.7, linestyle=(0, (4, 3)), transform=legend.transAxes)
        legend.text(.37, .36, "accessory crease", va="center", fontsize=12, color=INK, transform=legend.transAxes)
        legend.text(.12, .17, "Normalized geometry schematics\nNot simulation renders", fontsize=12, color=MUTED,
                    linespacing=1.5, transform=legend.transAxes)
    fig.subplots_adjust(left=.05, right=.96, bottom=.045, top=.88, wspace=.22, hspace=.32)
    destination = paths[0].parent / "overview.png"
    fig.savefig(destination, dpi=100, metadata={"Software": "render_cloth_designs.py"})
    plt.close(fig)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("designs", nargs="*", type=Path, help="YAML paths; defaults to every catalog design")
    args = parser.parse_args()
    matplotlib.rcParams.update({"font.family": "DejaVu Sans"})
    paths = args.designs or sorted((CONFIG_DIR / "rigid_cloth_designs").glob("*.yaml"))
    if not paths:
        parser.error("No design YAML files found")
    for path in paths:
        print(render(path))
    if len(paths) > 1:
        print(overview(paths))


if __name__ == "__main__":
    main()
