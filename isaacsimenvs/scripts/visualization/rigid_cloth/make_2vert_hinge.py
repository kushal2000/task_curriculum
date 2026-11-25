"""Write the 2vert+hinge fold-shape group: four vertical strips, the centre crease fixed and one outer
crease on each side on a 5 x 5 grid.

    .venv_isaacsim/bin/python isaacsimenvs/scripts/visualization/rigid_cloth/make_2vert_hinge.py [--out DIR]

Writes ``tasks/cloth/config/designs/2vert+hinge/2vert_hinge-L{a}-R{b}.yaml`` (25 designs) and the group
diagram ``2vert+hinge.png`` beside them. ``a`` / ``b``: the left / right outer crease at x = -a / +b
hundredths of L, each from 20, 35, 50, 65, 80. box4-mid's layout with the outer creases moved: rooted
next to the centre crease, so the half fold is one hinge and the outer strips ride along flat. Only
the half fold is listed (training starts flat). Shape names use ``_``, not ``+``: ``+`` is not allowed
in a USD prim name.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
OUT_DIR = REPO / "isaacsimenvs/tasks/cloth/config/designs/2vert+hinge"
POS = (20, 35, 50, 65, 80)  # |outer crease x| in hundredths of L


def _num(v: int) -> str:
    return f"{v / 100:g}"


def design_yaml(pa: int, pb: int) -> tuple[str, str]:
    """``(name, YAML text)`` of the variant with outer creases at x = -pa / +pb hundredths of L."""
    a, b = _num(-pa), _num(pb)
    name = f"2vert_hinge-L{pa}-R{pb}"
    return name, f"""\
# {name}: four vertical strips, centre crease + outer creases at x = {a} L, {b} L, 1 mm plates
#
# One of the 25 2vert+hinge variants (diagram: 2vert+hinge.png in this folder, rows = left crease -0.2 .. -0.8,
# columns = right crease 0.2 .. 0.8, steps of 0.15). box4-mid's layout with the two outer creases
# moved: rooted next to the centre crease, so the half fold is one hinge and the outer strips ride
# along flat. Only the half fold is listed: training starts flat and folds the half. The sheet size
# is set per env by the recipe (fold_pool.size_range); `size` here is nominal.
# Coordinates are in units of L = size / 2 (the sheet frame is [-1, 1]^2, +z up); fold angles in
# degrees, +180 = a full fold up and over onto the parent. Schema: fold_shapes.load_design.
name: {name}
label: 2vert+hinge, outer creases at {a} / {b} (4 plates)
size: 0.1
thickness: 0.001
hinge: mid
plates:  # first = root (the stationary piece); CCW polygons
  s2: [[{a}, -1], [0, -1], [0, 1], [{a}, 1]]
  s1: [[-1, -1], [{a}, -1], [{a}, 1], [-1, 1]]
  s3: [[0, -1], [{b}, -1], [{b}, 1], [0, 1]]
  s4: [[{b}, -1], [1, -1], [1, 1], [{b}, 1]]
hinges:  # child: [parent, line start, line end]
  s1: [s2, [{a}, -1], [{a}, 1]]
  s3: [s2, [0, -1], [0, 1]]
  s4: [s3, [{b}, -1], [{b}, 1]]
seams: []
folds:  # name: [stage, ...]; a stage turns its hinges together
  "half: right over left": [{{s3: 180}}]
grasp: {{plate: s4, xy: [1, 0], inset: [-1, 0]}}  # viewer pinch point: xy * L + inset * 6 mm
"""


def draw(out: Path) -> None:
    """The 5 x 5 overview: rows = left crease, columns = right crease; the folding half in orange,
    the centre crease solid, the outer creases dashed."""
    navy, blue, orange = "#1f3a52", "#c9dceb", "#f3c99b"
    n = len(POS)
    fig, axes = plt.subplots(n, n, figsize=(9, 9))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, hspace=0.12, wspace=0.12)
    for i, pa in enumerate(POS):
        for j, pb in enumerate(POS):
            a, b, ax = -pa / 100, pb / 100, axes[i, j]
            for x0, x1, c in [(-1, a, blue), (a, 0, blue), (0, b, orange), (b, 1, orange)]:
                ax.add_patch(Rectangle((x0, -1), x1 - x0, 2, fc=c, ec="none"))
            ax.add_patch(Rectangle((-1, -1), 2, 2, fc="none", ec=navy, lw=1))
            for x in (a, b):
                ax.plot([x, x], [-1, 1], color=navy, lw=1.1, ls=(0, (4, 3)))
            ax.plot([0, 0], [-1, 1], color=navy, lw=2.6)
            ax.set_xlim(-1.03, 1.03)
            ax.set_ylim(-1.03, 1.03)
            ax.set_aspect("equal")
            ax.axis("off")
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, default=OUT_DIR)
    out = p.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    for pa in POS:
        for pb in POS:
            name, text = design_yaml(pa, pb)
            (out / f"{name}.yaml").write_text(text)
    draw(out / "2vert+hinge.png")
    print(f"wrote {len(POS) ** 2} designs + 2vert+hinge.png to {out}")


if __name__ == "__main__":
    main()
