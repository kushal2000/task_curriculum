"""Write the 2flaps+hinge fold-shape group: the half fold plus one random crease in each half.

    .venv_isaaclab3/bin/python isaacsimenvs/scripts/visualization/rigid_cloth/make_2flaps_hinge.py

Writes ``tasks/cloth/config/designs/2flaps+hinge/2flaps_hinge_NN.yaml`` (``N`` designs, seeded) and the
group diagram ``2flaps+hinge.png`` beside them, drawn from the written YAMLs through
``fold_shapes.load_design``.

Each half gets one crease, drawn independently (no mirroring) from four kinds:

    diag   25%  middle-crease end (0, +-1) -> the opposite outer corner
    fan    30%  middle-crease end -> a point on the outer edge or the far top / bottom edge
    cut    30%  top or bottom edge -> outer edge (cuts an outer corner)
    strip  15%  top edge -> bottom edge (a slanted strip; rare, it is close to box4-mid)

Edge points are uniform along the edge, kept 0.1 of it off the corners, and rounded to 0.01 L. A
crease is redrawn (same kind) until its flap -- the piece NOT holding the middle crease -- covers
15-60 % of the half and it is at least 0.8 L long. A crease touches the middle crease only at its
ends (0, +-1), so the four plates form a chain l_out - l_in - r_in - r_out and need no seam; every
plate is a convex piece of a rectangle cut by one line.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
OUT_DIR = REPO / "isaacsimenvs/tasks/cloth/config/designs/2flaps+hinge"
N, SEED = 32, 0
KINDS, KIND_P = ("diag", "fan", "cut", "strip"), (0.25, 0.30, 0.30, 0.15)
FLAP_FRAC, MIN_LEN = (0.15, 0.60), 0.8
GRASP_INSET_L = 0.12  # fold_shapes.GRASP_INSET (6 mm) at the nominal L = 50 mm
IN_L, OUT_L, IN_R, OUT_R, INK = "#c9dcea", "#9fbfd8", "#f8e2c8", "#f2c79a", "#24384f"


def _fold_shapes():
    # By path: importing the package would register every gym task.
    path = REPO / "isaacsimenvs/tasks/cloth/mdp/geometry/fold_shapes.py"
    spec = importlib.util.spec_from_file_location("fold_shapes", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fold_shapes"] = mod
    spec.loader.exec_module(mod)
    return mod


def _cross(a, b) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _signed_area(poly) -> float:
    return 0.5 * sum(_cross(poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly)))


def split(sx: int, p, q):
    """Cut the half ``x in [0, sx]`` along ``p -> q`` (both on its boundary): ``(inner, flap)``, CCW.

    inner holds both ends of the middle crease (0, -1) and (0, 1); the flap is the rest."""
    square = [(0.0, -1.0), (sx, -1.0), (sx, 1.0), (0.0, 1.0)]
    p, d = np.array(p, float), np.subtract(q, p)

    def side(v):
        c = _cross(d, np.subtract(v, p))
        return 0 if abs(c) < 1e-9 else (1 if c > 0 else -1)

    a, b = [], []
    for i in range(4):
        v, w = square[i], square[(i + 1) % 4]
        sv, sw = side(v), side(w)
        if sv >= 0:
            a.append(tuple(v))
        if sv <= 0:
            b.append(tuple(v))
        if sv * sw < 0:  # the crease crosses this edge: at p or q
            x = p if abs(_cross(np.subtract(w, v), p - np.array(v))) < 1e-9 else np.array(q, float)
            a.append(tuple(x))
            b.append(tuple(x))
    holds = lambda poly: {(0.0, -1.0), (0.0, 1.0)} <= {tuple(map(float, v)) for v in poly}
    inner, flap = (a, b) if holds(a) else (b, a)
    ccw = lambda poly: poly if _signed_area(poly) > 0 else poly[::-1]
    return ccw(inner), ccw(flap)


def _edge_point(rng, edge: str):
    t = round(float(rng.uniform(0.1, 0.9)), 2)
    return {"top": (-t, 1.0), "bottom": (-t, -1.0), "outer": (-1.0, round(2 * t - 1, 2))}[edge]


def sample_half(rng):
    """``(kind, p, q)``: one crease of the LEFT half (mirror x for the right)."""
    kind = str(rng.choice(KINDS, p=KIND_P))
    while True:
        if kind == "diag":
            s = float(rng.choice([-1.0, 1.0]))
            p, q = (0.0, s), (-1.0, -s)
        elif kind == "fan":
            s = float(rng.choice([-1.0, 1.0]))
            p, q = (0.0, s), _edge_point(rng, str(rng.choice(["outer", "bottom" if s > 0 else "top"])))
        elif kind == "cut":
            p, q = _edge_point(rng, str(rng.choice(["top", "bottom"]))), _edge_point(rng, "outer")
        else:
            p, q = _edge_point(rng, "top"), _edge_point(rng, "bottom")
        _, flap = split(-1, p, q)
        if FLAP_FRAC[0] <= _signed_area(flap) / 2.0 <= FLAP_FRAC[1] and math.dist(p, q) >= MIN_LEN:
            return kind, p, q


def _inside(poly, xy, margin=0.02) -> bool:
    """Strictly inside a CCW convex polygon, at least ``margin`` from every edge."""
    for i in range(len(poly)):
        a, b = np.array(poly[i]), np.array(poly[(i + 1) % len(poly)])
        e = b - a
        if _cross(e, np.subtract(xy, a)) / np.linalg.norm(e) < margin:
            return False
    return True


def _grasp(r_in, r_out):
    """Viewer pinch point on the moving (+x) half: an outer corner or the outer edge's middle."""
    for xy, inset in (((1, -1), (-1, 1)), ((1, 1), (-1, -1)), ((1, 0), (-1, 0))):
        pt = (xy[0] + GRASP_INSET_L * inset[0], xy[1] + GRASP_INSET_L * inset[1])
        for name, poly in (("r_out", r_out), ("r_in", r_in)):
            if _inside(poly, pt):
                return name, xy, inset
    raise RuntimeError("no grasp point")


def _num(v: float) -> str:
    v = round(float(v), 2) + 0.0
    return str(int(v)) if v == int(v) else f"{v:g}"


def _pts(poly) -> str:
    return "[" + ", ".join(f"[{_num(x)}, {_num(y)}]" for x, y in poly) + "]"


def design_yaml(i: int, left, right) -> tuple[str, str]:
    (lk, lp, lq), (rk, rp, rq) = left, right
    l_in, l_out = split(-1, lp, lq)
    r_in, r_out = split(1, rp, rq)
    gp, gxy, gin = _grasp(r_in, r_out)
    name = f"2flaps_hinge_{i:02d}"
    text = f"""\
# {name}: half fold + one random crease in each half (left {lk}, right {rk}), 1 mm plates
#
# One of the {N} 2flaps+hinge variants (diagram: 2flaps+hinge.png in this folder, numbered row by
# row), written by isaacsimenvs/scripts/visualization/rigid_cloth/make_2flaps_hinge.py (seed {SEED}).
# Each half has one crease, drawn independently of the other half; it meets the centre crease only
# at its ends (0, +-1), so the plates form a chain l_out - l_in - r_in - r_out and need no seam.
# The target is the half fold; the flap folds give folded starts. The sheet size is set per env by
# the recipe (fold_pool.size_range); `size` here is nominal.
# Coordinates are in units of L = size / 2 (the sheet frame is [-1, 1]^2, +z up); fold angles in
# degrees, +180 = a full fold up and over onto the parent. Schema: fold_shapes.load_design.
name: {name}
label: 2flaps+hinge {i:02d}, left {lk} / right {rk} crease (4 plates)
size: 0.1
thickness: 0.001
hinge: mid
plates:  # first = root (the stationary piece); CCW polygons
  l_in:  {_pts(l_in)}
  l_out: {_pts(l_out)}
  r_in:  {_pts(r_in)}
  r_out: {_pts(r_out)}
hinges:  # child: [parent, line start, line end]
  l_out: [l_in, {_pts([lp, lq])[1:-1]}]
  r_in:  [l_in, [0, -1], [0, 1]]
  r_out: [r_in, {_pts([rp, rq])[1:-1]}]
seams: []
folds:  # name: [stage, ...]; a stage turns its hinges together
  "half: right over left": [{{r_in: 180}}]
  "left flap": [{{l_out: 180}}]
  "right flap": [{{r_out: 180}}]
  "both flaps": [{{l_out: 180, r_out: 180}}]
grasp: {{plate: {gp}, xy: [{gxy[0]}, {gxy[1]}], inset: [{gin[0]}, {gin[1]}]}}  # viewer pinch point: xy * L + inset * 6 mm
"""
    return name, text


def draw(fs, paths: list[Path], out: Path, cols: int = 8) -> None:
    rows = math.ceil(len(paths) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.55 * rows))
    for ax in axes.flat:
        ax.axis("off")
    colour = {"l_in": IN_L, "l_out": OUT_L, "r_in": IN_R, "r_out": OUT_R}
    for i, (ax, path) in enumerate(zip(axes.flat, paths)):
        s = fs.load_design(path)
        L = s.size / 2.0
        for p in s.plates:
            ax.add_patch(Polygon(np.asarray(p.polygon) / L, fc=colour[p.name], ec=INK, lw=0.7))
        for h in s.hinges:
            a, b = np.asarray(h.a) / L, np.asarray(h.b) / L
            ax.plot([a[0], b[0]], [a[1], b[1]], color=INK, lw=2.6 if h.child == "r_in" else 1.0,
                    ls="-" if h.child == "r_in" else (0, (4, 2)))
        ax.text(-1.05, 1.08, f"{i:02d}", fontsize=10, color=INK, va="bottom")
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.3)
        ax.set_aspect("equal")
    fig.subplots_adjust(0.01, 0.01, 0.99, 0.99, 0.05, 0.05)
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main() -> None:
    fs = _fold_shapes()
    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(N):
        left = sample_half(rng)
        k, p, q = sample_half(rng)
        right = (k, (-p[0] + 0.0, p[1]), (-q[0] + 0.0, q[1]))
        name, text = design_yaml(i, left, right)
        path = OUT_DIR / f"{name}.yaml"
        path.write_text(text)
        fs.load_design(path)  # validates: CCW plates, tree, area = sheet
        paths.append(path)
        print(f"{name}: left {left[0]:5s} right {right[0]}")
    out = OUT_DIR / "2flaps+hinge.png"
    draw(fs, paths, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
