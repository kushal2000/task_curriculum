"""Custom fold-shape designs from the viewer's Design panel, built with the family generators
(``scripts/visualization/rigid_cloth/make_2vert_hinge.py`` / ``make_2flaps_hinge.py``) and written to
``config/designs/custom/``, where ``fold_shapes`` picks them up at the next start.

Coordinates are the designs' own: the sheet is [-1, 1]^2 in units of L = size / 2.
"""

from __future__ import annotations

import re
from pathlib import Path

from isaacsimenvs.tasks.cloth.mdp.geometry import fold_shapes as fs

CUSTOM_DIR = Path(fs.DESIGN_DIR) / "custom"
FAMILIES = ("2vert+hinge", "2flaps+hinge")

#: 2flaps crease kinds for one half, in the LEFT half's frame (x in [-1, 0]); t1, t2 in (0, 1).
FLAP_KINDS = ("strip", "fan from top centre", "fan from bottom centre", "cut from top", "cut from bottom", "diagonal")


def base_names() -> list[str]:
    """The pool's own designs (the top level of config/designs: box2-mid, box4-mid, ...)."""
    return [n for n, p in fs.design_paths().items() if p.parent == Path(fs.DESIGN_DIR)]


def family_names() -> list[str]:
    return [n for n, p in fs.design_paths().items() if p.parent.name in FAMILIES or p.parent.name == "custom"]


def _rename(text: str, old: str, new: str, note: str) -> str:
    text = text.replace(old, new)
    return f"# CUSTOM (live viewer Design panel): {note}\n" + text


def _write(name: str, text: str) -> str:
    CUSTOM_DIR.mkdir(parents=True, exist_ok=True)
    path = CUSTOM_DIR / f"{name}.yaml"
    existing = fs.design_paths()
    if name in existing and existing[name] != path:
        return name  # already a family member: use it as it is
    path.write_text(text)
    fs.load_design(path)  # validates: CCW plates, a tree of hinges, plates covering the sheet
    return name


def make_2vert(left: float, right: float) -> str:
    """Outer creases at x = -left L and +right L (0 < left, right < 1)."""
    from isaacsimenvs.scripts.visualization.rigid_cloth.make_2vert_hinge import design_yaml

    pa, pb = int(round(100 * left)), int(round(100 * right))
    family, text = design_yaml(pa, pb)
    if family in fs.design_paths():
        return family
    name = f"2vert_custom-L{pa}-R{pb}"
    return _write(name, _rename(text, family, name, f"2vert outer creases at -{left:.2f} / +{right:.2f} L"))


def _half(kind: str, t1: float, t2: float):
    """``(kind, p, q)`` of one crease in the LEFT half's frame."""
    if kind == "strip":
        return "strip", (-t1, 1.0), (-t2, -1.0)
    if kind == "fan from top centre":
        return "fan", (0.0, 1.0), (-1.0, round(2 * t1 - 1, 3))
    if kind == "fan from bottom centre":
        return "fan", (0.0, -1.0), (-1.0, round(2 * t1 - 1, 3))
    if kind == "cut from top":
        return "cut", (-t1, 1.0), (-1.0, round(2 * t2 - 1, 3))
    if kind == "cut from bottom":
        return "cut", (-t1, -1.0), (-1.0, round(2 * t2 - 1, 3))
    if kind == "diagonal":
        return "diag", (0.0, 1.0), (-1.0, -1.0)
    raise ValueError(kind)


def make_2flaps(left: tuple[str, float, float], right: tuple[str, float, float]) -> str:
    """One crease in each half: ``(kind, t1, t2)`` each, the right half given in its own mirrored frame."""
    from isaacsimenvs.scripts.visualization.rigid_cloth.make_2flaps_hinge import design_yaml

    lk, lp, lq = _half(*left)
    rk, rp, rq = _half(*right)
    r = (rk, (-rp[0] + 0.0, rp[1]), (-rq[0] + 0.0, rq[1]))
    family, text = design_yaml(0, (lk, lp, lq), r)
    tag = lambda k, a, b: re.sub(r"[^a-z0-9]+", "", k.split()[0] + k.split()[-1][:3]) + f"{int(100 * a):02d}{int(100 * b):02d}"
    name = f"2flaps_custom_{tag(*left)}_{tag(*right)}"
    return _write(name, _rename(text, family, name, f"2flaps left {left}, right {right}"))


def make_2vert_x(left_x: float, right_x: float) -> str:
    """Outer creases at x = left_x (in (-1, 0)) and x = right_x (in (0, 1)), sheet units of L."""
    if not (-1.0 < left_x < 0.0 and 0.0 < right_x < 1.0):
        raise ValueError(f"need -1 < left x < 0 < right x < 1, got {left_x}, {right_x}")
    return make_2vert(-left_x, right_x)


def make_2flaps_points(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> str:
    """One crease per half as its endpoints ``(x1, y1, x2, y2)`` in sheet coordinates (units of L): the left
    crease on the boundary of x in [-1, 0], the right one on x in [0, 1]."""
    from isaacsimenvs.scripts.visualization.rigid_cloth.make_2flaps_hinge import design_yaml

    def on_half(pt, sx):
        x, y = pt
        lo, hi = min(0.0, sx), max(0.0, sx)
        inside = lo - 1e-9 <= x <= hi + 1e-9 and -1 - 1e-9 <= y <= 1 + 1e-9
        on_edge = abs(abs(y) - 1) < 1e-6 or abs(x - sx) < 1e-6 or abs(x) < 1e-6
        return inside and on_edge

    lp, lq, rp, rq = (left[0], left[1]), (left[2], left[3]), (right[0], right[1]), (right[2], right[3])
    for name, pts, sx in (("left", (lp, lq), -1.0), ("right", (rp, rq), 1.0)):
        if not all(on_half(p, sx) for p in pts):
            raise ValueError(f"the {name} crease's ends must lie on the {name} half's boundary "
                             f"(x = 0, x = {sx:+g}, or y = +-1): {pts}")
    family, text = design_yaml(0, ("custom", lp, lq), ("custom", rp, rq))
    tag = lambda v: "".join(f"{'m' if c < 0 else 'p'}{abs(int(round(100 * c))):03d}" for c in v)
    name = f"2flaps_custom_{tag(left)}_{tag(right)}"
    return _write(name, _rename(text, family, name, f"2flaps creases left {left}, right {right} (L units)"))
