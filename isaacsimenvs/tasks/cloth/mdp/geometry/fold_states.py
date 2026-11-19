"""One-fold tasks on the pool shapes: reachable folded states, single folds between them, and
which folded states look the same.

A **state** is a set of fully folded hinges (every hinge at 0 or +-pi) that a sequence of single
folds reaches from the flat sheet. A **move** is one fold: a set of hinges turned together by
+-pi that carries its plates as ONE rigid piece about one crease line, up off the table, with no
other plate lying on top of that piece. A **task** is (start state, move): fold once.

Two states are **congruent** when their sheets, seen from above without point identities, are the
same point set up to a rotation and a translation in the table plane. A fold "to the left" and
the same fold "to the right", or along the other mid-line of a square, land in one class; so does
any other fold sequence that ends in the same outline (a gate fold and a half fold are both a
double-layer half-width rectangle). ``fold_pool.goal_task.accept_lookalike`` accepts every member.
The goal tables built from these per env are ``mdp/goal_tasks.py``.

Pure numpy, no torch, no simulator (unit-tested on CPU). Mid-plane hinges only: every folded plate
lies in the root's plane, so a state is planar and layer order is bookkeeping, not geometry.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from . import fold_shapes as fs

__all__ = ["FoldState", "FoldMove", "ShapeTasks", "enumerate_tasks", "state_grid", "task_arrays"]

PI = math.pi
_CHECK_DEG = (45.0, 90.0, 135.0, 180.0)
_FINE = 25


@dataclass(frozen=True)
class FoldState:
    angles: tuple[float, ...]
    """Signed hinge angle (0 or +-pi), in ``shape.hinges`` order."""
    layers: tuple[int, ...]
    """Stacking rank of each plate, in ``shape.plates`` order (0 = on the table)."""

    @property
    def folded(self) -> frozenset[int]:
        return frozenset(i for i, a in enumerate(self.angles) if abs(a) > 1e-9)


@dataclass(frozen=True)
class FoldMove:
    start: int
    goal: int
    hinges: tuple[tuple[str, float], ...]
    """(hinge child, signed angle) of the hinges this fold turns."""


@dataclass(frozen=True)
class ShapeTasks:
    shape: fs.FoldShape
    states: tuple[FoldState, ...]
    moves: tuple[FoldMove, ...]
    classes: tuple[tuple[int, ...], ...]
    """``classes[i]``: every state congruent to state ``i``, itself first."""


def _angles(shape: fs.FoldShape, state_angles) -> dict[str, float]:
    return {h.child: float(a) for h, a in zip(shape.hinges, state_angles)}


def _polys(shape: fs.FoldShape, tf) -> dict[str, np.ndarray]:
    """Each plate's polygon in the root frame, CCW, ``(k, 2)``."""
    out = {}
    for p in shape.plates:
        r, t = tf[p.name]
        v = np.array([r @ np.array([x, y, 0.0]) + t for x, y in p.polygon])[:, :2]
        x, y = v[:, 0], v[:, 1]
        if float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)) < 0.0:
            v = v[::-1]
        out[p.name] = v
    return out


def _overlap(p: np.ndarray, q: np.ndarray) -> float:
    """Area of the intersection of two convex CCW polygons (Sutherland-Hodgman)."""
    out = [tuple(v) for v in p]
    for a, b in zip(q, np.roll(q, -1, axis=0)):
        if not out:
            return 0.0
        e = b - a
        inside = lambda v: e[0] * (v[1] - a[1]) - e[1] * (v[0] - a[0]) >= -1e-12  # noqa: E731
        nxt = []
        for i, cur in enumerate(out):
            prev = out[i - 1]
            ci, pi_ = inside(cur), inside(prev)
            if ci != pi_:
                d = (cur[0] - prev[0], cur[1] - prev[1])
                den = e[0] * d[1] - e[1] * d[0]
                if abs(den) > 1e-15:
                    s = (e[0] * (a[1] - prev[1]) - e[1] * (a[0] - prev[0])) / den
                    nxt.append((prev[0] + s * d[0], prev[1] + s * d[1]))
            if ci:
                nxt.append(cur)
        out = nxt
    if len(out) < 3:
        return 0.0
    v = np.array(out)
    return 0.5 * abs(float(np.dot(v[:, 0], np.roll(v[:, 1], -1)) - np.dot(np.roll(v[:, 0], -1), v[:, 1])))


def _try_move(shape: fs.FoldShape, state: FoldState, hinges: tuple[int, ...]):
    """The fold of ``hinges`` from ``state`` if it is one legal fold: ``(angles, layers)`` or ``None``."""
    names = [h.child for h in shape.hinges]
    plates = [p.name for p in shape.plates]
    if any(abs(state.angles[i]) > 1e-9 for i in hinges):
        return None
    base = _angles(shape, state.angles)
    tf0 = shape.transforms(base)
    eps = 1e-9 * shape.size
    area_eps = 1e-6 * shape.size**2
    for signs in itertools.product((1.0, -1.0), repeat=len(hinges)):
        moving, ok = None, True
        for deg in _CHECK_DEG:
            ang = dict(base)
            ang.update({names[i]: s * math.radians(deg) for i, s in zip(hinges, signs)})
            if shape.seam_error(ang) > eps:
                ok = False
                break
            tf = shape.transforms(ang)
            delta = {}
            for p in shape.plates:
                r0, t0 = tf0[p.name]
                r1, t1 = tf[p.name]
                dr = r1 @ r0.T
                dt = t1 - dr @ t0
                if np.abs(dr - np.eye(3)).max() > 1e-9 or np.abs(dt).max() > eps:
                    delta[p.name] = (dr, dt)
                for x, y in p.polygon:
                    if (r1 @ np.array([x, y, 0.0]) + t1)[2] < -eps:
                        ok = False
            if not ok or not delta:
                ok = False
                break
            first = next(iter(delta.values()))
            if any(np.abs(dr - first[0]).max() > 1e-9 or np.abs(dt - first[1]).max() > eps for dr, dt in delta.values()):
                ok = False                      # two pieces turning about two lines: not one fold
                break
            if moving is None:
                moving = set(delta)
            elif moving != set(delta):
                ok = False
                break
        if not ok:
            continue
        # Nothing that stays may lie on top of the piece that moves.
        poly0 = _polys(shape, tf0)
        layer = dict(zip(plates, state.layers))
        still = [p for p in plates if p not in moving]
        if any(
            layer[q] > layer[m] and _overlap(poly0[m], poly0[q]) > area_eps
            for m in moving for q in still
        ):
            continue
        ang = dict(base)
        ang.update({names[i]: s * PI for i, s in zip(hinges, signs)})
        poly1 = _polys(shape, shape.transforms(ang))
        under = [layer[q] for m in moving for q in still if _overlap(poly1[m], poly1[q]) > area_eps]
        floor = max(under) if under else -1
        top = max(layer[m] for m in moving)
        new_layers = tuple(floor + 1 + (top - layer[p]) if p in moving else layer[p] for p in plates)
        return tuple(ang[n] for n in names), new_layers
    return None


def _fine_xy(shape: fs.FoldShape, state_angles, resolution: int = _FINE) -> np.ndarray:
    """The state seen from above: its distinct grid points (a stack of plies is one point)."""
    xy = state_grid(shape, state_angles, resolution)[:, :2]
    step = shape.size / (resolution - 1)
    _, keep = np.unique(np.round(xy / (1e-3 * step)).astype(np.int64), axis=0, return_index=True)
    return xy[np.sort(keep)]


def _congruent(a: np.ndarray, b: np.ndarray, tol: float) -> bool:
    if len(a) != len(b):
        return False
    a0, b0 = a - a.mean(axis=0), b - b.mean(axis=0)
    ra, rb = np.sort(np.linalg.norm(a0, axis=1)), np.sort(np.linalg.norm(b0, axis=1))
    if np.abs(ra - rb).max() > tol:
        return False
    for k in range(8):
        c, s = math.cos(k * PI / 4.0), math.sin(k * PI / 4.0)
        rot = a0 @ np.array([[c, s], [-s, c]])
        d = np.linalg.norm(rot[:, None, :] - b0[None, :, :], axis=-1)
        if d.min(axis=1).max() < tol and d.min(axis=0).max() < tol:
            return True
    return False


def state_grid(shape: fs.FoldShape, state_angles, resolution: int) -> np.ndarray:
    """``(resolution^2, 3)`` the cloth grid (``grid.grid_mesh`` order) of ``shape`` at a
    state, in the root plate's frame."""
    tf = shape.transforms(_angles(shape, state_angles))
    out = np.zeros((resolution * resolution, 3))
    for k, (plate, local) in enumerate(fs.grid_plate_map(shape, resolution)):
        r, t = tf[plate]
        out[k] = r @ np.asarray(local) + t
    return out


def enumerate_tasks(shape: fs.FoldShape) -> ShapeTasks:
    if shape.hinge != "mid":
        raise ValueError(f"{shape.name}: one-fold tasks need mid-plane hinges")
    names = [h.child for h in shape.hinges]
    cand = {tuple(sorted(names.index(c) for c in st)) for stages in shape.folds.values() for st in stages}
    cand |= {(i,) for i in range(len(names))}
    cand = sorted(cand, key=lambda h: (len(h), h))
    states = [FoldState((0.0,) * len(names), (0,) * len(shape.plates))]
    index = {states[0].folded: 0}
    moves: list[FoldMove] = []
    queue = [0]
    while queue:
        i = queue.pop(0)
        for hinges in cand:
            got = _try_move(shape, states[i], hinges)
            if got is None:
                continue
            new = FoldState(*got)
            j = index.get(new.folded)
            if j is None:
                j = len(states)
                states.append(new)
                index[new.folded] = j
                queue.append(j)
            moves.append(FoldMove(i, j, tuple((names[h], got[0][h]) for h in hinges)))
    tol = 1e-6 * shape.size
    fine = [_fine_xy(shape, s.angles) for s in states]
    classes = []
    for i in range(len(states)):
        same = [j for j in range(len(states)) if j != i and _congruent(fine[i], fine[j], tol)]
        classes.append((i, *same))
    return ShapeTasks(shape, tuple(states), tuple(moves), tuple(classes))


def _spread(pts: np.ndarray) -> float:
    """Second singular value of the centred points: 0 for a point or a line."""
    if len(pts) < 3:
        return 0.0
    return float(np.linalg.svd(pts - pts.mean(axis=0), compute_uv=False)[1])


def _farthest(pts: np.ndarray, k: int) -> list[int]:
    pick = [int(np.argmax(np.linalg.norm(pts - pts.mean(axis=0), axis=1)))]
    while len(pick) < k:
        d = np.min(np.linalg.norm(pts[:, None, :] - pts[pick][None, :, :], axis=-1), axis=1)
        pick.append(int(np.argmax(d)))
    return pick


def task_arrays(st: ShapeTasks, resolution: int, only_half: bool = False) -> list[dict]:
    """The tasks of one shape as arrays on the env's cloth grid, nominal size (``shape.size``).

    Each entry: ``start`` / ``goal`` (state index), ``angles`` (start hinge angles, ``shape.hinges``
    order), ``p_start`` / ``p_goal`` ``(P, 3)``, ``moving`` ``(P,)`` bool, ``kp`` (4 grid indices on
    the moving piece), ``rot`` ``(3, 3)`` taking the keypoints' start offsets to their goal offsets,
    ``direction`` [rad] the way the moving piece travels in the root frame, and the congruent goals
    ``members`` = ``[(state', p_goal', frame')]`` (``frame'``: the particles that goal leaves where they are),
    the task's own goal first. ``only_half``: the flat sheet's half fold alone (the stage-1 task).

    A task whose moving piece or frame has fewer than three non-collinear grid points is left out
    (the rigid fits need them); so is a class member with such a frame.
    """
    shape = st.shape
    names = [h.child for h in shape.hinges]
    grids = [state_grid(shape, s.angles, resolution) for s in st.states]
    tol = 1e-7 * shape.size
    min_spread = 0.05 * shape.size
    half = tuple(sorted(names.index(c) for c in shape.folds[fs.HALF_FOLD][0]))
    out = []
    for mv in st.moves:
        if only_half and not (mv.start == 0 and tuple(sorted(names.index(c) for c, _ in mv.hinges)) == half):
            continue
        pa, pb = grids[mv.start], grids[mv.goal]
        moving = np.linalg.norm(pb - pa, axis=1) > tol
        if _spread(pa[moving]) < min_spread or _spread(pa[~moving]) < min_spread:
            continue
        idx = np.nonzero(moving)[0]
        kp = idx[_farthest(pa[idx], 4)]
        a, b = pa[kp] - pa[kp].mean(axis=0), pb[kp] - pb[kp].mean(axis=0)
        u, _, vt = np.linalg.svd(a.T @ b)
        d = np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))])
        rot = vt.T @ d @ u.T
        travel = pb[moving].mean(axis=0) - pa[moving].mean(axis=0)
        members = []
        for j in st.classes[mv.goal]:
            frame = np.linalg.norm(grids[j] - pa, axis=1) <= tol
            if _spread(pa[frame]) < min_spread or frame.all():
                continue
            members.append((j, grids[j], frame))
        if not members or members[0][0] != mv.goal:
            continue
        out.append({
            "start": mv.start, "goal": mv.goal, "angles": np.array(st.states[mv.start].angles),
            "p_start": pa, "p_goal": pb, "moving": moving, "kp": kp, "rot": rot,
            "direction": math.atan2(travel[1], travel[0]), "members": members,
            "name": "+".join(c for c, _ in mv.hinges),
        })
    return out
