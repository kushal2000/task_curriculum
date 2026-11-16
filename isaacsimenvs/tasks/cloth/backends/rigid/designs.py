"""YAML-defined convex plate cloths, geometry scaling, binding and exact mass properties.

The central x=0 crease defines the half-fold task. Other complete crease cuts
create a tree of rigid convex prisms. Body frames share the cloth rest origin.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from isaacsimenvs import CONFIG_DIR

LENGTH_RANGE = (0.10, 0.30)
WIDTH_RANGE = (0.10, 0.30)
THICKNESS_RANGE = (0.002, 0.020)

# Physical moving-half corners, independent of the diagnostic surface grid.
MOVING_KEYPOINTS = ((0.0, -0.5, 0.0), (0.0, 0.5, 0.0),
                    (0.5, -0.5, 0.0), (0.5, 0.5, 0.0))


def design_grid(
    n: int = 16,
    length: tuple[float, float] = LENGTH_RANGE,
    width: tuple[float, float] = WIDTH_RANGE,
    thickness: tuple[float, float] = THICKNESS_RANGE,
) -> np.ndarray:
    """``(n**3, 3)`` designs ``[length, width, thickness]`` [m]; design ``d`` is
    ``(length[d // n**2], width[d // n % n], thickness[d % n])`` on the evenly spaced levels."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    levels = [np.linspace(lo, hi, n) if n > 1 else np.array([lo]) for lo, hi in (length, width, thickness)]
    for name, (lo, hi) in zip(("length", "width", "thickness"), (length, width, thickness)):
        if not 0.0 < lo <= hi:
            raise ValueError(f"{name} range must be 0 < lo <= hi, got {(lo, hi)}")
    li, wi, ti = np.unravel_index(np.arange(n**3), (n, n, n))
    return np.stack([levels[0][li], levels[1][wi], levels[2][ti]], axis=1)


def design_of(env_ids, n_designs: int, seed: int = 0) -> np.ndarray:
    """Design index of each env: a fixed permutation of ``env_id % n_designs``.

    Every run of ``n_designs`` consecutive envs still holds each design exactly once; the
    permutation only spreads a SMALL run (a smoke test, an eval of a few hundred envs) across all
    three dimensions instead of the first lengths of the grid order."""
    perm = np.random.default_rng(int(seed)).permutation(int(n_designs))
    return perm[np.asarray(env_ids, dtype=np.int64) % int(n_designs)]


@dataclass(frozen=True)
class Box:
    """Legacy rectangular plate description, retained for analytical callers."""
    name: str
    center: tuple[float, float, float]
    half_extents: tuple[float, float, float]


@dataclass(frozen=True)
class Plate:
    """Convex CCW polygon extruded symmetrically around z=0 in the shared rest frame."""
    name: str
    vertices: tuple[tuple[float, float], ...]
    thickness: float

    @property
    def center(self) -> tuple[float, float, float]:
        _, center, _ = _polygon_integrals(self.vertices)
        return float(center[0]), float(center[1]), 0.0

    @property
    def half_extents(self) -> tuple[float, float, float]:
        """Bounding-box half extents; only rectangles are represented by these alone."""
        size = np.ptp(np.asarray(self.vertices), axis=0)
        return float(size[0] / 2), float(size[1] / 2), self.thickness / 2


@dataclass(frozen=True)
class Hinge:
    parent: str
    child: str
    anchor: tuple[float, float, float]
    axis: tuple[float, float, float]
    limits_deg: tuple[float, float] = (0.0, 180.0)
    edge: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None


@dataclass(frozen=True)
class SheetSpec:
    length: float
    width: float
    thickness: float
    plates: tuple[Plate, ...]
    hinges: tuple[Hinge, ...] = field(default=())
    ply_gap: float = 0.0
    fold_hinge: int = 0

    @property
    def joint_names(self) -> tuple[str, ...]:
        return _joint_names(self.plates, self.hinges)

    def point_owners(self, points: np.ndarray) -> np.ndarray:
        return point_owners(self, points)

    def plate(self, name: str) -> Plate:
        for plate in self.plates:
            if plate.name == name:
                return plate
        raise KeyError(name)


@dataclass(frozen=True)
class ClothDesign:
    """YAML-derived topology in unit length/width/thickness coordinates."""
    name: str
    plates: tuple[Plate, ...]
    hinges: tuple[Hinge, ...]
    fold_hinge: int
    source_path: Path | None = None

    @property
    def joint_names(self) -> tuple[str, ...]:
        return _joint_names(self.plates, self.hinges)

    def resolve(self, length: float, width: float, thickness: float) -> SheetSpec:
        scale = np.asarray([length, width, thickness], dtype=float)
        if not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError(f"dimensions must be finite and positive, got {tuple(scale)}")
        plates = tuple(Plate(p.name, tuple(map(tuple, np.asarray(p.vertices) * scale[:2])),
                             p.thickness * thickness) for p in self.plates)
        hinges = []
        for h in self.hinges:
            edge = None if h.edge is None else tuple(map(tuple, np.asarray(h.edge) * scale))
            direction = np.asarray(h.axis) * scale if edge is None else np.subtract(edge[1], edge[0])
            axis = tuple(direction / np.linalg.norm(direction))
            hinges.append(Hinge(h.parent, h.child, tuple(np.asarray(h.anchor) * scale), axis, h.limits_deg, edge))
        return SheetSpec(float(length), float(width), float(thickness), plates, tuple(hinges),
                         float(thickness), self.fold_hinge)

    def point_owners(self, points: np.ndarray) -> np.ndarray:
        return point_owners(self, points)


def _joint_names(plates, hinges) -> tuple[str, ...]:
    if tuple(p.name for p in plates) == ("minus", "plus") and len(hinges) == 1:
        return ("hinge",)
    return tuple(f"hinge_{i}" for i in range(len(hinges)))


DEFAULT_DESIGN_DIR = CONFIG_DIR / "rigid_cloth_designs"
_EPS = 1e-10


def discover_designs(directory: str | Path | None = None) -> tuple[Path, ...]:
    root = DEFAULT_DESIGN_DIR if directory is None else Path(directory)
    return tuple(sorted((*root.glob("*.yaml"), *root.glob("*.yml"))))


def resolve_design_path(path: str | Path) -> Path:
    text = str(path)
    candidate = Path(path).expanduser()
    if not candidate.suffix:
        candidate = candidate.with_suffix(".yaml")
    if not candidate.is_absolute() and "/" not in text and "\\" not in text:
        candidate = DEFAULT_DESIGN_DIR / candidate
    if not candidate.is_file():
        raise ValueError(f"cloth design YAML does not exist: {candidate}")
    return candidate.resolve()


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _clean_polygon(points):
    out = []
    for point in points:
        if not out or np.linalg.norm(point - out[-1]) > _EPS:
            out.append(point)
    if len(out) > 1 and np.linalg.norm(out[0] - out[-1]) < _EPS:
        out.pop()
    return np.asarray(out, dtype=float).reshape(-1, 2)


def _clip(polygon, start, direction, sign):
    result = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        da, db = sign * _cross(direction, a - start), sign * _cross(direction, b - start)
        if da >= -_EPS:
            result.append(a)
        if (da > _EPS and db < -_EPS) or (da < -_EPS and db > _EPS):
            result.append(a + (b - a) * da / (da - db))
    return _clean_polygon(result)


def _crease_segments(creases):
    if not isinstance(creases, dict) or not creases:
        raise ValueError("creases must be a nonempty mapping of names to two endpoints")
    result = []
    for name, endpoints in creases.items():
        try:
            edge = np.asarray(endpoints, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"crease {name!r} must contain two numeric endpoints") from exc
        if edge.shape != (2, 2) or not np.isfinite(edge).all() or np.abs(edge).max() > .5 + _EPS:
            raise ValueError(f"crease {name!r} must have two finite endpoints inside [-0.5, 0.5]²")
        if np.linalg.norm(edge[1] - edge[0]) <= _EPS:
            raise ValueError(f"crease {name!r} has zero length")
        result.append((str(name), edge))
    return result


def split_panels(creases) -> list[np.ndarray]:
    """Split the square by complete segments, rejecting partial or crossing cuts.

    Each segment must span exactly one current convex panel. No cut is extended
    beyond the authored endpoints. Graph validation happens in load_design.
    """
    polygons = [np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]])]
    for name, edge in _crease_segments(creases):
        direction = edge[1] - edge[0]
        candidates = []
        for i, polygon in enumerate(polygons):
            distances = _cross(direction, polygon - edge[0])
            if distances.min() < -_EPS and distances.max() > _EPS:
                candidates.append(i)
        if len(candidates) != 1:
            raise ValueError(f"crease {name!r} must cut exactly one panel; crossing or duplicate cuts are unsupported")
        i = candidates[0]
        halves = [_clip(polygons[i], edge[0], direction, sign) for sign in (1, -1)]
        on_line = np.concatenate([p[np.abs(_cross(direction, p - edge[0])) < _EPS] for p in halves])
        t = (on_line - edge[0]) @ direction / (direction @ direction)
        if not np.isclose(t.min(), 0.0, atol=_EPS) or not np.isclose(t.max(), 1.0, atol=_EPS):
            raise ValueError(f"crease {name!r} is a partial cut; its endpoints must span the panel boundary")
        polygons[i:i + 1] = halves
    return polygons


def _shared_edge(a, b):
    for p, q in zip(a, np.roll(a, -1, axis=0)):
        d = q - p
        length = np.linalg.norm(d)
        if length <= _EPS:
            continue
        u = d / length
        for r, s in zip(b, np.roll(b, -1, axis=0)):
            if abs(_cross(u, r - p)) > _EPS or abs(_cross(u, s - p)) > _EPS:
                continue
            lo = max(0.0, min((r - p) @ u, (s - p) @ u))
            hi = min(length, max((r - p) @ u, (s - p) @ u))
            if hi - lo > _EPS:
                return np.stack([p + lo * u, p + hi * u])
    return None


def load_design(path: str | Path) -> ClothDesign:
    """Read current YAML geometry every call; descriptive simulation metadata is ignored."""
    source = resolve_design_path(path)
    try:
        document = yaml.safe_load(source.read_text())
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid cloth design YAML {source}: {exc}") from exc
    if not isinstance(document, dict) or document.get("shape") != "rectangle":
        raise ValueError("cloth designs require shape: rectangle")
    polygons = split_panels(document.get("creases"))
    # Canonical geometry ordering makes names independent of YAML crease order.
    polygons.sort(key=lambda p: tuple(np.round(_polygon_integrals(p)[1], 12)))
    centers = np.array([_polygon_integrals(p)[1] for p in polygons])
    edges = {}
    adjacency = [[] for _ in polygons]
    for i in range(len(polygons)):
        for j in range(i + 1, len(polygons)):
            edge = _shared_edge(polygons[i], polygons[j])
            if edge is not None:
                edges[i, j] = edge
                adjacency[i].append(j)
                adjacency[j].append(i)
    if len(edges) != len(polygons) - 1:
        raise ValueError("crease graph must be a connected tree; crossing cuts and closed articulation loops are unsupported")
    central = [(pair, edge) for pair, edge in edges.items()
               if np.abs(edge[:, 0]).max() < _EPS and np.allclose(sorted(edge[:, 1]), [-.5, .5], atol=_EPS)]
    if len(central) != 1:
        raise ValueError("current half-fold task requires one complete center crease at x=0")
    pair, _ = central[0]
    root = pair[0] if centers[pair[0], 0] < 0 else pair[1]
    moving = pair[1] if root == pair[0] else pair[0]
    order, links, seen = [root], [], {root}
    for parent in order:
        neighbors = sorted(adjacency[parent], key=lambda j: (j != moving if parent == root else True, tuple(centers[j])))
        for child in neighbors:
            if child in seen:
                continue
            seen.add(child)
            order.append(child)
            links.append((parent, child))
    if len(seen) != len(polygons):
        raise ValueError("cloth articulation graph is disconnected")
    names = {index: ("minus", "plus")[i] if len(order) == 2 else f"panel_{i}" for i, index in enumerate(order)}
    plates = tuple(Plate(names[i], tuple(map(tuple, polygons[i])), 1.0) for i in order)
    hinges, fold_hinge = [], None
    for parent, child in links:
        edge = edges[tuple(sorted((parent, child)))].copy()
        is_fold = {parent, child} == set(pair)
        # Stable axis direction; central -Y gives positive angles an upward fold.
        if tuple(edge[0][::-1]) < tuple(edge[1][::-1]):
            edge = edge[::-1]
        endpoints = np.c_[edge, np.full(2, .5)]
        direction = endpoints[1] - endpoints[0]
        if is_fold:
            fold_hinge = len(hinges)
        hinges.append(Hinge(names[parent], names[child], tuple(endpoints.mean(axis=0)),
                            tuple(direction / np.linalg.norm(direction)),
                            (-90., 180.) if is_fold else (-90., 90.), tuple(map(tuple, endpoints))))
    return ClothDesign(source.stem, plates, tuple(hinges), int(fold_hinge), source)


def get_design(name: str | Path) -> ClothDesign:
    return load_design(name)


def strip(num_panels: int) -> ClothDesign:
    if num_panels not in (2, 4, 8):
        raise ValueError(f"strip panels must be 2, 4, or 8, got {num_panels}")
    return load_design("single_hinge" if num_panels == 2 else f"strip_{num_panels}")


def point_owners(spec: ClothDesign | SheetSpec, points: np.ndarray, *, plate_ids=None) -> np.ndarray:
    """Bind samples to polygons, optionally restricting candidate plate indices.

    Lower-x panels win shared seams. Restricting candidates to the moving half
    binds its exact crease endpoints without displacing the physical points.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("surface points must be a finite (N, 3) array")
    owners = np.full(len(points), -1, dtype=np.int64)
    candidates = range(len(spec.plates)) if plate_ids is None else plate_ids
    priority = sorted(candidates, key=lambda i: (*spec.plates[i].center[:2], i))
    for i in priority:
        vertices = np.asarray(spec.plates[i].vertices)
        edges = np.roll(vertices, -1, axis=0) - vertices
        distances = _cross(edges[None], points[:, None, :2] - vertices[None])
        tolerance = 1e-7 * np.max(np.linalg.norm(edges, axis=1)) ** 2
        inside = np.all(distances >= -tolerance, axis=1)
        owners[(owners < 0) & inside] = i
    if (owners < 0).any():
        raise ValueError("surface points fall outside the cloth polygons")
    return owners


def surface_mesh(plate: Plate) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.c_[np.asarray(plate.vertices), np.zeros(len(plate.vertices))]
    faces = np.array([(0, i, i + 1) for i in range(1, len(vertices) - 1)], dtype=np.int32)
    return vertices, faces


def extrusion_mesh(plate: Plate) -> tuple[np.ndarray, np.ndarray]:
    """Closed convex prism in the BODY LINK rest frame, not relative to its COM."""
    mid, top_faces = surface_mesh(plate)
    bottom, top = mid.copy(), mid.copy()
    bottom[:, 2] = -plate.thickness / 2
    top[:, 2] = plate.thickness / 2
    n = len(mid)
    sides = [(i, (i + 1) % n, (i + 1) % n + n) for i in range(n)]
    sides += [(i, (i + 1) % n + n, i + n) for i in range(n)]
    return np.r_[bottom, top], np.concatenate([top_faces[:, ::-1], top_faces + n, np.asarray(sides, dtype=np.int32)])


def _polygon_integrals(vertices):
    p = np.asarray(vertices, dtype=float)
    q = np.roll(p, -1, axis=0)
    cross = _cross(p, q)
    area = cross.sum() / 2
    if area <= 0:
        raise ValueError("plate polygon must have positive area and counterclockwise vertices")
    center = ((p + q) * cross[:, None]).sum(axis=0) / (6 * area)
    second = ((p * p + p * q + q * q) * cross[:, None]).sum(axis=0) / 12
    mixed = ((2*p[:, 0]*p[:, 1] + p[:, 0]*q[:, 1] + q[:, 0]*p[:, 1] + 2*q[:, 0]*q[:, 1]) * cross).sum() / 24
    covariance = np.array([[second[0], mixed], [mixed, second[1]]]) - area * np.outer(center, center)
    return area, center, covariance


def polygon_mass_properties(plate: Plate, density: float) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Exact prism mass/COM, principal inertia and its wxyz orientation in the body frame."""
    if not np.isfinite(density) or density <= 0 or not np.isfinite(plate.thickness) or plate.thickness <= 0:
        raise ValueError("areal density and thickness must be finite and positive")
    area, center, covariance = _polygon_integrals(plate.vertices)
    mass = float(density * area)
    covariance *= density
    xx, yy, xy = covariance[0, 0], covariance[1, 1], covariance[0, 1]
    zz = mass * plate.thickness**2 / 12
    tensor = np.array([[yy + zz, -xy, 0], [-xy, xx + zz, 0], [0, 0, xx + yy]])
    if abs(xy) <= 1e-14 * max(float(np.max(np.diag(tensor))), 1e-30):
        diagonal, rotation = np.diag(tensor).copy(), np.eye(3)
    else:
        diagonal, rotation = np.linalg.eigh(tensor)
        if np.linalg.det(rotation) < 0:
            rotation[:, -1] *= -1
    if (diagonal <= 0).any():
        raise ValueError("plate has non-positive principal inertia")
    from scipy.spatial.transform import Rotation
    xyzw = Rotation.from_matrix(rotation).as_quat()
    return mass, np.r_[center, 0.0], diagonal, xyzw[[3, 0, 1, 2]]


def single_hinge(length: float, width: float, thickness: float) -> SheetSpec:
    return load_design("single_hinge").resolve(length, width, thickness)


def box_mass_properties(box: Box | Plate, density: float) -> tuple[float, np.ndarray, np.ndarray]:
    """Legacy axis-aligned rectangle formula; polygon callers need principal axes too."""
    if isinstance(box, Plate):
        area, _, _ = _polygon_integrals(box.vertices)
        if not np.isclose(area, 4 * box.half_extents[0] * box.half_extents[1], rtol=1e-10):
            raise ValueError("box_mass_properties requires a rectangle; use polygon_mass_properties")
    a, b, c = (2 * h for h in box.half_extents)
    mass = density * a * b
    return mass, np.asarray(box.center), mass / 12 * np.array([b*b + c*c, a*a + c*c, a*a + b*b])


def _rot(axis, theta: float) -> np.ndarray:
    u = np.asarray(axis, dtype=float)
    u = u / np.linalg.norm(u)
    k = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)


def hinge_transform(hinge: Hinge, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """``(R, t)`` taking a child point at rest to its position with the hinge at ``angle`` [rad],
    both in the parent's (= cloth) frame: ``p' = R p + t``."""
    r = _rot(hinge.axis, angle)
    a = np.asarray(hinge.anchor, dtype=float)
    return r, a - r @ a


def plate_points(spec: SheetSpec, points: np.ndarray, angle: float) -> np.ndarray:
    """Sheet-frame rest points ``(P, 3)`` -> their positions with the single hinge at ``angle``:
    points with ``x > 0`` ride the ``+x`` plate, the rest stay. Single-hinge specs only."""
    if len(spec.hinges) != 1:
        raise ValueError("plate_points handles single-hinge specs only")
    pts = np.asarray(points, dtype=float)
    r, t = hinge_transform(spec.hinges[0], angle)
    out = pts.copy()
    moving = pts[:, 0] > 0.0
    out[moving] = pts[moving] @ r.T + t
    return out

