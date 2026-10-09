"""Which of a robot's link pairs to exclude from self-collision, measured from its URDF.

Tasks with ``robot_self_collision`` on (PlaySelfCollision, PlayFlexivWuji) let every robot body
collide except the pairs in ``RobotSpec.self_collision_filter``, as SimToolReal does in Isaac Gym
with its hand-written ``adjacent_links.py``. The Kuka + Sharpa uses that file verbatim. This
derives the list for a new robot:

  1. every jointed pair -- a moving body and the body its joint hangs off;
  2. every other pair whose collision hulls overlap at the reset pose (gen_mechanics'
     ``unified_commercial_hands/onboard.py`` rule): they would start every episode in contact;
  3. a body and its grandparent across a two-joint knuckle, if their hulls overlap anywhere along
     either joint's range (one joint swept at a time from the reset pose). SimToolReal filters
     these on the Sharpa: palm to proximal phalanx across each MCP.

On the Kuka + Sharpa this reproduces 33 of SimToolReal's 35 pairs. It misses palm-middle PP and
pinky MC-PP, which do not touch in its one-joint-at-a-time sweep, and adds the iiwa wrist
(link5-link7, which SimToolReal leaves colliding). Hence the Kuka keeps the upstream list.

Bodies are the ones left after the importer's ``merge_fixed_joints`` (a link joins the body its
fixed joint hangs off), named as Isaac Lab names them. The reset pose is the spec's default arm
pose with the hand at 0, which is what ``reset_utils`` puts the robot in.

    .venv_isaaclab3/bin/python scripts/flexiv_wuji/self_collision_pairs.py flexiv_wuji
    .venv_isaaclab3/bin/python scripts/flexiv_wuji/self_collision_pairs.py kuka_sharpa --compare-simtoolreal

``tests/test_robot_specs.py`` reruns this for the Flexiv and checks the spec's list against it.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import trimesh
import yourdfpy

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import flexiv_wuji as fw  # noqa: E402

sys.path.insert(0, str(fw.REPO_ROOT / "isaacsimenvs" / "tasks" / "play"))
import robots  # noqa: E402  (stdlib-only; imported by path so Isaac Lab is not needed)

URDFS = {"kuka_sharpa": fw.KUKA_SHARPA_URDF, "flexiv_wuji": fw.COMPOSED_URDF}
#: Deeper than this counts as overlapping (onboard.py's threshold).
OVERLAP_M = 5e-4
#: Samples per joint range in the knuckle sweep.
SWEEP_STEPS = 9


def _merged_bodies(urdf: yourdfpy.URDF) -> dict[str, str]:
    """Link -> the body it ends up in under merge_fixed_joints."""
    parent_joint = {j.child: j for j in urdf.robot.joints}

    def body(link: str) -> str:
        while link in parent_joint and parent_joint[link].type == "fixed":
            link = parent_joint[link].parent
        return link

    return {link: body(link) for link in urdf.link_map}


def _body_hulls(urdf: yourdfpy.URDF, body_of: dict[str, str]) -> dict[str, list[trimesh.Trimesh]]:
    """Each body's collision geometry, as convex hulls in the body's own frame."""
    hulls: dict[str, list[trimesh.Trimesh]] = {b: [] for b in set(body_of.values())}
    for link_name, link in urdf.link_map.items():
        T_link = urdf.get_transform(link_name, body_of[link_name])  # fixed, so any cfg
        for c in link.collisions:
            g = c.geometry
            if g.mesh is not None:
                m = trimesh.load(urdf._filename_handler(g.mesh.filename), force="mesh")
                if g.mesh.scale is not None:
                    m.apply_scale(g.mesh.scale)
            elif g.box is not None:
                m = trimesh.creation.box(extents=g.box.size)
            elif g.cylinder is not None:
                m = trimesh.creation.cylinder(radius=g.cylinder.radius, height=g.cylinder.length)
            elif g.sphere is not None:
                m = trimesh.creation.icosphere(radius=g.sphere.radius)
            else:
                continue
            T0 = c.origin if c.origin is not None else np.eye(4)
            hulls[body_of[link_name]].append(m.convex_hull.apply_transform(T_link @ T0))
    return hulls


def _penetration(a: list[trimesh.Trimesh], b: list[trimesh.Trimesh]) -> float:
    """Deepest point of either body's hulls inside the other's, in metres (0 if apart)."""
    best = 0.0
    for ma, mb in itertools.product(a, b):
        if (ma.bounds[1] < mb.bounds[0]).any() or (mb.bounds[1] < ma.bounds[0]).any():
            continue
        for p, q in ((ma, mb), (mb, ma)):
            np.random.seed(0)  # sample() draws from the global generator
            pts = np.vstack([p.vertices, p.sample(200)])
            best = max(best, float(trimesh.proximity.signed_distance(q, pts).max()))
    return best


def derive(robot: str) -> tuple[dict[str, tuple[str, ...]], dict[tuple[str, str], tuple[str, float]]]:
    """``(filter map, {non-jointed pair: (why, depth in m)})`` for a robot spec name."""
    spec = robots.ROBOT_SPECS[robot]
    urdf = yourdfpy.URDF.load(str(URDFS[robot]), load_collision_meshes=False)
    names = urdf.actuated_joint_names
    home = np.array([spec.arm_default_joint_pos.get(n, 0.0) for n in names])
    body_of = _merged_bodies(urdf)
    hulls = _body_hulls(urdf, body_of)
    bodies = sorted(b for b in hulls if hulls[b])

    def depth(a: str, b: str, cfg: np.ndarray) -> float:
        urdf.update_cfg(cfg)
        posed = {x: [m.copy().apply_transform(urdf.get_transform(x, urdf.base_link)) for m in hulls[x]]
                 for x in (a, b)}
        return _penetration(posed[a], posed[b])

    moving = {j.child: j for j in urdf.robot.joints if j.type != "fixed"}
    jointed = {frozenset((body_of[j.parent], c)) for c, j in moving.items()}
    extra: dict[tuple[str, str], tuple[str, float]] = {}
    # Rule 2: overlapping at the reset pose.
    for a, b in itertools.combinations(bodies, 2):
        if frozenset((a, b)) not in jointed and (d := depth(a, b, home)) > OVERLAP_M:
            extra[(a, b)] = ("at reset", d)
    # Rule 3: a body and its grandparent across a two-joint knuckle, if they overlap anywhere
    # along either joint's range (one joint swept at a time from the reset pose). Where the two
    # axes meet, as at the Sharpa's MCPs, the flexing phalanx rubs the palm through the whole
    # range; SimToolReal filters exactly these.
    for child, j in moving.items():
        parent = body_of[j.parent]
        if parent not in moving:
            continue
        grand = body_of[moving[parent].parent]
        key = tuple(sorted((grand, child)))
        if key in extra or not hulls[grand] or not hulls[child]:
            continue
        worst = 0.0
        for joint in (moving[parent], j):
            i = names.index(joint.name)
            for q in np.linspace(joint.limit.lower, joint.limit.upper, SWEEP_STEPS):
                cfg = home.copy()
                cfg[i] = q
                worst = max(worst, depth(grand, child, cfg))
        if worst > OVERLAP_M:
            extra[key] = ("across knuckle", worst)

    pairs = jointed | {frozenset(p) for p in extra}
    adjacency: dict[str, list[str]] = {}
    for pair in pairs:
        a, b = sorted(pair)
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)
    return {k: tuple(sorted(v)) for k, v in sorted(adjacency.items())}, extra


def _simtoolreal_left_map() -> dict[str, tuple[str, ...]]:
    path = fw.REPO_ROOT.parent / "simtoolreal" / "isaacgymenvs" / "tasks" / "simtoolreal" / "adjacent_links.py"
    scope: dict = {}
    exec(path.read_text(), scope)
    return {k: tuple(sorted(v)) for k, v in scope["LEFT_SHARPA_KUKA_LINK_TO_ADJACENT_LINKS"].items()}


def _pairs(adjacency) -> set[frozenset]:
    return {frozenset((a, b)) for a, bs in adjacency.items() for b in bs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("robot", choices=sorted(URDFS))
    parser.add_argument("--compare-simtoolreal", action="store_true",
                        help="diff against SimToolReal's hand-written map (kuka_sharpa only)")
    args = parser.parse_args()

    adjacency, overlapping = derive(args.robot)
    print(f"# {args.robot}: {len(_pairs(adjacency))} filtered pairs, "
          f"{len(overlapping)} of them not jointed")
    for (a, b), (why, d) in sorted(overlapping.items()):
        print(f"#   {a} x {b}: {why}, {d * 1000:.2f} mm")
    print("{")
    for k, v in adjacency.items():
        print(f"    {k!r}: {v!r},")
    print("}")

    spec_map = robots.ROBOT_SPECS[args.robot].self_collision_filter
    if spec_map:
        same = _pairs(spec_map) == _pairs(adjacency)
        print(f"# robots.{args.robot.upper()}.self_collision_filter "
              + ("matches" if same else "DIFFERS from this"))
    if args.compare_simtoolreal:
        ours, theirs = _pairs(adjacency), _pairs(_simtoolreal_left_map())
        print(f"# vs SimToolReal LEFT map: {len(ours & theirs)} shared, "
              f"only derived {sorted(map(sorted, ours - theirs))}, "
              f"only SimToolReal {sorted(map(sorted, theirs - ours))}")


if __name__ == "__main__":
    main()
