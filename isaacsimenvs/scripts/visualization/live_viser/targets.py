"""The viewer's targets: the evaluation targets of ``experiments/eval.sh`` (keep the two in step).

The rigid targets run in the policy's own training world (its recipe, ``app.recipe``), as the
evaluation does. Two differences, both on the rigid sheet: every env carries all seven shapes
(``one_shape_per_env=false``, so any env can run any shape), and the per-env sheet sizes are spaced
evenly instead of drawn (``fold_pool.even_env_sizes``).
"""

from __future__ import annotations

RIGID_TASK = "Isaacsimenvs-RigidCloth-Direct-v0"
VBD_TASK = "Isaacsimenvs-Cloth-Direct-v0"

SHAPES = ("box2-mid", "box2diag4-mid", "box2x8-mid", "cross4-mid", "corners8-mid", "box4-mid", "star8-mid")
_RIGID = (f"env.fold_pool.shapes=[{','.join(SHAPES)}]", "env.fold_pool.one_shape_per_env=false",
          "env.fold_pool.init_fold_prob=0.0")

#: target -> (gym task, hydra overrides)
TARGETS = {
    # Flat starts, the half fold.
    "rigidbody_flat": (RIGID_TASK, [*_RIGID, "env.fold_pool.goal_task.mode=half_fold"]),
    # From a random reachable folded state, with one more fold as the goal.
    "rigidbody_folded": (RIGID_TASK, [*_RIGID, "env.fold_pool.goal_task.mode=any_fold"]),
    # The VBD task's own physics; without robot self-collision (with it, ~4x slower).
    "vbd": (VBD_TASK, ["env.robot_self_collision=false", "env.newton.njmax=32768", "env.cloth.start_height=0.20"]),
}

#: The start each target is evaluated from (rand: all domain randomisation on; pinned: off). The
#: evaluation has the rigid sheet at rand only; the viewer also allows pinned there.
START = {"rigidbody_flat": "rand", "rigidbody_folded": "rand", "vbd": "pinned"}

#: Every target, as the evaluation runs it: a fold HELD 10 steps within 1.5 cm, NaN -> reset.
COMMON = ("env.cloth.keypoint_tolerance=0.015", "env.cloth.nan_policy=reset", "env.termination.success_steps=10")


def overrides(target: str, extra: list[str]) -> tuple[str, list[str]]:
    """``(gym task, hydra overrides)`` of ``target``; for a key set twice the later value wins, so
    ``extra`` (the command line's) overrides the target's."""
    task, own = TARGETS[target]
    by_key: dict[str, str] = {}
    for o in (*COMMON, *own, *extra):
        by_key.pop(o.split("=", 1)[0], None)  # re-insert: the later value, in the later place
        by_key[o.split("=", 1)[0]] = o
    return task, list(by_key.values())
