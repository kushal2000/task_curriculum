"""ClothEnvCfg — fold half a cloth onto the other half.

Subclasses :class:`PlayNewtonEnvCfg`, so the robot, table, action space and coupled-solver
machinery are exactly the ones measured on the cable task. Only the manipuland changes, from a
rigid tool to a VBD cloth sheet.

**The task.** A square sheet lies flat on the table. Its *far* half (``+x``) is tracked by a
handful of keypoints; the goal is to bring those keypoints onto the mirror image of their rest
positions across the fold line, i.e. to fold the far half back over the near half. Success is the
usual keypoint criterion: every tracked keypoint within tolerance, held for ``success_steps``.

**Why keypoints rather than a pose.** A cloth has no rigid frame. The cable task could fake one
from its first-to-last span, and that estimator degenerated as soon as the cable bent
(``docs/phase4_cable_env.md``). A sheet has no meaningful single axis at all, so the state *is* the
keypoint set -- there is nothing to reduce it to.

**Coupling defaults are inherited from the cable work, not guessed.** ``proxy_mode="staggered"``,
``cable_substeps=2``-equivalent substepping and the damping stack are what took the cable from 0 to
~14 goals per 64 envs; `lagged` coupling pairs the hand's begin-pose with its end-velocities, an
error that scales with hand speed and which shook the cable loose during transport. A cloth is more
compliant than a cable, so the same defect should hurt at least as much here.
"""

from __future__ import annotations

# IMPORT ORDER IS LOAD-BEARING, as in `cable_env_cfg`: importing `isaaclab_physx` binds
# `isaaclab.utils.configclass` to the same-named *submodule*, shadowing the decorator. Importing
# PlayNewtonEnvCfg first runs the compat shim that re-asserts it.
from isaacsimenvs.tasks.play_newton.play_newton_env_cfg import PlayNewtonEnvCfg  # isort: skip

from isaaclab.utils import configclass  # noqa: E402

from isaacsimenvs.tasks.cloth.cloth_cfg import ClothCfg  # noqa: E402

__all__ = ["ClothCfg", "ClothEnvCfg"]


@configclass
class ClothEnvCfg(PlayNewtonEnvCfg):
    """Play task with a folding cloth in place of the rigid tool."""

    cloth: ClothCfg = ClothCfg()
