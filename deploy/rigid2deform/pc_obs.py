"""The point-cloud fold policies' cloud input: robot-base-frame points -> the network's (M, D) points + mask.

From pc_deploy_handoff (2026-11-17): pc_student_ref.py's featurisation, the training code's
scene_pointcloud.py (branch pc_round2). The network is deploy/utils/pointcloud_net.py; the robot's own
103 numbers, deploy/utils/observation.py build_proprio.

    check_supported(net.pc_args, net.point_dim)            # once, at load
    points, mask = featurize(live_base, net.pc_args, ref_base, ref_moving)   # mode label
    points, mask = featurize(live_base, net.pc_args, goal_base=goal_base)    # mode goal
    action = net.act((proprio, points, mask))

The mode says which non-live group the policy takes: label (rigidcloth_pc_label_*: the flat cloth, its
points labelled by half) or goal (rigidcloth_pc_goal_*: the folded sheet). The checkpoint stores it as
pc_args.goal by training's names, first = label, scene = goal; mode_of translates, and only it and
labels_of (training's code as is) see training's names. Supported so far, no fk points. The cloud is the
live group, then one of the other two:

| group     | slots | label                       | mode  | what                                              |
|-----------|-------|-----------------------------|-------|---------------------------------------------------|
| live      | 1024  | live                        | both  | what both cameras see in the box now: cloth, hand, |
|           |       |                             |       | arm                                               |
| reference | 192   | ref_moving / ref_stationary | label | the flat cloth alone, each point labelled by the  |
|           |       |                             |       | half it is on (moving = the half folded over)     |
| goal      | 192   | goal                        | goal  | the sheet as it is to be when folded (training:   |
|           |       |                             |       | synthesised, both layers; real: the folded sheet, |
|           |       |                             |       | photographed before the run)                      |

The reference / goal group is all masked before the policy step it comes in (training: 12).

Per point: [(p_base - CENTRE_BASE) * 10, one-hot label]; an empty slot is all zero and masked. Point
order does not matter (the PointNet max-pools).

The frame: the training env (PhysX fold pool) has the robot base at ROOT_ENV, identity rotation, the
table top at TABLE_Z_ENV. That is NOT the deploy spec's env frame (Play's, base at (0, 0.8, 0)); this
file works in the robot base frame only.
"""

from __future__ import annotations

import numpy as np

#: the robot base in the TRAINING env frame (identity rotation): env = base + ROOT_ENV
ROOT_ENV = np.array([0.0, 0.51, 0.20])
TABLE_Z_ENV = 0.53                    # the table top in that frame (= 0.33 above the base)
#: the box: the 22 x 15 inch table top, from its edge next to the robot, 2 cm below the top, no ceiling
BOX_HALF_X, BOX_Y_NEAR, BOX_DEPTH, BOX_BELOW = 0.2794, 0.20, 0.381, 0.02   # env frame
CENTRE_ENV = np.array([0.0, BOX_Y_NEAR - 0.5 * BOX_DEPTH, TABLE_Z_ENV])     # (0, 0.0095, 0.53)
CENTRE_BASE = CENTRE_ENV - ROOT_ENV                                          # (0, -0.5005, 0.33)
BOX_LO_BASE = np.array([-BOX_HALF_X, BOX_Y_NEAR - BOX_DEPTH, TABLE_Z_ENV - BOX_BELOW]) - ROOT_ENV
BOX_HI_BASE = np.array([BOX_HALF_X, BOX_Y_NEAR, np.inf]) - ROOT_ENV
SCALE = 10.0
N_POINTS, N_REF = 1024, 192


def labels_of(pc: dict) -> tuple[str, ...]:
    """The one-hot label columns for these pc_args (none when the cloud has one group). Training's code as
    is, so training's pc_args.goal names (first | scene | shape)."""
    out = ["live"]
    if pc["goal"] == "first":
        out += ["ref_moving", "ref_stationary"]
    elif pc["goal"] in ("scene", "shape"):
        out += ["goal"]
    if int(pc.get("fk_points", 0)) > 0:
        out.append("fk")
    return tuple(out) if len(out) > 1 else ()


#: pc_args.goal (training's name, as the checkpoint stores it) -> the mode: which non-live group
MODE_OF_GOAL = {"first": "label", "scene": "goal"}
MODES = tuple(MODE_OF_GOAL.values())      # label: the reference group | goal: the goal group


def mode_of(pc: dict) -> str:
    """'label' | 'goal': the non-live group these pc_args take. SystemExit for a pc_args.goal not built here."""
    mode = MODE_OF_GOAL.get(pc.get("goal"))
    if mode is None:
        raise SystemExit(f"[pc_obs] pc_args.goal={pc.get('goal')}: supported "
                         + ", ".join(f"{g} (mode {m})" for g, m in MODE_OF_GOAL.items()))
    return mode


def check_supported(pc: dict, point_dim: int) -> None:
    """SystemExit unless the checkpoint's cloud is one this file builds, the network's width agreeing."""
    mode_of(pc)
    if int(pc.get("fk_points", 0)) > 0 or int(pc.get("n_points", N_POINTS)) != N_POINTS:
        raise SystemExit(f"[pc_obs] supported: no fk points, {N_POINTS} points; the checkpoint has {pc}")
    if point_dim != 3 + len(labels_of(pc)):
        raise SystemExit(f"[pc_obs] the network takes {point_dim} numbers per point, {pc} gives {3 + len(labels_of(pc))}")


def featurize(live_base: np.ndarray, pc: dict, ref_base: np.ndarray | None = None,
              ref_moving: np.ndarray | None = None, goal_base: np.ndarray | None = None
              ) -> tuple[np.ndarray, np.ndarray]:
    """(M, D) float32 points and (M,) bool mask, groups in training's order and sizes (the table above).

    live_base (K, 3): the live cloud in the robot base frame, already cropped, table-free and
        downsampled; K <= N_POINTS (more are cut off, missing slots masked; non-finite rows masked).
    mode label: ref_base (R, 3) + ref_moving (R,) bool: the reference cloth cloud and which half each
        point is on. mode goal: goal_base (R, 3): the goal cloud. R <= N_REF. None = the whole group
        masked (training: the policy steps before ref_step).
    """
    groups = []                                        # (xyz (n, 3), valid (n,), label index (n,))

    def put(xyz, lab, size):
        xyz = np.zeros((0, 3)) if xyz is None else np.asarray(xyz, np.float64)[:size]
        lab = np.broadcast_to(np.asarray(lab), (len(xyz),))
        pad = size - len(xyz)
        groups.append((np.concatenate([xyz, np.zeros((pad, 3))]),
                       np.concatenate([np.isfinite(xyz).all(-1), np.zeros(pad, bool)]),
                       np.concatenate([lab, np.zeros(pad, int)])))

    put(live_base, 0, N_POINTS)
    if mode_of(pc) == "label":
        lab = None if ref_base is None else np.where(np.asarray(ref_moving, bool)[:len(ref_base)], 1, 2)
        put(ref_base, 0 if lab is None else lab, N_REF)
    else:
        put(goal_base, 1, N_REF)

    xyz = np.concatenate([g[0] for g in groups])
    ok = np.concatenate([g[1] for g in groups])
    lab = np.concatenate([g[2] for g in groups])
    feat = (xyz - CENTRE_BASE) * SCALE                 # == (xyz_env - CENTRE_ENV) * SCALE
    feat = np.concatenate([feat, np.eye(len(labels_of(pc)))[lab]], axis=-1)
    feat = np.where(ok[:, None], feat, 0.0)
    return feat.astype(np.float32), ok
