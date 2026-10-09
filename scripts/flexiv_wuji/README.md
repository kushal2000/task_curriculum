# Flexiv Rizon 4s + Wuji Hand 2

The robot for pretraining our own rigid-body prior with SimToolReal's recipe, in place of its
Kuka iiwa14 + Sharpa. The task is `Isaacsimenvs-PlayFlexivWuji-Direct-v0`: the Play task with
`robots.FLEXIV_WUJI` (`isaacsimenvs/tasks/play/robots.py`) instead of `KUKA_SHARPA`, and
`cfg/task/PlayFlexivWuji.yaml`, a copy of `Play.yaml` that differs only in the robot.

The robot has 27 joints (7 arm + 20 hand) against the Kuka's 29, so its policy has a 134-dim
observation and 27 actions. SimToolReal's released checkpoint cannot be loaded; this robot trains
from scratch.

## Setup

```bash
scripts/flexiv_wuji/fetch_assets.sh
```

Fetches the Rizon 4s (flexivrobotics/flexiv_description, xacro expanded with upstream's own
script) and the Wuji Hand 2 Beta 2 (wuji-technology/wuji-description) at pinned commits into
`assets/flexiv_wuji/`. It then writes `rizon4s_left_wuji.urdf`, the two joined at
`flexiv_wuji.MOUNT_*`.

The left-hand robot's files (~12 MB) are committed, because the training-time pose viewer loads
the robot from raw.githubusercontent.com at the run's branch. Rerun the script only to move to a
new upstream commit or mount, then commit the result.

## Tools

| | |
|---|---|
| `viser_compare.py` | Both robots side by side where the env puts them, each behind its own table, with joint, base and mount sliders, and IK to match the Kuka's hand pose. `.venv_isaaclab3` |
| `reachability.py` | IK for the palm centre over the task's goal volume and the table, for both robots. `.venv_isaaclab3` |
| `smoke_env.py` | Builds a Play-family task in Isaac Sim and checks sizes, palm/fingertip positions, arm and hand step responses, and random-action stability. `.venv_isaacsim` |

The two hands are compared in one canonical frame built from their knuckles (`hand_frames.py`),
so the vendors' different link conventions do not matter.

## What is measured, and what is provisional

**Geometry, checked:**
- The default arm pose (`FLEXIV_DEFAULT_ARM_POS`) is the IK match of the Kuka's: the hand frames
  agree to 0.0 mm and 0.1 deg.
- The palm centre the policy sees is the Kuka's, carried over through the hand frame. In Isaac
  Sim, at reset, the two robots' palm centres agree to 0.1 mm.
- The fingertip pad offset is the Sharpa's share of the way to the tip.

`tests/test_robot_specs.py` recomputes all of this from the models.

**Provisional, to be refined against the hardware:**
- **The mount** (`flexiv_wuji.MOUNT_*`): the hand is flipped onto the flange, with no adapter
  plate and no yaw. After changing it, re-match the default pose in the viewer, update
  `palm_center_offset` (the test prints the new value), and rerun `fetch_assets.sh` or
  `flexiv_wuji.py`.
- **The arm gains**: stiffness is 2 Nm/rad per Nm of Rizon torque limit, and damping keeps the
  Kuka's damping ratio. The arm lags about twice as much as the Kuka's while moving (11.5 against
  5.4 mrad).
- **The hand gains**: the vendor MJCF's position servos, much softer than the Sharpa's. They
  track well in free motion; nothing has tested them under grasp load.

## Reachability

From `reachability.py`, palm centre within 5 mm:

| | Kuka + Sharpa | Flexiv + Wuji |
|---|---|---|
| goal volume, any orientation | 109/140 | 108/140 |
| goal volume, palm down (30 deg) | 109/140 | 104/140 |
| table layer, either | 25/25 | 25/25 |

The goal volume bounds the *object* goal, not the palm, so neither robot covers all of it.
