# rigid2deform deploy: Flexiv Rizon 4s + right Wuji Hand 2

Runs the rigid2deform policies (branch `yiboc_wuji`: rigid plates standing in for a deformable sheet)
on the real robot. `task` in [config.yaml](config.yaml) picks the task; so far there is one, `fold`.

The robot, perception, viewer, recorder and playback are the shared nodes in `deploy/nodes/`, the
same as SimToolReal's; [deploy/simtoolreal/README.md](../simtoolreal/README.md) describes them
(setup on the robot PC, robot modes, safety, recording, playback). This folder has the task's own
parts:

| file | what |
|---|---|
| [run.py](run.py) | the entry point: the shared nodes + policy.py on config.yaml (deploy/utils/runner.py) |
| [policy.py](policy.py) | the policy loop of each task and kind of network (`TASKS`): keypoints: perception → sheet particles → observation → LSTM → joint targets; point cloud: point_cloud (live cloud + the reference taken at the start) → proprio + cloud → PointNet + LSTM → joint targets |
| [fold.py](fold.py) | task=fold: the sheet's grid points → what the policy observes, and the success test |
| [sheets/](sheets/__init__.py) | where the grid points come from, by the object's `perception`: `rigid_plates.py` (two FoundationPose plates), `cloth_points.py` (CoTracker cloth: placeholder) |
| [perception/](perception/__init__.py) | the perception nodes (each in its camera environment). [main.py](perception/main.py) is the entry: it starts the node config.yaml's `mode` needs ([mode.py](mode.py): `pc_label` / `pc_goal` / `foundationpose` / `cotracker`; FoundationPose: `deploy/nodes/perception.py`; CoTracker: `perception/cotracker_node.py` / `perception/cotracker_stereo_node.py`; point cloud: `perception/pc_node.py`, its start `perception/pc_ref_goal.py`), or with `perception.source=fake` `fake_sheet.py` / `pc_fake_node.py` (no camera) |
| [pc_obs.py](pc_obs.py) | the point-cloud policies' cloud input: base-frame points → the network's points + mask (the network: `deploy/utils/pointcloud_net.py`) |
| [config.yaml](config.yaml) | every setting, complete (not an overlay of SimToolReal's) |
| `spec/` | the fold spec json + the robot URDF with the real joint limits |
| `objects/` | the plate meshes |
| `policy/` | the fold checkpoints; `policy/pc/`: the point-cloud ones (+ the golden file of `check_pointcloud_net.py`). `deploy/utils/policy_net.py` tells the two kinds apart by the file, and the kind picks the perception and the loop |
| `checks/` | the spec exporter, and the check of fold.py against the training code |

Every node reads config.yaml: run.py and policy.py directly, the shared nodes because run.py passes them
`config=deploy/rigid2deform/config.yaml` (start one by hand with that argument too). Override any value
on the command line with dotted keys. To print the merged config:
`.venv_deploy/bin/python deploy/utils/config.py config=deploy/rigid2deform/config.yaml <overrides>`.

## Mode

`mode` in config.yaml (default `pc_label`) picks the perception and, while `policy.checkpoint` is null,
the network (`policy.default_checkpoint`):

| mode | perception | default network |
|---|---|---|
| `pc_label` | point cloud (`perception/pc_node.py`), the reference | `policy/pc/rigidcloth_pc_label_noiseDR_dagger.pth` |
| `pc_goal` | point cloud, the goal | `policy/pc/rigidcloth_pc_goal_dagger.pth` |
| `foundationpose` | two plates (`deploy/nodes/perception.py`) | `policy/flat7RW_base.pth` |
| `cotracker` | a cloth, `object.name=[cloth]` (`perception/cotracker_node.py`) | `policy/flat7RW_base.pth` |

Give perception/main.py and run.py the same `mode` (or set it in config.yaml). A `policy.checkpoint` given
on the command line must be of the mode's kind (and point-cloud mode): policy.py refuses another.

## Objects and perception

`object.name` lists what is on the table, from `objects`. Each of those objects has a `perception`
(all of them the same), which picks both the perception node to run and the sheet source (`sheets/`)
that turns that node's messages into the sheet's grid points; the fold code after that is the same
for every sheet.

| perception | object.name | node (topic) | settings |
|---|---|---|---|
| `foundationpose` | `[moving plate, stationary plate]`, two rigid plates taped along one edge | `deploy/nodes/perception.py` (`object_poses`), one per plate | the sheet size and the hinge axis from the plates' meshes; `fold.hinge_z` (null: half the thickness), `fold.max_hinge_gap_m` |
| `cotracker` | `[cloth]` | `deploy/nodes/point_tracker.py` (`object_points`): not written yet | `fold.sheet_size` (required; more with the node) |

`mode=pc_label` / `pc_goal` ignores this table: perception/main.py runs
perception/pc_node.py instead, in the foundationstereo environment for either `depth` (it opens a window:
`.venv_deploy`'s OpenCV cannot); `perception.source=fake`: `perception/pc_fake_node.py`, a still flat sheet.

At its start, the non-live group of points the network gets from step 12 on, which the mode says
(`pc_label`: the reference | `pc_goal`: the goal; the network's own, `pc_args.goal`, training's names
`first` | `scene`, must agree):

Both from the flat cloth (the robot at home, hands away, nothing touching the cloth at table height:
keep it 1 cm from the tape): SPACE in the window freezes both cameras, then ONE click on the half that
folds over (about its middle, not near a corner). The cloth = the connected piece of the cloud at table
height (within `perception.pointcloud.ref_plane_m` of 0.33) under the click; the sheet = the smallest
rectangle around it; the fold line = its mid line across the side direction nearest the click. Check the
drawing (yellow outline, white fold line), SPACE accepts; only then does `point_cloud` start. To take it
again, restart perception/main.py (the policy keeps the first message's, so restart the run too).

- `label` (the reference; `rigidcloth_pc_label_*`): the cloth's own points, 192 at random, each labelled by
  its half (drawn orange moving, blue stationary) -> `ref_xyz`, `ref_moving`, `ref_corners`.
- `goal` (the goal; `rigidcloth_pc_goal_*`): synthesised as training does (handoff README 4.3): the moving
  half mirrored over the fold line lands on the stationary half at the same height, both layers in one,
  so 192 points uniform on the stationary half's rectangle, no noise (drawn green) -> `goal_xyz`,
  `goal_corners`. No folding by hand.

Warnings: a side outside 10-20 cm, not square (something touching the cloth?), a centre more than 10 cm
from (0, -0.51), fewer than 192 cloth points (label). Refused: a click not on the cloth or near a corner,
all points on one half.

Measured Nov 17 on the 5090, both cameras at 640 x 480, 60 fps:

| `depth` | clouds/s | cloud age when sent | table plane residual (std, per camera) |
|---|---|---|---|
| `raw` (spatial + temporal) | 30 (`rate_hz`) | ~42 ms | ~6 mm |
| `foundationstereo` (384x512 it4, 31 ms per camera) | ~13 | ~87 ms | 2-3 mm |

Both agree on objects to ~2 mm (median). Camera B uses `right_camera_nov14.json` here: with it B's table
plane sits within a few mm of A's (nov10: 1 cm lower). The table is removed by height + colour (black:
brightest - darkest channel <= 35, and brightest <= 100, or <= 140 if not warm: the black tape's highlights,
not the brown wood), so a black or grey cloth would be removed with it.

```bash
~/miniconda3/envs/foundationstereo/bin/python deploy/rigid2deform/perception/main.py [mode=pc_goal] [perception.pointcloud.depth=raw]
.venv_deploy/bin/python deploy/rigid2deform/run.py robot.arm=<ROBOT_SN> robot.hand=<HAND_SN> [mode=pc_goal] record.name=pc_try1
```

`cloth` is in the config as a placeholder: `object.name=[cloth]` stops with "not implemented yet".
`deploy/nodes/perception.py` refuses an object whose perception is not `foundationpose`.

## Fold: the fold-pool policies on two taped plates

Two plates taped along one edge make a sheet; the policy turns one plate over onto the other.

```bash
PY=.venv_deploy/bin/python; D=deploy/rigid2deform
ARM="robot.arm=<ROBOT_SN> robot.hand=<HAND_SN>"
$PY $D/run.py robot.mode=home $ARM
# perception: both plates, the MOVING one first (one box to click per plate)
~/miniconda3/envs/foundationpose/bin/python $D/perception/main.py mode=foundationpose
# the run (flat7RW_base by default; any of the five: policy.checkpoint=deploy/rigid2deform/policy/<name>.pth)
$PY $D/run.py $ARM mode=foundationpose record.name=fold_try1
```

Other plates: `object.name=[plate7_brown,plate7_white]` (no spaces), given to perception/main.py and run.py
alike; nothing else. The sheet size (the plates' long side) and the hinge height (half their
thickness: tape on the top face) come from the meshes, and the two plates must match (same size, long
side twice the short one). Tape elsewhere: `fold.hinge_z=<m>`.

`object.name: [plate5_brown, plate5_white]` is [moving, stationary]. Start with the two plates lying
flat, side by side, taped edge to edge, about where training put the sheet: centre 0.51 m in front of
the base (base frame (0, −0.51)) ± 10 cm, any heading. The policy refuses to start otherwise
(`start_frames` in `sheets/rigid_plates.py`).

Without hardware (MuJoCo arm, fake hand, a still sheet; the viewer at http://localhost:8091), any mode:

```bash
$PY $D/perception/main.py perception.source=fake mode=foundationpose
$PY $D/run.py robot.arm=mujoco robot.hand=fake mode=foundationpose run.duration_s=30
```

What the policy sees ([fold.py](fold.py), from the sheet's 7 x 7 grid points as training computes it;
every number checked against the training code by `checks/check_fold.py`):

| | |
|---|---|
| object | the moving plate's 4 corner keypoints: training's 7 x 7 grid, rows 16.7 and 50 mm from the hinge (10 cm sheet), so the object position is 8.3 mm off the plate centre, away from the hinge; orientation = the Kabsch fit of the corners (`fit_rotation_wxyz`; for rigid plates the plate's own, to 1e-5), with the sign rule of training's `_mat_to_wxyz` |
| goal | the same corners mirrored onto the stationary plate (training's hinge is on the mid-plane: no lift), recomputed every step from the stationary half's points (Kabsch, `_stationary_frame`). Its quaternion is relabelled (x, y, z, w) -> (w, x, y, z) as the PhysX training env does (`physx_envs._goal_quat_from_matrix`): keep it |
| object_scales | (sheet, sheet, 0.016) / 0.04 = (2.5, 2.5, 0.4) at 10 cm: the z is the sim's felt thickness, not the plate's |
| success | every corner within 1.5 cm of the fold target AND the sheet's extent along the stationary plate's x ≤ 0.65 x sheet. The episode ends there (as in training). The target is the REAL fold: tape on the top face puts the folded plate 2 x `fold.hinge_z` = 5 mm above training's (the source's `target_lift`) |
| stops | folded; 600 steps (10 s, `fold.timeout_steps`); a sheet point 5 cm below the table top; the source's own stop (plates: hinge edges more than `fold.max_hinge_gap_m`, 3 cm, apart: a tracker flipped or slipped) |

A plate is a box, so FoundationPose may report it turned 180° about any of its axes. `start_frames` / `update_frames` (`sheets/rigid_plates.py`)
take the pose that lies flat with the hinge edges together at the start, and after that, for each
plate, the symmetric pose closest to the last frame.

The viewer shows the fold error, the footprint and the hinge gap (goal_pose's `status` line); the
goal mesh is where the moving plate lands when folded.

The fold spec (`spec/spec_flexiv_wuji_right_fold.json`, `checks/export_fold_spec.py`) differs from
the Play spec in what the deploy code reads: the real joint limits (`rizon4s_right_wuji_real_limits.urdf`:
arm 5° inside, finger PIP ≤ 100°), the table top 0.33 m above the base, and the fingertip pad offsets
(per finger: the thumb gets 22.9 mm, not the ring finger as in Play). The five runs in `policy/`
share every one of these (`export_fold_spec.py --compare`):

| checkpoint | trained on |
|---|---|
| `flat7RW_base` | 7 shapes, S2R hand drive, 16 mm felt, no observation DR |
| `flat7RW_fricition` | + plate/robot/table friction DR |
| `flat7RW_sysid` | + the identified hand (friction, spring, armature, 1.0 A x Kt) and its DR |
| `flat7RW_allDR` | identified hand, 5 mm felt (3-7), point-level object DR, delays, noise, table height ± 1.5 cm |
| `box2RW_allDR` | as `flat7RW_allDR`, two-plate sheets only |

## Checks

| script | what | runs in |
|---|---|---|
| `checks/export_fold_spec.py` | the fold spec from `yiboc_wuji` (task YAML + robot overlay + experiment, as train.py layers them); `--compare` diffs the five runs | `.venv_deploy`, no Isaac Sim |
| `perception/pc_fake_node.py` | a stand-in for the point-cloud perception node: a still flat sheet as point_cloud (live cloud + reference), to run the point-cloud loop in MuJoCo without cameras | `.venv_deploy` |
| `checks/check_pointcloud_net.py` | the point-cloud networks (mode label: `rigidcloth_pc_label_noiseDR_dagger`, mode goal: `rigidcloth_pc_goal_dagger`) + `pc_obs.py` against pc_deploy_handoff's golden files (made by the training code): the points, the actions (< 1e-4), the 103 proprio numbers, the time per step | `.venv_deploy`, no Isaac Sim |
| `checks/check_fold.py` | `fold.py` against the training code's own methods (pulled from git, run with torch) on random plate sheets; the success test at 0-180°; the plate frames through random symmetric flips; the `rigid_plates` source end to end | `.venv_deploy`, no Isaac Sim |

## Adding a task or a perception

- A task: its loop in `TASKS` in policy.py (it gets cfg, spec, robot model, network, publisher), its
  settings as a section of config.yaml, its geometry next to fold.py.
- A perception (e.g. finishing cotracker): a sheet source module in `sheets/` with the functions in
  `sheets/__init__.py` (`TOPIC`, `init`, `start`, `update` → particles, stop, status, info,
  goal_mesh_pose), registered in `SOURCES`; its node in `deploy/nodes/` if another task could use it;
  its fake message in perception/fake_sheet.py.
- Code another deploy folder could use goes in `deploy/utils/`.

## Not verified yet

- No sim rollout of a fold policy has been run through the deploy code (the Play `check_rec1`
  equivalent); the observation is checked against the training code piece by piece
  (`check_fold.py`), and a MuJoCo run with fake plates moved the hand over the sheet. Not checked:
  FoundationPose on 5 mm plates while a finger covers them; the hinge-gap stop on a real fold; the
  hand's effort limit (`robot.hand_effort_limit` 1.0 A) against flat7RW_base / _fricition, which
  trained with the S2R drive's higher torque limits.
