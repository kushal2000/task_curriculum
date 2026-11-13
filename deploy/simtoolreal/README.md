# SimToolReal deploy: Flexiv Rizon 4s + right Wuji Hand 2

Runs the SimToolReal Play policy (`pretrained_policy/nov6_dr.pth`: move a tool to a goal pose) on the
real robot. Trained on branch `kushal/2026-10-30_pretraining_real`, task
`Isaacsimenvs-PlayFlexivWujiRightHumanLike-Direct-v0`.

## How it runs

Each part is its own process (a "node"). They talk over ZMQ on one machine: every publisher binds
its own port, and any number of processes can subscribe. Every pose on the bus is in the robot base
frame. The topics and their fields are listed in [utils/link.py](utils/link.py).

```
perception.py ──object_pose, camera_image──┐   (+ camera_frame: every frame, raw, for the recorder)
(RealSense + FoundationPose)               │
                                           ▼
robot.py ──joint_state──────────────▶ policy.py ──joint_target──▶ robot.py
(Flexiv + Wuji, 200 Hz)               (60 Hz)  ──goal_pose

viewer.py, recorder.py: subscribe to everything, publish nothing
run.py: the entry point; starts viewer, recorder, robot (in robot.mode), and the policy in policy mode
robot.mode=openloop: no policy, no perception; robot.py sends a sim rollout's joint_target itself
```

| node | runs in | what |
|---|---|---|
| [run.py](run.py) | `.venv_deploy` | the entry point: starts the nodes for a robot.mode, ends them on Ctrl-C |
| [perception.py](perception.py) | FoundationPose env, robot PC GPU | RealSense → FoundationPose → `object_pose` (+ `camera_image` for the viewer); `perception.source=fake`: a fixed pose, no camera (any env) |
| [robot.py](robot.py) | `.venv_deploy` | arm (aioflexiv; or its MuJoCo arm) + hand (aiowuji; or `fake`, no hand); modes `home`, `free_drive`, `policy`, `openloop` |
| [policy.py](policy.py) | `.venv_deploy` | observation → LSTM → joint targets; goals and the goal test |
| [viewer.py](viewer.py) | `.venv_deploy` | live viser page: robot, targets, object, goal, camera, topic status |
| [recorder.py](recorder.py) | `.venv_deploy` | every message → `recordings/<date>_<time>_<name>.jsonl` |
| [playback.py](playback.py) | `.venv_deploy` | plays a recording back in viser; exports an mp4; never moves the robot |

Other folders: `utils/` (helpers used by several nodes, see [utils/\_\_init\_\_.py](utils/__init__.py)),
`spec/` (facts exported from the training env, and the robot URDF), `objects/` (object meshes),
`checks/` (offline checks against a recorded sim rollout, and the scripts that make the spec and
the rollout).

## Config

Every node reads [config.yaml](config.yaml). Override any value on the command line with dotted keys:
`run.py robot.mode=home robot.arm=<ROBOT_SN>`. A key that is not in the file is an error (a typo
stops the run instead of being ignored). To print the merged config:
`.venv_deploy/bin/python deploy/simtoolreal/utils/config.py <overrides>`.

The spec json (`spec/`) is not a setting: it is copied from the training env. Never edit it by hand.

## Setup on the robot PC

- `.venv_deploy` (Python 3.12) with this repo's `third_party/aioflexiv` and `third_party/aiowuji`, plus
  `pip install pyzmq "viser==1.1.0" yourdfpy imageio-ffmpeg` (imageio-ffmpeg: the ffmpeg that
  recorder.py and playback.py encode mp4s with).
- The FoundationPose environment (conda env `foundationpose`, the `kushal2000/FoundationPose` checkout at
  `~/FoundationPose`), plus `pip install pyzmq omegaconf pyrealsense2`.
- The checkpoint at `pretrained_policy/nov6_dr.pth` (gitignored).
- The robot meshes for the viewer: `scripts/flexiv_wuji/fetch_assets.sh` (the right hand is gitignored).
- In `config.yaml`: `perception.foundationpose_dir` (the checkout) and `perception.extrinsics` (the
  camera pose from `deploy/calibration/calibrate_compare.py`; now `left_camera_nov7.json`, D415
  821312062243). perception.py opens the camera named in the extrinsics file and refuses another;
  after moving the camera, calibrate again and point `perception.extrinsics` at the new file.
- The hand and its mount must be set up as the Flexiv tool payload. Flexiv compensates gravity
  with it; without it the arm sags in free drive.

## Run without hardware

MuJoCo arm, fake hand, fixed object. Two terminals, from the repo root:

```bash
PY=.venv_deploy/bin/python; D=deploy/simtoolreal
$PY $D/perception.py perception.source=fake
$PY $D/run.py robot.arm=mujoco robot.hand=fake run.duration_s=30   # viewer at http://localhost:8091
```

The object does not move here, so no goal is ever reached; each goal times out after 10 s.

## Run on the robot

```bash
PY=.venv_deploy/bin/python; D=deploy/simtoolreal
ARM="robot.arm=<ROBOT_SN> robot.hand=<HAND_SN>"

# 1. get the robot to home (the training start pose, spec default_q)
$PY $D/run.py robot.mode=home $ARM             # move there (joint-space line, no table avoidance)
$PY $D/run.py robot.mode=free_drive $ARM       # or: move it by hand; prints how far each joint is from home

# 2. perception (FoundationPose env): click 4 corners of a box around the object, then it tracks
~/miniconda3/envs/foundationpose/bin/python $D/perception.py
#    several objects: object.name=[plate7_white,plate7_brown] (one box to click per object; the policy
#    sees the first; give the viewer / run.py the same object.name to draw them all)

# 3. the policy run (robot.mode=policy is the default)
$PY $D/run.py $ARM record.name=eraser_try1     # record.camera=true: also the camera images
                                               # run.viewer=false: no viewer; no record.name: no recording
```

| robot.mode | run.py starts | the robot |
|---|---|---|
| `policy` (default) | viewer, recorder, robot, then policy | refuses to start with the arm away from home; follows the policy |
| `home` | viewer, recorder, robot | moves to home, then robot.py exits |
| `free_drive` | viewer, recorder, robot | movable by hand until Ctrl-C |
| `openloop` | viewer, recorder, robot | refuses to start with the arm away from home; sends a sim rollout's joint targets, then holds the last one |

The viewer (http://<robot PC>:8091) starts unless `run.viewer=false`; the recorder only with
`record.name`. In every mode the viewer stays up after the robot stops, until Ctrl-C.

run.py asks you to type `go` on real hardware, with a warning for the mode. The robot refuses to start with the arm away from
home, then brings the hand home itself (it went limp when the last run ended). The policy waits
for the robot's first joint_state. For the first runs, use `robot.hand=dry_run robot.hand_sn=<HAND_SN>`:
the hand is connected but never enabled.

**How a run ends.**
- The policy ends (duration, max goals, goal timeout, object off the table, safety trip): the robot
  holds where it is.
- The Flexiv E-stop or enabling button: it stops only the arm. robot.py sees the arm's torque loop
  stop, stops the hand itself and exits; the policy then stops (no more joint_state).
- In both cases the viewer and the recorder stay up, so you can look at what happened.
- Ctrl-C on run.py ends everything, in this order: policy, robot (holds, then stops; stopping the
  hand disables it, so the fingers go limp), recorder, viewer.

The nodes can also run one by one, each in its own terminal (`robot.py robot.mode=...`,
`viewer.py`, `policy.py`, `recorder.py record.name=...`): that is all run.py does.

## Fold: the fold-pool policies on two taped plates

`task=fold` runs a fold-pool policy (`deploy/policy/*.pth`, branch `yiboc_wuji`): two plates taped
along one edge make a sheet; the policy turns one plate over onto the other. Everything else (robot,
viewer, recorder, safety) is the same as above.

```bash
# perception: both plates, the MOVING one first (one box to click per plate)
~/miniconda3/envs/foundationpose/bin/python $D/perception.py task=fold
# the run (flat7RW_base by default; any of the five: policy.checkpoint=deploy/policy/<name>.pth)
$PY $D/run.py task=fold $ARM record.name=fold_try1
```

`task=fold` applies `fold.overrides` in config.yaml (spec, checkpoint, `object.name:
[plate5_brown, plate5_white]` = [moving, stationary]), then the command line. Start with the two
plates lying flat, side by side, taped edge to edge, about where training put the sheet: centre 0.51 m
in front of the base (base frame (0, −0.51)) ± 10 cm, any heading. The policy refuses to start
otherwise (`PlateFrames.start`).

What the policy sees (`utils/fold.py`, every number checked against the training code by
`checks/check_fold.py`):

| | |
|---|---|
| object | the moving plate's 4 corner keypoints: training's 7 x 7 grid, rows 16.7 and 50 mm from the hinge (10 cm sheet), so the object position is 8.3 mm off the plate centre, away from the hinge; orientation = the plate's, with the sign rule of training's `_mat_to_wxyz` |
| goal | the same corners mirrored onto the stationary plate (training's hinge is on the mid-plane: no lift), recomputed every step from the stationary plate. Its quaternion is relabelled (x, y, z, w) -> (w, x, y, z) as the PhysX training env does (`physx_envs._goal_quat_from_matrix`): keep it |
| object_scales | (sheet, sheet, 0.016) / 0.04 = (2.5, 2.5, 0.4) at 10 cm: the z is the sim's felt thickness, not the plate's |
| success | every corner within 1.5 cm of the fold target AND the sheet's extent along the stationary plate's x ≤ 0.65 x sheet. The episode ends there (as in training). The target is the REAL fold: tape on the top face puts the folded plate 2 x `fold.hinge_z` = 5 mm above training's |
| stops | folded; 600 steps (10 s, `goals.timeout_steps`); a plate 5 cm below the table top; the plates' hinge edges more than `fold.max_hinge_gap_m` (3 cm) apart, which means a tracker flipped or slipped |

A plate is a box, so FoundationPose may report it turned 180° about any of its axes. `PlateFrames`
takes the pose that lies flat with the hinge edges together at the start, and after that, for each
plate, the symmetric pose closest to the last frame.

The fold spec (`spec/spec_flexiv_wuji_right_fold.json`, `checks/export_fold_spec.py`) differs from
the Play spec in what the deploy code reads: the real joint limits (`rizon4s_right_wuji_real_limits.urdf`:
arm 5° inside, finger PIP ≤ 100°), the table top 0.33 m above the base, and the fingertip pad offsets
(per finger: the thumb gets 22.9 mm, not the ring finger as in Play). The five runs in `deploy/policy/`
share every one of these (`export_fold_spec.py --compare`):

| checkpoint | trained on |
|---|---|
| `flat7RW_base` | 7 shapes, S2R hand drive, 16 mm felt, no observation DR |
| `flat7RW_fricition` | + plate/robot/table friction DR |
| `flat7RW_sysid` | + the identified hand (friction, spring, armature, 1.0 A x Kt) and its DR |
| `flat7RW_allDR` | identified hand, 5 mm felt (3-7), point-level object DR, delays, noise, table height ± 1.5 cm |
| `box2RW_allDR` | as `flat7RW_allDR`, two-plate sheets only |

Without hardware: `perception.py task=fold perception.source=fake` sends the plates of a flat sheet at
`perception.fake_pose_base` (the sheet's hinge middle), the moving one turned up by `fold.fake_fold_deg`.

## robot modes

| mode | what |
|---|---|
| `home` | Hand to home first (it opens and lets go), then the arm, both jerk-limited at `move_speed` / `hand_move_speed`; checks it got there, then exits |
| `free_drive` | Arm: kp 0, light damping, gravity compensated, movable by hand. Hand: connected, never enabled. Publishes `joint_state` and prints the distance to home every second |
| `policy` | Forwards `joint_target` at 200 Hz. No new target for 0.1 s: holds the measured pose until targets come back |
| `openloop` | Hand to home, then sends target k of `openloop.path` from k/60 s on, with no feedback, and the same safety checks as `policy`. After the last one it holds it until Ctrl-C |

Home is the training env's start pose: arm −120, 0, 30, 100, 150, 80, 120°, hand all 0. It is not
aioflexiv's own home.

## Open loop: lift the eraser 10 cm without camera or policy

The policy runs only in sim. Its joint targets are then sent to the real robot unchanged. The real
eraser must lie where the sim's did; nothing checks it.

1. Record the rollout in the training env (Isaac Sim, the training branch's worktree):
   ```bash
   OUT=$PWD/docs/results/simtoolreal/lift1 EXTRA_ARGS="--num_envs 256 --candidates 16" \
       sbatch scripts/cluster/bos14_simtoolreal_lift.sh
   ```
   [checks/record_lift.py](checks/record_lift.py): the env's eraser box pinned to the DexToolBench
   mesh (12.5 x 2.8 x 5 cm, 70 g), lying flat at the table centre, the goal 10 cm above it.
   - The policy runs in every env. It often drops the box and grasps it again; that needs
     feedback, so only envs that lift and hold without a drop are kept.
   - Each kept rollout's targets are sent open loop in sim, from the same start, to several envs;
     the one that works most often is saved to `openloop.npz`.
   - Its targets are then sent to boxes placed off by up to 2 cm / 20 deg, and with the hand's
     gains x0.5 to x2: `robustness.json` says how exactly the real eraser has to be placed.
2. On the robot PC (copy `openloop.npz` to `openloop.path`):
   ```bash
   $PY $D/run.py robot.mode=home $ARM
   $PY $D/run.py robot.mode=openloop $ARM record.name=lift_try1
   ```
   Before `go`, run.py prints where the eraser goes (in the robot base frame). The viewer draws the
   sim's eraser in blue and its goal in orange.
   The sim's eraser lies on the bare table (z 0.28). Measured on Nov 8, the black foam pad there has
   its top about 3 cm higher (base z ~0.32, the table ~0.29): take it away, or the eraser sits 3 cm
   higher than the targets expect.

## Safety (policy and openloop modes)

| check | when | limit |
|---|---|---|
| arm tracking: any arm joint vs the target it is being sent | every tick | 25° (training reaches ~19° when the hand presses) |
| target jump: a new target vs the last one sent | before it is sent | arm 2°, hand 20° per step (training's action pipeline: 0.14° and 12.5° at most) |

A trip holds the measured pose (the hand keeps its grip) and ignores every later target. robot.py puts
the reason in `joint_state` (`tripped`), so policy.py stops and the viewer shows it. Ctrl-C to end.

## Recording and playback

`run.py record.name=<name>` starts recorder.py for that run. It writes every message, as sent, to
`recordings/<date>_<time>_<name>.jsonl` (gitignored). Line 1 is the run's full config. Size: about
25 MB/min. recorder.py can also run on its own, e.g. to record a free-drive session.

`record.camera=true` also records the camera: perception.py sends every camera frame raw
(`camera_frame`, 640 x 480 at 60 fps, whatever FoundationPose's rate), and recorder.py encodes them
to `<same name>.camera.mp4` (H.264, `record.camera_codec`, `record.camera_quality`; libx264 at about
0.4 of a CPU core). The .jsonl gets each frame's time, so playback lines the video up with the rest.
Recordings from before Nov 9 evening have the viewer's 10 Hz JPEGs in the .jsonl instead; playback
reads those too.

```bash
$PY $D/playback.py $D/recordings/<file>.jsonl      # http://localhost:8091: time slider, play, speed
```

To export a video, set the 3D view in the browser, then click "export mp4". The browser tab renders
every frame, the camera image goes on the right, and the video is saved next to the recording
(`<recording>.mp4`, H.264 at `record.video_fps`).

## Checks

`checks/check_rec1.py` runs the deploy code on a rollout recorded in the training env
(`docs/results/simtoolreal/rec1`) and compares the numbers.

| | max error |
|---|---|
| FK (palm, fingertips), the 134 observation numbers | ~1e-6 |
| action → joint targets | 1e-7 |
| goal-test keypoint distance | 6e-8 |
| network actions, CPU vs the sim's GPU | mean 0.004, max 0.09 |
| own goal sampler vs training's rules (2000 goals) | no rule broken |

Run it after any change to `utils/observation.py`, `utils/robot_model.py`, `utils/goals.py` or
`policy.py`, and after a new checkpoint or spec. The rollout (32 MB) is gitignored, so it runs only
where the rollout is.

| script | what | runs in |
|---|---|---|
| `checks/record_lift.py` | the lift rollout for `robot.mode=openloop`, and its open-loop test in sim (cluster: `scripts/cluster/bos14_simtoolreal_lift.sh`) | the training branch's worktree, Isaac Sim |
| `checks/record_sim_rollout.py` | records the policy in the training env, every step → `rollout.npz` (cluster: `scripts/cluster/bos14_simtoolreal_record.sh`) | the training branch's worktree, Isaac Sim |
| `checks/export_spec.py` | reads the training branch from git → `spec/` (spec json + URDF); `--rollout_meta` adds the fingertip order | `.venv_deploy`, no Isaac Sim |
| `checks/export_fold_spec.py` | the fold spec from `yiboc_wuji` (task YAML + robot overlay + experiment, as train.py layers them); `--compare` diffs the five runs | `.venv_deploy`, no Isaac Sim |
| `checks/check_fold.py` | `utils/fold.py` against the training code's own methods (pulled from git, run with torch) on random sheets; the success test at 0-180°; `PlateFrames` through random symmetric flips | `.venv_deploy`, no Isaac Sim |

## Things the policy depends on that are easy to break

- **Fingertip order.** The observation lists the fingertips in PhysX body order (index, middle,
  pinky, ring, thumb), with the pad offsets applied by position, so the ring finger gets the thumb's
  22.9 mm. The spec's `fingertip_obs_*` fields hold this; do not "fix" it.
- **Object frame.** FoundationPose returns the pose of the mesh's own frame, and the keypoints sit
  around that frame's origin. Use DexToolBench's mesh unchanged (`objects/`). Its origin is not the
  mesh centre.
- **Gravity.** The sim robot has no gravity: Flexiv must compensate it, with the hand as the tool payload.

## Not verified yet

- perception.py has run on the robot PC (D415, 640 x 480: 60 Hz, ~12 ms from frame to pose), but
  only on a mask of the empty foam pad, not on the eraser. On its first run with the eraser, check in
  the viewer that the tracked mesh sits on the real object.
- Hand gains: converted with the sysid's per-joint Kt (`robot.hand_kt`); not yet checked on the robot
  that the hand tracks like the sim.
- Hand velocity: `hand_velocity()` in robot.py is a placeholder (a position difference) until it is
  measured on the hand.
- How long robot.py takes to see a Flexiv E-stop.
- The mp4 export's 3D rendering. It needs a browser, so it was tested only with a stand-in.
- Nothing has run on the real robot yet. All tests so far used the MuJoCo arm and a fake hand.
- Fold: no sim rollout of a fold policy has been run through the deploy code (the Play `check_rec1`
  equivalent); the observation is checked against the training code piece by piece
  (`check_fold.py`), and a MuJoCo run with fake plates moved the hand over the sheet. Not checked:
  FoundationPose on 5 mm plates while a finger covers them; the hinge-gap stop on a real fold; the
  hand's effort limit (`robot.hand_effort_limit` 1.0 A) against flat7RW_base / _fricition, which
  trained with the S2R drive's higher torque limits.
