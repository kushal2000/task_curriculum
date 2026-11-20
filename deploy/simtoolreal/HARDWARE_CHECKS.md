# Hardware checks: Wuji hand, Flexiv arm, SimToolReal deployment

One sim workflow, then the robot: **train in Isaac Sim → record in Isaac Sim → play in viser → hardware**.
The order:

1. Hand connection: the joint mapping in Isaac Sim, then the real hand.
2. Arm connection: the real arm, small moves.
3. Right hand, the validated `deployment` code and checkpoint: viser preview, then hardware.
4. Left hand, this branch (`joy/2026-11-18_deploy_left`): Isaac Sim recording, viser preview, then hardware.

Each step uses only what the steps before it proved works. Keep a hand on the e-stop whenever the arm
is enabled.

## 0. Setup

From the repo root, on the robot PC and on a laptop:

```bash
git submodule update --init third_party/aiowuji third_party/aioflexiv
uv pip install --python .venv_deploy/bin/python -e ./third_party/aioflexiv -e ./third_party/aiowuji \
    pyzmq "viser==1.1.0" yourdfpy imageio-ffmpeg
scripts/flexiv_wuji/fetch_assets.sh     # robot meshes (the right hand's are gitignored)
PY=.venv_deploy/bin/python; D=deploy/simtoolreal
```

- **macOS:**
  - `fetch_assets.sh` uses GNU `sed -i`, so run it with GNU sed on the `PATH`.
  - FoundationPose and Isaac Sim need Linux and an NVIDIA GPU, so neither runs on a Mac. Record on a Linux
    GPU machine (cluster, lambda-joy) and copy the recording over; viser runs anywhere.

### The sim workflow

| Step | Where | Command |
|---|---|---|
| Train | Linux + GPU, training branch `joy/2026-11-18_play_flexiv_left`, `.venv_isaacsim` | `scripts/cluster/sbatch_play_csail.sh` |
| Record a rollout in Isaac Sim | Linux + GPU, same venv | `checks/record_lift.py --headless ... --goals train` (4a) |
| Play the recording in viser | any machine, `.venv_deploy` | `$PY $D/checks/view_lifts.py <out>/top0 ... --port 8092` |
| Play the policy live in viser | Linux + GPU, `.venv_isaacsim` | `python -m isaacsimenvs.eval.viser_play --run <run dir or .pth>` (below) |

**Live Isaac Sim in a browser** (Linux + GPU; training branch `joy/2026-11-18_play_flexiv_left`, `.venv_isaacsim`):

```bash
OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python -m isaacsimenvs.eval.viser_play \
    --run <run dir or .pth> --port 8081                  # http://localhost:8081, or ssh -L 8081:<node>:8081
```

- `--run` takes a run name under `outputs/`, or a run directory or `.pth` anywhere (e.g.
  `runs/play_flexiv_left_dr/<job>/0_play_flexiv_left_dr`).
- It runs one environment per object, stepped together on the GPU.
- Controls: play, pause, step, reset, which object, its start orientation, the SAPG exploration block,
  and greedy or sampled actions.
- On the cluster, run it as a GPU job and connect through an ssh tunnel.

---

## 1. Wuji hand connection

**1a. Find the hand.** Read-only; nothing moves.

```bash
.venv_deploy/bin/aiowuji scan                 # every Wuji hand on USB, with serial numbers
.venv_deploy/bin/aiowuji status <HAND_SN>     # firmware, Side (left/right), joint and gain status
```

- Note the serial, e.g. `WH2KA01260825022`. Every later command picks the hand by its serial.
- `Side` must match the spec you deploy. `run.py` checks it too and refuses a hand of the wrong side.

**1b. Move one joint per finger: first the mapping in Isaac Sim, then on the hand.**

```bash
# sim (training branch, Linux + GPU): each hand joint +0.5 rad through the policy's own action pipeline
OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python scripts/flexiv_wuji/check_joint_mapping.py \
    --task Isaacsimenvs-PlayFlexivWujiHumanLike-Direct-v0 --headless --out left.json
#   compare_joint_mapping.py (same folder): left vs right (same index = mirror-image motion of the same
#   finger) and the sim vs the deploy code's URDF forward kinematics
$PY third_party/aiowuji/examples/00_move.py <HAND_SN> --slot 1 --delta 0.3 --max-velocity 0.5 --cycles 1  # real
```

- The real command moves joint S1 (flexion) of each finger out by 0.3 rad and back, one finger at a time,
  thumb first. A positive delta must **flex**, curling the finger toward the palm, as in the sim check.
- Each finger has joints S1–S4 (`--slot 1..4`). Joint index = finger × 4 + slot − 1.
- Below about 0.5 rad/s the real joints stick and slip, so slow moves look stepped.
- If a real joint doesn't move, rerun with `--disabled-joints <i>` to skip it, then check its cable.

---

## 2. Flexiv arm connection

**2a. Status.** Read-only.

```bash
.venv_deploy/bin/aioflexiv status <ARM_SN>    # e.g. Rizon4s-063533: connection, mode, faults, recent events
```

- Clear any fault or e-stop on the teach pendant first.
- If it can't find the robot, add `--network-interface <PC IPv4 on the robot network>`.

**2b. Move one joint, small.**

```bash
$PY third_party/aioflexiv/examples/00_move.py <ARM_SN> --joint 0 --delta 0.05     # from where the arm is, and back
```

- `--joint` is 0–6, for joint1–joint7. Keep `--delta` small.

**2c. Optional: move the arm by hand.**

```bash
.venv_deploy/bin/aioflexiv gravcomp <ARM_SN>   # zero stiffness, gravity compensated; Ctrl-C stops and goes IDLE
```

---

## 3. Right hand: the validated deployment

These files live on the robot PC and are gitignored. Copy them to a laptop to run 3a there.

```bash
CKPT=pretrained_policy/nov8_friction1p5_dr.pth
ROLL=docs/results/simtoolreal/lift_nov9/nov8_friction1p5_dr_train/top1      # holds openloop.npz
ARM="robot.arm=Rizon4s-063533 robot.hand=WH2KA01260825022"
```

**3a. Preview in viser.** Nothing connects to the robot. What the policy did in Isaac Sim, with physics:
the recorded rollout these goals came from.

```bash
$PY $D/checks/view_lifts.py $ROLL --port 8092         # http://localhost:8092; --camera none without the extrinsics file
```

- "policy rollout" plays the closed-loop run.
- "open loop replay" plays exactly what `robot.mode=openloop` would send to the robot.
- Solid robot = the sim's joints. Green ghost = the targets sent. Orange = the goal.

**3b. Hardware.** A dry run first (the hand is never enabled), then the real run.

```bash
$PY $D/run.py robot.mode=home $ARM                                   # hand home, then arm to (-120, 0, 30, 100, 150, 80, 120) deg
~/miniconda3/envs/foundationpose/bin/python $D/perception.py         # terminal 1: click the object's box
$PY $D/run.py robot.arm=Rizon4s-063533 robot.hand=dry_run robot.hand_sn=WH2KA01260825022 \
    policy.checkpoint=$CKPT goals.from_rollout=$ROLL/openloop.npz goals.timeout_steps=1200   # dry run: hand never enabled
$PY $D/run.py $ARM policy.checkpoint=$CKPT goals.from_rollout=$ROLL/openloop.npz goals.timeout_steps=1200
```

- `goals.from_rollout` uses the recorded rollout's goals in order. The run stops after the last one.
- `goals.timeout_steps=1200` gives each goal 20 s (60 steps = 1 s).
- **Automatic stops.** The run stops itself if any of these happens:
  - the joint state is older than 50 ms;
  - an arm joint is more than 25° from its target;
  - a target jumps by more than 2° (arm) or 20° (hand) in one step.

  Ctrl-C stops it at any time.
- The viewer at http://localhost:8091 stays open after the run ends.

---

## 4. Left hand: this branch

Right-hand runs on this branch behave the same as on `deployment`. The left hand adds `robot.side=left`,
the left checkpoint **and its config**, and goals recorded on the left task. The safety stops are the
right hand's (3b), unchanged. `run.py` also refuses `robot.side=left` with the right hand's default
`policy.checkpoint` / `policy.config`, or with a `policy.config` from a right-hand run.

**4a. Record in Isaac Sim.** Any Linux GPU machine with the training branch's `.venv_isaacsim`.

```bash
# CSAIL cluster
sbatch /data/scratch/joyzhuo/rec_left/rec_left.sbatch    # record_lift.py --goals train on the newest left checkpoint
rsync -a --partial joyzhuo@login.csail.mit.edu:/data/scratch/joyzhuo/rec_left/out/<job>/ docs/results/simtoolreal/left/<job>/
# or lambda-joy (no Slurm: tmux)
ssh lambda-joy 'cd ~/simtoolreal_left && tmux new -d -s rec_left "CKPT=<pth> TAG=<name> bash rec_left/rec_left_lambda.sh 2>&1 | tee logs/rec.log"'
rsync -a --partial --exclude=policy_envs.npz lambda-joy:simtoolreal_left/rec_left/out/<date>_<tag>/ docs/results/simtoolreal/left/<date>_<tag>/
```

- `--save_top 3` writes `top0`, `top1`, `top2` (best first), each with `openloop.npz`.

**4b. Preview in viser**, as 3a:

```bash
$PY $D/checks/view_lifts.py docs/results/simtoolreal/left/<rec>/top* --port 8092 --camera none   # picks the left hand from the joint names
```

**4c. Hardware.** As 3b:

```bash
LEFT="robot.side=left robot.arm=Rizon4s-063533 robot.hand=<LEFT_HAND_SN>"
POLICY="policy.checkpoint=<left.pth> policy.config=<left config.yaml>"
$PY $D/run.py robot.mode=home $LEFT
$PY $D/run.py robot.side=left robot.arm=Rizon4s-063533 robot.hand=dry_run robot.hand_sn=<LEFT_HAND_SN> \
    $POLICY goals.from_rollout=<left top0>/openloop.npz goals.timeout_steps=1200   # dry run: hand never enabled
$PY $D/run.py $LEFT $POLICY goals.from_rollout=<left top0>/openloop.npz goals.timeout_steps=1200
```

- Before the first run, `aiowuji status` must show `Side: left`, and step 1b on the real left hand must
  flex every finger. So far "+ = flexion" on the left hand is proven only in sim.
- The left hand uses the right hand's motor constants (`hand_kt`), because the left hand hasn't been measured yet.
- More in README.md, section "Left hand".
