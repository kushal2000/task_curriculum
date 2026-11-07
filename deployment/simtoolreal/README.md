# Play policy on the Flexiv Rizon 4s + right Wuji Hand 2

Deploys `pretrained_policy/nov6_dr.pth`: the SimToolReal-style Play policy (move a tool to a goal pose) trained on
branch `kushal/2026-10-30_pretraining_real`, task `Isaacsimenvs-PlayFlexivWujiRightHumanLike-Direct-v0`
(right human-like Wuji, the real setup's table and home pose, all Play DR + wrench impulses).

| file | what |
|---|---|
| `export_spec.py` | reads the training branch from git, writes `spec_flexiv_wuji_right_hl.json` + the robot URDF |
| `core.py` | numpy only: FK, the 134-dim observation, action pipeline, goal sampler, success test, policy loader |
| `run.py` | the 60 Hz deploy loop; backends `isaac` (sim_server) and `real` (aioflexiv + aiowuji) |
| `sim_server.py` | the training env as a plant, for `run.py --robot isaac` (runs on the branch's code) |
| `record_sim_rollout.py`, `check_offline.py` | record the policy in its training env; check `core.py` against it |
| `objects.json` | known objects: grasp-box scale (DexToolBench `Object.scale`) |

Checked so far (2026-11-07): `core.py` reproduces the training env's observation, targets and goal
test to ~1e-6 (`docs/results/simtoolreal/rec1`); `run.py` against the env as a plant reaches goals at
the env's own rate, also with training-level delays and noise (`docs/results/simtoolreal/s2s1`);
`run.py --robot real --arm mujoco --hand fake` holds 60 Hz (~5 ms read-to-send).

Things the policy depends on that are easy to break:
- **Fingertip order.** The observation lists the fingertips in PhysX body order (index, middle,
  pinky, ring, thumb) with the spec's pad offsets applied by position, so the ring gets the thumb's
  22.9 mm and the thumb 19 mm. The spec's `fingertip_obs_*` fields hold that; do not "fix" it.
- **Object frame.** Keypoints sit at the corners of the grasp box (scale x 4 cm) around the object
  frame's origin: the tracker's mesh frame must be DexToolBench's (box centre, x along the handle).
- **Hand gains.** The sim uses Wuji's Isaac Lab drives (Nm/rad); the firmware takes A/rad.
  `--hand_kp_per_nm 2.0` is a guess until the torque constant is measured (aiowuji
  `scripts/sysid_ref_collect.py` + `sysid_fit.py`): kp_A = kp_Nm / (1.95 Kt).
- **Gravity.** The sim robot has no gravity; the real arm's controller must compensate it, including
  the hand and its mount as the Flexiv tool payload.

## Real-robot sequence

Robot PC: `.venv_deploy` (Python 3.12), this repo with `third_party/aioflexiv`, `third_party/aiowuji`,
and the checkpoint at `pretrained_policy/nov6_dr.pth` (gitignored). Each run logs to `deployment/simtoolreal/logs/`.

```bash
R=deployment/simtoolreal/run.py; PY=.venv_deploy/bin/python

# 0. no hardware: MuJoCo arm + fake hand + fake object (the same async code path)
$PY $R --robot real --arm mujoco --hand fake --object fake --duration_s 20 --on_episode_end next_goal

# 1. arm only, hand connected but never enabled; replay a sim rollout open loop (17 s)
$PY $R --robot real --arm <ROBOT_SN> --hand dry_run --hand_sn <HAND_SN> \
    --mode replay --replay docs/results/simtoolreal/rec1/rollout.npz --replay_env 0
#    -> summary replay_err_deg_arm_*: arm tracking vs the sim's

# 2. hand joint directions: aiowuji's own tools, one joint at a time, against the sim URDF in viser
#    (scripts/flexiv_wuji/viser_compare.py); the URDF and aiowuji's right MJCF match exactly in software

# 3. both live, open-loop replay
$PY $R --robot real --arm <ROBOT_SN> --hand <HAND_SN> --mode replay --replay docs/results/simtoolreal/rec1/rollout.npz

# 4. policy, no object on the table, fixed fake object pose (flat at the table centre by default)
$PY $R --robot real --arm <ROBOT_SN> --hand <HAND_SN> --object fake --duration_s 30 --on_episode_end next_goal

# 5. policy with the tracked object (FoundationPose backend: to do)
```

Safety (abort by default, `--on_safety count` to only count):
- Arm target > 25 deg from the measured angle, or > 10 deg for 0.5 s.
- Palm centre < 2 cm above the table.
- Whole hand vs the real table (`table_guard.py`): the real table has training's top height and near edge
  but is larger (`--table_x_half` 1 m, `--table_far_y` -1 m, env frame). Every wrist/hand mesh point
  (convex hulls, 4 mm grid, 0.4 ms per pose) is checked each tick:
  - measured pose: >= `--min_hand_height` (-1 cm: contact allowed, training's own reaches -0.7 cm);
  - arm targets with the hand at its measured angles, over training's table: >= -`--max_press_depth`
    (2 cm; training presses up to ~6 cm while grasping: 2.5% of steps > 2 cm, so raise it for the policy
    once the hand is trusted); over the rest of the real table, which the policy never felt: >= -1 cm.
- Replay preflight: before anything moves, every recorded step is checked the same way (and, with
  `--hand dry_run`, with the hand open, since it does not follow its targets); a violation refuses the run
  and suggests `--replay_max_s`.
- Scripted moves (to the start pose, `go_home`: hand opens first, then a joint-space move) are checked
  along the joint-space line first; if unsafe the arm holds and says so.
- MuJoCo has no table, so policy runs there can trip the hand checks where the real table would stop
  the hand. Ctrl-C holds the measured pose.
