# Cloth folding

One fold task on two backends. The policy sees the same observation and its actions mean the same
thing on both, so a policy trained on the rigid pool is evaluated on the VBD cloth unchanged.

| Task id (`Isaacsimenvs-<name>-Direct-v0`) | Backend | Simulator | Manipuland |
|---|---|---|---|
| `Cloth` | `backends/vbd` | Newton (`.venv_isaaclab3`) | VBD particle cloth |
| `RigidCloth` | `backends/rigid` | PhysX (`.venv_isaacsim`) | pool of 7 hinged-plate fold shapes |

The robot is `env.robot_setup` (a file stem in `isaacsimenvs/scripts/configs/robot/`), set in the run recipe or on
the command line. The older ids stay registered as aliases that carry their setup:
`FoldPoolPhysx<Robot>` -> `RigidCloth`, `Cloth<Robot>` -> `Cloth`, e.g.
`Isaacsimenvs-FoldPoolPhysxFlexivWujiRightReal-Direct-v0` = `RigidCloth` + `flexiv_wuji_right_real`.

## Layout

```text
cloth/
  __init__.py         task registration (canonical ids + aliases)
  cloth_env.py        ClothEnv(DirectRLEnv): the lifecycle hooks, wired to mdp/
  env_cfg.py          ClothEnvCfg: robot, obs, action, reward, reset, termination, DR, cloth
  cloth_cfg.py        ClothCfg: the sheet the task is defined on
  mdp/                the task, simulator-independent
    task.py           the fold: targets, error, criterion, goal pose (ClothTaskMixin)
    robot.py          joint orders and limits, reset pose, the hand's passive spring
    actions.py        policy action -> joint targets
    observations.py   policy and critic observations, observation DR
    rewards.py        reward terms and the reward guard
    terminations.py   fold, fall, hand far, divergence, timeout
    reset.py          table, sheet and tracker resets
    randomization.py  per-episode DR draws (delays, biases, point DR, PD gains, hand dynamics)
    point_dr.py       point-level observation DR
    tracker.py        tracker observation DR (TrackerMixin) and the pipeline it replays
    geometry/         grid.py (sheet grid, keypoints, fold targets), fold_shapes.py (pool shapes),
                      quat.py (the quaternion helpers, one place for the Newton backend to wrap)
  backends/
    scene.py          robot, table, goal marker, ground, light; then the backend's sheet
    adapter.py        ClothAsRigidObject: the sheet as the rigid object the observation encodes
    rigid/            RigidClothEnv: env.py, config.py, scene.py, contacts.py, randomization.py,
                      checks.py (startup proofs), physx.py, pool.py (adapter), assets.py, usd.py
    vbd/              VbdClothEnv: env.py, config.py, newton_setup.py, pierce.py,
                      newton_support/ (Isaac Lab 3.0 compat, patches, USD cache)
  perception/         point clouds for the point-cloud students (pointcloud, scene_pointcloud,
                      robot_surface)
  pose_viewer.py      training-time interactive viewer
  tests/
```

Robots (`RobotSpec`s and the URDF -> USD pipeline) are `isaacsimenvs/robots/`, shared with the Play
task. The fold shapes are YAML designs with a PNG each in `isaacsimenvs/scripts/configs/rigid_cloth_designs/`
(regenerate the PNGs with `isaacsimenvs/scripts/visualization/rigid_cloth/render_designs.py`).

The observation is the encoding the Play prior was trained on: the sheet's moving half as a rigid
object (`backends/adapter.py`), the fold goal as a pose (`ClothTaskMixin._update_goal`), so every
cloth checkpoint, and the Play prior, reads the same vector.

## Training

Launchers are in `experiments/` (scheduler-agnostic; see `experiments/README.md`). Array index k =
seed k + 1.

### On PhysX (rigid plates)

```bash
RUN_TAG=base WANDB=1 experiments/train.sh --task Isaacsimenvs-RigidCloth-Direct-v0 \
    --experiment rigid_cloth_right_wuji \
    --checkpoint pretrained_policy/flexiv_right_wuji/nov8_dr_friction_0p5.pth
```

- The recipe supplies env count, epochs, minibatch and the robot setup; DR comes from its `dr:` blocks
  (`isaacsimenvs/scripts/configs/README.md`). Override anything with hydra arguments.
- `--checkpoint` loads model + normalizers only (`--checkpoint_load_mode weights`, the default here);
  `LOAD_MODE=resume` also loads the optimizer and epoch counter, `max_epochs` is then the total.
- `PRIORS="<s1> <s2> <s3>"` gives each array index its own start checkpoint.
- Output: `outputs/<date>/<time>/0_pxpool_<RUN_TAG>_s<seed>_<job>/nn/`. Files:
  `snap_*_ep_<N>.pth` (every `snapshot_every` epochs), `last_*`, `best_*`.

### On VBD cloth (Newton)

```bash
RUN_TAG=vbdRW_sharpa experiments/train.sh --task Isaacsimenvs-Cloth-Direct-v0 --experiment vbdRW_sharpa \
    --checkpoint pretrained_policy/flexiv_right_wuji/nov8_dr_friction_0p5.pth \
    env.robot_self_collision=false env.newton.njmax=32768 env.scene.num_envs=1488 \
    agent.params.config.minibatch_size=23808 agent.params.config.central_value_config.minibatch_size=23808 \
    agent.params.config.expl_coef_block_size=248
```

- One GPU per seed (`--distributed` for several GPUs on one machine). 1488 envs is the most the Flexiv
  fits (Warp's int32 contact buffer).
- Robot self-collision must be off for the Flexiv. With it on, MJWarp is ~26x slower.
- Output name: `0_pxvbd_<RUN_TAG>_s<seed>_<job>`.

## Evaluation

The metric is held folds on the VBD cloth, 32 episodes x 2 eval seeds per cell, in 4 cells:

- start: pinned (one fixed start state) or rand (`--randomize`)
- keypoint cutoff: 4 cm or 1.5 cm

The key metrics are 1.5 cm pinned and 1.5 cm rand. `experiments/eval.sh` runs them for one checkpoint,
a training array, or a list file (one checkpoint per line):

```
<tag> <seed> <checkpoint .pth>
<tag> <seed> <run name> <best | last | ep:N>      # resolved when the job runs
```

```bash
LIST=my_list.txt TARGETS=vbd INDEX=<k> experiments/eval.sh        # k = 0 .. 2 x lines - 1
```

- Index = (line x targets + target) x starts + start; `TARGETS` (default `pool7 vbd`), `STARTS`
  (default `pinned rand`).
- The policy's robot and spawn height: `ROBOT_SETUP` (default `flexiv_wuji_right_hl_reallimits`),
  `START_HEIGHT` (default 0.20). VBD runs at the ns6 protocol (`num_substeps 6`, cloth `substeps 2`).
- Output: `docs/results/px_matrix/<tag>_s<seed>__on_<target>__<pinned|rand>__<4cm|1p5cm>_s<es>.json`;
  `isaacsimenvs/scripts/evaluation/matrix_summary.py <tag>` tabulates them.
- `EVAL_SEEDS="3 4 5 6 7 8"` re-tests a picked checkpoint on fresh seeds.
