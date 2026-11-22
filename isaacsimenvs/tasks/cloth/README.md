# Cloth folding

One fold task on two backends. The policy sees the same observation and its actions mean the same
thing on both, so a policy trained on the rigid pool is evaluated on the VBD cloth unchanged.

| Task id (`Isaacsimenvs-<name>-Direct-v0`) | Backend | Simulator | Manipuland |
|---|---|---|---|
| `Cloth` | `backends/vbd` | Newton (`.venv_isaaclab3`) | VBD particle cloth |
| `RigidCloth` | `backends/rigid` | PhysX (`.venv_isaacsim`) | pool of 7 hinged-plate fold shapes |

The robot is the real setup's: `robots.FLEXIV_WUJI_RIGHT` (Wuji's HAND2_DRIVE on the hand) with the
real robot's joint limits (`rizon4s_right_wuji_real_limits.urdf`), its base where the real table puts
it. `env.hand_sysid=true` uses the identified hand instead (`flat7FlexivWujiRHL` sets it, with the
real setup's frictions and contact shell).

The configuration is `config/`: the fields in `config/schema/`, every value in `config/task.yaml`,
run recipes and DR blocks beside them (`config/README.md`).

## Layout

```text
cloth/
  __init__.py         task registration; make_env, the gym entry point
  cloth_env.py        ClothEnv(DirectRLEnv): owns a backend and the four task parts below
  config/             schema/ (the configclasses), task.yaml (their values), runs/, dr/, designs/
  mdp/                the task, simulator-independent
    state.py          SimState, SheetLayout, FoldView: what the parts hand each other
    robot.py          RobotLayout: body and joint indices, joint orders, limits, passive spring
    actions.py        JointController: policy action -> joint targets, action DR, reset joints
    fold.py           FoldGeometry: targets, stationary frame, fold error, footprint, goal pose,
                      the moving half as a rigid object
    task.py           ClothTask: episode start (table, sheet), dones, reward, the goal as state
    goal_tasks.py     GoalTasks: one-fold goal tasks on the rigid pool (fold_pool.goal_task)
    observations.py   ObservationBuilder: assembles the policy, critic and teacher vectors from the
                      sensors (perception/) and the task fields
    randomization.py  DomainRandomization: per-episode observation / action / robot DR draws
                      (fixed order); DelayLine
    geometry/         grid.py (sheet grid, keypoints, fold targets), fold_shapes.py (pool shapes),
                      fold_states.py (each shape's reachable folded states, one-fold moves,
                      look-alike classes), quat.py (the quaternion helpers, one place for the
                      Newton backend to wrap)
  backends/
    base.py           ClothBackend: build the scene, read the sheet's points, write resets;
                      reset_yaw / sheet_yaw
    scene.py          robot, table, goal marker, ground, light; then the backend's sheet
    rigid/            RigidBackend (backend.py) on PhysX: pool.py (PlatePool), scene.py
                      (building the pool, shape draws, folded starts), contacts.py,
                      randomization.py (PlateRandomization: the sheet-physics DR), checks.py (startup proofs), physx.py,
                      assets.py, usd.py
    vbd/              VbdBackend (backend.py) on Newton: physics.py (the coupled MJWarp + VBD
                      solve from the config), newton_setup.py, pierce.py,
                      newton_support/ (Isaac Lab 3.0 compat, patches, USD cache)
  perception/         the sensors: what the real robot would report, built from the sim state, each
                      with its per-step DR (noise, delay); never writes the sim
    proprio.py        ProprioObservation: joints, last targets, palm, fingertips (Play frame)
    keypoints.py      KeypointObservation: the sheet as a rigid object + 4 keypoints; OBJECT_FIELDS
    pointcloud_obs.py PointCloudObservation: the cloud with obs.object: pointcloud
    pointcloud.py     the two cameras' cloud (visibility, noise, crop; torch only)
    robot_surface.py  samples on the robot's link meshes
  tests/
```

Each hook of `ClothEnv` reads the simulator once (`sim_state()`), derives the fold from it
(`fold_view()`), and hands both to the parts:

```text
_pre_physics_step   controller.process                       (+ backend.pre_physics_step)
_apply_action       targets + passive spring to the robot    (every physics step)
_get_dones          task.compute_dones(state, view)
_get_rewards        task.compute_reward(state, view)
_reset_idx          backend.begin_reset -> controller / task draws, written to the scene
                    -> dr.reset -> goal -> backend.end_reset
_get_observations   task.update_goal, observations.build
```

For evaluation and viewers the env answers `fold_error()`, `fold_targets_w()`,
`cloth_keypoints_w()`, `footprint_ratio()`, `is_folded()`, `folded_half_w()`, `cloth_points_w()`,
`object_pos_w()`; anything else is on `env.task`, `env.backend` or `env.robot_layout`.

The robot (`RobotSpec` and the URDF -> USD pipeline) is `isaacsimenvs/robots/`, shared with the Play
task; the training-time wandb viewer is `isaacsimenvs/scripts/visualization/wandb_viewer/`.
The fold shapes are YAML designs with a PNG each in `isaacsimenvs/tasks/cloth/config/designs/`
(regenerate the PNGs with `isaacsimenvs/scripts/visualization/rigid_cloth/render_designs.py`).

The observation is the encoding the Play prior was trained on: the sheet's moving half as a rigid
object (`FoldView.object_*`), the fold goal as a pose (`ClothTask.goal_*`), so every cloth
checkpoint, and the Play prior, reads the same vector. `obs.object: pointcloud` replaces the sheet's
fields (`OBJECT_FIELDS`) with a point cloud of what two depth cameras see over the table
(`obs.pointcloud`, `perception/`): the policy then reads `[proprio | points | mask]`, and the env also
returns the clean keypoint vector as `teacher` (for DAgger, `scripts/distillation/`). The
proprioception's DR (`domain_randomization.proprio`) is the same either way; the sheet's is
`keypoints` or `pointcloud`, with its own, independent lag.

## Training

Launchers are in `experiments/` (scheduler-agnostic; see `experiments/README.md`). Array index k =
seed k + 1.

### On PhysX (rigid plates)

```bash
RUN_TAG=base WANDB=1 experiments/train.sh --task Isaacsimenvs-RigidCloth-Direct-v0 \
    --experiment rigid_cloth_right_wuji \
    --checkpoint pretrained_policy/flexiv_right_wuji/nov8_dr_friction_0p5.pth
```

- The recipe supplies env count, epochs and minibatch; DR comes from its `dr:` blocks
  (`config/README.md`). Override anything with hydra arguments.
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

The metric is held folds within 1.5 cm (held 10 steps), 2 eval seeds per cell. Every fold policy goes
through `experiments/eval.sh`: keypoint policies and point-cloud students (distilled or RL-trained; the
checkpoint says which, `eval.py`). Three cells per checkpoint by default, plus one opt-in:

| cell | world | episodes |
|---|---|---|
| `rigidbody_flat:rand` | PhysX, the policy's OWN training recipe (physics, DR, start height, accept_lookalike), the 7 shapes, flat starts, the half fold (`goal_task` `half_fold`) | 448 |
| `rigidbody_folded:rand` (opt-in) | the same from a random reachable folded state with one more fold as the goal (`goal_task` `any_fold`, below) | 448 |
| `vbd:pinned` / `vbd:rand` | the VBD cloth with the VBD task's own physics for every policy (ns6: `num_substeps 6`, cloth `substeps 2`) | 32 |

Rigid body is rand only. A goal point-cloud student has no VBD cell (no goal cloud there).
The key metrics are VBD pinned and rand. The recipe is found next to the checkpoint (`run.json`, written
by `train.py` since 2026-11-22; `args.json` of `distill_pc`); an older checkpoint names it. One checkpoint,
a training array (`RUN_TAG` + `TRAIN_JOB`), or a list file, one checkpoint per line:

```
<tag> <seed> <recipe | auto> <checkpoint .pth>
<tag> <seed> <recipe | auto> <run name> <best | last | ep:N>      # resolved when the job runs
```

```bash
LIST=my_list.txt INDEX=<k> experiments/eval.sh        # k = 0 .. 3 x lines - 1
CKPT=<pth> TAG=<tag> RECIPE=experiments/nov15_dr_delay/flat7RW_fricSysid INDEX=<0..2> experiments/eval.sh
CKPT=<pth> TAG=<tag> CELLS=rigidbody_folded:rand INDEX=0 experiments/eval.sh
```

- Index = line x cells + cell; `CELLS` (default `rigidbody_flat:rand vbd:pinned vbd:rand`).
- The sheet's spawn height on VBD: `START_HEIGHT` (default 0.20); rigid uses the recipe's.
- Output: `docs/results/px_matrix/<tag>_s<seed>__on_<target>__<start>__1p5cm_s<es>.json`;
  `isaacsimenvs/scripts/evaluation/matrix_summary.py <tag>` tabulates them (one row per tag).
- `EVAL_SEEDS="3 4 5 6 7 8"` re-tests a picked checkpoint on fresh seeds.

## Goal tasks (rigid pool)

`fold_pool.goal_task` gives each env a start state and, as its goal, that state plus one more fold
(`mdp/goal_tasks.py`, states from `mdp/geometry/fold_states.py`), so a point-cloud student can be told
the goal as a cloud of the folded sheet (`obs.pointcloud.goal` `goal_fixed` / `goal_any`, `perception/pointcloud.py`):

- `mode`: `off` (the stage-1 task), `half_fold` (the stage-1 task with the goal tables built),
  `any_fold` (a random reachable folded start, flat included, plus one more fold; the observed
  keypoints, the target, the frame, the error and the goal pose follow the task);
- `accept_lookalike`: a folded state that looks like the goal from above also counts;
- `sector_deg` / `sector_centre_deg` / `sector_margin_deg`: the designated fold's travel direction
  drawn in a sector instead of at any yaw.

Recipes in `experiments/nov17_goal_pointcloud/` (`rigidcloth_pc_half_fold`, `rigidcloth_pc_any_fold`,
the hard-mode teacher `rigidcloth_any_fold_teacher`). `any_fold` is also the `rigidbody_folded` cell of
`experiments/eval.sh`.
