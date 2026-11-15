# task_curriculum

RL for cloth folding with a Flexiv Rizon 4s + Wuji Hand 2 (right hand), trained in Isaac Lab and
deployed on the real robot. The package is `isaacsimenvs`; training is one `train.py` on a vendored
`rl_games` fork (PPO / SAPG), with the config layering of
[play2perfect](https://github.com/kushal2000/play2perfect).

## Three env families

| Family | Task ids | Backend | What it is |
|---|---|---|---|
| **Play** | `Isaacsimenvs-Play-Direct-v0` (Kuka + Sharpa), `Isaacsimenvs-PlayFlexivWujiRightHumanLike-Direct-v0` (the real setup: table, goals off the table) | PhysX | SimToolReal's rigid-object goal-reaching task: the tool-use prior every fold policy starts from. |
| **Rigid fold** | `Isaacsimenvs-RigidClothPhysx-Direct-v0` (one hinged-plate chain), `Isaacsimenvs-FoldPoolPhysx-Direct-v0` (a shape drawn per episode from the fold-shape pool) | PhysX | The cloth approximated by hinged rigid plates; where the fold policies are trained. |
| **VBD fold** | `Isaacsimenvs-Cloth-Direct-v0` | Newton (MJWarp + VBD) | A real deformable sheet; where rigid-trained policies are evaluated, zero-shot or fine-tuned. |

Every cloth task also exists per robot, `<Task><Robot>`, with `<Robot>` one of `FlexivWujiRightHumanLike`
(human-like joint limits, Wuji's HAND2 drive), `FlexivWujiRightReal` (the real hand: sysid dynamics and
the real joint limits) and `FlexivWujiRightHLRL` (the human-like spec on the real joint limits), e.g.
`Isaacsimenvs-FoldPoolPhysxFlexivWujiRightReal-Direct-v0`. The robot is a YAML overlay
(`isaacsimenvs/cfg/robot/`) on the task's own YAML; the specs are in `isaacsimenvs/tasks/play/robots.py`.

`Isaacsimenvs-JointReachFlexivWujiRightHumanLike-Direct-v0` is a deploy test, not a learning task:
reach commanded joint configurations through the Play action pipeline.

The rigid and VBD fold envs share one task definition (`tasks/cloth/cloth_task.py`: fold targets,
reward, success, the observation), so a policy moves between them unchanged. Tests diff their YAMLs:
`RigidClothPhysx.yaml` is `Cloth.yaml` with the manipuland and solver swapped, and `FoldPoolPhysx.yaml`
is the chain YAML plus `fold_pool:`.

## Layout

```
isaacsimenvs/
  train.py              the one training entry point
  tasks/play/           PlayEnv, robots.py (robot specs), play utils
  tasks/cloth/          shared fold task; physx_envs.py (rigid fold), cloth_env.py + newton_base_env.py (VBD)
  tasks/joint_reach/    the deploy test task
  cfg/task|robot|experiment|train|fabric/
  eval/                 episodes.py (evals), live_viser.py (live viewer), player.py (loads a checkpoint)
  newton/               Newton backend patches and the URDF->USD cache
  curriculum/           difficulty curricula (the cloth unfold task's start-angle curriculum)
deployment/             the real robot: simtoolreal, rigid2deform, joint_reach, calibration, shared nodes/utils
assets/                 urdf/ (robots: kuka_sharpa_description/, flexiv_wuji/; tables), fabric textures
scripts/slurm/          submit.sh + train/eval/viser job scripts
scripts/flexiv_wuji/    fetch the robot assets, compose the arm + hand URDFs
scripts/newton_py       launch Python in .venv/isaaclab3 (isaacsim_py: the Engaging container)
rl_games/               vendored rl_games fork
third_party/            vendored upstream source at pinned commits: IsaacLab, aioflexiv, aiowuji
docs/installation/      venv install scripts (+ engaging/ for the Engaging cluster) and instructions
docs/reports/           write-ups
debug_outputs/          (gitignored) EVERY run artifact: train_logs/, videos/, deployment/
.cache/                 (gitignored) tool caches: usd/ (URDF->USD for Newton), pytest/, ruff/
.venv/                  (gitignored) isaacsim/ and isaaclab3/
```

## Setup

Two venvs on one source tree (both editable-install it):

| venv | stack | runs |
|---|---|---|
| `.venv/isaacsim` | Python 3.11, Isaac Sim 5.1 / Isaac Lab 2.3 | Play, rigid fold, joint_reach (PhysX) |
| `.venv/isaaclab3` | Python 3.12, Isaac Lab 3.0 + Newton (kit-less) | VBD fold |

Install scripts and instructions are in [`docs/installation/`](docs/installation/README.md):
`docs/installation/install_isaacsim.sh`, `docs/installation/install_isaaclab3.sh`, then bake the Newton
USD cache.

The Play priors are in `pretrained_policy/` (gitignored, or set `$PRETRAINED_POLICY_DIR`), one folder per
robot, each with `model.pth` and `config.yaml`: `kuka_sharpa/` (the generic SimToolReal prior; on portal
symlinks into `/share/portal/kk837/simtoolreal/pretrained_policy`) and `flexiv_wuji/` (the right-Wuji
human-like Play prior, best checkpoint of run play_scratch_flexiv_right_dr_cont_41213). Flexiv + right Wuji
runs start from `flexiv_wuji/`, Kuka + Sharpa runs from `kuka_sharpa/`.

## Train, evaluate, watch

Config layers, each overriding the last: configclass defaults, `cfg/task/<Task>.yaml`, the robot
overlay `cfg/robot/<robot>.yaml`, an experiment recipe `cfg/experiment/<name>.yaml` (`--experiment`;
recipes chain through `base:` and name the task they were written for), then Hydra CLI overrides.

```bash
# on Slurm: each run gets one folder, debug_outputs/train_logs/<month>/<date>_<name>_<job id>/,
# holding its checkpoints, config, wandb/ and Slurm log; the venv follows the task id
scripts/slurm/submit.sh scripts/slurm/train.sh \
    --task Isaacsimenvs-FoldPoolPhysxFlexivWujiRightHLRL-Direct-v0 --experiment flat7RW_obsActDR \
    --checkpoint <play run>/nn/<name>.pth --checkpoint_load_mode weights --single_variant \
    agent.params.config.name=0_my_run
scripts/slurm/submit.sh --array=0-2 scripts/slurm/train.sh ...          # seeds 1-3
scripts/slurm/submit.sh --partition=gpu_a100 --gres=gpu:rtxpro6000:1 ... # another cluster

scripts/slurm/submit.sh scripts/slurm/eval.sh --experiment_dir experiments/<day> --task <id> --checkpoint <pth>
                                          # -> experiments/<day>/results/ (JSON + Slurm log)
scripts/slurm/submit.sh scripts/slurm/viser.sh --task <id>                     # live viewer; the log prints the tunnel
```

Run names must start with an integer index (`0_...`): rl_games parses it. Kit exits 0 even when the
script inside it raised, so check a job's log, not its exit code.

## Outputs

Everything a run produces is under `debug_outputs/` (`isaacsimenvs/utils/paths.py`).

- `debug_outputs/train_logs/<month>/<date>_<run name>_<job id>/` is one training run, all of it:
  `nn/` (checkpoints: `<name>.pth` is the best by mean reward, `last_<name>_ep_*.pth` periodic),
  `summaries/`, `.hydra/` (the resolved config), `wandb/`, `interactive_viewer/` and
  `slurm_<job id>.out`. Outside Slurm the time stands in for the job id. Older runs keep the
  `<date>/<time>/0_<name>/` layout; `deprecated/` holds runs of removed robots and tasks.
- `debug_outputs/train_logs/.slurm/`: where sbatch starts each log, before the job moves it into its
  folder (a symlink stays).
- `debug_outputs/videos/`: rendered videos. `debug_outputs/deployment/`: real-robot recordings and logs.
- Eval results go to the experiment's folder, `experiments/<day>/results/`.

## Deploy

`deployment/` runs a trained policy on the robot: `deployment/simtoolreal/` (Play), `deployment/rigid2deform/`
(folding) and `deployment/joint_reach/`, as ZMQ nodes (`deployment/nodes/`). The drivers aioflexiv and aiowuji
are vendored in `third_party/` (below). See `deployment/simtoolreal/README.md`.

## Third-party code

`third_party/` is committed: plain copies of Isaac Lab (the tree `.venv/isaaclab3` installs) and the
robot drivers aioflexiv and aiowuji, each at the commit in `third_party/VENDORED.md`, so a clone is
complete with nothing to fetch. Don't edit them in place; to move a pin, change it in
`scripts/vendor_third_party.sh`, rerun it and commit the diff.

## Tests

Each package keeps its tests beside it (`isaacsimenvs/**/tests/`); `isaacsimenvs/conftest.py` is
the shared setup. Kit-free in both venvs.

```bash
.venv/isaaclab3/bin/python -m pytest     # everything
.venv/isaacsim/bin/python -m pytest      # the Isaac-Lab-backed tests skip here instead of booting Kit
```
