# task_curriculum

Cloth folding with a Flexiv Rizon 4s + right Wuji Hand 2: pretrain on rigid objects (Play), train
the fold on a pool of hinged-plate "rigid cloths" in PhysX, evaluate and fine-tune on a deformable
VBD cloth in Newton, and distil the state policy into a point-cloud student for the real robot.

## Setup

Two venvs, one source tree:

```bash
docs/installation/install_isaacsim.sh      # .venv_isaacsim  (Isaac Sim 5.1 / Isaac Lab 2.3, PhysX: Play, rigid cloth)
docs/installation/install_isaaclab3.sh     # .venv_isaaclab3 (Isaac Lab 3 / Newton: VBD cloth, kit-less)
isaacsimenvs/scripts/robots/fetch_assets.sh   # Rizon 4s + Wuji Hand 2 vendor meshes into assets/flexiv_wuji/
```

Details in [`docs/installation/`](docs/installation/README.md). Newton reads the robot as
a USD baked by Kit, once per URDF and self-collision value: `SC=false experiments/bake_robot.sh`.
Policies are not in git: the Play prior goes in `pretrained_policy/flexiv_right_wuji/nov8_dr_friction_0p5.pth`,
the network config in `pretrained_policy/config.yaml`.

## Tasks

| Task id (`Isaacsimenvs-<name>-Direct-v0`) | Simulator | What |
| --- | --- | --- |
| `PlayFlexivWujiRightHumanLike` | PhysX | rigid-object goal reaching, the prior |
| `RigidCloth` | PhysX | fold a pool of 7 hinged-plate shapes (10-20 cm) |
| `Cloth` | Newton | fold a VBD particle cloth |

One robot, `robots.FLEXIV_WUJI_RIGHT`: the Flexiv Rizon 4s + right Wuji Hand 2 with Wuji's sim2real
hand drive (HAND2_DRIVE). The cloth tasks use the real robot's joint limits and table geometry; Play
keeps the human-like limits its priors were trained with. `env.hand_sysid=true` (the `sysid` DR
block) swaps in the real hand's identified joint dynamics. See
[`isaacsimenvs/tasks/cloth/README.md`](isaacsimenvs/tasks/cloth/README.md).

## Running

```bash
# Fold on the rigid pool (PhysX), from the Play prior: the base recipe, or one with DR blocks
experiments/train.sh --task Isaacsimenvs-RigidCloth-Direct-v0 --experiment rigid_cloth_right_wuji \
    --checkpoint pretrained_policy/flexiv_right_wuji/nov8_dr_friction_0p5.pth
experiments/train.sh --task Isaacsimenvs-RigidCloth-Direct-v0 \
    --experiment experiments/nov11_right_wuji_dr_ablation/flat7RW_allDR.yaml --checkpoint <prior.pth>

# Evaluate a checkpoint on the rigid pool and the VBD cloth (held folds, 1.5 / 4 cm)
CKPT=<pth> TAG=<name> INDEX=<k> experiments/eval.sh

# Distil into a point-cloud student (DAgger), then evaluate it
TEACHER=<pth> NAME=<name> VIS=cams GOAL=g1 experiments/distill.sh
TEACHER=<pth> MODE=evalall STUDENT=<student.pth> TAG=<name> experiments/distill.sh

# Watch a policy live in a browser
experiments/viewer.sh --task Isaacsimenvs-RigidCloth-Direct-v0
```

The launchers are scheduler-agnostic: on a cluster, submit them as the batch command (array index =
seed - 1); keep the cluster's own settings in a git-ignored `cluster/` ([`experiments/README.md`](experiments/README.md)).
Runs write to `outputs/<date>/<time>/<run name>/` (checkpoints in `nn/`).

## Configuration

A run's config is the task's Python defaults (its config class), a run recipe and the command line,
in that order. A recipe names a base recipe and DR blocks:

```yaml
base: rigid_cloth_right_wuji
dr: [sysid, friction, contact_shell, proprio_noise, object_points]
```

See [`isaacsimenvs/tasks/cloth/config/README.md`](isaacsimenvs/tasks/cloth/config/README.md).

## Layout

```text
isaacsimenvs/
  tasks/cloth/      the fold task: ClothEnv owning a backend (backends/{rigid,vbd}) and mdp/ (task, obs, actions, DR), perception/
  tasks/play/       the Play task (the prior)
  robots/           the robot spec and the URDF -> USD pipeline
  scripts/
    training/       train.py and its rl_games / wandb glue
    evaluation/     eval.py, protocol, checkpoint lookup, summaries
    distillation/   point-cloud student: DAgger (distill_pc.py) and RL (pc_rl.py)
    visualization/  live viewer, interactive captures, design diagrams
    robots/         asset fetch, self-collision pairs, Wuji hand sysid replay
experiments/        launchers, and the recipes of past studies (<date>_<study>/)
rl_games/           vendored rl_games (PPO, SAPG)
docs/               installation, phase notes, reports
```

Tests are Kit-free and beside the code: `experiments/newton_py -m pytest`.
