# Experiments

Launchers that run on any GPU machine, and the recipes of past studies.

| Script | Runs |
| --- | --- |
| `train.sh` | `isaacsimenvs/scripts/training/train.py` (any task; seed, run name, SAPG by default) |
| `eval.sh` | fold evaluation: checkpoints x targets (PhysX pool, VBD cloth) x starts x cutoffs |
| `distill.sh` | point-cloud student distillation and its evaluations |
| `viewer.sh` | the live policy viewer (`live_viser.py`) |
| `bake_robot.sh` | a robot's URDF -> USD cache for the kit-less Newton runs |
| `newton_py` | the Newton venv's python (with the C++ runtime it needs) |

Each script picks the interpreter from `--task` (`Isaacsimenvs-Cloth*` -> Newton, `.venv_isaaclab3`;
every other id -> PhysX, `.venv_isaacsim`), logs where it ran (host, commit, uncommitted changes,
GPU) and its peak GPU / host memory, and on PhysX retries a run whose simulator fails to start.
Their headers list the options.

```bash
experiments/train.sh --task Isaacsimenvs-RigidCloth-Direct-v0 --experiment rigid_cloth_right_wuji \
    --checkpoint <prior.pth>
SMOKE=1 experiments/train.sh --task Isaacsimenvs-RigidCloth-Direct-v0 \
    --experiment experiments/nov15_cloth_tracker/flat7RW_trkCr.yaml --checkpoint <prior.pth>
CKPT=<pth> TAG=mypolicy INDEX=2 experiments/eval.sh
```

## On a cluster

The scripts know nothing about a particular cluster. A batch system only has to run one of them on a
GPU node from the repository root; arrays work through the index:

- `INDEX` (else `SLURM_ARRAY_TASK_ID`, else 0) picks the seed (`train.sh`: seed = index + 1) or the
  evaluation cell (`eval.sh`);
- `JOB_ID` (else `SLURM_ARRAY_JOB_ID` / `SLURM_JOB_ID`) goes into run names.

For example, three seeds with Slurm:

```bash
sbatch --array=0-2 --gres=gpu:1 --output=slurm_logs/px_train_%A_%a.out --wrap \
    "experiments/train.sh --task Isaacsimenvs-RigidCloth-Direct-v0 --experiment rigid_cloth_right_wuji --checkpoint <prior.pth>"
```

Keep a cluster's own settings (partition, GPU type, nodes to avoid, chained submissions) in a local
`cluster/` directory at the repository root; it is git-ignored.

## Studies

`<date>_<study>/` holds the run recipes of one study (base recipe + DR blocks, see
`isaacsimenvs/tasks/cloth/config/README.md`); pass one with `--experiment <path>`.
