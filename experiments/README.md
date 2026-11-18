# Experiments

Group experiments by date, then by experiment name:

```text
experiments/
  _common.sh
  submit.sh
  train.sh
  eval.sh
  nov15/
    rigid_cloth_init/
    rigid2deform/
    fabric_eval/
    ...
  nov16/
    cloth_from_scratch/
    yibo_our_dr/
  nov17/
    cloth_resets/
    contact_offset_20mm/
    contact20mm_rest5mm/
    yibo_prior_finetune/
    yibo_offsets/
    pressing_audit/
```

New work uses the same layout, for example `experiments/nov18/<experiment_name>/`.
The date belongs to the enclosing directory, rather than being repeated in every
experiment name. Shared scheduler helpers stay at this root.

Run launchers from any working directory; each resolves the repository root:

```bash
experiments/nov17/contact_offset_20mm/submit_prior.sh
```

Recipes inherit other YAML files using paths relative to their own directory.
The shared submission wrapper remains available for custom scheduler options:

```bash
experiments/submit.sh --job-name=cloth \
  experiments/train.sh --task Isaacsimenvs-RigidBodyCloth-Direct-v0 \
  --experiment experiments/nov17/contact_offset_20mm/prior.yaml
```

Training outputs, checkpoints, W&B data and periodic interactive HTML captures
live under `debug_outputs/train_logs/`. Experiment-local `results/` directories
retain evaluation artifacts; their local ignore rules move with the experiment.
The date-folder reorganization preserves those artifacts and updates archived
Slurm log symlinks. Historical command strings inside saved run outputs remain
records of the original launches.
