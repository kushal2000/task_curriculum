# Cloth task configuration

Everything the cloth task is configured with is in this folder: what the fields are (Python), and
what they are set to (YAML).

| Path | Contents |
| --- | --- |
| `schema/` | The configclasses: every field, its type, what it means and why it has its value, the checks. `env.py` (`ClothEnvCfg`, both tasks), `cloth.py` (the sheet), `rigid_cloth.py` (`RigidClothEnvCfg`: plates, pool, PhysX contacts), `vbd.py` + `vbd_cloth.py` (`VbdClothEnvCfg`: the VBD sheet, MJWarp). A task value has no default here (`MISSING`) |
| `task.yaml` | The values: `common` (both tasks), `rigid`, `vbd`. Every value either task starts from is here |
| `runs/` | Run recipes: `rigid_cloth_right_wuji.yaml`, the training setup of every PhysX run |
| `dr/` | DR blocks: named env overlays a recipe switches on with `dr: [...]` |
| `designs/` | The fold-pool shapes (plates, hinges, seams, folds) with PNG diagrams |
| `ClothSAPG.yaml` | Learner settings (SAPG); PPO is Play's (`tasks/play/config/`) |
| `experiments/<date>_<study>/` | The recipes of one study, beside its launchers |

## Order

A run's config is composed in this order; each layer overrides the ones before it:

1. `task.yaml` on the configclass, chosen by the gym task id (`config/__init__.py`): what the task is,
   for training, evaluation and viewers alike. Loading fails if a field has no value.
2. the run recipe (`--experiment`): its `base:` recipe, then its DR blocks in order, then its own `env:` / `agent:`
3. command-line overrides (`env.scene.num_envs=512`)

`tasks/loader.py` applies the layers; `tasks/recipes.py` composes a recipe (plain YAML, no simulator
needed).

## Recipes

```yaml
# a keypoint policy with every DR group, on the real setup's centre values
base: rigid_cloth_right_wuji
dr: [proprio, keypoints, action, robot, cloth]
env:
  hand_sysid: true                  # centre values: configuration, not DR
  assets: {finger_tip_friction: 0.5, table_friction: 1.0}
  rigid_cloth: {slat_friction: 1.0, contact_margin: 0.0015}
  physx_chain: {friction_model: cloth_min}
  reset:
    table_reset_z_range: 0.015      # not DR: the episode's start distribution
  domain_randomization:
    action:
      noise_std: 0.02               # a recipe's own values override its blocks'
```

```yaml
# a point-cloud policy (DAgger student or point-cloud RL): the same proprioception, the cloud instead
# of the keypoints
base: rigid_cloth_right_wuji
dr: [proprio, pointcloud, action]
env:
  obs:
    object: pointcloud
    pointcloud:
      goal: label
      ref_step: 12
      fk_points: 128
```

`--experiment` takes a name in `runs/` or a path (`experiments/<study>/<name>.yaml`). A `base:` is
looked up next to the recipe first, then in `runs/`. Long-lived recipes live in `runs/`; the
recipes of one study live beside it in `experiments/<date>_<study>/`.

## Centre values and DR

A centre value -- what the robot, the sheet or a contact *is* -- is configuration: `hand_sysid`,
`assets.*_friction`, `rigid_cloth.*`, `physx_chain.friction_model`; `task.yaml` has the task's, and a
recipe changes them in its `env:` (`experiments/nov10_right_wuji_obs_dr/flat7FlexivWujiRHL.yaml`
has the real setup's, with where each comes from). DR is a range around those centres, under
`domain_randomization`, in six groups (`schema/env.py`); every range is neutral by default, and
`task.yaml` sets one: the rigid sheet's contact-shell jitter (`cloth.contact_margin_offset`, +-3 mm).

## DR blocks

One block per group; each turns its whole group on with the standard values. The policy sees the
sheet either as keypoints or as a point cloud (`obs.object`), so `keypoints` and `pointcloud` are
parallel: a recipe uses one of them, never both. The sheet's lag and the proprioception's are
independent in both.

| Block | Group | What it randomizes |
| --- | --- | --- |
| `proprio` | `domain_randomization.proprio` | every field but the sheet's 0-3 steps late, joint position / velocity noise |
| `keypoints` | `domain_randomization.keypoints` | the sheet pose the keypoints come from: 0-5 steps late, noise 1 cm / 5 deg, per-episode bias 5 / 5 / 2 mm, 3 deg |
| `pointcloud` | `domain_randomization.pointcloud` | the cloud: 0-6 steps late, camera position, per-camera calibration, depth noise, dropped points |
| `action` | `domain_randomization.action` | action delay 0-3 steps, action noise 0.05 |
| `robot` | `domain_randomization.robot` | PD gains x0.7-1.3; hand joint friction x0.7-1.3 and torque limit x0.85-1.15 (needs `hand_sysid`) |
| `cloth` | `domain_randomization.cloth` | plate / robot / table friction (needs `cloth_min`); hinge damping, stiffness, plate mass (PhysX only) |

Each file's header says what the values are and where they come from. A recipe changes any value
of a block in its own `env:` (it is applied after the blocks), or sets a group's fields without the
block. A delay `delay_max: N` is a lag of 0 .. N policy steps (1/60 s), drawn once per episode.
