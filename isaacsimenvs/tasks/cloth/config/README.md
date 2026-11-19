# Cloth task configuration (YAML)

Python configclasses beside the code hold every parameter with its default and its documentation;
the defaults are the task (`RigidClothEnvCfg`, `VbdClothEnvCfg`; Play: `PlayEnvCfg`), robot included.
The YAML files only change them for a run.

| Path | Contents |
| --- | --- |
| `runs/` | Run recipes: `rigid_cloth_right_wuji.yaml` is the base of every PhysX run, `rigid_cloth_right_wuji_real.yaml` the same on the real setup's centre values (the base for DR runs) |
| `dr/` | DR blocks: named env overlays a recipe switches on with `dr: [...]` |
| `designs/` | The fold-pool shapes (plates, hinges, seams, folds) with PNG diagrams |
| `ClothSAPG.yaml` | Learner settings (SAPG); PPO is Play's (`tasks/play/config/`) |
| `experiments/<date>_<study>/` | The recipes of one study, beside its launchers |

## Order

A run's config is composed in this order; each layer overrides the ones before it:

1. configclass defaults (Python), chosen by the gym task id: what the task is, for training, evaluation
   and viewers alike
2. the run recipe (`--experiment`): its `base:` recipe, then its DR blocks in order, then its own `env:` / `agent:`
3. command-line overrides (`env.scene.num_envs=512`)

`tasks/loader.py` applies the layers; `tasks/recipes.py` composes a recipe (plain YAML, no simulator
needed).

## Recipes

```yaml
# an all-DR run on the real setup
base: rigid_cloth_right_wuji_real
dr: [observation, action, robot, cloth]
env:
  reset:
    table_reset_z_range: 0.015      # not DR: the episode's start distribution
  domain_randomization:
    action:
      noise_std: 0.02               # a recipe's own values override its blocks'
```

`--experiment` takes a name in `runs/` or a path (`experiments/<study>/<name>.yaml`). A `base:` is
looked up next to the recipe first, then in `runs/`. Long-lived recipes live in `runs/`; the
recipes of one study live beside it in `experiments/<date>_<study>/`.

## Centre values and DR

A centre value -- what the robot, the sheet or a contact *is* -- is configuration: `hand_sysid`,
`assets.*_friction`, `rigid_cloth.*`, `physx_chain.friction_model`. `rigid_cloth_right_wuji_real` sets
the real setup's. DR is a range around those centres, under `domain_randomization`, in four groups
(`env_cfg.py`); every range is neutral by default except the rigid sheet's contact-shell jitter
(`cloth.contact_margin_offset`, +-3 mm), which belongs to the shell's configuration.

## DR blocks

One block per group; each turns its whole group on with the standard values.

| Block | Group | What it randomizes |
| --- | --- | --- |
| `observation` | `domain_randomization.observation` | observation delay 0-3 steps, joint position / velocity noise; the sheet pose: delay 0-5 steps, noise 1 cm / 5 deg, per-episode bias 5 / 5 / 2 mm, 3 deg |
| `action` | `domain_randomization.action` | action delay 0-3 steps, action noise 0.05 |
| `robot` | `domain_randomization.robot` | PD gains x0.7-1.3; hand joint friction x0.7-1.3 and torque limit x0.85-1.15 (needs `hand_sysid`) |
| `cloth` | `domain_randomization.cloth` | plate / robot / table friction (needs `cloth_min`); hinge damping, stiffness, plate mass (PhysX only) |

Each file's header says what the values are and where they come from. A recipe changes any value
of a block in its own `env:` (it is applied after the blocks), or sets a group's fields without the
block.
