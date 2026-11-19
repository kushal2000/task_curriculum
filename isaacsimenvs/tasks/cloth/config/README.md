# Cloth task configuration (YAML)

Python configclasses beside the code hold every parameter with its default and its documentation;
the defaults are the task (`RigidClothEnvCfg`, `VbdClothEnvCfg`; Play: `PlayEnvCfg`). The YAML files
only change them for a robot or a run.

| Path | Contents |
| --- | --- |
| `runs/` | Run recipes; `rigid_cloth_right_wuji.yaml` is the base of every right-Wuji PhysX run |
| `dr/` | DR blocks: named env overlays a recipe switches on with `dr: [...]` |
| `designs/` | The fold-pool shapes (plates, hinges, seams, folds) with PNG diagrams |
| `ClothSAPG.yaml` | Learner settings (SAPG); PPO is Play's (`tasks/play/config/`) |
| `isaacsimenvs/robots/setups/` | Robot setups (`env.robot_setup`), shared with Play: robot spec, URDF, self-collision, base position |
| `experiments/<date>_<study>/` | The recipes of one study, beside its launchers |

## Order

A run's config is composed in this order; each layer overrides the ones before it:

1. configclass defaults (Python), chosen by the gym task id: what the task is, for training, evaluation
   and viewers alike
2. `robots/setups/<setup>.yaml`, chosen by `env.robot_setup` (command line, else the recipe, else an
   alias task id)
3. the run recipe (`--experiment`): its `base:` recipe, then its DR blocks in order, then its own `env:` / `agent:`
4. command-line overrides (`env.scene.num_envs=512`)

`tasks/loader.py` applies the layers; `tasks/recipes.py` composes a recipe (plain YAML, no simulator
needed).

## Recipes

```yaml
# experiments/nov11_right_wuji_dr_ablation/flat7RW_allDR.yaml
base: rigid_cloth_right_wuji
dr: [sysid, friction, contact_shell, proprio_noise, object_points, action_delay, pd_gains, action_noise, hinge_mass]
env:
  reset:
    table_reset_z_range: 0.015    # what no block covers
```

`--experiment` takes a name in `runs/` or a path (`experiments/<study>/<name>.yaml`). A `base:` is
looked up next to the recipe first, then in `runs/`. Long-lived recipes live in `runs/`; the
recipes of one study live beside it in `experiments/<date>_<study>/`.

## DR blocks

| Block | What it changes |
| --- | --- |
| `sysid` | identified hand (`flexiv_wuji_right_real`) + hand friction / torque-limit DR |
| `friction` | 2026-11-10 contact frictions (cloth_min) + their DR |
| `contact_shell` | 5 mm felt: contact shell 1.5 +- 1 mm |
| `proprio_noise` | observation delay 0-3 steps per episode, joint position / velocity noise |
| `object_points` | camera model on the object points (realistic lags, noise, biases) |
| `tracker` | cloth keypoints through the stereo + CoTracker model, self- and hand occlusion |
| `tracker_reacquire` | on top of `tracker`: re-acquisition failure |
| `action_delay` | action delay 0-3 steps |
| `action_noise` | action noise 0.05 |
| `pd_gains` | PD gains x0.7-1.3 |
| `hinge_mass` | hinge damping / stiffness, plate mass |
| `dr_delay` | flat7DRABC's A+B layers (delays, noise, PD gains x0.8-1.2); overlaps the delay / noise blocks |

Each file's header says what the values are and where they come from.
