# Run configurations

Python configclasses beside each environment hold every parameter with its default and its
documentation. The YAML files here only change those defaults for a task, a robot or a run.

| Directory | Contents |
| --- | --- |
| `task/` | One overlay per task id: only what differs from the configclass defaults |
| `robot/` | Robot setups (`env.robot_setup`): robot spec, URDF, self-collision, base position |
| `runs/` | Run recipes; `rigid_cloth_right_wuji.yaml` is the base of every right-Wuji PhysX run |
| `dr/` | DR blocks: named env overlays a recipe switches on with `dr: [...]` |
| `rigid_cloth_designs/` | The fold-pool shapes (plates, hinges, seams, folds) with PNG diagrams |
| `train/` | Learner settings (PPO, SAPG) |

## Order

A run's config is composed in this order; each layer overrides the ones before it:

1. configclass defaults (Python)
2. `task/<Task>.yaml`, chosen by the gym task id
3. `robot/<setup>.yaml`, chosen by `env.robot_setup` (command line, else the recipe, else an alias task id)
4. the run recipe (`--experiment`): its `base:` recipe, then its DR blocks in order, then its own `env:` / `agent:`
5. command-line overrides (`env.scene.num_envs=512`)

`loader.py` applies the layers; `recipes.py` composes a recipe (plain YAML, no simulator needed).

## Recipes

```yaml
# experiments/nov11_right_wuji_dr_ablation/flat7RW_allDR.yaml
base: rigid_cloth_right_wuji
dr: [sysid, friction, contact_shell, proprio_noise, object_points, action_delay, pd_gains, action_noise, hinge_mass]
env:
  cloth:
    soft_contact_mu: 0.265        # what no block covers
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
