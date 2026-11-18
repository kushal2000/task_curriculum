# Cloth manipulation

Start with [cloth_env.py](cloth_env.py). `ClothEnv` is the single Isaac Lab `DirectRLEnv` for
both articulated plates and VBD cloth. The policy receives the same observation
layout and its joint actions have the same meaning with either approximation.

```text
policy actions
    → JointController.process
    → JointController.commands
    → backend.apply_joint_commands
    → Isaac Lab advances physics
    → backend.read_state
    → ClothTask.evaluate (goals, reward, termination)
    → reset finished episodes
    → ObservationBuilder.build
    → next policy observation
```

## Ownership

The public package exports `ClothEnv` and `ClothCfg` lazily. The root has two
implementation files; task behavior and physics live below them:

```text
cloth/
  cloth_env.py           # DirectRLEnv hooks
  cloth_cfg.py           # shared environment configuration
  mdp/                  # actions, observations, goals/rewards/dones, state, geometry
  backends/             # robot/assets and rigid/VBD scene adapters
    vbd/newton_support/ # simulator compatibility and cached asset reads
  tests/
```

The runners and viewers live in `isaacsimenvs/scripts/` and support cloth and
Play where applicable. Their YAML presets and run recipes are described in
[the configuration guide](../../scripts/configs/README.md).


| Component | Owns | Does not depend on |
| --- | --- | --- |
| [cloth_env.py](cloth_env.py) | Isaac hooks, component construction, public viewer/evaluation queries | Backend physics details |
| [mdp/actions.py](mdp/actions.py) | Joint mapping, target smoothing, action delays, passive springs, robot reset draws | Cloth representation |
| [mdp/observations.py](mdp/observations.py) | Observation layout, noise, observation history | Simulator APIs |
| [mdp/task.py](mdp/task.py) | Fold/unfold goals, reward, termination, release/persistence, episode history | Simulator APIs |
| [mdp/state.py](mdp/state.py) | Common state, joint commands, reset data, task transition | Isaac Lab |
| [backends/robot.py](backends/robot.py) | Robot constants and the policy's joint/fingertip order | Isaac Lab |
| [mdp/geometry/](mdp/geometry/) | Surface grids, fold geometry, quaternion math | Simulator APIs |
| [backends/base.py](backends/base.py) | Shared robot/table reads and writes, coordinate conversion | Task rewards and policy history |
| [backends/rigid/](backends/rigid/) | Articulated plates, size designs, PhysX materials and scene | Policy observations |
| [backends/vbd/](backends/vbd/) | Mesh cloth, Newton coupling, friction, contacts, plasticity | Policy observations |
| [backends/assets.py](backends/assets.py) | Shared robot/table USD preparation | Task logic |

`SimState.points_w` means **cloth surface samples**, not necessarily particles.
The rigid adapter computes them from plate transforms; VBD reads nodal positions.
All common state is in world coordinates, with wxyz quaternions and joints in
Isaac Lab order. The observation builder converts to the pretrained policy's
canonical joint order, local coordinates, and xyzw quaternion convention.

## Isaac Lab lifecycle

`DirectRLEnv` owns `step`, physics decimation, scene updates, rendering, and
autoresets. `ClothEnv` implements its hooks:

| Hook | Work |
| --- | --- |
| `_setup_scene` | Adapter builds and registers scene assets |
| `_pre_physics_step` | Controller interprets the policy action once |
| `_apply_action` | Controller computes commands using current joint positions; adapter writes them |
| `_get_dones` | Read state and evaluate the task once; retain reward and done flags |
| `_get_rewards` | Return the retained reward and publish metrics |
| `_reset_idx` | Isaac reset bookkeeping, robot/task reset draws, physical writes, history reset |
| `_get_observations` | Read current state and construct policy/critic observations |

Rewards and termination describe the same pre-reset transition. Observations read
fresh state after resets. `ClothTask.derive` and viewer queries never advance task
history. Resetting a subset clears only those environments' task and delay buffers.
The ordering of random reset draws is preserved across the refactor.

The adapter is constructed before `DirectRLEnv.__init__`, which invokes
`_setup_scene`. After Isaac starts simulation, the adapter resolves body IDs and
surface geometry; the environment then constructs the controller, task, and
observation builder. Newton's temporary construction hooks surround Isaac startup
and are restored even on failure. Final material refreshes happen after startup.

## Configuration and entry points

For Python callers, construct a complete config with the public API, after starting
the appropriate Isaac runtime:

```python
from isaacsimenvs.tasks.cloth import ClothCfg

cfg = ClothCfg.for_backend("vbd")  # or "rigid" in the PhysX runtime
from isaacsimenvs.tasks.cloth import ClothEnv

env = ClothEnv(cfg)
```

`ClothCfg()` alone holds shared settings; `for_backend` adds the physical settings
and simulator defaults. Select the config before importing `ClothEnv`, so VBD can
install its Isaac compatibility setup first.

Both existing Gym IDs use the same environment factory:

| Gym ID | Config | Adapter/runtime |
| --- | --- | --- |
| `Isaacsimenvs-RigidBodyCloth-Direct-v0` | `backends/rigid/config.py:RigidClothEnvCfg` | `rigid`, `.venv/isaacsim` |
| `Isaacsimenvs-Cloth-Direct-v0` | `backends/vbd/config.py:DeformableClothEnvCfg` | `vbd`, `.venv/isaaclab3` |

[cloth_cfg.py](cloth_cfg.py) declares `ClothCfg`, the shared environment configuration.
[mdp/config.py](mdp/config.py) holds its task/policy sections; `ClothTaskCfg` configures
the fold/unfold goal under `task`; `task.mode` selects fold, unfold, or mixed. Backend configs add
their physical parameters and select the adapter through `backend`. The
`rigid_cloth`, `deformable_cloth`, and `newton` sections hold physical settings.
For task overrides use `env.task.mode=unfold` or
`env.task.keypoint_tolerance=0.015`.
Use each config with its matching runtime; this is a construction-time choice.
The factory installs Newton compatibility before importing the environment.

Additional VBD approximations can generally be expressed as material/solver
configuration. Add another adapter only when scene construction, state readout,
or simulator writes actually differ. Adapters do not implement their own step loop.

## Articulated folding designs

The [design catalog](../../scripts/configs/rigid_cloth_designs/README.md) contains
five runnable YAML designs and matching PNG diagrams. Each YAML describes crease
segments in normalized sheet coordinates; the loader derives the rigid panels
and their hinge connections.

| YAML | Plates | Hinges |
| --- | ---: | ---: |
| `single_hinge.yaml` | 2 | 1 |
| `strip_4.yaml` | 4 | 3 |
| `strip_8.yaml` | 8 | 7 |
| `flexible_corners.yaml` | 6 | 5 |
| `hinge_band.yaml` | 4 | 3 |

All designs use the same task: fold the positive-x half onto the negative-x half
about the central y-axis crease. Extra hinges allow bending while attempting that
fold. The supplied designs are symmetric across both sheet axes.

Select a list before constructing the environment, after launching Isaac Sim:

```python
cfg = ClothCfg.for_backend("rigid")
cfg.rigid_cloth.designs = ["single_hinge.yaml", "flexible_corners.yaml", "hinge_band.yaml"]
cfg.scene.num_envs = 64
cfg.task.mode = "fold"
env = ClothEnv(cfg)
```

`ClothEnv` passes this list to `RigidBackend(cfg, design_paths)`. Names resolve
under `scripts/configs/rigid_cloth_designs`; explicit YAML paths also work. Existing
runners accept, for example:

```sh
'env.rigid_cloth.designs=[single_hinge.yaml,strip_4.yaml,flexible_corners.yaml]'
```

The backend cycles through the list across environments. Environments sharing a
design form one articulation group with that design's native body/joint counts.
Length, width and thickness vary independently through the size grid. State
readout restores the original environment order and exposes the same surface
samples and observation layout for every design.

[designs.py](backends/rigid/designs.py) loads each YAML into a `ClothDesign`.
`design.resolve(length, width, thickness)` produces a `SheetSpec` in meters:
convex plate polygons, hinge anchors, axes, limits and the designated fold hinge.
The scene builder extrudes these polygons and computes their mass and inertia.
The loader currently supports rectangular sheets whose crease cuts form a tree,
including one complete center crease at `x = 0`.

All policies observe a common moving-half object representation: center,
orientation and velocity, dimensions `[length/2,width,thickness]`, and four 3D
keypoints in Play's diagonal top/bottom order. `obs.keypoint_scale` defaults to
1.5 and expands local marker offsets; `reward.object_base_size` normalizes the
size observation independently. Current and target markers use consistent object
poses. The single-hinge moving frame is exact; additional hinges and deformable
cloth use a fitted moving-half frame.

Physical rigid fold landmarks remain the exact four moving-half corners attached
to their owning plates. Their rest center is `(length/4,0,0)` regardless of grid
resolution. These landmarks drive physical fold error and reward. The viewer
switches between physical corners and the policy's 3D markers.

Reset sets the designated center hinge angle and leaves other hinges straight.
Passive hinge targets remain flat. A shared sheet rigidity and viscosity are
converted to per-hinge stiffness/damping using crease length and neighboring panel
spacing, so discretization changes do not silently change the material draw.

The static-robot viewer discovers the catalog in its **Design YAML** dropdown.
Select a design and press **Apply geometry and restart**; this reloads its YAML.

```sh
.venv/isaacsim/bin/python isaacsimenvs/scripts/visualization/viser_rigidCloth_viewer.py --design flexible_corners
```

Add `--policy` to evaluate the bundled Flexiv–Wuji Play prior. Policy mode uses the
real `ClothEnv` observation/action path and animates the robot. The setup selector
switches between the checkpoint's human-like hand and trained table placement,
and the current real-limits hand with identified dynamics. See
[policy viewer setup](../../scripts/visualization/rigid_cloth/README.md) for the
training table dimensions, limits, and evaluation controls.

Newton compatibility belongs to [backends/vbd/newton_support/](backends/vbd/newton_support/).
Evaluation overflow reporting lives in `isaacsimenvs/scripts/evaluation/utils.py`.
Prepare the default VBD robot/table cache under Isaac Sim with:

```sh
.venv/isaacsim/bin/python -m isaacsimenvs.scripts.assets.prepare_usd_cache --populate \
  --task Isaacsimenvs-RigidBodyCloth-Direct-v0 \
  env.assets.table_urdf=assets/urdf/table_narrow.urdf
```

Use the same robot/table paths and conversion options as the intended VBD run.

Historical physics measurements are in
[backends/vbd/physics_notes.md](backends/vbd/physics_notes.md). Fabric presets live
in `isaacsimenvs/scripts/configs/deformable_cloth_designs`; print their Hydra overrides with:

```sh
.venv/isaaclab3/bin/python -m isaacsimenvs.tasks.cloth.backends.vbd.fabrics cotton_measured --plastic
```

The old rigid/deformable environment subclasses and `utils/mdp` modules have been replaced by
these components. Internal Python imports moved; Gym IDs, configuration keys,
policy widths (134 actor, 156 critic, 27 actions), metrics, and evaluation queries
are preserved. `viewer_plates` returns `None` for VBD.

## Checks

```sh
.venv/isaaclab3/bin/python -m pytest isaacsimenvs -q
```

Tests cover geometry, policy conventions, task progression, partial resets,
backend state normalization, solver wiring, construction-hook cleanup, and contact
semantics. GPU smoke runs additionally exercise real scene creation and stepping.

The default rigid DR recipe is
[`scripts/configs/runs/rigid_body_cloth.yaml`](../../scripts/configs/runs/rigid_body_cloth.yaml).
It explicitly lists all five designs, fixed 2 mm thickness, density, sheet bending
rigidity/viscosity, disabled armature, effective cloth/table and cloth/robot static
friction, dynamic/static ratio, reset bounds/yaw and observation/action noise.
Fold entry pays 1,000 once; a still-valid fold with the entire robot at least 10 cm
clear pays another 1,000 and completes immediately, without a hold timer.
The Viser inspector loads this same recipe with `--training_config`; its physics
sliders set fixed values and its reset controls can sample the listed placement ranges.
