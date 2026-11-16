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

Newton compatibility belongs to [backends/vbd/newton_support/](backends/vbd/newton_support/).
Evaluation overflow reporting lives in `isaacsimenvs/scripts/evaluation/contact_guard.py`.
Prepare the default VBD robot/table cache under Isaac Sim with:

```sh
.venv/isaacsim/bin/python -m isaacsimenvs.scripts.assets.prepare_usd_cache --populate \
  --task Isaacsimenvs-RigidBodyCloth-Direct-v0 \
  env.assets.table_urdf=assets/urdf/table_narrow.urdf
```

Use the same robot/table paths and conversion options as the intended VBD run.

Historical physics measurements are in
[backends/vbd/physics_notes.md](backends/vbd/physics_notes.md). Fabric presets live
in `isaacsimenvs/scripts/configs/fabric`; print their Hydra overrides with:

```sh
.venv/isaaclab3/bin/python -m isaacsimenvs.tasks.cloth.backends.vbd.fabrics cotton --plastic
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
