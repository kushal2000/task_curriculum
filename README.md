# task_curriculum

**Does learning an RL task through a curriculum of progressively harder variants beat
learning the hard variant directly?**

Testing that needs environments whose difficulty is a *knob*, not a fixed property. This
repo provides two of them in Isaac Sim (Isaac Lab), a shared curriculum controller, and a
driver that runs the A/B.

The structure follows [play2perfect](https://github.com/kushal2000/play2perfect) — same
`isaacsimenvs` package layout, same three-layer config, same single `train.py`, same
vendored `rl_games` fork.

## The two environments

| Task | id | Difficulty knob |
|------|----|-----------------|
| Multi-link cartpole balance | `Isaacsimenvs-MultiLinkCartpole-Direct-v0` | number of free pole joints `n`, and the effective link lengths `L` |
| Bottle flipping (Kuka + Sharpa hand) | `Isaacsimenvs-BottleFlip-Direct-v0` | fill level, required turns, landing tolerance |

### Multi-link cartpole

Every env spawns the *same* `n_max`-link pole. Difficulty is realised by **locking a
per-env subset of the pole joints** at reset: a locked joint fuses its two links into one
rigid segment, so an env with only joint 0 free is a classic single inverted pendulum,
while an env with every joint free is an `n_max`-link chain **of the same total length and
mass**. Difficulty is therefore purely "how many unactuated DOF must one actuator
stabilise", with the geometry held constant.

Two consequences worth knowing:

- Because *which* joints are free is randomised, the effective link lengths vary too —
  with `n_max=4`, freeing `{0,1}` gives a `(0.25, 0.75)` double pendulum and freeing
  `{0,3}` gives `(0.75, 0.25)`. Both `n` and `L` vary per-env and per-episode from one
  spawned USD, with no `MultiUsdFileCfg` and no rebuild.
- A locked joint is *stiff, not rigid*. It is held by a narrow position limit (a solver
  constraint) plus a high-gain PD. `geometry.use_joint_limit_lock=false` turns the
  constraint off, which is how you measure what the approximation is worth.

Observations are padded to `n_max` and carry the segment lengths, so one network spans the
whole curriculum and a stage-2 run can restore a stage-1 checkpoint unchanged.

### Bottle flipping

`BottleFlipEnv` subclasses play2perfect's `PlayEnv`, inheriting the Kuka + Sharpa robot,
the 29-dim action pipeline, and the observation layout **byte-identical**. That is what
lets a play2perfect checkpoint initialise a bottle-flip run:

```bash
isaacsimenvs/train.py --task Isaacsimenvs-BottleFlip-Direct-v0 \
  --checkpoint <play2perfect.pth> --checkpoint_load_mode weights
```

The bottle is a single rigid link whose mass, centre of mass and inertia are assembled
from a wall, a base disc, and a liquid column. Fill level is written into PhysX per-env at
every reset, so it is a genuine per-episode curriculum knob rather than something frozen
into the USD.

One finding worth flagging: **CoM height is not monotonic in fill.** An empty bottle's CoM
sits near its centre, a little liquid drags it down, and more liquid pushes it back up —
the minimum is around 24% for the default bottle. That is the familiar "about a third full
is easiest" result, recovered from the mass distribution alone with no sloshing model. So
difficulty interpolates fill over `[fill_fraction_easy, fill_fraction_hard]` starting *at*
the CoM minimum; setting `fill_fraction_easy` lower would make the "easy" end harder.

Sloshing is not modelled — the liquid is rigid, so in-flight redistribution and the
damping it gives on landing are both absent. Bottles here are uniformly harder to land
than real ones.

## Project structure

```
task_curriculum/
├── isaacsimenvs/
│   ├── train.py              # single training entry point
│   ├── curriculum/           # shared, task-agnostic: cfg, core, schedulers
│   ├── cfg/{task,train}/     # Hydra task + rl_games configs
│   └── tasks/
│       ├── multilink_cartpole/
│       ├── bottle_flip/      # subclasses play
│       └── play/             # play2perfect base env, verbatim
├── rl_games/                 # vendored fork (PPO + SAPG), verbatim
├── assets/urdf/              # Kuka + Sharpa robot, table
├── experiments/run_curriculum.sh
├── tests/                    # Kit-free unit tests
└── docs/isaacsim_installation.md
```

## The curriculum API

Every env carries `env._difficulty`, shape `(num_envs, difficulty_dim)`, in `[0, 1]` where
0 is the easiest instance and 1 the hardest. `isaacsimenvs/curriculum/` owns *where that
vector is sampled from and when the range moves*; each task owns *what the numbers mean*,
in its `reset_utils.apply_difficulty`.

Wiring, identical in both envs:

```python
__init__      allocate_curriculum_buffers(self)
_reset_idx    record_episode_success(...) → sample_difficulty(...) → apply_difficulty(...)
_get_dones    update_curriculum(self)
_get_rewards  reward_scale(self, "<term>")  →  log_curriculum(self)
```

Three schedulers, selected by `env.curriculum.mode`:

- `fixed` — the range never moves. The control arm and eval runs.
- `linear` — interpolates `init_range → final_range` over `anneal_steps`.
- `adaptive` — advances only when the recent success rate clears
  `adapt_success_threshold`, and backs off if it collapses. Self-paced.

`curriculum/*` scalars land in `extras`, which the existing `EnvStatsAlgoObserver` already
forwards to tensorboard and wandb — no observer changes.

## Installation

See [docs/isaacsim_installation.md](docs/isaacsim_installation.md): Python 3.11, `uv`,
a `.venv_isaacsim/`, `isaaclab[isaacsim,all]==2.3.2.post1`, `-e ./rl_games/`, then
`-e . --no-deps`. Keep the pins exact.

```bash
export OMNI_KIT_ACCEPT_EULA=YES
```

## Running

```bash
# Cartpole
.venv_isaacsim/bin/python isaacsimenvs/train.py \
  --task Isaacsimenvs-MultiLinkCartpole-Direct-v0 --headless \
  env.scene.num_envs=4096 agent.params.config.max_epochs=1000

# Bottle flip (SAPG)
.venv_isaacsim/bin/python isaacsimenvs/train.py \
  --task Isaacsimenvs-BottleFlip-Direct-v0 \
  --agent rl_games_sapg_cfg_entry_point --headless \
  env.scene.num_envs=8192
```

Smoke test at small scale — note that `num_actors * horizon_length` must be divisible by
`minibatch_size`, so shrink both together:

```bash
.venv_isaacsim/bin/python isaacsimenvs/train.py \
  --task Isaacsimenvs-MultiLinkCartpole-Direct-v0 --headless \
  env.scene.num_envs=64 \
  agent.params.config.minibatch_size=256 \
  agent.params.config.max_epochs=2
```

## The experiment

```bash
export OMNI_KIT_ACCEPT_EULA=YES
WANDB_PROJECT=task_curriculum experiments/run_curriculum.sh MultiLinkCartpole 42
```

Three arms, matched on total environment steps, all scored with difficulty pinned to 1.0:

| Arm | What it is |
|-----|-----------|
| **A** curriculum | one run, `mode=adaptive` — the range starts easy and advances on demonstrated competence |
| **B** control | one run, `curriculum.enabled=false` — every env at difficulty 1.0 from step 0 |
| **C** staged | three runs at fixed increasing difficulty, each restoring the previous stage's weights |

A and B differ in exactly one config field, so nothing else can explain a gap between
them. Stage definitions live in the script as Hydra CLI overrides rather than in the task
YAML, so both arms are visible side by side and auditable in one place.

Watch `curriculum/range_hi`, `curriculum/window_success_rate` and `episode_final/success`
in tensorboard: an adaptive run whose `range_hi` never leaves `init_range[1]` is not a
curriculum, it is a fixed easy task, and the threshold needs lowering.

## Tests

```bash
.venv_isaacsim/bin/python -m pytest tests/ -q     # 37 tests, no Kit, no GPU
```

The modules under test — the asset generators, the cartpole difficulty maths, the
schedulers — are deliberately free of Isaac imports, and `tests/conftest.py` loads them by
file path so the package `__init__` chain (which needs a booted Kit) never runs. If a test
here starts needing Kit, the module it covers has grown a dependency it should not have.

`test_config_overlays.py` additionally checks statically that every key in every task YAML
names a real configclass field — otherwise a typo only surfaces after Kit has booted and
the assets have been generated.

Registration and scene setup do need Kit, and have their own check:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
.venv_isaacsim/bin/python experiments/check_registration.py          # ids resolve
.venv_isaacsim/bin/python experiments/check_registration.py --make   # + build each scene
```

## Formatting

```bash
./format.sh     # ruff check --fix + ruff format, then xmllint over assets/**/*.urdf
```

## Acknowledgements

Environment scaffolding, the `rl_games` SAPG fork, and the Kuka + Sharpa assets come from
[play2perfect](https://github.com/kushal2000/play2perfect). The cartpole dynamics and
reward are generalised from Isaac Lab's `direct/cartpole`.
