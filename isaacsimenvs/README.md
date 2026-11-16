# Environment package

The main path through this package is configuration → environment → observations
and joint actions → training/evaluation. Importing `isaacsimenvs` registers Gym
IDs without starting a simulator.

| Package | Purpose |
| --- | --- |
| [tasks/cloth/](tasks/cloth/README.md) | One cloth manipulation loop with rigid and VBD scene adapters |
| `tasks/play/` | Existing rigid-object manipulation task and its robot variants |
| `scripts/` | Run entry points, visualization tools, and YAML configurations |

The policy runs outside the environments. `scripts/evaluation/player.py` loads checkpoint
policies and `scripts/evaluation/protocol.py` sets shared evaluation conditions;
the training scripts use rl_games. Environment observations and action
semantics define their interface, so physics choices stay inside the task package.

Run configuration lives in `scripts/configs/`: task overlays,
robot/fabric presets, learner settings, and named recipes under `runs/`. Python
configuration classes remain beside the environments they describe.

Configuration precedence is Python config defaults → task YAML → robot overlay
→ run YAML → CLI overrides. `scripts/configs/loader.py` applies those layers. Shared locations are
in the package `__init__.py`; fabric loading and Gym registration use `CONFIG_DIR` rather
than reconstructing paths from their module depth.

Start with [the cloth code-flow guide](tasks/cloth/README.md) for cloth work.
`tasks/cloth/backends/vbd/newton_support/patches.py` contains runtime compatibility details, not cloth reward or
observation logic. The two simulator runtimes use separate virtual environments:
`.venv/isaacsim` for PhysX and `.venv/isaaclab3` for Newton.

Run package tests with `.venv/isaaclab3/bin/python -m pytest isaacsimenvs -q`.

Script helpers live beside their callers: learner integration and logging in
`scripts/training/`, policy loading in `scripts/evaluation/`, and the pose/HTML
viewer in `scripts/visualization/`. Run-output paths and Slurm log handling live
in `scripts/output.py`. Task code does not import script helpers.
