# Run configurations

This is the YAML configuration root for the cloth and Play runners.
Python configuration classes remain beside each environment and backend; these
files select presets and override their defaults for a run.

| Directory | Contents |
| --- | --- |
| `task/` | Environment overlays selected by the registered Gym task |
| `robot/` | Robot variants layered over the task settings |
| `fabric/` | Cloth material presets |
| `train/` | Learner settings, including PPO and SAPG |
| `runs/` | Named experiment recipes, optionally inheriting another recipe with `base:` |

Existing experiment names are unchanged. For example, `--experiment rigid_body_cloth`
loads `runs/rigid_body_cloth.yaml`:

```sh
bash experiments/train.sh --experiment rigid_body_cloth
```

Environment settings are composed in this order: Python defaults → task YAML →
robot overlay → run recipe → command-line overrides. A run recipe's `env` and
`agent` sections configure the environment and learner, respectively.

`loader.py` applies these layers. `formatting.py` converts resolved OmegaConf
values into plain Python containers for policy loading and logging.
