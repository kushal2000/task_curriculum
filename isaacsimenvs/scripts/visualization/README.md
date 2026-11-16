# Visualization

Start the rigid-cloth inspector with the existing entry point:

```sh
.venv/isaacsim/bin/python isaacsimenvs/scripts/visualization/viser_rigidCloth_viewer.py
```

Add `--policy` to run the pretrained Flexiv–Wuji policy. Static preview and policy
mode use the training robot/table setup by default.

| Folder | Purpose |
| --- | --- |
| [`rigid_cloth/`](rigid_cloth/README.md) | Cloth policy adapter, reset/recipe helpers, design PNG renderer and tests |
| [`policies/`](policies/) | Live policy viewers and PhysX/Newton rollout videos |
| [`robots/`](robots/) | Robot comparison and Wuji joint-limit inspectors |
| [`interactive_viewer/`](interactive_viewer/) | Recorded HTML viewers and training capture wrappers |

Other entry points:

```sh
# Browse trained Play policies.
.venv/isaacsim/bin/python -m isaacsimenvs.scripts.visualization.policies.viser_play --help

# Compare robot geometry and joint limits.
.venv/isaaclab3/bin/python -m isaacsimenvs.scripts.visualization.robots.viser_compare --help

# Render the cloth YAML catalog to PNG.
.venv/isaaclab3/bin/python -m isaacsimenvs.scripts.visualization.rigid_cloth.render_designs --help
```

The default rigid DR recipe is
[`rigid_body_cloth.yaml`](../configs/runs/rigid_body_cloth.yaml).
