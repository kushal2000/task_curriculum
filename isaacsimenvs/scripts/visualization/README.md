# Visualization

Start the rigid-cloth inspector with the existing entry point:

```sh
.venv/isaacsim/bin/python isaacsimenvs/scripts/visualization/viser_rigidCloth_viewer.py
```

Add `--policy` to run the pretrained Flexiv–Wuji policy. Static preview and policy
mode use the training robot/table setup by default.

Play the Flexiv-Wuji checkpoint on a live deformable cloth with Newton/VBD:

```sh
.venv/isaaclab3/bin/python isaacsimenvs/scripts/visualization/viser_deformableCloth_viewer.py
```

Open `http://localhost:8080`, forwarding that port over SSH when needed. The default
is the 10 cm base cloth with the Yibo branch's reset ranges and a tabletop centered
33 cm above the robot base (with ±1 cm height variation). Use `--preset materials`
for the 15 cm material-evaluation setup, or `--material cotton_measured` to start
with another YAML fabric. All eight materials are available in the browser.

Controls include play/pause, single step, episode reset, speed, particles,
wireframe, and policy markers versus physical fold corners/targets. The sheet
mesh follows the live VBD particles. Material, side length, mesh resolution and
drop-height edits take effect with **Apply and restart**; the browser reconnects
on the same port. Material changes start from a fresh config and reset the policy.

The viewer uses the environment's observations, controller and episode resets.
It loads `flat7RW_fricition.pth` by default; `--checkpoint` and `--policy-config`
select another compatible Flexiv-Wuji policy. `--paused` starts without stepping,
`--port` changes the port, and `--steps 120` performs a finite smoke run. Policy
markers preserve Yibo's historical goal encoding; physical corners/targets use
the current cloth and the physical fold criterion. VBD release scoring is disabled.

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
