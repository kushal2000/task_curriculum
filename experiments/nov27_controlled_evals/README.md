# Controlled zero-shot evals on VBD cloth (2026-11-27)

The two point-cloud policies (`pretrained_policy/play_pc_rigidclothRL_64live.pth`, `..._64live_inplace.pth`),
trained on hinged rigid plates in PhysX, evaluated zero-shot on Newton (MuJoCo-Warp robot + VBD cloth).

## Protocol

- World: `isaacsimenvs/tasks/cloth/config/eval/default.yaml` = `vbd.yaml` + the contact-stability settings of
  `stable.yaml` (training hand `hand_sysid: false`, 16 mm particles, 3 Aitken-relaxed coupler iterations from
  0.5), frictions left at the PhysX training defaults (cloth-table 0.354, cloth-fingertip 0.612). The robot
  collides PhysX's convex hulls (`NEWTON_PUSH_APPROXIMATION`, on by default since this date; `=0` turns it off).
- Start: pinned (no `--randomize`): default robot pose at rest, cloth centred flat on the table at the fixed
  180 deg heading (`backends/base.reset_yaw`), no reset noise, no observation / action DR. `--randomize` is a
  separate robustness check (it adds start-state noise and the policy's training DR together).
- Actions: sampled (`--stochastic`), never the mean.
- Rollouts: each env scores its first episode (<= 900 steps); 64 per number (32 envs x 2 seeds so far).
  95% CI ~ +-12 pp near 50%, ~ +-7 pp near 10 / 90%.
- Metric: the Hausdorff distance to the goal configuration of the cloth (aligned in the table plane, each ply
  with its own ply; `mdp/surface_match.py`, `surface_layered.py`). Success: < 1 cm (0.1 x side), held 10 steps.
  Continuous metric: its best value per episode (`surface.best_layered_dist`, summary `best_dist_median` /
  `best_dist_mean`). Folded episodes land at 3-7 mm; an unfolded cloth scores ~30 mm.
- Layering (`layering` in the eval JSON): the half that turned over is detected from the grid normals, so a
  mirrored fold (the "stationary" half folded over) is scored as the valid fold it is (`successes_mirrored`).

`eval.sbatch` here runs this protocol for one line of a config list.

## Results so far (pinned, 64 rollouts each)

| cloth | 64live: best distance median / success | inplace |
|---|---|---|
| default | 21.2 mm / 8% | 30.5 mm / 0% |
| halves x10000 bending stiffness, default crease (`VBD_STIFF_FACTOR=10000`) | 6.8 mm / 100% | 29.9 mm / 45% |

PhysX plates from the same pinned start: 64/64 and 64/64. The policies fold by pushing the sheet until it
buckles and one half pops up; soft VBD cloth slides instead (observations and hand path match the plates to
the millimetre until the buckle). Raw results: `outputs/vbd_transfer/cr4_*`; renders: `outputs/renders/`.
Open: a uniformly stiffer cloth with bending plasticity (yield 20 / 45 deg like the plates' hinges),
`outputs/vbd_transfer/pl_*`.
