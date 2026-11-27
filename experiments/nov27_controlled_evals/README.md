# Controlled zero-shot evals on VBD cloth (2026-11-27)

The two point-cloud policies (`pretrained_policy/play_pc_rigidclothRL_64live.pth`, `..._64live_inplace.pth`),
trained on hinged rigid plates in PhysX, evaluated zero-shot on Newton (MuJoCo-Warp robot + VBD cloth).

## Protocol

- Worlds (`isaacsimenvs/tasks/cloth/config/eval/`):
  - `rigid_cloth.yaml`: the training world, two 2 mm plates on a free hinge in PhysX (box2-mid, 10 cm), every
    plate-physics value written out, the contact shell pinned at 7 mm.
  - `vbd.yaml`: the VBD evaluation world as it was.
  - `default.yaml`: `vbd.yaml` + `stable.yaml`'s contact-stability settings (training hand, 16 mm particles,
    3 Aitken-relaxed coupler iterations), frictions at the PhysX training values (cloth-table 0.354, fingertip 0.612).
  - `default_table_friction_0p7.yaml`, `default_high_table_friction.yaml`: cloth-table friction 0.7 / 1.0.
  - The robot collides PhysX's convex hulls on Newton by default (`NEWTON_PUSH_APPROXIMATION=0` turns it off).
- Start: pinned (no `--randomize`): default robot pose at rest, the cloth centred flat on the table, no reset
  noise, no observation / action DR. `PINNED_YAW_DEG` sets the cloth's heading (default 180: a left-to-right fold
  from the robot's view; 90: the half on the robot's side folds away from it, bottom to top). Non-default
  headings write to `<eval>_yaw<deg>/`.
- Actions: sampled (`--stochastic`), never the mean.
- Rollouts: each env scores its first episode (<= 900 steps); 64 per number (32 envs x 2 seeds). 95% CI ~ +-12 pp
  near 50%, ~ +-7 pp near 10 / 90%. 64 envs x 1 seed costs about the same GPU time as 32 (7.9 GB at 128 envs).
- Metric: the Hausdorff distance to the goal configuration of the cloth (aligned in the table plane, each ply
  with its own ply; `mdp/surface_match.py`, `surface_layered.py`). Success: < 1 cm (0.1 x side), held 10 steps.
  Continuous metric: its best value per episode (`surface.best_layered_dist`, `best_dist_median` / `_mean`).
  Folded episodes land at 3-7 mm on VBD, 0.1 mm on the plates; an unfolded cloth scores ~30 mm.
- Layering: the half that turned over is detected from the grid normals; a mirrored fold is a valid fold.

## Scripts

- `eval_pinned_<world>.sbatch`: 64 pinned rollouts per policy (both policies x 2 seeds) -> `outputs/controlled_evals/`.
  `VBD_STIFF_FACTOR=k` (env) makes the VBD sheet's halves k x stiffer in bending with the crease left soft
  (`backends/vbd/backend.py _install_crease_stiffness`); its results go to `<eval>_stiffhalves<k>/`.
- `render_pinned_rigid_cloth.sbatch` (+ `render_rigid_cloth.py`): pinned PhysX plate episodes, recorded.
- `../nov25_better_rigidcloth/eval/render_xfer.sbatch` (+ `render_vbd.py`): pinned VBD episodes, recorded.
- `render_recording.py`: every recording drawn in one look (URDF arm / hand colours, wood table, same floor,
  light, camera; VBD cloth blue, rigid plates purple / green), with the Hausdorff distance in the label. The
  render jobs call it, so each render folder holds `<episode>.npz` and `<episode>.mp4`.

## Results (pinned, heading 180, 64 rollouts each; success / mean best Hausdorff)

| world | 64live | inplace |
|---|---|---|
| rigid plates (PhysX) | 100% / 0.1 mm | 100% / 0.1 mm |
| vbd.yaml | 6% / 26 mm | 5% / 29 mm |
| default.yaml | 12% / 22 mm (15/128, two runs) | 0% / 30 mm |
| default + table friction 0.7 | 2% / 26 mm | 0% / 30 mm |
| default + table friction 1.0 | 2% / 26 mm | 0% / 31 mm |
| default + stiff halves x10000 (soft crease) | 100% / 7 mm | 45% / 18 mm |
| uniform stiffness x1-x1000 + plasticity (20 / 45 deg yield) | <= 30% (x1000, 20 deg) | <= 2% |

The policies fold by pushing the sheet until it buckles and one half pops up (observations and hand path match
the plates to the millimetre until the buckle); soft VBD cloth slides or crumples instead. Raising table friction
hurts. A uniformly stiff or plastic sheet does not buckle like hinged plates either; only plate-like halves with a
soft crease transfer. inplace's failures on stiff halves never start the fold (best Hausdorff ~31 mm).
Raw results: `outputs/controlled_evals/`, `outputs/vbd_transfer/`; renders: `outputs/renders/`.
