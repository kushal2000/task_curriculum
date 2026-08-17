# Cable-task renders

Folders are ordered as the investigation happened, so this reads as a narrative: what the scene
looked like when nothing worked, the defects found along the way, and what changed when it started
scoring. Every clip is reproducible — see the command at the bottom.

**Replays do not reproduce the measured episodes, and the captions had a bug on top of that.**
Both, independently:

1. Until 2026-08-17 the renderer counted goals as `_successes[world]` end-minus-start, and
   `_successes` is **zeroed when an episode terminates**. Scoring extends episodes, so a clip whose
   episode ended mid-render captioned `scored ~0 goals` regardless. Fixed (accumulates positive
   deltas).
2. **With the fix in place, envs that scored 7, 6 and 5 goals in the eval harness replayed at 0.**
   So the divergence is real and was not merely a counting artefact -- a correction to an earlier
   claim in this file that the videos "had been showing goals all along".

The render and eval paths were checked to use the same `eval_success_tolerance`, the same
`disable_randomization` / `use_single_object_variant`, the same env count and the same seed. The
cause is therefore **unexplained**: either physics nondeterminism compounding over ~1800 steps in
episodes that only continue while the grasp holds, or a remaining difference between the two paths.

**Treat the videos as evidence of behaviour, not of goal counts.** The numbers live in
`docs/results/*.json` (`per_env_goals`), which the eval harness snapshots at the termination step
and which was never affected by either issue. `07_replays_final_config/cable_w0_s100.mp4` is the
best on-camera evidence: 2 goals, scored despite the counter bug.

---

## 01_early_cable_geometry
First cable envs on a flat table. 10 / 30 / 50 mm thickness and a 2-segment variant, plus the
first goal-marker renders. **Nothing scores.** The hand presses down and the cable skids off the
table within a second — the failure mode that persisted through 22 configurations.

## 02_scene_defects
Renders that exposed bugs rather than physics.
* `cable_lift_*` — the "lifts" here are the cable **pivoting on end**, not being grasped; a 0.30 m
  cable stood upright raises its centre by exactly the lift threshold.
* `cable_nohammer` — the orphaned rigid tool, visible for hours as an object lying on the floor
  beside the table, finally made inert.
* `cable_goalviz*` — the goal marker, which had never been drawn because it has no collision
  geometry and the viewer renders collision shapes.

## 03_proxy_links
Whole-hand proxying. The cable could previously only be touched by five fingertips, so an enclosing
grasp was geometrically impossible. Correct, but it did **not** measurably help.

## 04_clearance_supports
Cable raised on rails so the fingers have somewhere to close. **First goals ever scored.** Later
banned as a configuration, but it isolated the grasp-geometry question.

## 05_flat_table_first_goals
First goals on a flat table with no clearance, via the AVBD contact ramp. Later understood to be
scoring partly *because* the solver was running hot (39.9 m/s peak against a rigid baseline of
1.68), so this is a milestone rather than a good configuration.

## 06_staggered_coupling
`proxy_mode=staggered` — the change that actually worked. The exchange between the MJWarp and VBD
solvers pairs the hand's **end** pose with its end velocities instead of its begin pose. Falls drop
from ~46/64 to ~13/64 and the cable is carried instead of dropped.

## 07_replays_final_config
Per-env replays under the committed defaults. Captions here are trustworthy (post-fix).

---

## Reproducing

    sbatch scripts/cluster/sbatch_render.sh <world> <steps>          # one env, on the cluster
    xvfb-run -a scripts/newton_py -m isaacsimenvs.eval.render_newton \
        --task Isaacsimenvs-Cable-Direct-v0 --num_envs 64 --world 29 --steps 1900 \
        'env.assets.handle_head_types=[hammer]' env.cable.thickness=0.03

Replays do **not** reproduce a specific measured episode: physics nondeterminism compounds over
~1500 steps, and these episodes only continue while the grasp holds, so small divergences change
the outcome. The videos show behaviour; the numbers live in `docs/results/`.
