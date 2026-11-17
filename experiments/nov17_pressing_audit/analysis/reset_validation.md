# Reset-condition verification — 17 November 2026

**The proposed checks are not validated as a selective pressing-strategy detector.** Thresholds that intercept most reverse-half lifts also reject intended-half lifts. Hinge anchor and axis checks do not detect the behavior. These checks may still be useful as explicitly chosen simulation-quality limits, but their ability to discourage pressing during new training has not been tested.

## What was actually run

Four current periodic checkpoints, 64 first episodes each, all five training designs, full saved training DR, seed 17, deterministic actions and zero exploration coefficient. Fixed 2 mm and 2–20 mm thickness profiles both included. Each rollout ran up to 600 policy steps and recorded every 120 Hz physics step. Actual training physics, rewards and reset logic were unchanged. The candidate resets were evaluated afterward on the complete trajectories, so their first firing time and subsequent motion can both be inspected.

The exact checkpoints and hashes are in [the manifest](reset_checkpoint_manifest.json). Full raw recordings and pinned checkpoints live under `../results/reset_validation/`, which is ignored by git. [All thresholds, per-run counts, event identities and physical maxima](reset_validation.json).

| Current run | Epoch | Reverse-half first lifts | Intended-half first lifts | Unclassified/excluded |
|---|---:|---:|---:|---:|
| scratch_2mm | 4040 | 43 | 0 | 21 |
| prior_2mm | 3980 | 43 | 0 | 21 |
| scratch_2to20mm | 4020 | 52 | 0 | 12 |
| prior_2to20mm | 3480 | 41 | 7 | 16 |

## Results before the first clear lift

A lift event is the first time one material half reaches 60° from table-up while the other is within 20° of flat and near tabletop height. A height threshold excludes small rotations. Every cloth vertex must have stayed over the tabletop through the event, within the declared 2 mm footprint margin; earlier edge excursions are excluded. Half identity and geometry nominate comparison cohorts without using the tested load/depth/constraint signals.

**“Intended-half lift” is not a certificate of a clean, successful fold. “Reverse-half lift” does not prove exploitation.** Their labels distinguish motion paths; force/penetration observations cannot by themselves certify physical correctness. Seven intended-half controls are a small sample, all from the current thick fine-tuned checkpoint. Unclassified episodes are not counted as successes or failures.

| Candidate reset | Reverse-half lifts intercepted | Intended-half lifts intercepted |
|---|---:|---:|
| Opposed load >1 N, trailing 50 ms mean | 56/179 | 3/7 |
| Cloth below tabletop >0.5 mm | 139/179 | 6/7 |
| Cloth below tabletop >1 mm | 76/179 | 3/7 |
| Either load >1 N/50 ms or table depth >0.5 mm | 151/179 | 7/7 |
| Hinge anchor separation >0.5 mm | 0/179 | 0/7 |
| Hinge axis misalignment >1° | 0/179 | 0/7 |
| Joint angle exceeds limit by >1° | 66/179 | 2/7 |
| Robot–cloth reported contact depth >0.5 mm | 179/179 | 7/7 |
| Opposed load >50× sheet weight, 50 ms mean | 28/179 | 0/7 |
| Table depth >10% thickness (minimum tolerance 0.1 mm) | 137/179 | 4/7 |

No tested threshold catches all 179 reverse-half events while retaining all seven intended-half events. Normalizing loads by sheet weight or penetration by thickness does not remove the overlap. A 50× weight load limit catches 28/179 while retaining seven of seven in this sample; that is low coverage and does not establish a transferable threshold.

These are causal interception counts, not retraining results. The 50 ms mean contains only current and earlier physics steps. A trigger is counted only when it precedes or coincides with the first 60° event. The JSON also contains whole-first-episode triggers and effects on natural release terminations. Actual early termination would discard later trajectory states; this analysis does not claim to simulate the policy after a reset.

## Concrete comparison

Current thick fine-tuned checkpoint, single-hinge worlds:

- [Reverse-half world 0](reset_reverse_world0_geometry.png), [forces](reset_reverse_world0_traces.png): before its 1.8 s lift event, maximum 50 ms opposed load 0.578 N, maximum table penetration 1.098 mm. Orange is pinned while blue rises.
- [Intended-half world 60](reset_intended_world60_geometry.png), [forces](reset_intended_world60_traces.png): before its 0.683 s lift event, maximum 50 ms opposed load 0.436 N, maximum table penetration 1.214 mm. Orange rises while blue stays down. A table-depth threshold that preserves this example also misses the reverse example's table-depth excursion. This example itself has collision-depth violations; it is a comparison of motion, not a claim that it is physically clean.

Hinge anchors remain within 0.853 µm and axes within 0.000054° over all valid frames in the four captures. The plates can therefore execute this motion while satisfying the articulation's anchor/axis constraints. Joint limits do overshoot in some cases, so a joint-limit guard catches a subset but not the general strategy.

## Measurement limits

Opposed load is the sum over physical plates of `min(downward robot normal load, upward table normal load)`; loads are assigned by canonical plate identity, never noisy rest-x signs near a crease. This measures simultaneous opposing loads, not actual continuum contact pressure or all force transmission through neighboring plates. Friction forces are not included, and sleeping contact reports can be zero.

Cloth–table depth is computed from the actual extruded plate vertices transformed by their recorded link poses. Its plane test is restricted to the tabletop footprint; an independent table-edge reset is a separate condition. Hinge anchors/axes use resolved per-environment metadata recorded during scene setup. Joint limits use actual design limits including the configured central-hinge override.

Robot–cloth depth here is **PhysX's reported separation for loaded contact points**, not an independent post-solve exact collision-mesh intersection. Such a separation can refer to contact generation before the recorded post-step poses. Robot–table mesh penetration and adjacent-cloth solid overlap were not independently reconstructed by this sweep. These unmeasured quantities are not reported as passing.

The earlier 32-world fine-tuned recording was analyzed separately as an archived reference (its older checkpoint was already removed by training retention). It supplied no additional first-lift controls after the full-prefix edge exclusion, so it is not included in the 256-episode totals or the table above. Missing archived hinge metadata is explicitly unknown, not zero.

## Recommendation

Do not enable these thresholds as an “airtight” exploit classifier. Decide whether compression/penetration limits are desired operating constraints in their own right; then evaluate their effect through retraining. To establish an approximation or solver defect, independently check the collision geometry and replay offending action tapes with smaller timesteps/more solver iterations. No additional contact-history rule was added.

## Reproduction and checks

`evaluate_pressing.py` now records resolved anchors, axes, plate ownership and actual limits for each environment. Run it with an explicit pinned `--checkpoint`, then run:

```bash
.venv/isaacsim/bin/python experiments/nov17_pressing_audit/validate_resets.py \
  experiments/nov17_pressing_audit/results/reset_validation/scratch_2mm \
  experiments/nov17_pressing_audit/results/reset_validation/prior_2mm \
  experiments/nov17_pressing_audit/results/reset_validation/scratch_2to20mm \
  experiments/nov17_pressing_audit/results/reset_validation/prior_2to20mm \
  --out experiments/nov17_pressing_audit/analysis/reset_validation.json

.venv/isaacsim/bin/python -m pytest -q \
  experiments/nov17_pressing_audit/tests_contact_readout.py \
  experiments/nov17_pressing_audit/tests_reset_predicates.py
```

All 256 first episodes completed. Contact pair-matrix checks passed. Fourteen focused tests passed, including known anchor/limit errors, a physically folded reference pose, per-plate opposed load, causal timing, and explicit missing-data handling. No training jobs or production reset/reward code were changed.
