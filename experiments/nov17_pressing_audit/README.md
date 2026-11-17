# Pressing strategy audit — 17 November 2026

Follow-up: [reset conditions tested against four current checkpoints](analysis/reset_validation.md). The sweep did not validate a selective pressing-strategy detector.

Actual saved rollouts and 96 fresh first episodes show the hand can pin the designated moving half to the tabletop while the opposite half rises and folds over it. A table-edge reset will not catch the clearest witness. A hard force threshold does not cleanly distinguish this motion from intended-half lifting.

## What the rollouts show

[Saved reverse-fold storyboard](analysis/saved_reverse_fold.png): scratch job 710889, single hinge, captured step 108599. Orange denotes +X moving material; blue denotes −X stationary material. At 0.483 s the blue half tilts 88.97° and its far edge is 111.8 mm above the table, while orange stays flat. The thumb approaches orange; all cloth vertices remain inside the tabletop throughout the 0–0.98 s sequence. These saved captures contain poses, not forces.

[Saved intended-fold comparison](analysis/saved_intended_fold.png): fine-tuned job 710887, frame 247. Orange lifts while blue tilts 7.94°. Blue's center is nevertheless 10.53 mm above its flat-sheet height. A strict requirement for the stationary half to sit within 2 mm of the table would reject this example.

Fresh normal-contact recordings use the actual saved run configurations, current trained checkpoints, normal actions/controller, all robot links, and first episodes only. They sample every 120 Hz physics step, before automatic resets. These are diagnostic samples, not general success-rate estimates:

| Dataset under `results/` | Initialization | Thickness | First episodes |
|---|---|---|---:|
| `scratch_single_hinge` | Scratch 710889 | 2 mm | 32 |
| `prior_single_hinge` | Fine-tuned 710887 | 2 mm | 32 |
| `scratch_thick_single_hinge` | Scratch 711273 | 2–20 mm | 32 |

A concrete fresh witness is scratch 2 mm world 3. At 0.692 s the stationary half has tilted 62.24°, its sampled mean midplane height is 42.8 mm above table, and the moving half's is 0.964 mm. During the preceding 0.1 s, peak downward robot load on the moving material is 0.694 N; there is no robot contact on the stationary material. The whole-sheet weight is 0.0236 N. This establishes simultaneous downward pinning and opposite-half lift, rather than inferring force from hand proximity.

A second fresh witness, scratch world 1 at 2.208 s, is shown in [the contact storyboard](analysis/scratch_reverse_geometry.png) and [force traces](analysis/scratch_reverse_traces.png). Only the orange moving plate receives appreciable robot pressure: 5.043 N downward while it stays flat, versus whole-sheet weight 0.0702 N. Blue reaches 90.49°. Correctly assigning crease contacts by plate ownership gives 0 N table support on blue and 5.081 N on orange at that instant.

Force ranges overlap. With stricter pose/support filtering, the fine-tuned 2 mm sample has reverse-pose windows in 14 worlds and intended-lift windows in 7 worlds. Across the selected physics frames, reverse downward load has median 0.404 N, p90 2.771 N and maximum 6.856 N; intended load has median 0.453 N, p90 1.984 N and maximum 3.938 N. All vertices remain within table bounds in each included 100 ms window, and robot contact is present currently or recently. These nominate geometry/interaction states, not manually labeled complete episodes or calibrated force thresholds. Full criteria and exact checkpoint provenance are in [the comparison data](analysis/fresh_comparison.json). [Intended-lift comparison](analysis/prior_intended_geometry.png).

## Why this is currently rewarded

`mdp/geometry/fold.py` reflects the moving material in the **live stationary-half frame**. This is necessary to avoid rewarding a simple whole-sheet slide. `ClothTask.is_folded` checks relative corner error and footprint; it does not require a particular half to remain table-supported or specify how folding occurs. Consequently, keeping +X down and folding −X over it can satisfy the current shape objective. Final inverted stationary material is a particularly strong discriminator of which half folded, but not proof of nonphysical motion.

A reverse-half fold can be valid manipulation of real cloth. Since designs are symmetric, an unconditional named-stationary-half uprightness gate would also remove that valid behavior. Avoid treating the nominal half labels as a physical validity test.

## Recommended implementation

1. **Measure pressure where it happens.** Use robot–cloth contacts, not robot net forces or actuator effort. For each material patch near and supported by the table, sum the downward component of force ON cloth. Keep robot and table contacts separate and retain material coordinates so all symmetric designs work. A short temporal average or integrated excess load avoids classifying one contact impulse as sustained pressing. Record absolute N and load relative to sheet weight; weight alone is a poor threshold for lightweight fabrics.
2. **Penalize sustained excessive supported pressing softly.** This discourages forceful pinning without resetting ordinary grasps or counting lateral lifting forces as pressure. Pair it with progress diagnostics: the contacted half stays table-flat while the uncontacted half rises. That conjunction flags the observed strategy, but should not by itself be called a solver failure or automatically terminate the episode.
3. **Make the motion requirement explicit if needed.** For a designated-half, table-supported fold, require meaningful moving-half lift history and tolerant stationary-half tilt/lift bounds when paying fold/release bonuses. Keep the live relative targets. For a task allowing either half to fold, use a symmetric supported-final-stack condition instead; this alone does not prohibit forceful pressing during the approach.
4. **Use physical invalidity for hard resets.** Deep robot/table penetration, large joint constraint error, or severe persistent compression is a more defensible termination criterion than raw force. Calibrate against accepted rollouts and test offending action tapes at smaller dt / more solver iterations before changing contact geometry. An exact hand-mesh penetration/energy/convergence test remains to be done; these recordings do not establish numerical energy creation.

This gives two distinct controls: a soft pressure penalty for the unwanted behavior, and physically grounded terminations for invalid states. A force-only reset cannot reliably do both.

## Physics interpretation and limits

The existing controlled downward-force probe did not reproduce a large flip. Current creases lie on the top face; central limits permit negative bending and adjacent plates do not collide, so neighboring prisms can overlap for negative angles. That is a concrete approximation defect worth testing, not yet a demonstrated cause of this rollout. A few newtons acting through submillimeter displacement can supply millijoules, enough to lift a gram-scale half; visually small displacement does not establish energy creation.

Contact instrumentation reads **normal forces only**, not friction or complete contact work. Settled smoke calibration gave table support 0.014758/0.053868/0.193302 N versus weights 0.014758/0.053868/0.193303 N in three uncontacted worlds. Per-point force vectors reproduce the PhysX pair-force matrix at every recorded sample; incomplete buffers raise. Sleep thresholds, material properties, rewards, production backend, and running jobs are unchanged. All three full captures completed with finite cloth geometry and verified contact matrices.

## Reproduce

```bash
OMNI_KIT_ACCEPT_EULA=YES OMP_NUM_THREADS=2 .venv/isaacsim/bin/python \
  experiments/nov17_pressing_audit/evaluate_pressing.py --headless \
  --run debug_outputs/train_logs/nov/2026-11-16_yibo_our_dr_scratch_710889 \
  --num-envs 32 --steps 600 --design single_hinge.yaml \
  --out-dir experiments/nov17_pressing_audit/results/scratch_single_hinge

.venv/isaacsim/bin/python -m pytest -q experiments/nov17_pressing_audit/tests_contact_readout.py
```

Each dataset's `metadata.json` records the exact selected checkpoint, dimensions, meshes, mass, configuration, terminal reasons and contact verification. `trajectory.npz` stores geometry and per-half forces; `contacts.npz` stores raw point records. Local results are ignored by git.

`analyze_pressing.py RESULTS_DIRECTORY ...` generates area-weighted geometry metrics, contact/force storyboards and traces. `render_rollouts.py --render` scans the saved training HTML captures. Focused contact-buffer/coordinate tests pass in the IsaacSim environment.
