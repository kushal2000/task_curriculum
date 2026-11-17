# Better reset candidate: direct robot lifting work

**A work-source test is substantially more selective than force or penetration limits on the recorded checkpoints. It remains a task rule, not an airtight physics-validity certificate.** All results below are shadow reset evaluations on complete trajectories. No training reward, termination, solver or collision settings were changed.

## The condition

Apply the same check to either material half. Reset when all of the following occur:

1. One half is rising through 30–90° while the other remains approximately flat over the tabletop.
2. The lower half currently has opposing loads: the robot exerts a net downward force and the table supplies an upward reaction. An upward pinch is not classified as downward pressing.
3. The rising half has gained appreciable gravitational potential energy, but the robot has not supplied meaningful **opening work directly to that material half**.

The research threshold is 10% of the rising half's gravitational energy gain. Other constants and the precise eligibility conditions are in [the candidate configuration](lift_work_provisional_rule.json). This fraction is an explicit task-design parameter, not a derived physical constant.

Work is measured at the **robot side** of each contact: force on cloth dotted with the robot contact point's velocity, integrated over physics steps. Only contact forces whose moment about the central crease favors opening that material half earn lifting credit. Normal and friction forces are both included. A static hand receives zero credit even if the solver moves cloth out of penetration. A side pull that supplies an opening moment can receive credit; this is not restricted to vertical fingertip motion.

Credit begins after the entire material half is observed flat, table-supported and nearly stationary. It survives letting go, and is discarded when the half settles flat again. This avoids a fixed touch-history timeout and prevents old pressing losses or old lift credit from contaminating a new attempt. The candidate is inactive until a valid reference has been observed; invalid initialization needs a separate reset/settling rule.

## What was tested

640 instrumented first episodes with the same four pinned checkpoints used in the earlier reset study:

- 256 recaptured episodes across four policies and five designs, seed 17. Cloth points and robot joint trajectories match the earlier recordings **exactly**, confirming the extra friction/state reads did not change those rollouts.
- 128 episodes at seed 29, initially a validation sample and then used to improve the rule.
- 128 episodes at seed 41, used to inspect remaining mechanisms and the limitation of measuring work only on the receiving cloth.
- **128 episodes at seed 53 reserved for the final robot-source candidate.** Its parameters were fixed before these results were inspected.

The four checkpoint hashes remain in [the checkpoint manifest](reset_checkpoint_manifest.json). [Capture provenance](lift_work_capture_manifest.json), [development results](lift_work_all_development.json), [final held-out results](lift_work_seed53_holdout.json).

The geometry cohorts retain the previous definition: first clear 60° lift with the opposite half near-flat, and no preceding tabletop excursion within the stated 2 mm footprint margin. They distinguish which half rises. **They do not establish that every reverse lift is cheating or that every intended lift is clean.** Unclassified episodes are excluded from the cohort comparison, not counted as successes or failures.

## Final held-out comparison, seed 53

| Proposed reset | Reverse-half lifts intercepted before the cohort event | Intended-half lift trajectories flagged |
|---|---:|---:|
| Compression >1 N, trailing 50 ms mean | 39/70 | 15/26 |
| Cloth–table penetration >0.5 mm | 62/70 | 20/26 |
| Either of those limits | 65/70 | 24/26 |
| New robot-source lifting-work rule | **62/70** | **4/26** |

All 70 reverse and 26 intended cohort events had an observed flat starting reference. [The old-condition sweep on this exact sample](lift_work_seed53_old_conditions.json) supplies the comparison. The new rule reaches the same reverse-cohort interception count as the 0.5 mm penetration limit while flagging far fewer intended trajectories.

Development results for the final rule were 174/179 reverse and 0/7 intended on the original sample; 70/75 and 4/21 at seed 29; 60/66 and 2/22 at seed 41. These samples informed the research process and are not additional independent validation of the final choice.

## The cohort labels conceal earlier pressing

The four seed 53 intended trajectories flagged by the final rule are worlds 71,73,99,113. The trigger details are included in the held-out JSON:

| World | First trigger | Half rising at trigger | Opposite half's net supported load | Robot lifting work credited |
|---|---:|---|---:|---:|
| 71 | 1.325 s | Blue stationary material, 38.27° | 0.410 N | 0 J |
| 73 | 1.325 s | Blue stationary material, 30.20° | 0.583 N | 0 J |
| 99 | 2.475 s | Blue stationary material, 38.26° | 0.123 N | −0.0229 J |
| 113 | 1.433 s | Orange moving material, 73.09° | 0.405 N | −0.0175 J |

The first three are flagged during an earlier opposite-half rise, before the later intended-half event. World 113 requires interpretation of the hand contact and energy transfer; a negative credited work value alone is not a certificate of invalid physics. These counts should not be presented as a measured false-positive rate on human-certified good episodes.

An independently inspected seed 41 example makes the label problem concrete. World 1 is labeled intended because orange eventually reaches the cohort's 60° event. At **1.808 s**, however, orange is flat at 0.42°, the robot presses it downward with **2.001 N**, and untouched blue has already risen to **30.29°**, gaining 0.382 mJ with zero direct robot work. The receiver-side predecessor flags this earlier maneuver. Orange is lifted later. [Hand/cloth storyboard](lift_work_flagged_hand.png), [load and work traces](lift_work_flagged_case.png), [exact trigger](lift_work_seed41_flagged_case.json). The final source rule also flags this trajectory. [Another inspected early-press-then-lift sequence](lift_work_early_press_then_lift.png).

Storyboards show actual plate geometry with robot visual-mesh convex hulls for display speed. They are not exact cooked robot collision meshes and are not used to compute the detector.

## Measurement checks

- A controlled sliding-sheet test starts at +0.2 m/s with no robot contacts. Friction work computed from the mean pre/post contact-point velocities matches the loss of horizontal kinetic energy within **0.020%** in all four worlds. [Calibration](lift_work_calibration.json).
- A position-difference work estimator underestimated that loss because PhysX uses discrete stepping. It was replaced before selecting the work-source candidate.
- Robot point velocities use measured COM linear/angular velocities and actual body poses, not URDF visual origins. Source-side work and receiver-side work are kept separate.
- Normal per-point forces reproduce PhysX's pair-force matrix on every recorded step. Friction buffers have independent capacity/span/finite checks; friction vectors were checked through the controlled slide and a net dissipative hand/cloth slip diagnostic. This does not establish exact pointwise angular energy accounting through every contact transition.
- Contact coordinates near a crease are assigned by actual rigid-plate ownership. All five current designs split material consistently at x=0. General diagonal designs are outside this validation.
- The code refuses to read a capture whose NPZ files may still be writing. Missing pre-step information is not used to authorize a source-work reset before a flat reference has been observed.

## Limits and next implementation step

The work rule is a much closer match to “press one side to raise the other” than a raw force cap. It permits either material half to be manipulated and retains credit after release. It does not prove the motion is impossible for real cloth, solve neighboring-panel overlap, or establish that every unflagged rollout is acceptable. Eight reverse-cohort events in the final sample remain unflagged. Some involve direct robot input or lack the required supported-press context; they should not automatically be relabeled successful detection or clean manipulation.

The 10% energy fraction, angular bounds and numerical tolerances can still be exploited by a policy trained against them. The next meaningful test is a training comparison with this rule enabled, followed by rollout inspection. Initialization must establish a valid flat reference before policy control or veto success until that reference exists; leaving the rule unarmed is not a safe production fallback. Production integration needs GPU-side accumulation at physics rate, separate termination statistics, and vetoing success/release bonuses on the triggering step. Training throughput at 12,288 environments has not been benchmarked for these contact reads.

All current scripts remain diagnostic. Thirty-one focused tests cover source-versus-receiver work, static-hand zero credit, closing-force rejection, side-pull opening work, symmetric material ownership, release credit, flat-state rearming, force-buffer integrity, and capture completion.

## Reproduction

Record with `evaluate_pressing.py --record-work` and an explicit pinned checkpoint. The calibration uses `--calibration-slide`. Then run:

```bash
.venv/isaacsim/bin/python experiments/nov17_pressing_audit/lift_work.py \
  experiments/nov17_pressing_audit/results/work_validation/prior_2to20mm_holdout53 \
  --out experiments/nov17_pressing_audit/analysis/lift_work_seed53_holdout.json

.venv/isaacsim/bin/python -m pytest -q \
  experiments/nov17_pressing_audit/tests_contact_readout.py \
  experiments/nov17_pressing_audit/tests_reset_predicates.py \
  experiments/nov17_pressing_audit/tests_lift_work.py
```
