# Yibo checkpoint: isolate cloth collision offsets — November 17, 2026

Evaluate `pretrained_policy/flexiv_wuji_finetuned/flat7RW_fricition.pth`
using the current `yibo.yaml` recipe and a version changing only:

| Cloth setting | Yibo | Our offsets |
| --- | --- | --- |
| `contact_offset` | 12 mm | 2 mm |
| `rest_offset` nominal | 7 mm | 0 |
| `randomization.yibo_rest_offset` per reset | 4–10 mm | 0 |

The runtime reset range must also change; otherwise the historical randomizer
would overwrite zero rest offset. Friction stays on the Yibo dispatcher in both
cases. Neither the 2 mm physical plate thickness nor the observed 16 mm thickness
feature changes. Robot/table placement, hand dynamics, mass, gains, material
coefficients, initial pose, actions and observations are otherwise matched.

Each version uses three paired seeds (0, 1, 2), 256 environments per seed and up to
600 deterministic policy steps. All five designs are included. The task YAML is
applied before the recipe, matching training's 1.5 cm corner tolerance and 0.65
footprint limit. Each environment's first episode is the sampling unit; fold and
release metrics are captured before autoreset.

The requested comparison retains the recipe's two new reset checks. A second
matched comparison disables both checks to separate contact-physics effects from
guard compatibility. The initial evaluation exposed a bug: the settled-flat height
gate assumed zero rest offset, while Yibo's positive shell lifts the resting cloth.
The guard now includes the runtime cloth rest offset in its support-height check.
Actual COM heights and gravitational-work references remain unchanged. A CPU
regression covers independent 0, 4 and 10 mm shells. The requested pair is rerun
with this fix; the original invalid guarded result is preserved in `guard_bug.json`. Geometric fold,
guard-eligible fold, guard readiness, and full fold-and-release are reported
separately. The no-checks diagnostic can permit strategies that the new training
checks intentionally prohibit.

```bash
experiments/nov17_yibo_offsets/run.sh
experiments/nov17_yibo_offsets/run_without_checks.sh
.venv/isaacsim/bin/python experiments/nov17_yibo_offsets/summarize.py
```

Results are local ignored files: corrected checks under `results/offset_aware/`,
and the no-checks diagnostic under `results/strict/`. Earlier pilot results
under `results/` used the legacy runner's 4 cm tolerance and are excluded from the
final comparison. `summarize.py` requires identical initial states, identical
non-offset native property readbacks, and exactly the three intended config
changes. PhysX offset readbacks are checked for every cloth environment. Buffer
overflow warnings and nonfinite observations/rewards fail the evaluation.

These evaluations do not change or restart the four training jobs.


For future offset tuning, calibrate rest offset to intended surface geometry first,
then sweep the positive contact margin at the chosen timestep and expected speeds.
Check stationary settling, fingertip impacts, grasp/lift and panel self-contact;
measure penetration, jitter, tunnelling and runtime before choosing a DR interval.
Offsets near a panel width can also affect non-neighboring plates, so test every
topology. Keep rest-offset DR tied to actual geometry uncertainty. PhysX's primary
reference is [Tuning Shape Collision Behavior](https://nvidia-omniverse.github.io/PhysX/physx/5.1.2/docs/AdvancedCollisionDetection.html).
