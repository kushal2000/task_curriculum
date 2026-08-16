# Phase 1 — pretrained SimToolReal policy on the Play env (PhysX baseline)

**13.599 ± 0.832 goals/episode**, three runs of 64 envs, zero censoring in any of them.

Raw: `docs/results/phase1_play_physx_64env{,_s1,_s2}.json`.

```bash
export OMNI_KIT_ACCEPT_EULA=YES
for s in 0 1 2; do
  .venv_isaacsim/bin/python isaacsimenvs/eval/episodes.py \
    --task Isaacsimenvs-Play-Direct-v0 --num_envs 64 --num_assets_per_type 4 \
    --max_steps 12000 --seed $s --headless \
    --out docs/results/phase1_play_physx_64env_s$s.json
done
```

## Why the policy loads at all

`isaacsimenvs/tasks/play/` is upstream SimToolReal's Isaac Lab task package under a different
name. Diffed against `github.com/tylerlum/simtoolreal`: `action_utils.py` and
`termination_utils.py` are byte-identical, and `obs_utils.py`, `reset_utils.py`,
`goal_sampling.py`, `reward_utils.py` differ by a single docstring line. So the released
checkpoint's 140-dim observation layout is this env's layout, and no adapter is needed.

## Protocol

Every field below is recorded in the result JSON, because the number means nothing without them.

| setting | value | why |
|---|---|---|
| envs | 64 | |
| sampling | first completed episode per env | see below |
| `num_assets_per_type` | **4** (6 types → 24-object pool) | the reference protocol's value |
| success tolerance | `eval_success_tolerance = 0.01` | × `keypoint_scale` 1.5 = a **1.5 cm** keypoint threshold, held 10 consecutive steps |
| frictions | on (fingertip mu 1.5, else 0.5) | the regime the policy was trained in |
| domain randomisation | **off**, start pose pinned | |
| step budget | 12000 (used 6066) | |

**Sampling.** The episode deadline resets on every goal hit
(`play/utils/termination_utils.py:51`), so a successful episode runs for thousands of steps while
a failed one ends within 600. Taking "the first 64 completed episodes" would resample failures
repeatedly while the good episodes were still running; one episode per environment is unbiased.

**Pool size is part of the task, not a speed knob.** `generate_handle_head_urdfs` shuffles the
whole pool under a fixed seed before round-robin assignment, so changing
`num_assets_per_type` changes *which* objects envs 0–63 receive. `cfg/task/Play.yaml` ships 100
for training; evaluation uses 4 to match the reference. As a side effect startup converts 48
URDFs instead of 1200.

## Result

| seed | goals/episode | sd | ep length | lift | hit 50-cap | censored |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 12.516 ± 1.738 | 13.91 | 1787 | 0.828 | 3 | 0 |
| 1 | 13.047 ± 1.672 | 13.37 | 1827 | 0.828 | 2 | 0 |
| 2 | 15.234 ± 1.428 | 11.43 | 2025 | 0.859 | 1 | 0 |
| **3-run mean** | **13.599 ± 0.832** | | 1880 | 0.838 | | 0 |

| | here | reference |
|---|---:|---:|
| goals/episode | **13.599 ± 0.832** | 14.109 ± 1.575 |
| sd across episodes | 11.4 – 13.9 | 13.47 – 13.85 |
| mean episode length | 1880 | ~1806 |
| lift fraction | 0.838 | 0.80 – 0.86 |

Difference from the reference is **+0.51**, far inside noise. The episode-level spread, episode
length and lift rate all agree as well — the consistency check worth having, since a number
matching on the mean while disagreeing on the distribution would suggest two compensating errors
rather than agreement.

The reference is `isaac_newton`'s Isaac Lab 2.3 + PhysX run on the same protocol. Different
repository, different hardware (RTX 5090 there, RTX 6000 Ada here), so it orients rather than
certifies. **13.599 is the baseline phases 2–4 are measured against.**

### How much resolution this protocol actually has

Run-to-run sd across the three seeds is **1.44**, against a mean within-run sem of **1.61**. Run
noise and episode noise are therefore comparable — the same relationship `isaac_newton` measured
at 1024 envs — so the total sd on any *single* run is `sqrt(1.61² + 1.44²) = 2.16`, not 1.6.

Consequently a one-run-versus-one-run comparison at 64 envs resolves only about
**6 goals/episode at 95%**. Quoting a single run's own sem as the uncertainty on the measurement
understates it by about a third.

**Decision: later phases run once, not three times.** The cost is resolution, and it is worth
stating plainly so no one reads more into a phase-3 number than it carries. A single Newton run
against this 3-run baseline has `se = sqrt(2.16² + 0.83²) = 2.31`, so it resolves about
**4.5 goals/episode at 95%** — enough to catch the kind of failure this port is prone to (the
`max_triangle_pairs` defect took Newton from 14.3 to 0.12), and not enough to call a 2-goal
difference real. If a phase-3 result lands close to 13.6, that is "no gross defect detected",
not "parity established"; establishing parity would need the repeats.

## Caveats

- **Right-censored at the top.** 1–3 episodes per run ended on `max_consecutive_successes = 50`,
  so their true goal counts are higher and the mean is slightly understated. At 2–5% this is
  small, but it means a *better* backend gets compressed toward this number rather than scoring
  above it. Reported on every run for that reason.
- **The distribution is heavily skewed**, not concentrated: in seed 0, ten envs scored 0 and three
  hit the 50 cap, with sd ≈ mean. That skew is why the resolution is what it is (above).
- **Self-collision is off** in this env (`scene_utils.py:1613,1619`), whereas upstream commit
  `84058661` deliberately enabled it with adjacent-link filtering, on the grounds that the
  checkpoint is gym-trained with self-collision on. This was flagged in advance as a likely reason
  the baseline would come in low. **It did not materialise** at this resolution — the run matches
  the reference without it. That is not evidence the difference is zero, only that it is smaller
  than ~4 goals/episode here. The backport remains available as a one-flag A/B if a later phase
  needs the extra fidelity.

## Incidental fix

`play/utils/scene_utils.py` scaled object friction off `assets_cfg.robot_friction` instead of
`object_friction`. Dormant on both counts today — the branch needs
`object_friction_scale_range != (1.0, 1.0)` to execute, and both frictions are 0.5 — so it changes
no result here, but it would have applied the wrong base the moment either stopped being true.
