# Phase 3 — the pretrained policy on Newton/MJWarp

**The policy transfers. Newton runs the task and scores above matched PhysX, which is not a win.**

| | Newton | PhysX (matched) |
|---|---:|---:|
| goals/episode | **18.438 ± 1.921** | **13.859 ± 1.696** |
| sd across episodes | 15.37 | 13.57 |
| mean episode length | 2396 | 1783 |
| lift fraction | 0.906 | 0.859 |
| censored | 0 / 64 | 0 / 64 |
| hit the 50-goal cap | **7** | 4 |
| terminations (fall / cap / hand_far / timeout) | 7 / 7 / 1 / 49 | 22 / 4 / 1 / 37 |

Raw: `docs/results/phase3_{newton,physx}_64env_single.json`.

```bash
# Newton (kit-less; scripts/newton_py supplies the LD_PRELOAD the venv needs)
scripts/newton_py -m isaacsimenvs.eval.episodes \
  --task Isaacsimenvs-PlayNewton-Direct-v0 --num_envs 64 --num_assets_per_type 1 \
  --single_variant --max_steps 12000 'env.assets.handle_head_types=[hammer]'

# PhysX, identical protocol
.venv_isaacsim/bin/python -m isaacsimenvs.eval.episodes \
  --task Isaacsimenvs-Play-Direct-v0 --num_envs 64 --num_assets_per_type 1 \
  --single_variant --max_steps 12000 --headless 'env.assets.handle_head_types=[hammer]'
```

Both legs run the same harness file, same flags, and the JSONs agree on every protocol field
(`num_envs`, `num_assets_per_type`, `single_variant`, `randomization`, `success_tolerance`,
`max_steps`, `seed`) — asserted, not assumed.

## Why a single shared object

Not a simplification for convenience. ``SolverMuJoCo`` rejects worlds whose collision shapes
differ in *type*, and the procedural tools mix boxes and capsules — even one `handle_head`
request emits both a handle-only variant (capsule) and a handle+head one (capsule + box):

```
ValueError: SolverMuJoCo requires homogeneous worlds.
Shape types mismatch at position 36: world 0 has type 4, but other worlds have types [7, 4, 7].
```

The alternative is `meshify`, which makes every collider a convex mesh and therefore uniform in
type. It does let Newton run the full pool, but it changes the geometry *both* backends see, so it
would add a confound to the comparison being made. A shared object removes the blocker without
introducing one.

**Consequence: these numbers are not comparable to the phase-1 baseline of 13.599.** That was the
24-object pool; this is one hammer. The PhysX column here, not phase 1, is Newton's control.

## Reading the result

**The primary gate passes.** Newton constructs, resets, steps, and runs the policy to goal
completion with **zero overflow warnings** of any kind — triangle-pair, `nefc`, or contact-limit.
That is the gate that mattered: those failures are silent, and the reference project spent a day
misattributing the backend because a `max_triangle_pairs` default of 1e6 was quietly dropping
contacts above ~16 envs. The auto-sized budget here (64 × 65,536 = 4.19M) held.

Lift rate 0.906 corroborates it — the policy is genuinely grasping, not scoring through a metric
artifact.

**Newton scoring above PhysX is not evidence the port is better.** The PhysX env is the one whose
dynamics the policy was trained against, so scoring above it means the dynamics differ in a
direction that makes goals *easier*, not that Newton is more faithful. The termination histogram
shows the mechanism plainly: PhysX drops the tool three times as often (22 falls against 7), while
Newton holds it and keeps scoring.

**The difference is at the edge of what one run each can resolve.** Δ = +4.58 with se 2.56 gives
z = 1.79 — suggestive, not established at 95%. Phase 1 measured the resolution limit for a
one-run-vs-one-run comparison at 64 envs as roughly 6 goals/episode, and 4.58 sits inside that.

**And Newton's mean is right-censored.** Seven of its 64 episodes ended on
`max_consecutive_successes = 50` against PhysX's four, so its true mean is higher than 18.44 and
the gap is *understated*. The harness flagged this automatically rather than leaving it to be
noticed.

## Independent corroboration

The reference port measured the same thing on different hardware:

| | here | reference |
|---|---:|---:|
| Newton, single object, primitives | **18.438 ± 1.921** | 18.609 ± 2.071 |
| PhysX band, single object | 13.859 | 15.6 – 16.6 |

Newton agreeing to within 0.17 across two independent implementations on different GPUs is strong
evidence that this port reproduces the reference's Newton behaviour, including its offset.

That offset is **unexplained in the reference too**, and explicitly recorded there as open rather
than attributed — the project's own note being that attributing it prematurely had repeatedly
produced a retraction. Candidates it ruled out by direct measurement include torsional friction
(`condim=6` versus 3 is neutral, z = 0.34) and geometry (meshify is worth about +0.44 on PhysX).
Nothing here narrows it further, so it stays open.

## Open

* **The offset itself.** Not attributed. The clean discriminator the reference never ran is a
  matched-geometry comparison at scale.
* **Multi-asset on Newton** needs `meshify`; untested here.
* **One run each.** Repeats would settle whether +4.58 is real, at the cost the project decided
  not to pay for now.
