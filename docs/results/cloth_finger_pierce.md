# Fingers go through the VBD cloth, and the sheet then hangs on them

**Why.** The VBD cloth feels rigid bodies ONLY through its particles (Newton `SolverVBD`:
`accumulate_particle_body_contact_force_and_hessian`; no triangle or edge sees a shape). Pressed onto
the table, the particle under a fingertip has nowhere to go but sideways -- contact is a soft penalty
(`soft_contact_ke` 500 N/m) on a sheet with very low stretch stiffness (`tri_ke` 100) -- so it slides
out from under, and the finger reaches the table between particles. Nothing then pushes the sheet up
off the finger: when the hand lifts, the surrounding particles sit on top of the finger and ride up
with it.

**Measured** (`scripts/analysis/cloth_pierce_probe.py`, launcher `bos14_cloth_pierce.sh`; 64 envs,
one episode each, randomised start). A pierce = a finger link's axis crossing its own env's cloth
triangles (impossible in ordinary contact: the particle radius keeps the finger surface 8 mm off the
sheet's mid-surface and the axis is ~7 mm inside the finger).

| | stage-2 pool, box3 prior s3 | cloth-trained reference |
|---|---|---|
| episodes with a pierce | 57/64 | 41/64 |
| ... lasting >= 30 steps | 38/64 | 5/64 |
| pierce onsets with the finger ON the table (axis < 12 mm up) | 68% (median 8.0 mm) | 87% (median 7.0 mm) |
| pierced finger-steps lifted > 30 mm (sheet hanging on the finger) | 36% | 30% |
| finger speed at onset, median / p90 | 0.16 / 0.30 m/s | 0.20 / 0.31 m/s |

Slow onsets and finger-on-table: this is pressing through, not fast tunnelling. The cloth-trained
policy pierces too, but briefly (and folds 98% of its pierced episodes); the stage-2 policy, trained
on rigid plates that cannot be pierced, presses and stays through.

**No solver knob fixes it** (fp box3 s3, % of episodes pierced; base 88%): substeps 12 89%, VBD
iterations 24 89%, `soft_contact_ke` 5e3 91%, proxy mode `lagged` 89%. `tri_ke` 1e3 88%; `tri_ke`
1e4 59% but the sheet stops folding (reference 98% -> 32% of its pierced episodes). A finer grid does
not either -- resolution 9 / 12.5 mm 59/64, resolution 13 / 8.3 mm 55/64 -- because the particle radius
shrinks with the spacing; it does cut the long hangs (>= 30 steps: 30 -> 21 -> 10).

The fix is a contact model in which the sheet's FACES collide with the hand (triangle/edge vs shape),
which Newton's VBD does not have -- or a policy that learns not to press through (the reference
mostly does). The live viewer (`isaacsimenvs/eval/live_viser.py`) draws a pierced finger's axis red
and names it in the status panel.
