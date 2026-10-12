# Newton fold pool: stiffer contacts remove the visual overlap and cost nothing

**Why.** In the live viewer (Newton `FoldPool`, PhysX-trained run 2 policy) fingers sink into the
plates and plates sink into the table. It is not a missing contact: the contacts are always present
(`scripts/analysis/rigid_overlap_probe.py`). It is MJWarp's soft contact -- shape `ke` 2500 / `kd`
100 = solref 0.02 -- acting on 7-27 g plates, which gives mm-scale penetration.

**Overlap probe** (8 envs x 1500 steps, PhysX run 2 policy; share of env-steps with a finger vertex
at the plate mid-plane / a plate > 2 mm into the table):

| contact | fingers in plate | plate in table |
|---|---|---|
| default (solref 0.02) | 34% | 10% (median 6 mm, p90 17 mm) |
| `contact_ke=1e4 contact_kd=200` (solref 0.01) | 1.0% | 2.8% |
| `contact_ke=4e4 contact_kd=400` + `num_substeps=4` (solref 0.005) | 0% | 0.5% |
| plate mass x3 | 2.6% | 8.2% |

The Newton-trained stage-2 policy (156746 s1) on default physics: 7.2% / 6.0%. The PhysX policy
presses harder, since it trained on rigid contacts, but both policies do it.

## Fold rate with stiffer contacts (existing checkpoints, zero-shot)

`scripts/cluster/bos14_contact_stiffness_bench.sh MODE=eval` (arrays 158687 + 158691): 5-shape
Newton pool, 320 envs x 2 eval seeds x 3 policy seeds, 1.5 cm cutoff, 0 non-finite episodes.
Collated by `scripts/analysis/contact_stiffness_summary.py`; raw JSONs in
`docs/results/contact_stiffness/`.

Config names: in `stiff2` / `stiff4` the digit is the **substep count**, not a stiffness multiple
(`stiff2` = 4x ke, solref 0.01; `stiff4` = 16x ke + 4 substeps, solref 0.005).

| policy | config | pinned | randomised | median best error (pinned) |
|---|---|---|---|---|
| Newton stage 2, box3 prior (156746) | base | 97.1% | 81.5% | 3.2 mm |
| | stiff2 | 98.7% | 83.3% | 2.1 mm |
| | stiff4 | 99.5% | 83.2% | 2.1 mm |
| PhysX run 2, flat pool (158376) | base | 92.8% | 72.8% | 3.0 mm |
| | stiff2 | **98.9%** | **77.3%** | 1.6 mm |
| | stiff4 | 94.9% | 76.9% | 0.8 mm |

Stiffer contacts never lowered the overall fold rate, and the fold gets tighter (PhysX policy: median
error 3.0 -> 1.6 -> 0.8 mm pinned). The PhysX-trained policy gains the most, as expected from a policy
that trained on rigid contacts.

## Speed

`MODE=speed`, 1536 envs, 80 epochs, all on one node (158740; 158688 on mixed nodes agrees): base
5213, stiff2 5734 (+10%), stiff4 4978 (-4.5%) env-steps/s. Doubling the substeps costs only ~5%.

## Not measured

Only zero-shot evaluation of existing checkpoints on the rigid pool. Whether a policy **trained**
with stiffer contacts transfers better to the VBD cloth has not been run. Knob:
`env.rigid_cloth.contact_ke` / `contact_kd` (0 = Newton default; Newton only, PhysX refuses it).
