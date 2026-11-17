# Realistic VBD cloth: free fold lines, wrinkles, curvature plasticity (2026-11-16)

Page: https://claude.ai/artifact/6J1mX1u4MwbKx4zvxobfYR. Raw outputs: `outputs/cloth_probe/final/`, `outputs/vbd_eval/`.
Branch `vbd-cloth`.

## What changed

| | Today (`<fab>_measured.yaml`) | New (`<fab>_vbd.yaml`) |
|---|---|---|
| Mesh | 7x7 grid at every size; a vertex row on x=0 picks the crease | rotated, jittered triangle lattice, 10 mm edges (`cloth_geometry.irregular_mesh`) |
| Bending | `edge_ke` per hinge, mesh dependent | measured rigidity B [N m] per hinge as B / w_e, w_e = (A1+A2)/l_e, times a fitted solver factor (0.87-2.17) |
| Plasticity | 2-stage angle gate on one hinge | hinge rheology (`utils/hinge_rheology.py`): elastic + Dahl friction + curvature yield + creep, all in material units |
| Membrane | tri_ke 28-200 N/m | 10 N/m (see "Resolution") |
| Contact | 8 mm particle radius, 2 mm self contact | same |

## Resolution sweep

Cantilever tip angle at 5.25 cm overhang, 12 VBD iterations, dt 1/1440 s, membrane at the old stiffness
(`outputs/cloth_probe/cal2`):

| Edge | cotton | chiffon | napkin | towel | denim |
|---|---|---|---|---|---|
| 6 mm | 83 | 70 | 68 | 86 | 77 |
| 6 mm, 48 iters | 74 | 84 | 80 | 79 | 84 |
| 8 mm | 71 | 76 | 73 | 78 | 77 |
| 12 mm | 62 | 63 | 64 | 64 | 63 |
| swatch | 30 | 45 | 25 | 75 | 80 |

Every mesh droops every fabric about the same: the sheet is insensitive to B. This is VBD conditioning:
when the membrane stiffness per vertex is far above m/dt^2, the soft bending modes do not converge in 12
iterations. More iterations do not fix it, and a smaller dt (sub48, 1/2880 s) crumples resting sheets.
Softening the membrane to 10 N/m on a 10 mm mesh restores the B dependence. A per-fabric factor
`bending_stiffness_scale` then fits the swatches:

| | chiffon | silk | napkin | cotton | big_napkin | towel | denim |
|---|---|---|---|---|---|---|---|
| sim (deg) | 44 | 29 | 23 | 36 | 37 | 77 | 79 |
| swatch (deg) | 45 | 30 | 25 | 30 | 30 | 75 | 80 |

10 mm is the chosen fidelity mesh (130 / 491 / 1076 particles at 10 / 20 / 30 cm). A 17 mm coarse
variant (`<fab>_vbd_coarse.yaml`) exists for cheaper training.

## Properties (cloth-only probes, `tools/cloth_probe.py`)

- **Free fold line:** creases placed at offsets of 0/1/2 cm and angles of 15/45 deg land within 2.2 mm
  of the placed line. The grid snaps them by +6.7 / +13.3 mm.
- **Plasticity matches the swatches:** chiffon reopens even after a press. Silk and napkin reopen
  unpressed (73 deg) and hold when pressed (6 deg). Cotton, big napkin, towel and denim hold without a
  press. A light touch leaves no set. Every fold that holds stays within 4 cm for 180 frames after release.
- **Wrinkles:** under side shear on the table, cotton and silk buckle into ridges that grow with sheet
  size (max height 23 / 42 / 43 mm for cotton at 10 / 20 / 30 cm). The legacy cloth stays flat.
  Hung by two corners, every new fabric curves (p90 curvature 8-24 1/m), and dropped 30 cm sheets fold.
  See "Denim" below.

## Rigid-checkpoint eval (224 envs per cell, 8 mm / 2 mm contact)

| Cloth | Touch 4 cm | Hold 180 steps 4 cm | Touch 1.5 cm | Hold 180 steps 1.5 cm |
|---|---|---|---|---|
| Today | 46% | 25% | 29% | 2% |
| New | 84% | 8% | 29% | 0% |

- The mesh explains most of the touch gain: today's cloth with plasticity off scores 48%, the new cloth
  with tri 100 scores 71%.
- The hold drop comes from the hand: with actions frozen after the touch (`eval.py --freeze_after_touch`),
  hold at 4 cm is 29% (today) and 43% (new).

## Denim

Denim stays exactly flat under table shear, even 12x stiffer, at 15% squeeze, or with friction off
(`outputs/cloth_probe/denim_diag`). A perfectly flat, symmetric sheet has no out-of-plane perturbation,
and denim has no yield branch to supply one (cotton and silk buckle because plastic flow breaks the
symmetry). It does curve when hung and folds when dropped. A random rest curvature per hinge
(`RheologyParams.natural_curvature`, default 0, off in every preset) fixes it at 5 /m: ridges of 11 / 17.5 /
21 mm at 10 / 20 / 30 cm (`outputs/cloth_probe/natcurv`). It is left off: only denim needs it and the
magnitude is not measured.
