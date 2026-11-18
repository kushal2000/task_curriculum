# Contact 20 mm, rest 5 mm, fixed 2 mm plates — November 17, 2026

Two additional fresh SAPG runs inherit our material DR, Yibo robot/perception/table
interface and fold/release rewards. One initializes from original Yibo friction
checkpoint weights (`flat7RW_fricition.pth`); the other starts from scratch. Neither
resumes an existing optimizer or epoch.

- Cloth contact offset: **20 mm**, fixed.
- Cloth rest offset: **5 mm**, fixed; no offset randomization.
- Plate thickness: **2 mm**, fixed; thickness curriculum disabled.
- Both physics-rate resets enabled: outside-table and press-driven folding.
- 12,288 environments; W&B project `rigidClothDR`.

The cloth underside starts 7 mm above the visible tabletop: 2 mm above its intended
5 mm resting gap. This preserves the previous runs' 2 mm clearance above the
support surface instead of spawning inside the new rest shell. The support-height
guard already accounts for positive rest offset. Robot/table offsets stay at
2 mm contact and zero rest. Physical goals and hinge geometry retain the configured
2 mm plate thickness. All existing training runs are retained.

```bash
experiments/nov17_contact20mm_rest5mm/submit_prior.sh
experiments/nov17_contact20mm_rest5mm/submit_scratch.sh
```
