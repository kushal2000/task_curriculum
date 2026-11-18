# 20 mm cloth contact offset, fixed 2 mm geometry — November 17, 2026

Two additional fresh SAPG runs: initialize from Yibo's friction checkpoint, or
train from scratch. Both inherit the Yibo robot/observation/table interface and
our material DR from `yibo_our_dr.yaml`. Only the cloth contact offset is enlarged:

- Cloth `contact_offset`: **20 mm**.
- Cloth `rest_offset`: **0**; no rest-offset DR.
- Physical cloth thickness: **2 mm**, fixed; no thickness curriculum.
- Both physics-rate resets enabled: outside-table and press-driven folding.
- 12,288 environments; existing 1,000 fold + 1,000 release rewards.
- W&B project `rigidClothDR`, group `nov17_contact_offset_20mm`.
- Prior starts from original `flat7RW_fricition.pth` weights, with a fresh optimizer
  and epoch; scratch starts from random weights.

Robot/table collider offsets retain their 2 mm contact / 0 rest settings. Thus
cloth–robot/table pairs start generating contacts within 22 mm, while their target
resting surface gap remains zero. This changes contact detection, not mesh size.

```bash
experiments/nov17/contact_offset_20mm/submit_prior.sh
experiments/nov17/contact_offset_20mm/submit_scratch.sh
```

The four existing training runs are retained.

## Jobs

| Initialization | Job | W&B |
| --- | --- | --- |
| prior | 765823 | [run](https://wandb.ai/kk837/rigidClothDR/runs/rigidClothDR_prior_contact20mm_thickness2mm_resets_20261117_215416) |
| scratch | 765824 | [run](https://wandb.ai/kk837/rigidClothDR/runs/rigidClothDR_scratch_contact20mm_thickness2mm_resets_20261117_215347) |
