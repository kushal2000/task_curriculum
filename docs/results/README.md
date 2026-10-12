# Raw result data (for the scripts, not for reading)

The reports are in **`docs/reports/`**. Start with `docs/reports/README.md`.

This folder holds the raw output of evals and probes: one JSON per eval task, per-env tensors
(`.pt`), and the eval manifests (`rc_eval_manifest*.tsv`). The `scripts/analysis/*_summary.py`
scripts read these files and generate the tables in `docs/reports/`.

JSON and TSV files that were already committed stay in git. New raw output (`*.json`, `*.pt`,
`*.png` here) is ignored by `.gitignore` and stays on the local machine. To reproduce a number in a
report, keep the local files or rerun the Reproduce commands at the end of that report.

Contents:
- `rc_eval_*.json`: cross-evaluation of rigid chains and the cloth (`docs/reports/2_rigid_chain/`, `4_physx/`)
- `fp_eval_*.json`: Newton shape pool, stage 2 (`3_fold_pool_newton/`)
- `px_matrix/`: full eval matrix of the PhysX runs and ft7 (`4_physx/eval_matrix.md`, `finetune_7shapes.md`)
- `large_cloth/`: larger cloths (`4_physx/large_cloth.md`)
- `contact_stiffness/`: contact stiffness (`4_physx/contact_stiffness.md`)
- `fingerprint_*.pt`, `pxscreen_*`, `rigid_physics_*`: physics comparisons from the PhysX port (`4_physx/physx_port.md`)
- `cloth_pierce_*`: fingers through the cloth (`4_physx/cloth_finger_pierce.md`)
- `fold_start_probe_*`, `one_shape_parity/`, `ft7_diag/`, `physx_*`, `newton_replay*`, `px_success_lines_*.png`: debugging and validation probes and plots
