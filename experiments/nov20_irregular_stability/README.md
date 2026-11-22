# VBD cloth physics and probes

The VBD task defaults to a 7×7 checkerboard sheet, an 8 mm collision diameter, 12 robot substeps
with one VBD substep each, 12 VBD iterations, full-surface contact, effective cloth–table friction
0.6, and soft contact stiffness 500. The cloth timestep is 1/1440 s.

There are three presets for each of denim, towel, chiffon, cotton, silk, napkin, and big napkin.
The suffixes are `_measured` for the 7×7 grid, `_vbd_coarse` for 17 mm irregular edges, and `_vbd`
for 10 mm irregular edges. The irregular presets include per-hinge bending and fabric-specific
contact or membrane settings. Named presets override the default table friction with the measured
slide-onset coefficient, calculated as tan(slide angle). To evaluate a preset with the cleanup branch's held-fold protocol:

```bash
FABRIC_ARGS=$(experiments/newton_py isaacsimenvs/tasks/cloth/utils/fabrics.py denim_vbd_coarse)
CKPT=/path/to/policy.pth TAG=coarse_denim TARGETS=vbd INDEX=0 \
    EXTRA_ARGS="$FABRIC_ARGS" experiments/eval.sh
```

Use `--plastic` with the preset helper to reproduce the measured grid presets' bending plasticity.
To isolate mesh topology, start from the same measured preset on both runs and add only
`env.cloth.mesh_kind=irregular env.cloth.edge_length=0.017` to the irregular run.

`probes.sh` runs the standalone cantilever, fold, and press/drag sweep; submit it with
`sbatch --array=1-4 experiments/nov20_irregular_stability/probes.sh`. The CSVs in `results/`
are the original sweep results from `vbd-cloth`. For policy rollouts on the cleanup branch,
the following probes accept the same trailing Hydra overrides as evaluation:

- `python -m isaacsimenvs.tasks.cloth.tools.cloth_pierce_probe` measures finger–sheet crossings and their duration.
- `python -m isaacsimenvs.tasks.cloth.tools.hand_tracking_probe` measures tracking error, joint-limit overrun, and uncommanded joint jumps.
