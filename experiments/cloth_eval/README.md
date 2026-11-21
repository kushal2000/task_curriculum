# Standard VBD fold evaluation

Use an integration checkout that contains both `evals` and `cloth-physics`. The latter supplies
full-surface contact and the seven measured 7×7 fabric presets. The historical evaluation launcher
is `experiments/eval.sh`.

```bash
git switch -c cloth-eval-integration evals
git merge cloth-physics
CKPT=/path/to/policy.pth OUTDIR=outputs/cloth_eval/policy bash experiments/cloth_eval/run.sh
experiments/newton_py -m experiments.cloth_eval.summarize outputs/cloth_eval/policy
```

The default is 256 matched starts per fabric in batches of 32, all with deterministic actions, neutral DR and
identical robot initial states. For a smoke test, set `FABRIC=denim ENVS=2`. For a centered start,
pass `--cloth-pinned` to the launcher. Use a unique output directory for each checkpoint and profile.

Both thresholds use the same 10-second episodes. The JSON includes per-start acquisition, release
and persistence times, terminal reasons, checkpoint and simulator settings, and per-fabric
summaries. Failure diagnostics can overlap. A re-touch or loss of the geometric fold resets the
three-second persistence clock. An acquired fold stays credited if it subsequently fails.
