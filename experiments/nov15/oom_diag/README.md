# nov15_oom_diag: GPU memory growth at 24576 envs

Both 7-day rigid-cloth runs (658160 prior, 658161 scratch) ran out of GPU memory at epochs 29 / 21:
~39 GiB when training started (like the 11-14 fold job, 635994), then growing to 48 GiB on a 47.4 GiB
RTX 6000 Ada, until the LSTM's 1.45 GiB rollout allocation failed.

`submit.sh` runs four 4-hour jobs at 24576 envs, each changing one thing:

| Job | Change | Tests |
|---|---|---|
| `oom_asis` | none | the growth reproduces |
| `oom_noviewer` | no pose viewer (`VIEWER=0`) | the 7 stacked viewers |
| `oom_foldonly` | `task.mode=fold`, flat starts | the mixed fold/unfold resets |
| `oom_expandseg` | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | allocator fragmentation |

Read each run folder's `gpumem_<job>.csv` (every 30 s).

## Results

| Job | id | memory at training start | peak | OOM? |
|---|---|---|---|---|
