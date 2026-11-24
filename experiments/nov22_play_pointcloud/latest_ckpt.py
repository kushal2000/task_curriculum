"""Print the newest loadable rl_games checkpoint of a run, for chaining a resumed job after a time limit.

    python experiments/nov22_play_pointcloud/latest_ckpt.py <slurm job id of the previous run>
    python experiments/nov22_play_pointcloud/latest_ckpt.py <job> --tolerance

``--tolerance``: the Play success-tolerance curriculum's last value (``current_success_tolerance`` in the
run's wandb summary), or nothing for a task without one -- an rl_games checkpoint does not carry
the env's curriculum, so a resumed run would otherwise restart at the 7.5 cm start.

Looks in debug_outputs/train_logs/*/*_<job>/**/nn/*.pth, newest first by mtime, and prints the first that
torch.load opens (a job killed at its time limit can leave the last file half-written).
"""

import json
import sys
from pathlib import Path

job = sys.argv[1]
roots = sorted(Path("debug_outputs/train_logs").glob(f"*/*_{job}"))
if not roots:
    sys.exit(f"[latest_ckpt] no run directory for job {job}")
if "--tolerance" in sys.argv:
    # The run's wandb summary, through the API: the local wandb-summary.json is only written when a run
    # finishes, which a job killed at its time limit may never do. Run id: wandb/run-<date>_<time>-<id>.
    import os

    import wandb

    runs = sorted((p for r in roots for p in r.glob("wandb/run-*") if p.is_dir()), key=lambda p: p.name)
    if runs:
        run_id = runs[-1].name.split("-", 2)[2]
        api = wandb.Api()
        project = os.environ.get("WANDB_PROJECT", "play_pc")
        tol = api.run(f"{api.default_entity}/{project}/{run_id}").summary.get("current_success_tolerance")
        if tol is not None:
            print(f"{float(tol):.6g}")
    sys.exit(0)

import torch  # noqa: E402 - only the checkpoint mode needs it

ckpts = sorted((p for r in roots for p in r.rglob("nn/*.pth") if not p.name.startswith("snap_")),
               key=lambda p: p.stat().st_mtime, reverse=True)
for path in ckpts:
    try:
        torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - any unreadable file is skipped
        print(f"[latest_ckpt] skip {path}: {exc}", file=sys.stderr)
        continue
    print(path)
    sys.exit(0)
sys.exit(f"[latest_ckpt] no loadable checkpoint under {roots}")
