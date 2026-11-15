"""Upload engaging_fabric_finetune.sh runs (TensorBoard logs) to W&B, with their fabric-eval results.

    WANDB_BASE_URL=https://api.forge.coreweave.com .venv/isaaclab3/bin/python -I \
        scripts/analysis/ftfab_to_wandb.py --job 28276252 [--fabrics denim cotton]

One W&B run per fabric, id `ftfab_<fabric>_<job>`. A preempted task that restarted from the prior
leaves an aborted attempt dir behind; the attempt with the most logged epochs is the run. Scalars
are replayed with the TensorBoard step (env-steps / frames) as the W&B step, plus `epoch`.
Summary gets the 1.5 cm pinned fabric eval: untuned (c0, all seeds; best block on seed 3) and
finetuned (per-block seed-3 probe; chosen block, all seeds). Re-running a fabric resumes its run:
steps already logged are skipped, new ones appended, the summary refreshed.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path

FABRICS = ["big_napkin", "chiffon", "cotton", "denim", "napkin", "silk", "towel"]
COEFS = [50, 40, 30, 20, 10, 0]
RES = Path("debug_outputs/train_logs/results/px_matrix")
PRIOR = "checkpoints/last_0_pxpool_flat7DRABC_s1_175385_ep_3052_rew__1198.3557_.pth"
FRAMES_PER_EPOCH = 768 * 32


def held(tag: str, fabric: str, coef: int, seeds=None):
    """(held folds, episodes) over the given seeds' JSONs, or None if none exist."""
    files = sorted(RES.glob(f"fabric_{tag}__{fabric}_measured_plast_c{coef}_s*.json"))
    if seeds is not None:
        files = [f for f in files if int(f.stem.rsplit("_s", 1)[1]) in seeds]
    if not files:
        return None
    runs = [json.loads(f.read_text()) for f in files]
    return (sum(r["termination_reasons"].get("fold_held", 0) for r in runs),
            sum(r["num_envs"] for r in runs), len(runs))


def eval_summary(fabric: str) -> dict:
    s = {}
    b = held("flat7DRABC_s1", fabric, 0)
    if b:
        s["eval/untuned_c0_held"], s["eval/untuned_c0_episodes"] = b[0], b[1]
        s["eval/untuned_c0_rate"] = b[0] / b[1]
    for tag, key in (("flat7DRABC_s1", "untuned"), ("ft", "finetuned")):
        probe = {c: held(tag, fabric, c, seeds={3}) for c in COEFS}
        probe = {c: v for c, v in probe.items() if v}
        for c, v in probe.items():
            s[f"probe_s3/{key}_c{c}_held"] = v[0]
        if probe:
            best = max(sorted(probe), key=lambda c: probe[c][0])  # ties -> smaller coef
            s[f"probe_s3/{key}_best_coef"] = best
            s[f"probe_s3/{key}_best_held"] = probe[best][0]
    # Final: the chosen block is the one with all seeds present.
    for c in COEFS:
        v = held("ft", fabric, c)
        if v and v[2] >= 6:
            s["eval/finetuned_coef"] = c
            s["eval/finetuned_held"], s["eval/finetuned_episodes"] = v[0], v[1]
            s["eval/finetuned_rate"] = v[0] / v[1]
            clean = held("ft", fabric, c, seeds={4, 5, 6, 7, 8})
            if clean:
                s["eval/finetuned_rate_seeds4to8"] = clean[0] / clean[1]
    return s


def run_dir(fabric: str, job: str) -> Path | None:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator as EA

    best, best_n = None, -1
    for d in glob.glob(f"debug_outputs/train_logs/*/*/0_ftfab_{fabric}_{job}"):
        ev = glob.glob(f"{d}/summaries/events.out.tfevents*")
        if not ev:
            continue
        ea = EA(sorted(ev)[-1], size_guidance={"scalars": 0})
        ea.Reload()
        n = len(ea.Scalars("fold_rate")) if "fold_rate" in ea.Tags()["scalars"] else 0
        if n > best_n:
            best, best_n = Path(d), n
    return best


def upload(fabric: str, job: str, args) -> None:
    import wandb
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator as EA

    d = run_dir(fabric, job)
    if d is None:
        print(f"[wandb] {fabric}: no run dir")
        return
    ev = sorted(glob.glob(f"{d}/summaries/events.out.tfevents*"))
    rows: dict[int, dict] = defaultdict(dict)
    for f in ev:
        ea = EA(f, size_guidance={"scalars": 0})
        ea.Reload()
        for t in ea.Tags()["scalars"]:
            for x in ea.Scalars(t):
                rows[x.step][t] = x.value
    nn = sorted(p.name for p in (d / "nn").glob("*.pth"))
    finished = any(n.startswith("last_") for n in nn)
    overrides = subprocess.run(
        [".venv/isaaclab3/bin/python", "-I", "isaacsimenvs/tasks/cloth/utils/fabrics.py",
         f"{fabric}_measured", "--plastic"], capture_output=True, text=True).stdout.split()
    config = {
        "fabric": fabric, "preset": f"{fabric}_measured", "plasticity": True,
        "fabric_overrides": overrides, "prior": PRIOR, "checkpoint_load_mode": "weights",
        "task": "Isaacsimenvs-Cloth-Direct-v0", "sim": "Newton VBD", "num_envs": 768,
        "horizon_length": 32, "max_epochs": 2035, "pinned": True, "domain_randomization": False,
        "keypoint_tolerance": 0.015, "single_variant": True, "slurm_array_job": job,
        "run_dir": str(d), "launcher": "scripts/cluster/engaging_fabric_finetune.sh",
    }
    hydra_cfg = d.parent / ".hydra" / "overrides.yaml"
    if hydra_cfg.exists():
        config["hydra_overrides"] = hydra_cfg.read_text()
    run = wandb.init(
        entity=args.entity, project=args.project, id=f"ftfab_{fabric}_{job}",
        name=f"ftfab_{fabric}", group=f"ftfab_{job}", job_type="finetune",
        tags=["ftfab", fabric, "vbd", "pinned", "no_dr", "from_flat7DRABC_s1"],
        config=config, resume="allow", reinit=True,
    )
    for step in sorted(rows):
        r = rows[step]
        r["epoch"] = step / FRAMES_PER_EPOCH
        run.log(r, step=step)
    run.summary.update(eval_summary(fabric))
    run.summary["training_finished"] = finished
    run.summary["epochs_logged"] = len(rows)
    run.finish()
    print(f"[wandb] {fabric}: {len(rows)} steps from {d} finished={finished}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--fabrics", nargs="*", default=FABRICS)
    ap.add_argument("--entity", default="hailey6171-massachusetts-institute-of-technology")
    ap.add_argument("--project", default="rigid2deform")
    args = ap.parse_args()
    os.environ.setdefault("WANDB_BASE_URL", "https://api.forge.coreweave.com")
    os.environ.setdefault("WANDB_SILENT", "true")
    for f in args.fabrics:
        upload(f, args.job, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
