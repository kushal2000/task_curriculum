"""The top-face point cloud in the real simulator: samples on the plates, goals where the cloth goes.

    .venv/isaacsim/bin/python experiments/nov18/point_cloud/check.py --headless \
        --config <run>/.hydra/config.yaml --checkpoint <ckpt> --num_envs 64 --out_dir <dir> \
        obs.num_points=64 task.mode=mixed task.unfold_prob=0.5 task.release.enabled=false

With task.mode=mixed, fold envs start flat and unfold envs start folded at 180 deg, so each env's
start pose IS the goal of the other task. For every env and every check step this compares the
samples against both goals, computed from the same placed pose:
  * the goal of the env's start shape (flat for fold envs, folded for unfold envs) -> ~0 mm
    (positive control: samples sit on the top face, goals are the exactly folded cloth);
  * the env's own task goal -> large (negative control: the distance is not trivially small).
It also checks the samples lie on the top face (z of each plate's top surface) and stay within the
footprint, and saves a top and side view of one env of each task to <out_dir>/env_*.png.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from isaacsimenvs.scripts.evaluation.eval_cloth import add_common_args, build, launch  # noqa: E402


def main():
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser()
    add_common_args(p, num_envs=64)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--steps", type=int, nargs="+", default=[0, 1, 5, 20])
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    app = launch(args)
    import numpy as np
    import torch

    from isaacsimenvs.tasks.cloth.mdp.geometry import point_cloud

    b = build(args)
    env, inner = b.env, b.inner
    task, sf = inner.task, inner.backend.surface
    assert sf is not None, "run with obs.num_points > 0"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ids = torch.arange(inner.num_envs, device=inner.device)
    unfold = task._unfold_env

    def goals(as_unfold):
        rest = point_cloud.goal_rest(sf.rest, sf.moving, inner.backend.cloth_grid.ply_gap, as_unfold)
        return point_cloud.place(rest, sf.placed_pos_w, sf.placed_quat_w)

    report = {}
    with torch.inference_mode():
        obs, _ = env.reset()
        # The start shape: flat for fold envs (goal_rest with unfold=True), folded for unfold envs.
        start_goal, task_goal = goals(~unfold), task._goal_points.clone()
        assert torch.allclose(task_goal, goals(unfold)), "task goals differ from goal_rest at the placed pose"
        placed_before = sf.placed_pos_w.clone()
        step = 0
        for target in sorted(args.steps):
            while step < target:
                obs, *_ = env.step(torch.zeros_like(b.act(obs)))
                step += 1
            if not torch.equal(sf.placed_pos_w, placed_before):
                print(f"[check] some envs reset before step {step}; their rows are excluded", flush=True)
            same = (sf.placed_pos_w == placed_before).all(dim=1)
            pts = inner.surface_points_w()
            d_start = point_cloud.matched_distance(pts, start_goal)[same]
            d_task = point_cloud.matched_distance(pts, task_goal)[same]
            chamfer_start = point_cloud.chamfer_distance(pts, start_goal)[same]
            per_point = (pts - start_goal).norm(dim=-1)[same]
            row = {}
            for name, m in (("fold envs (start flat)", ~unfold[same]), ("unfold envs (start folded)", unfold[same])):
                if not m.any():
                    continue
                row[name] = {
                    "envs": int(m.sum()),
                    "start-shape goal matched mm (median/max)": [round(float(d_start[m].median()) * 1e3, 2),
                                                               round(float(d_start[m].max()) * 1e3, 2)],
                    "start-shape goal chamfer mm (median)": round(float(chamfer_start[m].median()) * 1e3, 2),
                    "worst single point mm": round(float(per_point[m].max()) * 1e3, 2),
                    "task goal matched mm (min/median)": [round(float(d_task[m].min()) * 1e3, 1),
                                                         round(float(d_task[m].median()) * 1e3, 1)],
                }
            report[f"step {step}"] = row
            print(f"[check] step {step}: {json.dumps(row)}", flush=True)
            if step == 0:
                # Samples on the top face: z of the plate's top surface at the start pose.
                dims = inner.backend._dims
                table_top = sf.placed_pos_w[:, 2] - dims[:, 2] / 2
                flat_z = (table_top + dims[:, 2]).unsqueeze(1)
                off = (pts[..., 2] - flat_z).abs()
                report["step 0 fold envs |z - top face| mm (max)"] = round(float(off[~unfold].max()) * 1e3, 2)
                                snapshot = {k: v.cpu().numpy() for k, v in
                            (("pts", pts), ("start", start_goal), ("task", task_goal),
                             ("moving", sf.moving), ("unfold", unfold))}
        (args.out_dir / "report.json").write_text(json.dumps(report, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = snapshot
    for label, e in (("fold", int(np.flatnonzero(~s["unfold"])[0])), ("unfold", int(np.flatnonzero(s["unfold"])[0]))):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        c = s["pts"][e].mean(axis=0)
        for ax, (i, j, title) in zip(axes, ((0, 1, "top view (x, y)"), (0, 2, "side view (x, z)"))):
            colors = np.where(s["moving"][e], "#e63ca0", "#3c78e6")
            ax.scatter(s["pts"][e][:, i] - c[i], s["pts"][e][:, j] - c[j], c=colors, s=18, label="samples (step 0)")
            ax.scatter(s["task"][e][:, i] - c[i], s["task"][e][:, j] - c[j], facecolors="none",
                       edgecolors="#28c85a", s=40, label="task goal")
            for a, g in zip(s["pts"][e], s["task"][e]):
                ax.plot([a[i] - c[i], g[i] - c[i]], [a[j] - c[j], g[j] - c[j]], color="0.8", lw=0.5, zorder=0)
            ax.set_title(title)
            if j == 1:
                ax.set_aspect("equal")  # the side view keeps a readable z scale
        axes[0].legend(loc="upper right", fontsize=8)
        fig.suptitle(f"env {e}: {b.designs[e]}, {label} task (blue stationary, pink moving; lines = matching)")
        fig.tight_layout()
        fig.savefig(args.out_dir / f"env_{e}_{label}.png", dpi=110)
    print(f"[check] wrote {args.out_dir}", flush=True)
    env.close()
    if app is not None:
        app.close()


if __name__ == "__main__":
    main()
