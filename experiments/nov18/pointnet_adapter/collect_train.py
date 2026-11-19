"""Teach a point-cloud encoder to produce the friction prior's cloth observation.

    .venv/isaacsim/bin/python experiments/nov18/pointnet_adapter/collect_train.py --headless \
        --config <prior run>/.hydra/config.yaml --checkpoint <prior .pth> --num_envs 4096 \
        --out_dir <dir> --stochastic env.obs.num_points=64 'env.obs.state_list=[...point actor fields...]'

1. Roll the prior out on its own states. The env's critic observation is set to the point-cloud
   actor's input (robot 103 + cloth points + goals), so every step yields the encoder's input and,
   from the prior's own policy observation, its target: the 31 cloth values (object_rot,
   keypoints_rel_palm, keypoints_rel_goal, object_scales) normalised by the prior's running stats.
2. Train ``AdapterEncoder`` on it, held out by env. The loss is matched in the prior's input
   embedding: ||W (f_hat - f)||^2 with W the prior LSTM's input weights for the 31 cloth columns, so
   errors count by how much they move what the LSTM sees (plus a small plain MSE term).
3. Save the encoder and a report: per-field errors in physical units and the embedding error.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from isaacsimenvs.scripts.evaluation.eval_cloth import add_common_args, build, launch  # noqa: E402

# The prior's observation (134) as robot / cloth runs, with each cloth field's slice.
PRIOR_FIELDS = (("joint_pos", "robot"), ("joint_vel", "robot"), ("prev_action_targets", "robot"),
                ("palm_pos", "robot"), ("palm_rot", "robot"), ("object_rot", "object"),
                ("fingertip_pos_rel_palm", "robot"), ("keypoints_rel_palm", "object"),
                ("keypoints_rel_goal", "object"), ("object_scales", "object"))


def layout(field_size):
    """``segments`` for the adapter network, the prior's cloth columns, and named cloth slices."""
    segments, cloth_cols, named, at, o = [], [], {}, 0, 0
    for name, kind in PRIOR_FIELDS:
        n = field_size(name)
        if segments and segments[-1][0] == kind:
            segments[-1][1] += n
        else:
            segments.append([kind, n])
        if kind == "object":
            cloth_cols += list(range(at, at + n))
            named[name] = (o, o + n)
            o += n
        at += n
    return segments, cloth_cols, named


def main():
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser()
    add_common_args(p, num_envs=4096)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--steps", type=int, default=1200, help="policy steps to roll out")
    p.add_argument("--keep_every", type=int, default=4, help="keep every k-th step")
    p.add_argument("--train_steps", type=int, default=30000)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--mse_weight", type=float, default=0.1)
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    app = launch(args)
    import torch

    from isaacsimenvs.scripts.training.pointnet import AdapterEncoder
    from isaacsimenvs.tasks.cloth.mdp.observations import FIELD_SIZES, PER_JOINT_FIELDS

    args.out_dir.mkdir(parents=True, exist_ok=True)
    b = build(args)
    env, inner = b.env, b.inner
    P = int(inner.cfg.obs.num_points)
    nj = inner.robot_spec.num_joints
    size = lambda f: nj if f in PER_JOINT_FIELDS else FIELD_SIZES[f]
    segments, cloth_cols, named = layout(size)
    assert sum(n for _, n in segments) == int(inner.cfg.observation_space), "prior layout mismatch"
    actor_fields = list(inner.cfg.obs.state_list)
    assert actor_fields[-2:] == ["cloth_points", "goal_points"], actor_fields
    robot_dim = sum(n for k, n in segments if k == "robot")
    palm_at = sum(size(f) for f in actor_fields[:actor_fields.index("palm_pos")])
    model = b.player.player.model
    rms = model.running_mean_std
    w_cloth = model.a2c_network.rnn.rnn.weight_ih_l0[:, cloth_cols].detach().float()  # (4H, 31)
    cols = torch.as_tensor(cloth_cols, device=inner.device)
    print(f"[adapter] segments {segments}, palm at {palm_at}, {P} points, cloth cols {len(cloth_cols)}", flush=True)

    # ---------------------------------------------------------------- 1. collect
    xs, ys, env_of = [], [], []
    t0 = time.time()
    with torch.inference_mode():
        obs, _ = env.reset()
        b.player.reset_rnn()
        for step in range(args.steps):
            if step % args.keep_every == 0 and step >= 10:
                x = obs["critic"]
                y = rms(obs["policy"].clamp(-10, 10))[:, cols]  # eval mode: no stat update
                xs.append(x.clone())
                ys.append(y.clone())
                env_of.append(torch.arange(x.shape[0], device=x.device))
            obs, _, terminated, truncated, _ = env.step(b.act(obs))
            done = (terminated | truncated).nonzero(as_tuple=True)[0]
            if done.numel():
                b.player.reset_rnn(done)
    # Plain tensors (the rollout ran under inference mode, whose tensors cannot enter autograd).
    X, Y, E = torch.cat(xs).clone(), torch.cat(ys).clone(), torch.cat(env_of).clone()
    del xs, ys
    assert X.shape[1] == robot_dim + 6 * P, (X.shape, robot_dim, P)
    print(f"[adapter] collected {len(X)} samples in {time.time() - t0:.0f} s", flush=True)

    # ---------------------------------------------------------------- 2. train
    torch.manual_seed(0)
    val = E >= int(0.9 * inner.num_envs)
    tr_idx, va_idx = (~val).nonzero(as_tuple=True)[0], val.nonzero(as_tuple=True)[0]
    enc = AdapterEncoder(len(cloth_cols)).to(inner.device)
    opt = torch.optim.Adam(enc.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.train_steps)
    m = w_cloth.T @ w_cloth
    m = m / (torch.trace(m) / m.shape[0])  # same overall scale as a plain MSE

    def run(idx):
        x = X[idx]
        cloud = x[:, robot_dim:].reshape(len(idx), 2, P, 3)
        return enc(cloud[:, 0], cloud[:, 1], x[:, palm_at:palm_at + 3])

    def losses(d):
        emb = ((d @ m) * d).sum(-1).mean() / d.shape[1]
        return emb, (d ** 2).mean()

    for it in range(args.train_steps):
        idx = tr_idx[torch.randint(len(tr_idx), (args.batch,), device=X.device)]
        emb, mse = losses(run(idx) - Y[idx])
        loss = emb + args.mse_weight * mse
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if it % 2000 == 0 or it == args.train_steps - 1:
            with torch.no_grad():
                vi = va_idx[:65536]
                v_emb, v_mse = losses(run(vi) - Y[vi])
            print(f"[adapter] it {it}: train emb {emb.item():.4f} mse {mse.item():.4f} | "
                  f"val emb {v_emb.item():.4f} mse {v_mse.item():.4f}", flush=True)

    # ---------------------------------------------------------------- 3. report
    enc.eval()
    with torch.no_grad():
        preds = torch.cat([run(va_idx[i:i + 65536]) for i in range(0, len(va_idx), 65536)])
    target = Y[va_idx]
    std = torch.sqrt(rms.running_var[cols].float() + rms.epsilon)
    raw_err = (preds - target) * std  # back to physical units
    rel = (raw_err.abs())
    report = {"samples": {"train": len(tr_idx), "val": len(va_idx)}, "segments": segments,
              "palm_slice": [palm_at, palm_at + 3], "num_points": P}
    for name, (a, c) in named.items():
        e = raw_err[:, a:c]
        if name in ("keypoints_rel_palm", "keypoints_rel_goal"):
            per_kp = e.reshape(len(e), -1, 3).norm(dim=-1)
            report[name + " error mm (median/p90)"] = [round(float(per_kp.median()) * 1e3, 2),
                                                       round(float(per_kp.quantile(.9)) * 1e3, 2)]
        elif name == "object_rot":
            mean = rms.running_mean[cols][a:c].float()
            q_hat = torch.nn.functional.normalize(preds[:, a:c] * std[a:c] + mean, dim=-1)
            q = torch.nn.functional.normalize(target[:, a:c] * std[a:c] + mean, dim=-1)
            ang = 2 * torch.acos((q_hat * q).sum(-1).abs().clamp(max=1.0)) * 180 / math.pi
            report[name + " error deg (median/p90)"] = [round(float(ang.median()), 2), round(float(ang.quantile(.9)), 2)]
        else:
            report[name + " abs error (median)"] = round(float(rel[:, a:c].median()), 5)
    emb_err = (preds - target) @ w_cloth.T
    emb_ref = target @ w_cloth.T
    report["embedding relative error (median)"] = round(float((emb_err.norm(dim=-1) / emb_ref.norm(dim=-1).clamp(min=1e-6)).median()), 4)
    report["normalised mse (val)"] = round(float(((preds - target) ** 2).mean()), 5)
    print("[adapter] " + json.dumps(report), flush=True)
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2))
    torch.save({"state_dict": enc.state_dict(), "segments": segments, "palm_slice": [palm_at, palm_at + 3],
                "num_points": P, "cloth_cols": cloth_cols, "robot_cols": [i for i in range(sum(n for _, n in segments)) if i not in cloth_cols],
                "actor_fields": actor_fields}, args.out_dir / "encoder.pt")
    print(f"[adapter] wrote {args.out_dir}", flush=True)
    env.close()
    if app is not None:
        app.close()


if __name__ == "__main__":
    main()
