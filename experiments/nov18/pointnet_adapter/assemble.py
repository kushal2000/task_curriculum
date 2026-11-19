"""Combine the friction prior and a trained adapter encoder into one point-cloud checkpoint.

    python experiments/nov18/pointnet_adapter/assemble.py --prior <prior.pth> \
        --encoder <dir>/encoder.pt --out <dir>/adapter_prior.pth

Actor (``pointnet_adapter_actor_critic``): every prior weight unchanged, plus
``a2c_network.adapter_encoder.*``. Its input statistics are rebuilt for the new observation
(robot inputs + 6P point values): robot slots carry the prior's statistics, point slots are
identity (the model does not normalise them anyway).

Critic: the prior's central value net with two zero-weight input columns for ``point_dist`` and
``closest_point_dist`` appended to its state, so it starts out computing exactly the prior's value.
"""
import argparse
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prior", type=Path, required=True)
    p.add_argument("--encoder", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--critic_extra", type=int, default=2, help="state columns appended for the critic")
    args = p.parse_args()

    prior = torch.load(args.prior, map_location="cpu", weights_only=False)[0]
    enc = torch.load(args.encoder, map_location="cpu", weights_only=False)
    model = dict(prior["model"])
    robot_cols = torch.as_tensor(enc["robot_cols"])
    num_points = int(enc["num_points"])

    # Actor input statistics: robot slots from the prior, then 6P identity point slots.
    mean, var = model["running_mean_std.running_mean"], model["running_mean_std.running_var"]
    model["running_mean_std.running_mean"] = torch.cat([mean[robot_cols], torch.zeros(6 * num_points, dtype=mean.dtype)])
    model["running_mean_std.running_var"] = torch.cat([var[robot_cols], torch.ones(6 * num_points, dtype=var.dtype)])
    for k, v in enc["state_dict"].items():
        model[f"a2c_network.adapter_encoder.{k}"] = v

    # Critic: insert zero columns for the new state fields, before the SAPG embedding columns.
    critic = dict(prior["assymetric_vf_nets"])
    state_dim = critic["model.running_mean_std.running_mean"].shape[0]
    extra = args.critic_extra
    for key, fill in (("model.running_mean_std.running_mean", 0.0), ("model.running_mean_std.running_var", 1.0)):
        critic[key] = torch.cat([critic[key], torch.full((extra,), fill, dtype=critic[key].dtype)])
    w = critic["model.a2c_network.actor_mlp.0.weight"]
    critic["model.a2c_network.actor_mlp.0.weight"] = torch.cat(
        [w[:, :state_dim], torch.zeros(w.shape[0], extra, dtype=w.dtype), w[:, state_dim:]], dim=1)

    out = {0: {"model": model, "assymetric_vf_nets": critic, "epoch": 0, "frame": 0}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"[assemble] actor input {model['running_mean_std.running_mean'].shape[0]} "
          f"(robot {len(robot_cols)} + points {6 * num_points}), critic state {state_dim} -> {state_dim + extra}; "
          f"wrote {args.out}")


if __name__ == "__main__":
    main()
