"""Mean goals per episode, sampled one episode per environment.

    python isaacsimenvs/eval/episodes.py --task Isaacsimenvs-Play-Direct-v0 --num_envs 64

The metric is the number of sequential goals a policy reaches in one episode. The episode is the
honest unit here because the deadline **resets on every goal hit**
(``termination_utils.py:51``), so an episode lasts exactly as long as the policy keeps
succeeding: a failed episode ends within ``episode_length`` steps while a good one runs for
thousands.

That is also why this samples **the first completed episode from each environment**, rather than
the first N completed episodes overall. The latter is biased -- failures finish soonest, so they
would be resampled repeatedly while the successful episodes were still in progress, and the mean
would be dragged toward the failures.

Everything that defines the protocol is a flag with a recorded default, and every run writes the
resolved values into its JSON, so two numbers produced by this file are comparable by
construction.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

from isaacsimenvs.newton.contact_guard import (
    assert_no_buffer_overflow,
    capture_fd_output,
    raise_if_overflowed,
)
from isaacsimenvs.eval.protocol import disable_randomization, use_single_object_variant

DEFAULT_CHECKPOINT = "/share/portal/kk837/simtoolreal/pretrained_policy/model.pth"
DEFAULT_POLICY_CFG = "/share/portal/kk837/simtoolreal/pretrained_policy/config.yaml"

# Termination causes published by `play/utils/logging_utils.py`. Reported as a histogram: two
# backends can agree on the mean while failing for entirely different reasons, and that is not
# parity.
REASONS = ("fall", "max_successes", "hand_far", "timeout")

# iiwa14 fully extended plus the hand. An object above this is not being held -- it has been
# ejected by a solver blow-up. The distinction matters because `_lifted_object` latches on
# `z > threshold` with no upper bound (reward_utils.lifting_reward), so an ejection sets it
# permanently True and reads as the hoped-for result. Deformable objects are where this bites.
REACH_M = 1.40


def _summarise(goals: list[int], lengths: list[int]) -> dict:
    n = len(goals)
    if n == 0:
        return {"n": 0}
    mean = statistics.fmean(goals)
    sd = statistics.pstdev(goals) if n > 1 else 0.0
    return {
        "n": n,
        "mean_goals_per_episode": round(mean, 4),
        "sem": round(sd / (n**0.5), 4) if n > 1 else None,
        "sd": round(sd, 4),
        "min": min(goals),
        "max": max(goals),
        "mean_episode_length": round(statistics.fmean(lengths), 1),
    }


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Mean goals per episode, one episode per env.")
    parser.add_argument("--task", default="Isaacsimenvs-Play-Direct-v0")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=12000,
        help="Safety bound on policy steps. Episodes still running at this point are reported "
        "as censored rather than truncated into the mean.",
    )
    parser.add_argument(
        "--num_assets_per_type",
        type=int,
        default=4,
        help="Procedural object variants per handle+head type; 6 types, so the default is a "
        "24-object pool. This is the reference protocol's value (isaac_newton `run.py episodes` "
        "defaults to 4, and its tables call 6x4 the 'full pool'), and it is part of the task "
        "definition rather than a performance knob: `generate_handle_head_urdfs` shuffles the "
        "whole pool under a fixed seed, so changing the count changes *which* objects the envs "
        "get, not just how many exist. cfg/task/Play.yaml ships 100 for training.",
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--policy_config", default=DEFAULT_POLICY_CFG)
    parser.add_argument(
        "--success_tolerance",
        type=float,
        default=0.01,
        help="Pins `termination.eval_success_tolerance`. Multiplied by reward.keypoint_scale "
        "(1.5) inside the env, so 0.01 is a 1.5 cm keypoint threshold -- the curriculum floor "
        "the checkpoint was trained down to. Unpinned, the criterion drifts with the tolerance "
        "curriculum and the number depends on how long the process has been running.",
    )
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Keep domain randomisation on (default: off; see disable_randomization).",
    )
    parser.add_argument(
        "--single_variant",
        action="store_true",
        help="Give every env the same object. Required for Newton (SolverMuJoCo rejects "
        "worlds whose collision shapes differ in type), and the matched setting to use on "
        "PhysX so the two are comparable.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rl_device", default="cuda:0")
    parser.add_argument("--sim_device", default="cuda:0")
    parser.add_argument("--out", default=None, help="JSON output path.")
    AppLauncher.add_app_launcher_args(parser)
    parser.add_argument(
        "--allow_overflow",
        action="store_true",
        help="skip the contact-buffer overflow check (results will be measured on dropped contacts)",
    )
    args_cli, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args

    # Kit is required by the PhysX backend and impossible for the Newton one, which runs kit-less
    # and has no Isaac Sim in its venv. Rather than a flag the caller has to remember to match to
    # the task, ask: AppLauncher raises ImportError precisely when the runtime is absent.
    #
    # Only that specific failure is tolerated. If Isaac Sim *is* installed, any launch error is a
    # real problem and propagates -- silently falling back to kit-less there would produce a run
    # that looks fine and used a different asset pipeline.
    try:
        app = AppLauncher(args_cli).app
    except ImportError as exc:
        app = None
        print(f"[episodes] running kit-less (no Isaac Sim runtime): {exc}", flush=True)

    import gymnasium as gym
    import torch

    import isaacsimenvs  # noqa: F401  gym.register side effects
    from isaacsimenvs.eval.player import PretrainedPlayer
    from isaacsimenvs.utils.hydra_utils import hydra_task_config_with_yaml

    @hydra_task_config_with_yaml(args_cli.task, "")
    def run(env_cfg, agent_cfg) -> None:
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.assets.num_assets_per_type = args_cli.num_assets_per_type
        env_cfg.sim.device = args_cli.sim_device
        env_cfg.seed = args_cli.seed
        env_cfg.termination.eval_success_tolerance = args_cli.success_tolerance
        if not args_cli.randomize:
            disable_randomization(env_cfg)
        if args_cli.single_variant:
            use_single_object_variant()

        env = gym.make(args_cli.task, cfg=env_cfg)
        inner = env.unwrapped
        device = inner.device
        num_envs = inner.num_envs

        player = PretrainedPlayer(
            config_path=args_cli.policy_config,
            checkpoint_path=args_cli.checkpoint,
            num_envs=num_envs,
            device=args_cli.rl_device,
            num_observations=int(inner.cfg.observation_space),
            num_actions=int(inner.cfg.action_space),
        )

        # Reset, then one zero-action tick, matching the reference runner's timing. Getting this
        # wrong shifts the whole rollout by a step.
        obs, _ = env.reset()

        # Fail before measuring, not after. A contact-buffer overflow drops contacts silently --
        # the engine prints from a kernel straight to fd 1, which never reaches sys.stdout and
        # was greppable away by every result-line filter used here.
        if not args_cli.allow_overflow:
            assert_no_buffer_overflow(env)
            env.reset()

        obs, _ = env.reset()
        obs, *_ = env.step(torch.zeros((num_envs, inner.cfg.action_space), device=device))

        goals = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        length = torch.zeros(num_envs, dtype=torch.long, device=device)
        steps_alive = torch.zeros(num_envs, dtype=torch.long, device=device)
        lifted = torch.zeros(num_envs, dtype=torch.bool, device=device)
        max_obj_z = torch.full((num_envs,), -1e9, device=device)
        contributed = torch.zeros(num_envs, dtype=torch.bool, device=device)
        reason = {name: torch.zeros(num_envs, dtype=torch.bool, device=device) for name in REASONS}

        capture = None
        if not args_cli.allow_overflow:
            capture_cm = capture_fd_output()
            capture = capture_cm.__enter__()

        step = 0
        while step < args_cli.max_steps and not bool(contributed.all()):
            action = player.get_action(obs["policy"], deterministic=True)
            obs, _rew, terminated, truncated, extras = env.step(action.to(device))

            # Only while the env is still on its first episode. An env that finished early keeps
            # stepping until the others catch up, and counting those later lifts would report a
            # statistic about a set of episodes that no other number here describes.
            lifted |= inner._lifted_object.bool() & ~contributed
            obj_z = (inner.object.data.root_pos_w - inner.scene.env_origins)[:, 2]
            max_obj_z = torch.where(~contributed, torch.maximum(max_obj_z, obj_z), max_obj_z)
            steps_alive += (~contributed).long()

            done = (terminated.bool() | truncated.bool()) & ~contributed
            if bool(done.any()):
                idx = done.nonzero(as_tuple=True)[0]
                final = extras["episode_final"]
                # Pre-reset snapshot: log_step_metrics runs inside _get_rewards, which
                # DirectRLEnv.step calls before _reset_idx zeroes _successes.
                goals[idx] = final["successes"][idx].long()
                length[idx] = steps_alive[idx]
                for name in REASONS:
                    reason[name][idx] = final[f"done_{name}"][idx].bool()
                contributed |= done

            # An episode that has ended carries no hidden state worth keeping; the policy never
            # saw a rollout stitched across a reset.
            ended = terminated.bool() | truncated.bool()
            if bool(ended.any()):
                player.reset_rnn(ended.nonzero(as_tuple=True)[0])

            step += 1

            # The warm-up check steps with zero actions and cannot see contacts that only appear
            # once the hand closes on the object: a 24-segment cable passed it, then overflowed for
            # hundreds of steps. So poll the captured fd during the rollout too.
            if capture is not None and step % 200 == 0:
                raise_if_overflowed(capture(), num_envs, f"the rollout (step {step})")
            if step % 500 == 0:
                print(
                    f"[episodes] step {step}: {int(contributed.sum())}/{num_envs} envs done",
                    flush=True,
                )

        ejected = max_obj_z > REACH_M
        if capture is not None:
            captured = capture()
            capture_cm.__exit__(None, None, None)
            raise_if_overflowed(captured, num_envs, "the rollout")
            sys.stdout.write(captured)

        done_mask = contributed.cpu()
        goals_l = goals.cpu()[done_mask].tolist()
        length_l = length.cpu()[done_mask].tolist()
        censored = int(num_envs - done_mask.sum())

        # A strict lower bound: count every censored env at the goals it had banked when the
        # budget ran out. Reported alongside, never instead of, the completed-only mean -- if the
        # two disagree materially, the budget is too small and the headline is not trustworthy.
        partial = inner._successes.cpu()
        lb_goals = goals_l + partial[~done_mask].tolist()

        result = {
            "task": args_cli.task,
            "num_envs": num_envs,
            "max_steps": args_cli.max_steps,
            "steps_run": step,
            "checkpoint": args_cli.checkpoint,
            "success_tolerance": args_cli.success_tolerance,
            "keypoint_scale": float(inner.cfg.reward.keypoint_scale),
            "randomization": bool(args_cli.randomize),
            "single_variant": bool(args_cli.single_variant),
            "seed": args_cli.seed,
            "num_assets_per_type": int(inner.cfg.assets.num_assets_per_type),
            "handle_head_types": list(inner.cfg.assets.handle_head_types),
            "modify_asset_frictions": bool(inner.cfg.assets.modify_asset_frictions),
            "max_consecutive_successes": int(inner.cfg.termination.max_consecutive_successes),
            "completed": _summarise(goals_l, length_l),
            "lower_bound_including_censored": _summarise(lb_goals, length_l or [0]),
            "censored": censored,
            "lift_fraction": round(float((lifted & ~ejected).float().mean()), 4),
            "lift_fraction_unguarded": round(float(lifted.float().mean()), 4),
            "ejected": int(ejected.sum()),
            "max_obj_z": [round(float(v), 3) for v in max_obj_z.cpu()],
            # Flags are not mutually exclusive -- an env can trip several on its final step,
            # so these count reasons, not episodes, and may sum above n.
            "termination_reasons": {
                name: int(reason[name].cpu()[done_mask].sum()) for name in REASONS
            },
            "per_env_goals": goals.cpu().tolist(),
            "per_env_length": length.cpu().tolist(),
        }

        summary = result["completed"]
        print("\n" + "=" * 72)
        print(f"{args_cli.task}  |  {num_envs} envs  |  first episode per env")
        print(
            f"goals/episode = {summary.get('mean_goals_per_episode')} "
            f"+/- {summary.get('sem')}   (n={summary.get('n')}, censored={censored})"
        )
        print(f"lift fraction = {result['lift_fraction']}"
              + (f"   (unguarded {result['lift_fraction_unguarded']}, "
                 f"{result['ejected']} ejected above {REACH_M} m)" if int(ejected.sum()) else ""))
        print(f"termination   = {result['termination_reasons']}")
        print(f"mean ep length= {summary.get('mean_episode_length')} steps")
        if censored:
            lb = result["lower_bound_including_censored"]
            print(
                f"NOTE {censored} env(s) censored at the step budget; lower bound counting "
                f"their partial goals = {lb.get('mean_goals_per_episode')}. Raise --max_steps."
            )
        cap_hits = result["termination_reasons"]["max_successes"]
        if cap_hits > 0.1 * max(summary.get("n", 1), 1):
            print(
                f"WARNING {cap_hits} episodes ended on max_consecutive_successes="
                f"{result['max_consecutive_successes']}: the mean is right-censored and "
                "cross-backend comparison understates any difference."
            )
        print("=" * 72 + "\n")

        if args_cli.out:
            out = Path(args_cli.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, indent=2))
            print(f"[episodes] wrote {out}")

        env.close()

    run()

    # Kit shutdown can hang; force-exit rather than wait for a clean teardown.
    if app is not None:
        del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
