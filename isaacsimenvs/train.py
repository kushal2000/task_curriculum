"""Train an isaacsimenvs task with our vendored rl_games.

Pipeline:
    argparse (--task, --agent, AppLauncher, wandb/video flags)
        ↓
    AppLauncher (boots Kit; must precede any isaaclab.* import)
        ↓
    @hydra_task_config_with_yaml  (configclass defaults ← task YAML overlay ← Hydra CLI)
        ↓
    gym.make(task_id, cfg=env_cfg)  → DirectRLEnv wrapped by gym.Wrapper
        ↓
    isaaclab_rl.RlGamesVecEnvWrapper (via register_rlgames_env — clipping,
                                      device bridging, obs-group routing)
        ↓
    rl_games.torch_runner.Runner (PPO / SAPG — both live in ./rl_games/)

CLI shape:
    python isaacsimenvs/train.py \
        --task Isaacsimenvs-Play-Direct-v0 \
        --agent rl_games_sapg_cfg_entry_point \   # or rl_games_cfg_entry_point
        --headless --capture_viewer \
        --wandb_activate --wandb_project X --wandb_name Y \
        env.scene.num_envs=4096 \
        agent.params.config.max_epochs=200 \
        agent.params.config.minibatch_size=16384 \
        agent.params.seed=42
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import math
import os
import sys

# Which interactive-viewer wrapper a task needs. Both render into the same three.js page
# (utils/interactive_viewer/index.template.html) and log it with wandb.Html; they differ
# only in what geometry they hand the browser.
#
#   Play / BottleFlip  the Kuka + SHARPA robot, whose STL meshes the browser must fetch
#                      from GitHub raw (hence --capture_viewer_github_raw_base)
#   MultiLinkCartpole  primitives only, so the URDF is embedded and the page needs no
#                      network at all
#
# Matched by substring so the registered task ids stay the single source of truth.
_POSE_VIEWERS = {
    "MultiLinkCartpole": "isaacsimenvs.tasks.multilink_cartpole.pose_viewer:CartpolePoseViewerWrapper",
    "BottleFlip": "isaacsimenvs.tasks.play.pose_viewer:PlayPoseViewerWrapper",
    # Cloth before Play: the cloth viewer adds the sheet and its fold target as deformable
    # channels, which a rigid-pose viewer cannot represent. Substring matching is first-wins, and
    # the cloth task id does not contain "Play", but keep it above for clarity.
    "Cloth": "isaacsimenvs.tasks.cloth.pose_viewer:ClothPoseViewerWrapper",
    "Play": "isaacsimenvs.tasks.play.pose_viewer:PlayPoseViewerWrapper",
}


def _pose_viewer_target(task_id: str) -> str | None:
    for name, target in _POSE_VIEWERS.items():
        if name in task_id:
            return target
    return None


def _arm_hang_watchdog() -> None:
    """Dump every thread's Python stack if the process stops making progress.

    rl_games' distributed path builds its own gloo group with a TWO HOUR timeout
    (`a2c_common.py:110`), so any failure in that region presents as a silent freeze carrying no
    information about its cause -- which is why the 4-GPU hang resisted several rounds of inference
    from the symptom alone. Set CLOTH_HANG_WATCHDOG=<seconds> to turn the freeze into a traceback.
    """
    import faulthandler

    secs = os.getenv("CLOTH_HANG_WATCHDOG")
    if not secs:
        return
    faulthandler.enable()
    faulthandler.dump_traceback_later(float(secs), repeat=True, exit=False)
    print(f"[train] hang watchdog armed: stacks every {secs}s", flush=True)


def main() -> None:
    _arm_hang_watchdog()

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Train isaacsimenvs task via rl_games.")
    # --- Task/agent selection ---
    parser.add_argument("--task", required=True, help="Gym task id, e.g. Isaacsimenvs-Cartpole-Direct-v0")
    parser.add_argument(
        "--agent",
        default="rl_games_cfg_entry_point",
        help="Key in gym.register kwargs for the rl_games YAML. "
        "Use rl_games_sapg_cfg_entry_point for SAPG.",
    )
    # --- Runtime toggles ---
    parser.add_argument("--test", action="store_true", help="Run inference (player) instead of training")
    parser.add_argument("--checkpoint", default=None, help="Path to .pth to restore")
    parser.add_argument(
        "--checkpoint_load_mode",
        choices=("resume", "weights"),
        default="resume",
        help="resume restores optimizer/rollout/env state; weights starts fresh from model weights.",
    )
    parser.add_argument(
        "--single_variant",
        action="store_true",
        help="Give every env the same object. REQUIRED on Newton: SolverMuJoCo rejects worlds whose "
        "collision shapes differ in type, and the procedural tool pool mixes boxes and capsules.",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Multi-GPU training under torchrun. Binds this rank to cuda:$LOCAL_RANK and sets "
        "agent.params.config.multi_gpu, without which every rank lands on cuda:0.",
    )
    parser.add_argument("--rl_device", default="cuda:0")
    parser.add_argument("--sim_device", default="cuda:0")
    # --- Video ---
    parser.add_argument("--capture_video", action="store_true", help="Attach recording camera; implies --enable_cameras")
    parser.add_argument("--video_interval", type=int, default=10)
    parser.add_argument("--video_capture_frames", type=int, default=120)
    parser.add_argument("--video_fps", type=int, default=30)
    # --- Pose-only interactive HTML viewer (no cameras / no renderer) ---
    parser.add_argument(
        "--capture_viewer",
        action="store_true",
        help="Write periodic pose-only interactive HTML viewers; does not enable Isaac cameras.",
    )
    parser.add_argument("--capture_viewer_len", type=int, default=600)
    parser.add_argument("--capture_viewer_interval", type=int, default=6000)
    parser.add_argument("--capture_viewer_env_id", type=int, default=0)
    parser.add_argument("--capture_viewer_wandb_key", default="interactive_viewer")
    parser.add_argument(
        "--capture_viewer_github_raw_base",
        default="",
        help="GitHub raw base URL used by the browser to fetch robot URDF meshes.",
    )
    parser.add_argument(
        "--capture_viewer_url_check",
        choices=("skip", "warn", "error"),
        default="skip",
        help="Whether to HEAD-check the robot URDF URL before writing viewer HTML.",
    )
    # --- wandb ---
    parser.add_argument("--wandb_activate", action="store_true")
    parser.add_argument("--wandb_project", default="isaacsimenvs")
    parser.add_argument("--wandb_group", default="")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--wandb_name", default="", help="Defaults to agent_cfg.params.config.name")
    parser.add_argument("--wandb_tags", nargs="*", default=[])
    parser.add_argument("--wandb_notes", default="")
    parser.add_argument("--wandb_logcode_dir", default="")
    # --- AppLauncher flags (--headless, --enable_cameras, etc.) ---
    AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args()

    # Recording a video requires cameras even if user forgot --enable_cameras.
    if args_cli.capture_video:
        args_cli.enable_cameras = True

    # Hand the leftover key=value tokens to Hydra via sys.argv.
    sys.argv = [sys.argv[0]] + hydra_args

    # Kit is required by the PhysX backend and impossible for the Newton one, which runs kit-less
    # in a venv with no Isaac Sim. Rather than a flag the caller must remember to match to the task,
    # ask: AppLauncher raises ImportError precisely when the runtime is absent. Only that specific
    # failure is tolerated -- if Isaac Sim IS installed, any launch error is real and propagates,
    # because silently falling back would train against a different asset pipeline than intended.
    # Mirrors `eval/episodes.py:149`.
    try:
        app = AppLauncher(args_cli).app
    except ImportError as exc:
        app = None
        print(f"[train] running kit-less (no Isaac Sim runtime): {exc}", flush=True)

    # 2. Safe to import isaaclab-backed modules now.
    import gymnasium as gym
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import OmegaConf
    from rl_games.torch_runner import Runner

    import isaacsimenvs  # noqa: F401  triggers gym.register side effects
    from isaacsimenvs.utils.hydra_utils import hydra_task_config_with_yaml
    from isaacsimenvs.utils.rlgames_utils import (
        EnvStatsAlgoObserver,
        MultiObserver,
        register_rlgames_env,
    )

    @hydra_task_config_with_yaml(args_cli.task, args_cli.agent)
    def run(env_cfg, agent_cfg: dict) -> None:
        hydra_run_dir = HydraConfig.get().runtime.output_dir

        # sim_device CLI flag still wins — it's a launcher-level concern, not
        # something we expect in the task YAML.
        # Under torchrun each rank MUST bind its own device. torchrun sets LOCAL_RANK but not
        # CUDA_VISIBLE_DEVICES, and both --rl_device and --sim_device default to cuda:0 -- so
        # without this every rank puts its whole scene on GPU 0 and dies in Warp's allocator,
        # which presents as an out-of-memory bug rather than a device-binding one. The same trap
        # was hit once already in the throughput benchmark (docs/phase4_cable_env.md:664).
        import torch

        from isaacsimenvs.eval.protocol import use_single_object_variant

        if args_cli.single_variant:
            use_single_object_variant()

        # Only rank 0 may own side-effecting, singleton outputs: wandb runs and viewer HTML.
        # Without this all four ranks create the SAME wandb run id and the server rejects the
        # duplicates with `409 ... Duplicate entry for key 'runs.PRIMARY'`, while four copies
        # of every capture race over the same files.
        is_main_rank = int(os.getenv("RANK", "0")) == 0

        rl_device = args_cli.rl_device
        if args_cli.distributed:
            local_rank = int(os.getenv("LOCAL_RANK", "0"))
            rl_device = f"cuda:{local_rank}"
            env_cfg.sim.device = rl_device
            torch.cuda.set_device(local_rank)
            # Offset the seed per rank, or every rank explores identically and the extra GPUs buy
            # nothing but a larger batch. Mirrors IsaacLab's own train_rl_games.py:122-126.
            agent_cfg["params"]["seed"] += int(os.getenv("RANK", "0"))
            agent_cfg["params"]["config"]["multi_gpu"] = True

            # Initialise the process group HERE, using torchrun's own rendezvous (env:// reads
            # MASTER_ADDR/MASTER_PORT/RANK/WORLD_SIZE). rl_games would otherwise build its own
            # `tcp://127.0.0.1:<hashed port>` group, which conflicts with torchrun's elastic agent
            # store and hangs in `_create_c10d_store` for two hours without printing anything.
            # `a2c_common.py:110` is guarded with `if not dist.is_initialized()`, so it defers.
            import torch.distributed as dist

            if not dist.is_initialized():
                dist.init_process_group("gloo", timeout=timedelta(minutes=30))
                print(
                    f"[train] process group ready: gloo, rank {dist.get_rank()}/"
                    f"{dist.get_world_size()}",
                    flush=True,
                )
            print(
                f"[train] distributed: rank {os.getenv('RANK', '0')}/"
                f"{os.getenv('WORLD_SIZE', '1')} bound to {rl_device}",
                flush=True,
            )
        else:
            env_cfg.sim.device = args_cli.sim_device

        # render_mode="rgb_array" makes DirectRLEnv.render() lazily create a
        # single omni.replicator render_product at cfg.viewer.cam_prim_path —
        # one buffer, num_envs-independent. The custom attach_record_camera
        # path we used before created a Camera sensor and called sim.reset()
        # *after* env init, which momentarily doubled the PhysX scene state
        # (~400 GB at 24576 envs) and OOM'd the slurm cgroup. The Lab-
        # canonical pattern is render_mode + gym.wrappers.RecordVideo (see
        # IsaacLab tests/test_record_video.py).
        env = gym.make(
            args_cli.task,
            cfg=env_cfg,
            render_mode="rgb_array" if args_cli.capture_video else None,
        )

        if args_cli.capture_video:
            from pathlib import Path

            video_folder = str(Path(hydra_run_dir) / "videos")
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=video_folder,
                step_trigger=lambda step: step % args_cli.video_interval == 0,
                video_length=args_cli.video_capture_frames,
                disable_logger=True,
            )

        if args_cli.capture_viewer and is_main_rank:
            from pathlib import Path

            target = _pose_viewer_target(args_cli.task)
            if target is None:
                print(
                    f"[train] --capture_viewer ignored: no viewer registered for "
                    f"{args_cli.task} (see _POSE_VIEWERS in train.py).",
                    flush=True,
                )
            else:
                module_name, _, class_name = target.partition(":")
                import importlib

                viewer_cls = getattr(importlib.import_module(module_name), class_name)
                viewer_kwargs = dict(
                    output_dir=Path(hydra_run_dir) / "interactive_viewer",
                    capture_len=args_cli.capture_viewer_len,
                    capture_interval=args_cli.capture_viewer_interval,
                    env_id=args_cli.capture_viewer_env_id,
                    wandb_key=args_cli.capture_viewer_wandb_key,
                )
                # Only the robot viewer has to fetch meshes over the network; the
                # cartpole's URDF is primitives and gets embedded, so it takes neither
                # of these.
                if "play.pose_viewer" in module_name:
                    viewer_kwargs["github_raw_base"] = args_cli.capture_viewer_github_raw_base
                    viewer_kwargs["url_check"] = args_cli.capture_viewer_url_check
                env = viewer_cls(env, **viewer_kwargs)

        # Clip bounds live in the rl_games YAML (params.env.*). Default to
        # +inf if absent so a task without clip YAML just runs unbounded —
        # matches isaacgymenvs's `cfg["env"].get("clipObservations", np.Inf)`.
        clip_obs = float(agent_cfg["params"]["env"].get("clip_observations", math.inf))
        clip_actions = float(agent_cfg["params"]["env"].get("clip_actions", math.inf))
        register_rlgames_env(
            env,
            rl_device=rl_device,
            clip_obs=clip_obs,
            clip_actions=clip_actions,
        )

        observers = [EnvStatsAlgoObserver()]
        if args_cli.wandb_activate and is_main_rank:
            from isaacsimenvs.utils.wandb_utils import WandbAlgoObserver

            # WandbAlgoObserver expects attribute access (cfg.wandb_project,
            # cfg.wandb_notes, …) and also passes cfg to omegaconf_to_dict
            # for wandb.config upload. OmegaConf satisfies both.
            wandb_cfg = OmegaConf.create(
                {
                    "wandb_activate": True,
                    "wandb_project": args_cli.wandb_project,
                    "wandb_group": args_cli.wandb_group,
                    "wandb_entity": args_cli.wandb_entity,
                    "wandb_name": args_cli.wandb_name or agent_cfg["params"]["config"]["name"],
                    "wandb_tags": list(args_cli.wandb_tags),
                    "wandb_notes": args_cli.wandb_notes,
                    "wandb_logcode_dir": args_cli.wandb_logcode_dir,
                }
            )
            observers.append(WandbAlgoObserver(wandb_cfg))

        runner = Runner(MultiObserver(observers))
        # Co-locate rl_games artifacts (checkpoints, summaries) with the Hydra
        # run dir so slurm logs + config + videos all live together.
        agent_cfg["params"]["config"]["train_dir"] = hydra_run_dir
        agent_cfg["params"]["config"]["device"] = rl_device
        agent_cfg["params"]["config"]["device_name"] = rl_device

        runner.load(agent_cfg)
        runner.reset()
        runner.run(
            {
                "train": not args_cli.test,
                "play": args_cli.test,
                "checkpoint": args_cli.checkpoint,
                "checkpoint_load_mode": args_cli.checkpoint_load_mode,
            }
        )

    run()

    # Kit shutdown can hang; force-exit instead of waiting for a clean teardown.
    if app is not None:
        del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
