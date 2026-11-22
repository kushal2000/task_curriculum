r"""Train an isaacsimenvs task with our vendored rl_games.

Pipeline:
    argparse (--task, --agent, AppLauncher, wandb/video flags)
        ↓
    AppLauncher (boots Kit; must precede any isaaclab.* import)
        ↓
    @hydra_task_config_with_yaml  (configclass defaults ← robot setup ← run recipe ← Hydra CLI)
        ↓
    gym.make(task_id, cfg=env_cfg)  → DirectRLEnv wrapped by gym.Wrapper
        ↓
    isaaclab_rl.RlGamesVecEnvWrapper (via register_rlgames_env — clipping,
                                      device bridging, obs-group routing)
        ↓
    rl_games.torch_runner.Runner (PPO / SAPG — both live in ./rl_games/)

CLI shape:
    python isaacsimenvs/scripts/training/train.py \
        --task Isaacsimenvs-PlayFlexivWujiRightHumanLike-Direct-v0 \
        --agent rl_games_sapg_cfg_entry_point \   # or rl_games_cfg_entry_point
        --headless --capture_viewer \
        --wandb_activate --wandb_project X --wandb_name Y \
        env.scene.num_envs=4096 \
        agent.params.config.max_epochs=200 \
        agent.params.config.minibatch_size=16384 \
        agent.params.seed=42

    A run recipe instead of the overrides (a name in isaacsimenvs/tasks/cloth/config/runs/ or a path,
    see tasks/recipes.py):
    python -m isaacsimenvs.scripts.training.train \
        --task Isaacsimenvs-RigidCloth-Direct-v0 --agent rl_games_sapg_cfg_entry_point --headless \
        --experiment experiments/nov11_right_wuji_dr_ablation/flat7RW_fric.yaml \
        --checkpoint <prior.pth> --checkpoint_load_mode weights agent.params.seed=1
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import math
import os
import sys

# Which wandb-viewer wrapper a task needs. Both render into the same three.js page
# (scripts/visualization/wandb_viewer/index.template.html) and log it with wandb.Html; they differ
# only in what geometry they hand the browser.
#
#   Play   the robot, whose STL meshes the browser must fetch from GitHub raw
#          (hence --capture_viewer_github_raw_base)
#   Cloth  the robot plus the sheet and its fold target as deformable channels
#
# Matched by id prefix (`Isaacsimenvs-<name>`). The PhysX sheet (`RigidCloth`) reads out as the same
# grid of points as the VBD one, so it takes the cloth viewer too.
_POSE_VIEWERS = {
    "RigidCloth": "isaacsimenvs.scripts.visualization.wandb_viewer.cloth_viewer:ClothPoseViewerWrapper",
    "Cloth": "isaacsimenvs.scripts.visualization.wandb_viewer.cloth_viewer:ClothPoseViewerWrapper",
    "Play": "isaacsimenvs.scripts.visualization.wandb_viewer.pose_viewer:PlayPoseViewerWrapper",
}


def _pose_viewer_target(task_id: str) -> str | None:
    for name, target in _POSE_VIEWERS.items():
        if task_id.startswith(f"Isaacsimenvs-{name}"):
            return target
    return None


def _arm_hang_watchdog() -> None:
    """Dump every thread's Python stack if the process stops making progress.

    rl_games' distributed path builds its own gloo group with a TWO HOUR timeout
    (`a2c_common.py:110`), so any failure in that region presents as a silent freeze carrying no
    information about its cause -- which is why the 4-GPU hang resisted several rounds of inference
    from the symptom alone. Set CLOTH_HANG_WATCHDOG=<seconds> to turn the freeze into a traceback.

    **DEBUGGING ONLY -- do not set it on a real run.** `dump_traceback_later(repeat=True)` walks
    every thread's stack from a timer thread, and doing that while a thread is executing inside
    Warp's C extension appears to be unsafe: the one run with it enabled died at
    `exitcode: -11` (SIGSEGV) around epoch 43 of 60, with the crashing thread inside
    `newton/_src/solvers/coupled/model_view.py`, while an otherwise identical 4-GPU run without it
    ran clean for longer. One crash is not proof of causation, but the tool exists to diagnose a
    hang and there is no reason to carry that risk through a 24-hour job.
    """
    import faulthandler

    secs = os.getenv("CLOTH_HANG_WATCHDOG")
    if not secs:
        return
    faulthandler.enable()
    faulthandler.dump_traceback_later(float(secs), repeat=True, exit=False)
    print(
        f"[train] hang watchdog armed: stacks every {secs}s "
        f"-- DEBUGGING ONLY, unsafe alongside Warp (see _arm_hang_watchdog)",
        flush=True,
    )


def main() -> None:
    _arm_hang_watchdog()

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Train isaacsimenvs task via rl_games.")
    # --- Task/agent selection ---
    parser.add_argument("--task", required=True, help="Gym task id, e.g. Isaacsimenvs-RigidCloth-Direct-v0")
    parser.add_argument(
        "--agent",
        default="rl_games_cfg_entry_point",
        help="Key in gym.register kwargs for the rl_games YAML. "
        "Use rl_games_sapg_cfg_entry_point for SAPG.",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="A run recipe: a name in isaacsimenvs/tasks/cloth/config/runs/ (e.g. rigid_cloth_right_wuji) or a "
        "YAML path; layered between the task defaults (and robot setup) and the Hydra CLI overrides (tasks/recipes.py).",
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
    # --- Point-cloud actor (a recipe with env.obs.object: pointcloud; scripts/distillation/pc_rl.py) ---
    parser.add_argument("--pc_init_student", default=None, help="Start the actor from this distilled student (.pth)")
    parser.add_argument("--pc_init_sigma", default=None, help="Take the exploration sigma of this state-policy checkpoint")
    parser.add_argument("--pc_freeze_norm", action="store_true", help="Keep the proprio normaliser fixed")
    parser.add_argument("--critic_checkpoint", default=None, help="Start the central value network from this full checkpoint")
    # --- Video ---
    parser.add_argument("--capture_video", action="store_true", help="Attach recording camera; implies --enable_cameras")
    parser.add_argument("--video_interval", type=int, default=10)
    parser.add_argument("--video_capture_frames", type=int, default=120)
    parser.add_argument("--video_fps", type=int, default=30)
    # --- Pose-only interactive HTML viewer (no cameras / no renderer) ---
    parser.add_argument(
        "--capture_viewer",
        action="store_true",
        help="Write a pose-only interactive HTML rollout (one episode of env --capture_viewer_env_id) at "
        "every --capture_viewer_epochs epochs and log it to wandb; does not enable Isaac cameras.",
    )
    parser.add_argument("--capture_viewer_len", type=int, default=600, help="Frame cap of one rollout")
    parser.add_argument(
        "--capture_viewer_epochs",
        type=int,
        default=500,
        help="A rollout every this many epochs (plus one at epoch 0)",
    )
    parser.add_argument("--capture_viewer_env_id", type=int, default=0)
    parser.add_argument("--capture_viewer_wandb_key", default="wandb_viewer")
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
    parser.add_argument("--wandb_activate", action="store_true",
                        help="Log to wandb; the best and last checkpoints are uploaded at the end")
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
    from isaacsimenvs.tasks.loader import hydra_task_config_with_yaml
    from isaacsimenvs.scripts.training.rlgames import (
        EnvStatsAlgoObserver,
        MultiObserver,
        register_rlgames_env,
    )

    @hydra_task_config_with_yaml(args_cli.task, args_cli.agent, experiment=args_cli.experiment)
    def run(env_cfg, agent_cfg: dict) -> None:
        hydra_run_dir = HydraConfig.get().runtime.output_dir

        # sim_device CLI flag still wins — it's a launcher-level concern, not
        # something we expect in the env config.
        # Under torchrun each rank MUST bind its own device. torchrun sets LOCAL_RANK but not
        # CUDA_VISIBLE_DEVICES, and both --rl_device and --sim_device default to cuda:0 -- so
        # without this every rank puts its whole scene on GPU 0 and dies in Warp's allocator,
        # which presents as an out-of-memory bug rather than a device-binding one. The same trap
        # was hit once already in the throughput benchmark (docs/phase4_cable_env.md:664).
        import torch

        from isaacsimenvs.scripts.evaluation.protocol import use_single_object_variant

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
        pointcloud_actor = getattr(getattr(env_cfg, "obs", None), "object", None) == "pointcloud"
        env = gym.make(
            args_cli.task,
            cfg=env_cfg,
            render_mode="rgb_array" if args_cli.capture_video else None,
        )

        if pointcloud_actor:
            from isaacsimenvs.scripts.distillation import pc_rl
            from isaacsimenvs.scripts.distillation.student import student_env

            inner = env.unwrapped
            layout = inner.observations.pointcloud_layout
            pc_rl.register_network()
            net = agent_cfg["params"]["network"]
            net["name"] = "pc_actor_critic"
            net["pc"] = {**layout, "init_student": args_cli.pc_init_student, "init_sigma": args_cli.pc_init_sigma,
                         "freeze_norm": bool(args_cli.pc_freeze_norm)}
            agent_cfg["params"]["config"]["normalize_input"] = False     # the network normalises the proprio part
            if is_main_rank:
                pc_rl.write_student_env(hydra_run_dir, student_env(env_cfg), layout, args_cli.pc_init_sigma)
            print(f"[train] point-cloud actor: {layout}, obs {inner.cfg.observation_space} columns", flush=True)

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

        viewer = None
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
                    output_dir=Path(hydra_run_dir) / "wandb_viewer",
                    capture_len=args_cli.capture_viewer_len,
                    env_id=args_cli.capture_viewer_env_id,
                    wandb_key=args_cli.capture_viewer_wandb_key,
                )
                # Only viewers that fetch meshes over the network take these; the cartpole's
                # URDF is primitives and gets embedded, so it accepts neither.
                #
                # Decided by INSPECTING THE SIGNATURE, not by matching the module name. The
                # previous test was `"play.pose_viewer" in module_name`, which is false for
                # `isaacsimenvs.scripts.visualization.wandb_viewer.cloth_viewer` even though ClothPoseViewerWrapper
                # subclasses PlayPoseViewerWrapper and needs both -- so cloth silently ran with
                # `github_raw_base=None` and `url_check="skip"`. The first was masked, because
                # `_derive_github_raw_base()` rebuilds the same URL from `git remote` + the
                # current branch; the second was a live loss, since the URL check is exactly what
                # would have reported the branch not being pushed instead of leaving every mesh
                # to 404 silently in the browser.
                import inspect

                accepted = inspect.signature(viewer_cls.__init__).parameters
                if "github_raw_base" in accepted:
                    viewer_kwargs["github_raw_base"] = args_cli.capture_viewer_github_raw_base
                if "url_check" in accepted:
                    viewer_kwargs["url_check"] = args_cli.capture_viewer_url_check
                env = viewer = viewer_cls(env, **viewer_kwargs)

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
        checkpoint_observer = None
        if is_main_rank and not args_cli.test:
            from isaacsimenvs.scripts.training.checkpoint_observer import CheckpointObserver

            checkpoint_observer = CheckpointObserver(viewer=viewer, rollout_every=args_cli.capture_viewer_epochs)
            observers.append(checkpoint_observer)
        if args_cli.wandb_activate and is_main_rank:
            from isaacsimenvs.scripts.training.wandb_observer import WandbAlgoObserver

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
            # What a run is, for filtering on wandb: the full env and agent config plus how it was launched.
            try:
                env_cfg_dict = env_cfg.to_dict()
            except Exception as exc:  # a config value wandb cannot take must not cost the run its logging
                print(f"[train] env config not uploaded to wandb: {exc}", flush=True)
                env_cfg_dict = {}
            run_config = {
                "task": args_cli.task,
                "experiment": args_cli.experiment,
                "checkpoint": args_cli.checkpoint,
                "checkpoint_load_mode": args_cli.checkpoint_load_mode if args_cli.checkpoint else None,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "overrides": hydra_args,
                "env": env_cfg_dict,
                "agent": agent_cfg,
            }
            observers.append(WandbAlgoObserver(wandb_cfg, run_config=run_config))

        if args_cli.critic_checkpoint:
            # Start the central value network (and its normalisers) from another run's FULL checkpoint,
            # after rl_games' own restore. Patched in here rather than in rl_games.
            import rl_games.torch_runner as _tr

            _restore = _tr._restore

            def _restore_with_critic(agent, args):
                _restore(agent, args)
                weights = _tr._load_checkpoint_weights(agent, args_cli.critic_checkpoint)
                if "assymetric_vf_nets" not in weights:
                    raise ValueError(f"{args_cli.critic_checkpoint} has no central value network (a snap_* file?)")
                agent.central_value_net.load_state_dict(weights["assymetric_vf_nets"])
                print(f"[train] central value network <- {args_cli.critic_checkpoint}", flush=True)

            _tr._restore = _restore_with_critic

        if pointcloud_actor:
            pc_rl.patch_gradient_step()

        runner = Runner(MultiObserver(observers))
        # Co-locate rl_games artifacts (checkpoints, summaries) with the Hydra
        # run dir so slurm logs + config + videos all live together.
        agent_cfg["params"]["config"]["train_dir"] = hydra_run_dir
        agent_cfg["params"]["config"]["device"] = rl_device
        agent_cfg["params"]["config"]["device_name"] = rl_device

        runner.load(agent_cfg)
        runner.reset()
        _run_args = {
            "train": not args_cli.test,
            "play": args_cli.test,
            "checkpoint": args_cli.checkpoint,
            "checkpoint_load_mode": args_cli.checkpoint_load_mode,
        }
        if os.getenv("PC_PROFILE"):          # cProfile of the whole run (set CUDA_LAUNCH_BLOCKING=1 with it)
            import cProfile
            import pstats

            _prof = cProfile.Profile()
            try:
                _prof.runcall(runner.run, _run_args)
            finally:
                _prof.dump_stats(os.getenv("PC_PROFILE"))
                pstats.Stats(os.getenv("PC_PROFILE")).sort_stats("cumulative").print_stats(70)
        else:
            runner.run(_run_args)

        if checkpoint_observer is not None:
            checkpoint_observer.upload_checkpoints()
        if args_cli.wandb_activate and is_main_rank:
            import wandb

            if wandb.run is not None:
                wandb.finish()  # waits for the uploads; os._exit below would cut them off

    run()

    # Kit shutdown can hang; force-exit instead of waiting for a clean teardown.
    if app is not None:
        del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
