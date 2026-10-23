"""Play trained Play-family policies live in Isaac Sim and watch them in viser.

Pick a run (and checkpoint) from ``outputs/``, and the objects it is tested on: the env is built
with the run's own config, one env per object, all stepped together. Objects come from training's
12 size distributions over the 6 handle-head types, ``--num_assets_per_type`` (default 1) from
each, so 12 envs by default. They are fresh seed-42 samples from those distributions; only at the
run's own ``num_assets_per_type`` (100) are they the exact objects it trained on, since the
generator's draws depend on the count. Viser draws the robot, object, goal and table of the env you are
watching; the panel counts goals and episodes per object type across every env.

Isaac Sim builds one robot per process. The run dropdown lists every run of every robot:
"load policy" swaps in a checkpoint of a run on the current robot (same task, same observation
layout) without touching the env; "load env" restarts this viewer in place -- same process, same
port -- on the chosen run's robot, and the browser reconnects once it is built (about 30 s).

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python -m isaacsimenvs.eval.viser_play \\
        --run play_scratch_flexiv_sharpa_ft05_nodr_249241 --port 8081

``--run`` takes a run name (or a unique part of one), a run directory, or a ``.pth``. The newest
checkpoint is loaded unless a ``.pth`` is given. By default the eval protocol applies, as in
``episodes.py``: randomization off (``protocol.disable_randomization``, which also pins the
object's start pose) and the success tolerance pinned (slider, default the curriculum's floor).
``--randomize`` keeps the run's training-time randomization instead.

Controls: play / pause, step, reset, speed; which object (type + variant) to watch; the object's
start orientation (random, as in training, or a fixed roll/pitch/yaw, applied at the next reset of
all envs); SAPG
exploration block, whose coefficient is appended to the observation (block 5, coefficient 0, is
trained on task reward alone; see player.SAPG_EXPL_COEF); greedy or sampled actions.

On the cluster, forward the port: ``ssh -L 8081:localhost:8081 <node>``.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUTPUTS = REPO / "outputs"
# isaacsim-kernel pins websockets==12.0 and viser needs >=13.1. scripts/install_isaacsim.sh
# installs a newer one here, outside the training venv; it must be first on the path before Kit
# or viser import websockets.
_WEBSOCKETS = REPO / "third_party" / "viser_websockets"
if _WEBSOCKETS.is_dir():
    sys.path.insert(0, str(_WEBSOCKETS))

#: (robot, robot_self_collision) in a run's config -> the task it trained on.
TASK_FOR = {
    ("kuka_sharpa", False): "Isaacsimenvs-Play-Direct-v0",
    ("kuka_sharpa", True): "Isaacsimenvs-PlaySelfCollision-Direct-v0",
    ("flexiv_wuji", True): "Isaacsimenvs-PlayFlexivWuji-Direct-v0",
    ("flexiv_wuji_human_like", True): "Isaacsimenvs-PlayFlexivWujiHumanLike-Direct-v0",
    ("flexiv_sharpa", True): "Isaacsimenvs-PlayFlexivSharpa-Direct-v0",
}
#: SAPG's per-block coefficients: linspace(50, 0, num_actors // expl_coef_block_size)
#: (rl_games a2c_common.py:342).
SAPG_MAX_COEF = 50.0


# --- runs on disk (no Isaac imports) -------------------------------------------------------------


class Run:
    """One training run: ``outputs/<date>/<time>/0_<name>/nn/*.pth`` plus its Hydra config."""

    def __init__(self, run_dir: Path) -> None:
        from omegaconf import OmegaConf

        self.dir = run_dir
        self.name = run_dir.name.removeprefix("0_")
        self.config_path = run_dir.parent / ".hydra" / "config.yaml"
        # Resolved: the agent config interpolates into the env's (num_actors: ${....env.scene.num_envs}).
        self.config = OmegaConf.to_container(OmegaConf.load(self.config_path), resolve=True)
        env = self.config["env"]
        self.robot = env.get("robot") or "kuka_sharpa"  # configs from before the field: the Kuka
        self.self_collision = bool(env.get("robot_self_collision", False))
        self.task = TASK_FOR.get((self.robot, self.self_collision))
        # Two jobs started in the same second share one outputs/<date>/<time>/ and so one
        # .hydra/config.yaml (2026-10-09/18-04-56 holds a Flexiv and a Kuka run). The Slurm log's
        # "[train] robot=... task=..." line is per job, so it decides; on disagreement the saved
        # env config is another run's and is not used.
        self.config_is_own = True
        logged = self._logged_robot_and_task()
        if logged is not None:
            robot, task = logged
            self.config_is_own = robot == self.robot
            self.robot, self.task = robot, task

    def _logged_robot_and_task(self) -> tuple[str, str] | None:
        m = re.search(r"_(\d+)$", self.name)
        log = REPO / "slurm_logs" / f"play_scratch_{m.group(1)}.out" if m else None
        if log is None or not log.is_file():
            return None
        with log.open(errors="replace") as f:
            for line in f:
                hit = re.match(r"\[train\] robot=(\S+) task=(\S+)", line)
                if hit:
                    return hit.group(1), hit.group(2)
        return None

    def checkpoints(self) -> list[Path]:
        """Periodic checkpoints by epoch, then the best-so-far one last."""
        def epoch(p: Path) -> int:
            m = re.search(r"_ep_(\d+)_", p.name)
            return int(m.group(1)) if m else 10**12
        return sorted((self.dir / "nn").glob("*.pth"), key=epoch)

    def num_blocks(self) -> int:
        cfg = self.config["agent"]["params"]["config"]
        return max(1, int(cfg["num_actors"]) // int(cfg.get("expl_coef_block_size") or cfg["num_actors"]))


def find_runs() -> list[Run]:
    runs = []
    for nn in sorted(OUTPUTS.glob("*/*/0_*/nn")):
        if any(nn.glob("*.pth")) and (nn.parent.parent / ".hydra" / "config.yaml").is_file():
            try:
                runs.append(Run(nn.parent))
            except Exception as exc:  # an unreadable config is skipped, not fatal
                print(f"[viser_play] skipping {nn.parent}: {exc}", flush=True)
    return runs


def newest_checkpoint(run: Run) -> Path:
    """The latest periodic (``_ep_``) checkpoint, else the best-so-far one."""
    periodic = [p for p in run.checkpoints() if "_ep_" in p.name]
    return (periodic or run.checkpoints())[-1]


def resolve_run(spec: str, runs: list[Run]) -> tuple[Run, Path]:
    path = Path(spec)
    if path.suffix == ".pth" and path.is_file():
        return Run(path.resolve().parents[1]), path.resolve()
    if path.is_dir() and (path / "nn").is_dir():
        run = Run(path.resolve())
        return run, newest_checkpoint(run)
    matches = [r for r in runs if spec in r.name]
    matches = [r for r in matches if r.name == spec] or matches
    if len(matches) != 1:
        names = "\n  ".join(r.name for r in (matches or runs))
        raise SystemExit(f"--run {spec!r} matches {len(matches)} runs:\n  {names}")
    return matches[0], newest_checkpoint(matches[0])


def object_type(urdf_path: str) -> str:
    """The handle-head type in a generated object URDF's name, e.g. ``007_hammer_handle_...``."""
    m = re.match(r"\d+_([a-z]+)_handle", Path(urdf_path).name)
    return m.group(1) if m else Path(urdf_path).stem


# --- main ----------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", required=True, help="run name (or a unique part), run dir, or .pth")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--num_assets_per_type", type=int, default=1,
                        help="objects per size distribution (12 distributions); 100 = the run's "
                        "exact training pool, 1200 envs")
    parser.add_argument("--randomize", action="store_true",
                        help="keep the run's training-time randomization (default: eval protocol)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    runs = find_runs()
    run, checkpoint = resolve_run(args.run, runs)
    if run.task is None:
        raise SystemExit(f"{run.name}: no task for robot {run.robot!r}, self-collision {run.self_collision}")
    siblings = [r for r in runs if r.task == run.task]
    playable = [r for r in runs if r.task is not None]
    run_label = {r.name: f"{r.robot}: {r.name}" for r in playable}
    print(f"[viser_play] {run.name} ({run.task}), {checkpoint.name}; "
          f"{len(siblings)} runs of this robot", flush=True)

    from isaaclab.app import AppLauncher

    app_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(app_parser)
    app_args = app_parser.parse_args([])
    app_args.headless, app_args.device = True, args.device
    app = AppLauncher(app_args).app

    import gymnasium as gym
    import numpy as np
    import torch
    import viser
    import yaml
    import yourdfpy
    from viser.extras import ViserUrdf

    import isaacsimenvs  # noqa: F401  gym.register side effects
    from isaacsimenvs.eval.player import PretrainedPlayer
    from isaacsimenvs.eval.protocol import disable_randomization
    from isaacsimenvs.utils.hydra_utils import hydra_task_config_with_yaml

    sys.argv = [sys.argv[0]]

    @hydra_task_config_with_yaml(run.task, "")
    def _run(env_cfg, agent_cfg) -> None:
        if run.config_is_own:
            saved_env = dict(run.config["env"])
            for key in ("observation_space", "state_space", "action_space", "num_observations",
                        "num_states", "num_actions"):
                saved_env.pop(key, None)
            env_cfg.from_dict(saved_env)
        else:
            print(f"[viser_play] {run.config_path} belongs to another run; building "
                  f"{run.task} from its defaults", flush=True)
        from isaacsimenvs.tasks.play.utils.object_size_distributions import OBJECT_SIZE_DISTRIBUTIONS

        per_type = args.num_assets_per_type
        env_cfg.assets.num_assets_per_type = per_type
        # One env per object: num_assets_per_type from every size distribution of the chosen types
        # (generate_handle_head_urdfs), several distributions per type.
        types_cfg = set(env_cfg.assets.handle_head_types)
        pool = per_type * sum(d.type in types_cfg for d in OBJECT_SIZE_DISTRIBUTIONS)
        env_cfg.scene.num_envs = pool
        env_cfg.seed = args.seed
        env_cfg.termination.eval_success_tolerance = float(env_cfg.termination.target_success_tolerance)
        if not args.randomize:
            disable_randomization(env_cfg)

        env = gym.make(run.task, cfg=env_cfg)
        inner = env.unwrapped
        num_envs, device = inner.num_envs, inner.device
        n_obs, n_act = int(inner.cfg.observation_space), int(inner.cfg.action_space)
        origins = inner.scene.env_origins

        # Which object each env holds; per type, its variants (every distribution's) in a fixed order.
        urdf_paths = list(inner._object_urdf_paths)
        asset_of_env = inner._object_asset_index_per_env.cpu().numpy()
        if len(set(asset_of_env.tolist())) != len(urdf_paths):
            raise RuntimeError(f"{num_envs} envs do not cover the {len(urdf_paths)}-object pool once each")
        type_of_env = [object_type(urdf_paths[a]) for a in asset_of_env]
        types = sorted(set(type_of_env))
        variants = {
            t: sorted((e for e in range(num_envs) if type_of_env[e] == t),
                      key=lambda e: Path(urdf_paths[asset_of_env[e]]).name)
            for t in types
        }

        # --- policy ---------------------------------------------------------------------------
        policy = {"player": None, "run": None, "ckpt": None, "blocks": 1}

        def load_policy(r: Run, ckpt: Path, block: int | None) -> None:
            agent = r.config["agent"]
            agent["params"]["config"]["multi_gpu"] = False
            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
                yaml.safe_dump({"train": agent}, f)
            blocks = r.num_blocks()
            block = blocks - 1 if block is None else min(block, blocks - 1)
            coef = float(np.linspace(SAPG_MAX_COEF, 0.0, blocks)[block]) if blocks > 1 else 0.0
            policy.update(
                player=PretrainedPlayer(f.name, ckpt, num_envs=num_envs, device=args.device,
                                        num_observations=n_obs, num_actions=n_act, expl_coef=coef),
                run=r, ckpt=ckpt, blocks=blocks, block=block, coef=coef,
            )
            print(f"[viser_play] loaded {r.name}/{ckpt.name}, SAPG block {block} (coef {coef:g})",
                  flush=True)

        load_policy(run, checkpoint, None)

        # --- viser scene ----------------------------------------------------------------------
        server = viser.ViserServer(port=args.port)
        server.scene.set_up_direction("+z")
        server.scene.add_grid("/ground", width=3.0, height=3.0, cell_size=0.1)
        robot_urdf = yourdfpy.URDF.load(str(REPO / inner.cfg.assets.robot_urdf))
        viser_robot = ViserUrdf(server, robot_urdf, root_node_name="/robot")
        robot_root = server.scene.add_frame("/robot", show_axes=False)
        table_urdf = yourdfpy.URDF.load(str(REPO / inner.cfg.assets.table_urdf))
        table_root = server.scene.add_frame("/table", show_axes=False)
        ViserUrdf(server, table_urdf, root_node_name="/table", mesh_color_override=(0.82, 0.56, 0.35))
        robot_joint_names = list(inner.robot.joint_names)
        viser_joint_order = list(robot_urdf.actuated_joint_names)
        nodes: dict = {}

        def show_object(env_id: int) -> None:
            for key in ("object", "goal"):
                if key in nodes:
                    nodes[key][1].remove()
                    nodes[key][0].remove()
            urdf = yourdfpy.URDF.load(urdf_paths[asset_of_env[env_id]])
            for key, color in (("object", (0.95, 0.6, 0.2, 1.0)), ("goal", (0.3, 0.85, 0.35, 0.35))):
                frame = server.scene.add_frame(f"/{key}", show_axes=False)
                nodes[key] = (frame, ViserUrdf(server, urdf, root_node_name=f"/{key}",
                                               mesh_color_override=color))

        # --- GUI ------------------------------------------------------------------------------
        # GUI callbacks run on viser's thread; they only queue requests ("want_*"), and the main
        # loop applies them between steps, so the scene is never edited while it is drawn.
        state = {"playing": True, "step_once": False, "reset": False, "env": 0, "t": 0,
                 "want_env": None, "want_policy": None, "want_start": None, "want_restart": None}
        rs = inner.cfg.reset
        # Where a fixed start puts the object: the table centre, dropped from training's height.
        start_xyz = (0.0, 0.0, float(rs.table_reset_z) + float(rs.table_object_z_offset))
        stats = {t: {"episodes": 0, "goals": 0} for t in types}
        ep_goals = np.zeros(num_envs, dtype=np.int64)

        with server.gui.add_folder("Policy"):
            g_run = server.gui.add_dropdown("run", [run_label[r.name] for r in playable],
                                            initial_value=run_label[run.name])
            g_ckpt = server.gui.add_dropdown("checkpoint", [p.name for p in run.checkpoints()],
                                            initial_value=checkpoint.name)
            g_block = server.gui.add_slider("SAPG block (last = task reward only)", 0,
                                            max(policy["blocks"] - 1, 1), 1, policy["block"])
            g_load = server.gui.add_button("load policy (same robot; resets all envs)")
            g_load_env = server.gui.add_button("load env (this run's robot; restarts, ~30 s)")
            g_policy_md = server.gui.add_markdown("")
            g_greedy = server.gui.add_checkbox("greedy actions", True)
        with server.gui.add_folder("Playback"):
            g_play = server.gui.add_button("play / pause")
            g_step = server.gui.add_button("step once")
            g_reset = server.gui.add_button("reset all envs + stats")
            g_speed = server.gui.add_slider("speed (x real time)", 0.1, 4.0, 0.1, 1.0)
            g_tol = server.gui.add_slider("success tolerance (m)", 0.005, 0.1, 0.001,
                                          float(inner.cfg.termination.eval_success_tolerance))
        with server.gui.add_folder("Object"):
            g_type = server.gui.add_dropdown("type", types, initial_value=types[0])
            g_variant = server.gui.add_slider("variant", 0, max(len(variants[types[0]]) - 1, 1), 1, 0)
            g_obj_md = server.gui.add_markdown("")
        with server.gui.add_folder("Object start"):
            start_modes = ("random (as in training)", "fixed")
            g_start = server.gui.add_dropdown(
                "orientation", start_modes,
                initial_value=start_modes[1] if rs.fixed_start_pose is not None else start_modes[0])
            g_rpy = [server.gui.add_slider(f"{a} (deg)", -180.0, 180.0, 5.0, 0.0)
                     for a in ("roll", "pitch", "yaw")]
            g_apply = server.gui.add_button("apply (resets all envs + stats)")
            g_start_md = server.gui.add_markdown("")
        md = server.gui.add_markdown("")

        def describe_start() -> str:
            if rs.fixed_start_pose is None:
                return "now: random orientation each reset, as in training"
            from scipy.spatial.transform import Rotation as Rot
            w, x, y, z = rs.fixed_start_pose[3:]
            rpy = Rot.from_quat([x, y, z, w]).as_euler("xyz", degrees=True)
            return f"now: fixed, roll/pitch/yaw {np.round(rpy).astype(int).tolist()} deg"

        g_start_md.content = describe_start()

        @g_apply.on_click
        def _(_) -> None:
            state["want_start"] = (g_start.value == "fixed", tuple(float(g.value) for g in g_rpy))
        def selected_run() -> Run:
            return next(x for x in playable if run_label[x.name] == g_run.value)

        @g_run.on_update
        def _(_) -> None:
            r = selected_run()
            g_policy_md.content = ("" if r.task == run.task else
                                   f"`{r.robot}` is not this env's robot: use **load env**")
            names = [p.name for p in r.checkpoints()]
            g_ckpt.options = names
            g_ckpt.value = newest_checkpoint(r).name

        @g_load.on_click
        def _(_) -> None:
            r = selected_run()
            if r.task != run.task:
                g_policy_md.content = f"`{r.robot}` is not this env's robot: use **load env**"
                return
            state["want_policy"] = (r, r.dir / "nn" / g_ckpt.value, int(g_block.value))

        @g_load_env.on_click
        def _(_) -> None:
            r = selected_run()
            state["want_restart"] = r.dir / "nn" / g_ckpt.value
            g_policy_md.content = f"restarting on `{r.robot}` ... the page reconnects when it is built"

        g_play.on_click(lambda _: state.update(playing=not state["playing"]))
        g_step.on_click(lambda _: state.update(step_once=True))
        g_reset.on_click(lambda _: state.update(reset=True))

        def watch(e: int) -> None:
            state["env"] = e
            show_object(e)
            g_obj_md.content = f"env {e}: `{Path(urdf_paths[asset_of_env[e]]).stem[:48]}`"

        @g_type.on_update
        def _(_) -> None:
            g_variant.max = max(len(variants[g_type.value]) - 1, 1)
            g_variant.value = min(int(g_variant.value), len(variants[g_type.value]) - 1)
            state["want_env"] = variants[g_type.value][int(g_variant.value)]

        @g_variant.on_update
        def _(_) -> None:
            vs = variants[g_type.value]
            state["want_env"] = vs[min(int(g_variant.value), len(vs) - 1)]

        def pose(data, env_id: int) -> tuple[tuple, tuple]:
            return (tuple((data.root_pos_w[env_id] - origins[env_id]).cpu().numpy()),
                    tuple(data.root_quat_w[env_id].cpu().numpy()))  # wxyz, as viser takes it

        def draw() -> None:
            e = state["env"]
            robot_root.position, robot_root.wxyz = pose(inner.robot.data, e)
            table_root.position, table_root.wxyz = pose(inner.table.data, e)
            q = dict(zip(robot_joint_names, inner.robot.data.joint_pos[e].cpu().numpy()))
            viser_robot.update_cfg(np.array([q[n] for n in viser_joint_order]))
            for key, data in (("object", inner.object.data), ("goal", inner.goal_viz.data)):
                nodes[key][0].position, nodes[key][0].wxyz = pose(data, e)
            p = policy
            rows = [
                f"| {t} ({len(variants[t])}) | {s['episodes']} | {s['goals'] / max(s['episodes'], 1):.2f} |"
                for t, s in stats.items()
            ]
            total_eps = sum(s["episodes"] for s in stats.values())
            total_goals = sum(s["goals"] for s in stats.values())
            md.content = "\n".join([
                f"**{p['run'].name}**  \n`{p['ckpt'].name}`  \nblock {p['block']} (coef {p['coef']:g}), "
                f"{'greedy' if g_greedy.value else 'sampled'}, "
                f"{'eval protocol' if not args.randomize else 'training randomization'}"
                + ("" if run.config_is_own else "; env from task defaults (run config shared)"),
                "",
                f"t = {state['t'] * float(inner.step_dt):.1f} s; watched env {e}: "
                f"**{int(ep_goals[e])}** goals this episode",
                "",
                "| object type (variants) | finished episodes | goals / episode |",
                "|---|---:|---:|",
                *rows,
                f"| **all** | {total_eps} | {total_goals / max(total_eps, 1):.2f} |",
            ])

        watch(variants[types[0]][0])
        obs, _ = env.reset()
        draw()
        port = server.get_port()
        if port != args.port:  # viser moves to the next free port rather than failing
            print(f"[viser_play] WARNING: port {args.port} is taken; serving on {port}", flush=True)
        print(f"[viser_play] {num_envs} envs, one per object ("
              + ", ".join(f"{t} {len(v)}" for t, v in variants.items())
              + f"); viser on http://localhost:{port}", flush=True)

        while app.is_running():
            inner.cfg.termination.eval_success_tolerance = float(g_tol.value)
            if state["want_restart"] is not None:
                # A new robot needs a new Isaac Sim stage, so a new process: replace this one with
                # the same command on the chosen checkpoint. Same PID and port; exec frees the GPU.
                argv = [sys.executable, "-m", "isaacsimenvs.eval.viser_play",
                        "--run", str(state["want_restart"]), "--port", str(args.port),
                        "--num_assets_per_type", str(args.num_assets_per_type),
                        "--seed", str(args.seed), "--device", args.device]
                if args.randomize:
                    argv.append("--randomize")
                print(f"[viser_play] load env: {' '.join(argv[3:])}", flush=True)
                server.stop()
                sys.stdout.flush()
                os.execv(sys.executable, argv)
            if state["want_env"] is not None:
                watch(state["want_env"])
                state["want_env"] = None
                draw()
            if state["want_start"] is not None:
                fixed, rpy = state["want_start"]
                if fixed:
                    from scipy.spatial.transform import Rotation as Rot
                    x, y, z, w = Rot.from_euler("xyz", rpy, degrees=True).as_quat()
                    rs.fixed_start_pose = (*start_xyz, float(w), float(x), float(y), float(z))
                else:
                    rs.fixed_start_pose = None  # _reset_object_pose draws random_orientation()
                g_start_md.content = describe_start()
                state.update(want_start=None, reset=True)
            if state["want_policy"] is not None:
                load_policy(*state["want_policy"])
                g_block.max = max(policy["blocks"] - 1, 1)
                state.update(want_policy=None, reset=True)
            if state["reset"]:
                obs, _ = env.reset()
                policy["player"].reset_rnn()
                ep_goals[:] = 0
                for s in stats.values():
                    s.update(episodes=0, goals=0)
                state.update(reset=False, t=0)
                draw()
            if not (state["playing"] or state["step_once"]):
                time.sleep(0.05)
                continue
            t0 = time.time()
            with torch.no_grad():
                action = policy["player"].get_action(obs["policy"], deterministic=g_greedy.value)
            obs, _, terminated, truncated, _ = env.step(action)
            ep_goals += inner._is_success.cpu().numpy().astype(np.int64)
            ended = (terminated | truncated)
            if ended.any():
                ids = ended.nonzero(as_tuple=True)[0]
                policy["player"].reset_rnn(ids)
                for e in ids.cpu().numpy():
                    s = stats[type_of_env[e]]
                    s["episodes"] += 1
                    s["goals"] += int(ep_goals[e])
                    ep_goals[e] = 0
            state["t"] += 1
            state["step_once"] = False
            draw()
            time.sleep(max(0.0, float(inner.step_dt) / g_speed.value - (time.time() - t0)))
        env.close()

    _run()
    app.close()


if __name__ == "__main__":
    main()
