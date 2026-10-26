"""Live viser viewer of a joint-reach policy: the simulated robot, its goal pose as a see-through ghost,
and the goal fingertip positions.

Goals follow held-out random-walk chains (as joint_reach_eval), or the real-robot test sequence
(real_test_sequence.pt) in every env. The menu sets the DR condition (joint_reach_eval's) and can fix
the obs / action delay; both restart the goal sequence.

    .venv_isaacsim/bin/python -m isaacsimenvs.eval.joint_reach_viser --headless \
        --checkpoint <run>/nn/<ckpt>.pth --real_sequence <dir>/real_test_sequence.pt --port 8083
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

from isaacsimenvs.eval.joint_reach_eval import CONDITIONS, TASK

ROOT = Path(__file__).resolve().parents[2]
GHOST_RGBA = (0.15, 0.85, 0.35, 0.35)
TIP_LINKS = ("l_thumb_tip", "l_index_finger_tip", "l_middle_finger_tip", "l_ring_finger_tip", "l_pinky_tip")


def _quat_to_mat(wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = wxyz
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--policy_config", default="pretrained_policy/config.yaml")
    parser.add_argument("--sapg_expl_coef", type=float, default=0.0)
    parser.add_argument("--real_sequence", default="", help="real_test_sequence.pt from joint_reach_eval")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--chain_len", type=int, default=200)
    parser.add_argument("--settle_s", type=float, default=2.0)
    parser.add_argument("--port", type=int, default=8083)
    AppLauncher.add_app_launcher_args(parser)
    args, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args
    os.chdir(ROOT)
    app = AppLauncher(args).app  # noqa: F841

    import gymnasium as gym
    import torch
    import viser
    import yourdfpy
    from viser.extras import ViserUrdf

    import isaacsimenvs  # noqa: F401
    from isaacsimenvs.eval.player import PretrainedPlayer
    from isaacsimenvs.utils.hydra_utils import hydra_task_config_with_yaml

    @hydra_task_config_with_yaml(TASK, "")
    def run(env_cfg, _agent_cfg) -> None:
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.episode_length_s = 1e6  # the viewer drives the goals and never ends an episode
        env_cfg.goal.default_start_prob = 1.0
        env_cfg.reset.reset_dof_pos_random_interval_arm = 0.0
        env_cfg.reset.reset_dof_pos_random_interval_fingers = 0.0
        env_cfg.reset.reset_dof_vel_random_interval = 0.0
        dr_on = env_cfg.domain_randomization.to_dict()
        urdf_path = ROOT / env_cfg.assets.robot_urdf

        env = gym.make(TASK, cfg=env_cfg)
        inner = env.unwrapped
        inner.external_goals = True
        dev, n_env = inner.device, inner.num_envs
        all_ids = torch.arange(n_env, device=dev)
        names = list(inner.robot_spec.joint_names_canonical)
        player = PretrainedPlayer(
            config_path=args.policy_config, checkpoint_path=args.checkpoint, num_envs=n_env,
            device=str(dev), num_observations=int(inner.cfg.observation_space),
            num_actions=int(inner.cfg.action_space), expl_coef=args.sapg_expl_coef,
        )
        arm_tol, hand_tol = inner._arm_tol, inner._hand_tol

        # Goal sources: held-out chains (half from the default pose), and the real-robot sequence.
        bank = inner.build_goal_bank(n_env, 11)
        chains = inner.build_goal_chains(inner.chain_starts(bank, 0.5, 11), args.chain_len + 1, 11)
        real = torch.load(args.real_sequence, map_location="cpu") if args.real_sequence else None

        # ------------------------------------------------------------------ viser scene
        server = viser.ViserServer(host="0.0.0.0", port=args.port)
        server.scene.set_up_direction("+z")
        fk = yourdfpy.URDF.load(str(urdf_path), build_scene_graph=True, load_meshes=False)
        robot_frame = server.scene.add_frame("/robot", show_axes=False)
        ghost_frame = server.scene.add_frame("/goal", show_axes=False)
        robot_vis = ViserUrdf(server, urdf_path, root_node_name="/robot")
        ghost_vis = ViserUrdf(server, urdf_path, root_node_name="/goal", mesh_color_override=GHOST_RGBA)
        order = [names.index(j) for j in robot_vis.get_actuated_joint_names()]
        fk_order = [names.index(j) for j in fk.actuated_joint_names]
        sx, sy, sz = env_cfg.goal.table_box_size
        table = server.scene.add_box("/table", dimensions=(sx, sy, sz), color=(0.55, 0.45, 0.35))
        server.scene.add_grid("/floor", width=3.0, height=3.0, cell_size=0.25)
        tip_goal = [server.scene.add_icosphere(f"/tips/goal_{t}", radius=0.008, color=(0.1, 0.8, 0.3))
                    for t in TIP_LINKS]
        tip_line = server.scene.add_line_segments(
            "/tips/err", points=np.zeros((5, 2, 3), np.float32), colors=(1.0, 0.2, 0.2), line_width=3.0)

        with server.gui.add_folder("Run"):
            g_play = server.gui.add_checkbox("playing", True)
            g_speed = server.gui.add_slider("speed x real time", 0.1, 2.0, 0.05, 1.0)
            g_env = server.gui.add_dropdown("shown env", [str(i) for i in range(n_env)], "0")
            g_ghost = server.gui.add_checkbox("show goal ghost", True)
            g_tips = server.gui.add_checkbox("show fingertip goals", True)
        with server.gui.add_folder("Goals / condition (restart)"):
            g_mode = server.gui.add_dropdown(
                "goals", ["random-walk chains"] + (["real test sequence"] if real is not None else []),
                "random-walk chains")
            g_cond = server.gui.add_dropdown("condition", list(CONDITIONS), "nodr")
            g_obs_d = server.gui.add_dropdown("obs delay [steps]", ["per-episode random", "0", "1", "2"], "0")
            g_act_d = server.gui.add_dropdown("action delay [steps]", ["per-episode random", "0", "1", "2"], "0")
            g_restart = server.gui.add_button("restart")
        hud = server.gui.add_markdown("")
        restart = threading.Event()
        restart.set()
        for h in (g_mode, g_cond, g_obs_d, g_act_d):
            h.on_update(lambda _e: restart.set())
        g_restart.on_click(lambda _e: restart.set())

        # ------------------------------------------------------------------ goal driver
        st = {}

        def start() -> None:
            dr = inner.cfg.domain_randomization
            for k, v in dr_on.items():
                setattr(dr, k, v)
            c = CONDITIONS[g_cond.value]
            if not c["dr"]:
                dr.use_obs_delay = dr.use_action_delay = False
                dr.action_noise_std = dr.joint_pos_obs_noise_std = dr.joint_velocity_obs_noise_std = 0.0
            inner.cfg.gain_dr.fixed_scales = tuple(c["gains"])
            for flag, menu in (("use_obs_delay", g_obs_d), ("use_action_delay", g_act_d)):
                if menu.value != "per-episode random":
                    setattr(dr, flag, True)
            env.reset()
            for idx, menu in ((inner._obs_delay_idx, g_obs_d), (inner._action_delay_idx, g_act_d)):
                if menu.value != "per-episode random":
                    idx[:] = int(menu.value)
            player.reset_rnn()
            if g_mode.value == "real test sequence":
                seq = torch.cat([real["start"][None], real["goals"]]).to(dev)
                st["chains"] = seq.unsqueeze(0).repeat(n_env, 1, 1)
                st["settle"] = float(real["settle_s"])
            else:
                st["chains"], st["settle"] = chains, args.settle_s
            inner.place_robot(all_ids, st["chains"][:, 0])
            st["seg"] = torch.ones(n_env, dtype=torch.long, device=dev)
            inner._goal[:] = st["chains"][:, 1]
            st["hold"] = inner.hold_steps(st["chains"][:, 0], st["chains"][:, 1],
                                          torch.full((n_env,), st["settle"], device=dev))
            st["t"] = torch.zeros(n_env, dtype=torch.long, device=dev)
            st["reached"] = torch.zeros(n_env, device=dev)
            st["done"] = torch.zeros(n_env, device=dev)
            st["obs"] = inner._get_observations()

        def advance(q: torch.Tensor, in_tol: torch.Tensor) -> None:
            st["t"] += 1
            ended = st["t"] >= st["hold"]
            if not bool(ended.any()):
                return
            ids = ended.nonzero(as_tuple=False).squeeze(-1)
            st["reached"][ids] += in_tol[ids].float()
            st["done"][ids] += 1
            k_max = st["chains"].shape[1] - 1
            st["seg"][ids] = torch.where(st["seg"][ids] >= k_max, 1, st["seg"][ids] + 1)
            wrap = ids[st["seg"][ids] == 1]
            if wrap.numel():  # back to the chain start: put the robot there
                inner.place_robot(wrap, st["chains"][wrap, 0])
            nxt = st["chains"][ids, st["seg"][ids]]
            prev = torch.where((st["seg"][ids] == 1).unsqueeze(1), st["chains"][ids, 0], inner._goal[ids])
            st["hold"][ids] = inner.hold_steps(prev, nxt, torch.full((ids.numel(),), st["settle"], device=dev))
            inner._goal[ids] = nxt
            st["t"][ids] = 0

        # ------------------------------------------------------------------ main loop
        dt = inner.step_dt
        print(f"[jr_viser] ready on port {args.port}", flush=True)
        n_steps, t_rate, rate = 0, time.time(), 0.0
        while True:
            if restart.is_set():
                restart.clear()
                start()
            if not g_play.value:
                time.sleep(0.05)
                continue
            t0 = time.time()
            with torch.no_grad():
                action = player.get_action(st["obs"]["policy"], deterministic=True)
                st["obs"], *_ = env.step(action.to(dev))
            q = inner.robot.data.joint_pos[:, inner._perm_lab_to_canon]
            err = (inner._goal - q).abs()
            arm_e, hand_e = err[:, :7].max(-1).values, err[:, 7:].max(-1).values
            in_tol = (arm_e < arm_tol) & (hand_e < hand_tol)
            advance(q, in_tol)

            w = int(g_env.value)
            origin = inner.scene.env_origins[w]
            root_p = (inner.robot.data.root_pos_w[w] - origin).cpu().numpy()
            root_q = inner.robot.data.root_quat_w[w].cpu().numpy()  # wxyz
            q_w = q[w].cpu().numpy()
            goal_w = inner._goal[w].cpu().numpy()
            with server.atomic():
                for frame in (robot_frame, ghost_frame):
                    frame.position, frame.wxyz = root_p, root_q
                robot_vis.update_cfg(q_w[order])
                ghost_vis.show_visual = g_ghost.value
                if g_ghost.value:
                    ghost_vis.update_cfg(goal_w[order])
                table.position = (inner.table.data.root_pos_w[w] - origin).cpu().numpy()
                table.wxyz = inner.table.data.root_quat_w[w].cpu().numpy()
                tips = []
                rot = _quat_to_mat(root_q)
                for cfg_q in (goal_w, q_w):
                    fk.update_cfg(cfg_q[fk_order])
                    tips.append([root_p + rot @ fk.get_transform(t, fk.base_link)[:3, 3] for t in TIP_LINKS])
                for h, p in zip(tip_goal, tips[0]):
                    h.position, h.visible = p, g_tips.value
                tip_line.points = np.stack([tips[1], tips[0]], axis=1).astype(np.float32)
                tip_line.visible = g_tips.value

            n_steps += 1
            if time.time() - t_rate > 1.0:
                rate, n_steps, t_rate = n_steps / (time.time() - t_rate), 0, time.time()
            e_deg = np.degrees(err[w].cpu().numpy())
            worst = int(np.argmax(e_deg))
            k_max = st["chains"].shape[1] - 1
            done = float(st["done"][w])
            hud.content = (
                f"**goal {int(st['seg'][w])}/{k_max}** · {int(st['hold'][w] - st['t'][w]) * dt:.1f} s left in hold\n\n"
                f"arm max err **{math.degrees(float(arm_e[w])):.2f}°** (tol {math.degrees(arm_tol):.1f}) · "
                f"hand max err **{math.degrees(float(hand_e[w])):.2f}°** (tol {math.degrees(hand_tol):.1f}) · "
                f"{'✅ in tol' if bool(in_tol[w]) else '… moving'}\n\n"
                f"worst joint: {names[worst]} {e_deg[worst]:.2f}°\n\n"
                f"reached {int(st['reached'][w])}/{int(done)} goals this run · "
                f"obs delay {int(inner._obs_delay_idx[w])} · act delay {int(inner._action_delay_idx[w])} · "
                f"gains (arm k, hand k, arm d, hand d) x{', '.join(f'{v:.2f}' for v in inner._gain_scale[w].tolist())}\n\n"
                f"{rate:.0f} steps/s (real time {1 / dt:.0f})"
            )
            lag = dt / max(g_speed.value, 1e-3) - (time.time() - t0)
            if lag > 0:
                time.sleep(lag)

    run()


if __name__ == "__main__":
    main()
