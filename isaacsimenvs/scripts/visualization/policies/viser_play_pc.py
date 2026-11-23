"""Play a point-cloud Play student live in Isaac Sim and watch it in viser (``viser_play`` for students).

The student (``distill_play_pc``: ``student.SplitStudentNet``) drives the point-cloud Play task
(``Isaacsimenvs-PlayFlexivWujiRightHumanLikePC-Direct-v0``) alone, one env per tool, all stepped
together, with the env's training randomization (the camera errors, drops and delays are what the
student reads). Tools are fresh seed samples of training's 12 size distributions over the 6
handle-head types, ``--num_assets_per_type`` (default 1) from each.

Viser draws, for the watched env: the robot, the tool, the goal (translucent green), the table and
what the network reads -- the live camera cloud (grey) and the goal cloud (green points) -- and, from
its auxiliary head, where the student thinks the tool's keypoints are (orange) and where their goal is
(magenta), against the truth (small white / green spheres). The HUD gives the surface distance to the
goal, the estimate errors, and goals / episode per tool type over every env.

    OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD .venv/isaacsim/bin/python \\
        -m isaacsimenvs.scripts.visualization.policies.viser_play_pc --headless \\
        --student play_pc_dagger_64pts_cont --port 8082

``--student`` takes a ``student_*.pth``, a run directory (its newest checkpoint) or part of a run's name.
The panel lists the checkpoints of every student run under ``debug_outputs/train_logs`` (and the older
``outputs/distill_play``) that reads the same observation; "load" swaps one in
and resets the envs. On the cluster, forward the port: ``ssh -N -L 8082:localhost:8082 <node>``.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
#: Where student runs live: one directory per run (dagger.sbatch), older runs under outputs/distill_play.
RUN_ROOTS = (REPO / "debug_outputs" / "train_logs", REPO / "outputs" / "distill_play")
# isaacsim-kernel pins websockets==12.0 and viser needs >=13.1 (see viser_play.py).
_WEBSOCKETS = REPO / "third_party" / "viser_websockets"
if _WEBSOCKETS.is_dir():
    sys.path.insert(0, str(_WEBSOCKETS))


def _iter_of(p: Path) -> int:
    m = re.search(r"student_it(\d+)\.pth$", p.name)
    return int(m.group(1)) if m else 1 << 30  # student_last sorts last


def checkpoints(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("student_*.pth"), key=_iter_of)


def resolve(spec: str) -> Path:
    p = Path(spec)
    if not p.exists():
        p = next((d for d in student_runs().values() if spec in d.name), p)
    if p.is_dir():
        cks = checkpoints(p)
        if not cks:
            raise SystemExit(f"{p}: no student_*.pth")
        return cks[-1]
    if not p.is_file():
        raise SystemExit(f"{spec}: not a checkpoint, a run directory, or (part of) a run name under {RUN_ROOTS}")
    return p


def student_runs() -> dict[str, Path]:
    """Every directory holding ``student_*.pth`` under ``RUN_ROOTS`` (two levels deep), by name."""
    out = {}
    for root in RUN_ROOTS:
        for d in sorted(root.glob("*")) + sorted(root.glob("*/*")) if root.is_dir() else []:
            if d.is_dir() and not d.is_symlink() and any(d.glob("student_*.pth")):
                out.setdefault(d.name, d)
    return out


def object_type(urdf_path: str) -> str:
    m = re.match(r"\d+_([a-z]+)_handle", Path(urdf_path).name)
    return m.group(1) if m else Path(urdf_path).stem


def main() -> None:
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--student", required=True, help="student_*.pth, or a run directory (newest checkpoint)")
    p.add_argument("--port", type=int, default=8082)
    p.add_argument("--num_assets_per_type", type=int, default=1, help="tools per size distribution (12 distributions)")
    p.add_argument("--tolerance", type=float, default=0.01, help="success tolerance [m], as the student was distilled")
    p.add_argument("--seed", type=int, default=0)
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    args.headless = True
    app = AppLauncher(args).app

    import gymnasium as gym
    import numpy as np
    import torch
    import viser
    import yourdfpy
    from viser.extras import ViserUrdf

    import isaacsimenvs  # noqa: F401  gym.register side effects
    from isaacsimenvs.scripts.distillation.student import SplitStudentNet
    from isaacsimenvs.tasks.play.utils.object_size_distributions import OBJECT_SIZE_DISTRIBUTIONS
    from isaacsimenvs.tasks.play.utils.obs_utils import _keypoints_world

    ckpt = resolve(args.student)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)

    # The env config of the student's own task (its gym registration), e.g. the real-limits variant.
    mod, cls = gym.spec(ck["env"]["task"]).kwargs["env_cfg_entry_point"].split(":")
    cfg = getattr(__import__(mod, fromlist=[cls]), cls)()
    cfg.assets.num_assets_per_type = int(args.num_assets_per_type)
    types_cfg = set(cfg.assets.handle_head_types)
    cfg.scene.num_envs = int(args.num_assets_per_type) * sum(d.type in types_cfg for d in OBJECT_SIZE_DISTRIBUTIONS)
    cfg.seed = int(args.seed)
    cfg.termination.eval_success_tolerance = float(args.tolerance)
    # The policy's observation only: no critic clouds, no teacher vector (the keypoint truth the
    # auxiliary estimates are drawn against is computed for the watched env alone, in draw()).
    cfg.pointcloud.extra_obs = ()
    env = gym.make(ck["env"]["task"], cfg=cfg)
    inner = env.unwrapped
    num_envs, device = inner.num_envs, inner.device
    lay = inner.layout
    n_p, m, n_live = int(lay["n_proprio"]), int(lay["num_points"]), int(lay["n_live"])
    n_act = int(inner.cfg.action_space)
    scale = float(inner.cfg.pointcloud.scale)

    urdf_paths = list(inner._object_urdf_paths)
    asset_of_env = inner._object_asset_index_per_env.cpu().numpy()
    type_of_env = [object_type(urdf_paths[a]) for a in asset_of_env]
    types = sorted(set(type_of_env))
    variants = {t: [e for e in range(num_envs) if type_of_env[e] == t] for t in types}

    # --- policy -------------------------------------------------------------------------------
    pol: dict = {}

    def load(path: Path) -> None:
        c = torch.load(path, map_location="cpu", weights_only=False)
        net_args = {**c["net_args"], "compile": False}   # same math; no compile wait at a handful of envs
        net = SplitStudentNet(n_p, n_act, 3, int(c["model"]["aux.2.weight"].shape[0]), **net_args).to(device)
        net.load_state_dict(c["model"])
        pol.update(net=net.eval(), path=path, iter=c.get("iter"), net_args=c["net_args"],
                   aux=float(c.get("aux_coef", 1.0)) > 0)   # an untrained aux head is not drawn
        print(f"[viser_play_pc] loaded {path} (iter {c.get('iter')})", flush=True)

    load(ckpt)

    # Runs whose student reads this env's layout (same task, same cloud sizes).
    runs = {}
    for name, d in student_runs().items():
        try:
            meta = __import__("json").loads((d / "student_meta.json").read_text())
            ok = meta["env"]["task"] == ck["env"]["task"] and meta["env"]["layout"]["policy_dim"] == lay["policy_dim"]
        except (OSError, KeyError, ValueError):
            ok = False
        if ok:
            runs[name] = d

    # --- viser scene --------------------------------------------------------------------------
    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/ground", width=3.0, height=3.0, cell_size=0.1)
    robot_urdf = yourdfpy.URDF.load(str(REPO / inner.cfg.assets.robot_urdf))
    robot_root = server.scene.add_frame("/robot", show_axes=False)
    viser_robot = ViserUrdf(server, robot_urdf, root_node_name="/robot")
    table_root = server.scene.add_frame("/table", show_axes=False)
    ViserUrdf(server, yourdfpy.URDF.load(str(REPO / inner.cfg.assets.table_urdf)), root_node_name="/table",
              mesh_color_override=(0.82, 0.82, 0.84))
    robot_joint_names = list(inner.robot.joint_names)
    viser_joint_order = list(robot_urdf.actuated_joint_names)
    nodes: dict = {}

    def show_object(e: int) -> None:
        for key in ("object", "goal"):
            if key in nodes:
                nodes[key][1].remove()
                nodes[key][0].remove()
        urdf = yourdfpy.URDF.load(urdf_paths[asset_of_env[e]])
        for key, color in (("object", (0.6, 0.35, 0.15, 1.0)), ("goal", (0.3, 0.85, 0.35, 0.3))):
            frame = server.scene.add_frame(f"/{key}", show_axes=False)
            nodes[key] = (frame, ViserUrdf(server, urdf, root_node_name=f"/{key}", mesh_color_override=color))

    far = np.zeros((1, 3), dtype=np.float32) - 10.0
    clouds = {
        "live": server.scene.add_point_cloud("/clouds/live", far, colors=(90, 90, 100), point_size=0.008),
        "goal": server.scene.add_point_cloud("/clouds/goal", far, colors=(40, 200, 90), point_size=0.006),
        "kp_true": server.scene.add_point_cloud("/kp/true", far, colors=(255, 255, 255), point_size=0.012),
        "kp_est": server.scene.add_point_cloud("/kp/est", far, colors=(255, 140, 0), point_size=0.018),
        "gkp_true": server.scene.add_point_cloud("/kp/goal_true", far, colors=(40, 220, 90), point_size=0.012),
        "gkp_est": server.scene.add_point_cloud("/kp/goal_est", far, colors=(230, 0, 200), point_size=0.018),
    }

    # --- GUI ----------------------------------------------------------------------------------
    # Callbacks only queue requests; the main loop applies them between steps.
    state = {"playing": True, "step_once": False, "reset": False, "env": 0, "t": 0, "want_env": None, "want_load": None}
    stats = {t: {"episodes": 0, "goals": 0} for t in types}
    ep_goals = np.zeros(num_envs, dtype=np.int64)

    run_names = list(runs) or [ckpt.parent.name]
    with server.gui.add_folder("Policy"):
        g_run = server.gui.add_dropdown("run", run_names, initial_value=ckpt.parent.name if ckpt.parent.name in run_names else run_names[0])
        g_ckpt = server.gui.add_dropdown("checkpoint", [c.name for c in checkpoints(ckpt.parent)], initial_value=ckpt.name)
        g_load = server.gui.add_button("load (resets all envs + stats)")
        g_policy_md = server.gui.add_markdown("")
    with server.gui.add_folder("Playback"):
        g_play = server.gui.add_button("play / pause")
        g_step = server.gui.add_button("step once")
        g_reset = server.gui.add_button("reset all envs + stats")
        g_speed = server.gui.add_slider("speed (x real time)", 0.1, 4.0, 0.1, 1.0)
        g_tol = server.gui.add_slider("success tolerance (m)", 0.005, 0.1, 0.001, float(args.tolerance))
    with server.gui.add_folder("Tool"):
        g_type = server.gui.add_dropdown("type", types, initial_value=types[0])
        g_variant = server.gui.add_slider("variant", 0, max(len(variants[types[0]]) - 1, 1), 1, 0)
        g_obj_md = server.gui.add_markdown("")
    with server.gui.add_folder("Show"):
        g_show = {k: server.gui.add_checkbox(label, True) for k, label in (
            ("live", "live camera cloud (grey)"), ("goal", "goal cloud (green)"),
            ("kp", "keypoints: estimate (orange) vs true (white)"),
            ("gkp", "goal keypoints: estimate (magenta) vs true (green)"))}
        g_show_goal_tool = server.gui.add_checkbox("goal tool (translucent)", True)
    md = server.gui.add_markdown("")

    @g_run.on_update
    def _(_) -> None:
        cks = checkpoints(runs.get(g_run.value, ckpt.parent))
        g_ckpt.options = [c.name for c in cks]
        g_ckpt.value = cks[-1].name

    @g_load.on_click
    def _(_) -> None:
        state["want_load"] = runs.get(g_run.value, ckpt.parent) / g_ckpt.value

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

    origins = inner.scene.env_origins

    def pose(data, e: int):
        return (tuple((data.root_pos_w[e] - origins[e]).cpu().numpy()), tuple(data.root_quat_w[e].cpu().numpy()))

    def set_points(key: str, pts: np.ndarray, show: bool) -> None:
        h = clouds[key]
        h.visible = bool(show) and len(pts) > 0
        if len(pts):
            h.points = pts.astype(np.float32)
            h.colors = np.tile(np.asarray(h.colors).reshape(-1, 3)[:1], (len(pts), 1))

    last = {"aux": None}
    timing: dict = {}

    def draw() -> None:
        e = state["env"]
        robot_root.position, robot_root.wxyz = pose(inner.robot.data, e)
        table_root.position, table_root.wxyz = pose(inner.table.data, e)
        q = dict(zip(robot_joint_names, inner.robot.data.joint_pos[e].cpu().numpy()))
        viser_robot.update_cfg(np.array([q[n] for n in viser_joint_order]))
        for key, data in (("object", inner.object.data), ("goal", inner.goal_viz.data)):
            nodes[key][0].position, nodes[key][0].wxyz = pose(data, e)
        nodes["goal"][0].visible = bool(g_show_goal_tool.value)

        xyz = inner.cloud.last_xyz[e, :n_live].float().cpu().numpy()
        ok = inner.cloud.last_mask[e, :n_live].cpu().numpy().astype(bool)
        set_points("live", xyz[ok], g_show["live"].value)
        set_points("goal", inner._last_goal_cloud[e].float().cpu().numpy(), g_show["goal"].value)

        err_kp = err_gkp = float("nan")
        for k in ("kp_true", "kp_est", "gkp_true", "gkp_est"):
            clouds[k].visible = False
        if last["aux"] is not None and pol["aux"]:
            # The truth, for the watched env only: the tool's keypoints, the palm (env frame, from the
            # clean keypoints_rel_palm) and the goal's keypoints at the symmetric goal nearest the tool,
            # as the teacher reads them (the auxiliary head's targets).
            sl = slice(e, e + 1)
            o_pos, o_quat, g_pos, g_quat = (x[sl] for x in inner._poses())
            kp_off = (inner._keypoint_offsets * inner._object_scale_multiplier.unsqueeze(1))[sl]
            kp_true = _keypoints_world(o_pos, o_quat, kp_off)[0]
            c_pos, c_quat = inner.shapes.canonical_goal_pose(inner._asset[sl], o_pos, o_quat, g_pos, g_quat, kp_off)
            gkp_true = _keypoints_world(c_pos, c_quat, kp_off)[0]
            palm = kp_true - inner._obs_fields[0]["keypoints_rel_palm"][e].reshape(-1, 3)
            est_aux = last["aux"][e].float().view(2, -1, 3) / scale
            kp_est = palm + est_aux[0]
            gkp_est = kp_est - est_aux[1]
            err_kp = float((kp_est - kp_true).norm(dim=-1).mean()) * 100
            err_gkp = float((gkp_est - gkp_true).norm(dim=-1).mean()) * 100
            set_points("kp_true", kp_true.cpu().numpy(), g_show["kp"].value)
            set_points("kp_est", kp_est.cpu().numpy(), g_show["kp"].value)
            set_points("gkp_true", gkp_true.cpu().numpy(), g_show["gkp"].value)
            set_points("gkp_est", gkp_est.cpu().numpy(), g_show["gkp"].value)

        rows = [f"| {t} ({len(variants[t])}) | {s['episodes']} | {s['goals'] / max(s['episodes'], 1):.2f} |"
                for t, s in stats.items()]
        eps = sum(s["episodes"] for s in stats.values())
        goals = sum(s["goals"] for s in stats.values())
        md.content = "\n".join([
            f"**{pol['path'].parent.name}**  \n`{pol['path'].name}` (iter {pol['iter']})",
            "",
            f"t = {state['t'] * float(inner.step_dt):.1f} s; env {e}: **{int(ep_goals[e])}** goals this episode  \n"
            f"surface distance to goal **{float(inner._keypoints_max_dist[e]) * 100:.1f} cm** "
            f"(success below {float(inner._current_success_tolerance) * float(inner.cfg.reward.keypoint_scale) * 100:.1f})  \n"
            f"live points {int(ok.sum())}/{n_live}; estimate error: keypoints {err_kp:.1f} cm, goal keypoints {err_gkp:.1f} cm",
            "",
            "| tool type (envs) | finished episodes | goals / episode |",
            "|---|---:|---:|",
            *rows,
            f"| **all** | {eps} | {goals / max(eps, 1):.2f} |",
        ])

    watch(variants[types[0]][0])
    obs, _ = env.reset()
    rnn = pol["net"].initial_state(num_envs, device)
    start = torch.ones(num_envs, dtype=torch.bool, device=device)
    draw()
    port = server.get_port()
    print(f"[viser_play_pc] {num_envs} envs ({', '.join(f'{t} {len(v)}' for t, v in variants.items())}); "
          f"viser on http://localhost:{port}", flush=True)

    while app.is_running():
        inner.cfg.termination.eval_success_tolerance = float(g_tol.value)
        if state["want_env"] is not None:
            watch(state["want_env"])
            state["want_env"] = None
            draw()
        if state["want_load"] is not None:
            load(state["want_load"])
            state.update(want_load=None, reset=True)
        if state["reset"]:
            obs, _ = env.reset()
            rnn = pol["net"].initial_state(num_envs, device)
            start[:] = True
            ep_goals[:] = 0
            for s in stats.values():
                s.update(episodes=0, goals=0)
            last["aux"] = None
            state.update(reset=False, t=0)
            draw()
        if not (state["playing"] or state["step_once"]):
            time.sleep(0.05)
            continue
        t0 = time.time()
        x = obs["policy"]
        with torch.no_grad():
            mu, aux, rnn = pol["net"].step(x[:, :n_p], x[:, n_p:n_p + 3 * m].reshape(-1, m, 3),
                                           x[:, n_p + 3 * m:n_p + 4 * m] > 0.5, rnn, start)
        last["aux"] = aux
        torch.cuda.synchronize()
        t1 = time.time()
        obs, _r, term, trunc, _ = env.step(mu.float().clamp(-1.0, 1.0))
        torch.cuda.synchronize()
        t2 = time.time()
        ep_goals += inner._is_success.cpu().numpy().astype(np.int64)
        start = term.bool() | trunc.bool()
        for e in start.nonzero(as_tuple=True)[0].cpu().numpy():
            s = stats[type_of_env[e]]
            s["episodes"] += 1
            s["goals"] += int(ep_goals[e])
            ep_goals[e] = 0
        state["t"] += 1
        # The scene at 30 Hz (every other 60 Hz policy step): drawing costs ~16 ms, the env ~25 ms.
        if state["t"] % 2 == 0 or state["step_once"]:
            draw()
        state["step_once"] = False
        t3 = time.time()
        for k, v in (("policy", t1 - t0), ("env", t2 - t1), ("draw", t3 - t2)):
            timing[k] = 0.95 * timing.get(k, v) + 0.05 * v
        if state["t"] % 200 == 0:
            print("[viser_play_pc] ms/step: " + ", ".join(f"{k} {v * 1e3:.1f}" for k, v in timing.items())
                  + f" (real time {float(inner.step_dt) * 1e3:.1f})", flush=True)
        time.sleep(max(0.0, float(inner.step_dt) / g_speed.value - (time.time() - t0)))
    env.close()
    app.close()


if __name__ == "__main__":
    main()
    sys.exit(0)
