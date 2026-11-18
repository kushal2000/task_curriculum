"""Watch a cloth checkpoint in the browser, built exactly as ``eval_cloth.py`` builds it.

    .venv/isaacsim/bin/python isaacsimenvs/scripts/evaluation/viser_eval_cloth.py \\
        --config <run>/.hydra/config.yaml --checkpoint <run>/nn/<name>.pth --num_envs 8 \\
        ['env.rigid_cloth.designs=[strip_4.yaml]' ...]

Then open http://localhost:8080 (on a cluster node: ``ssh -L 8080:localhost:8080 <node>``).

Same arguments and overrides as ``eval_cloth.py``, and the same env, policy, clipping and frozen
thickness (both call ``eval_cloth.build``), so what you watch is what the evaluation scores.
All ``--num_envs`` envs are stepped by the policy and reset automatically; one of them is drawn.
Drawn: the collision geometry PhysX simulates (robot, table, cloth), the cloth's current fold
keypoints (orange) and their fold targets (green).
"""

from __future__ import annotations

import argparse
import threading
import time

import numpy as np

from isaacsimenvs.scripts.evaluation.eval_cloth import PenetrationMeter, add_common_args, build, launch

ROBOT_RGB = (205, 208, 214)
TABLE_RGB = (96, 112, 138)
CLOTH_RGB = (242, 120, 60)
KEYPOINT_RGB = (255, 165, 25)
TARGET_RGB = (40, 200, 90)
# Top-face point cloud (obs.num_points > 0): samples on the stationary / moving half, and goals.
POINT_RGB = {False: (60, 120, 230), True: (230, 60, 160)}
POINT_GOAL_RGB = (40, 200, 90)


def _color(asset_name: str):
    if asset_name.startswith("cloth"):
        return CLOTH_RGB
    return TABLE_RGB if "table" in asset_name else ROBOT_RGB


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_common_args(parser, num_envs=8)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True  # the browser is the viewer
    app = launch(args)

    import torch
    import viser

    # Geometry from the USD stage and poses from the PhysX views, per Isaac Lab asset.
    from isaacsimenvs.scripts.visualization.policies.live_viser import _PhysxScene

    b = build(args)
    env, inner, n = b.env, b.inner, b.inner.num_envs
    origins = inner.scene.env_origins.cpu().numpy()
    scene = _PhysxScene(inner)

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/ground", width=3.0, height=3.0, cell_size=0.1)

    def label(e: int) -> str:
        d = b.dims[e]
        size = f"{d[0] * 100:.1f} x {d[1] * 100:.1f} cm" if d else ""
        return f"env {e}: {b.designs[e]} {size}"

    server.gui.add_markdown(f"**{b.task}**  \n`{args.checkpoint.name}`"
                            + (f"  \noverrides: `{' '.join(args.overrides)}`" if args.overrides else ""))
    playback = server.gui.add_button("Pause", icon=viser.Icon.PLAYER_PAUSE)
    step_button = server.gui.add_button("Single step", hint="Advance one policy step while paused")
    env_menu = server.gui.add_dropdown("Drawn env", tuple(label(e) for e in range(n)), initial_value=label(0))
    reset_button = server.gui.add_button("Reset all envs")
    speed = server.gui.add_slider("Playback speed", min=0.1, max=4.0, step=0.1, initial_value=1.0)
    has_points = inner.surface_points_w() is not None
    show_points = server.gui.add_checkbox("Show point cloud", initial_value=True, disabled=not has_points,
                                          hint="Top-face samples (blue stationary, pink moving) and goals (green)"
                                          if has_points else "Off: run with obs.num_points=64 to sample one")
    status = server.gui.add_markdown("")
    tally = server.gui.add_markdown("")

    lock = threading.Lock()
    requests = {"step": 0, "reset": False, "playing": True}

    @playback.on_click
    def _playback(_) -> None:
        requests["playing"] = not requests["playing"]
        # The button names the action it will take next.
        playback.label = "Pause" if requests["playing"] else "Play"
        playback.icon = viser.Icon.PLAYER_PAUSE if requests["playing"] else viser.Icon.PLAYER_PLAY
    step_button.on_click(lambda _: requests.__setitem__("step", requests["step"] + 1))
    reset_button.on_click(lambda _: requests.__setitem__("reset", True))

    drawn = {"env": None, "handles": {}, "keypoints": None, "targets": None, "points": None, "goals": None}

    def draw_env(w: int) -> None:
        with lock:
            for h in drawn["handles"].values():
                h.remove()
            drawn["handles"] = {}
            for i, s in enumerate(scene.shapes(w)):
                name = scene.assets[s.key[0]][0]
                drawn["handles"][s.key, i] = server.scene.add_mesh_simple(
                    f"/env/{i}", vertices=s.verts, faces=s.faces, color=_color(name))
            if drawn["keypoints"] is None:
                drawn["keypoints"] = server.scene.add_point_cloud(
                    "/keypoints", points=np.zeros((1, 3)), colors=KEYPOINT_RGB, point_size=0.012)
                drawn["targets"] = server.scene.add_point_cloud(
                    "/targets", points=np.zeros((1, 3)), colors=TARGET_RGB, point_size=0.012)
            if has_points and drawn["points"] is None:
                drawn["points"] = server.scene.add_point_cloud(
                    "/cloud/points", points=np.zeros((1, 3)), colors=POINT_RGB[False], point_size=0.005)
                drawn["goals"] = server.scene.add_point_cloud(
                    "/cloud/goals", points=np.zeros((1, 3)), colors=POINT_GOAL_RGB, point_size=0.005)
            drawn["env"] = w

    def update_poses(w: int) -> None:
        keys = {k for k, _ in drawn["handles"]}
        poses = scene.poses(w, keys)
        for (key, _i), h in drawn["handles"].items():
            p, q = poses[key]
            h.position, h.wxyz = tuple(p - origins[w]), tuple(q)
        drawn["keypoints"].points = (inner.cloth_keypoints_w()[w].cpu().numpy() - origins[w]).astype(np.float32)
        drawn["targets"].points = (inner.fold_targets_w()[w].cpu().numpy() - origins[w]).astype(np.float32)
        if has_points:
            moving = inner.backend.surface.moving[w].cpu().numpy()
            drawn["points"].points = (inner.surface_points_w()[w].cpu().numpy() - origins[w]).astype(np.float32)
            drawn["points"].colors = np.array([POINT_RGB[bool(m)] for m in moving], dtype=np.uint8)
            drawn["goals"].points = (inner.point_goals_w()[w].cpu().numpy() - origins[w]).astype(np.float32)
            drawn["points"].visible = drawn["goals"].visible = show_points.value

    @server.on_client_connect
    def _camera(client) -> None:
        center = inner.table.data.root_pos_w[drawn["env"] or 0].cpu().numpy() - origins[drawn["env"] or 0]
        client.camera.position = tuple(center + (0.85, -0.85, 0.67))
        client.camera.look_at = tuple(center + (0.0, 0.0, 0.1))
        client.camera.up_direction = (0.0, 0.0, 1.0)

    episodes = np.zeros(n, dtype=int)
    successes = np.zeros(n, dtype=int)
    episode_step = np.zeros(n, dtype=int)
    last_outcome = ["-"] * n
    pen = PenetrationMeter(inner)
    every_env = torch.ones(n, dtype=torch.bool, device=inner.device)

    def reset_all():
        nonlocal obs
        obs, _ = env.reset()
        b.player.reset_rnn()
        episode_step[:] = 0
        pen.reset(every_env.nonzero(as_tuple=True)[0])

    obs = None
    steps_run = 0
    with torch.inference_mode():
        reset_all()
    draw_env(0)
    print(f"[viser_eval_cloth] serving on http://localhost:{args.port}", flush=True)

    # Everything under inference mode: the policy creates its LSTM state there, and resetting
    # that state outside it raises.
    with torch.inference_mode():
        while app is None or app.is_running():
            start = time.perf_counter()
            w = [label(e) for e in range(n)].index(env_menu.value)
            if w != drawn["env"]:
                draw_env(w)
            if requests["reset"]:
                requests["reset"] = False
                reset_all()
            stepping = requests["playing"] or requests["step"] > 0
            if stepping:
                requests["step"] = max(0, requests["step"] - 1)
                obs, _, terminated, truncated, extras = env.step(b.act(obs))
                episode_step += 1
                pen.update(every_env)  # this step's depth enters the episode max before any reset
                ended = (terminated | truncated).cpu().numpy()
                if ended.any():
                    final = extras["episode_final"]
                    ok = (final["successes"] > 0).cpu().numpy()
                    fired = {k[5:]: v.bool().cpu().numpy() for k, v in final.items() if k.startswith("done_")}
                    for e in np.flatnonzero(ended):
                        episodes[e] += 1
                        successes[e] += int(ok[e])
                        flags = [k for k, v in fired.items() if v[e]]
                        last_outcome[e] = ("SUCCESS (fold + release)" if ok[e] else "failed") + \
                            f" at step {episode_step[e]} -- {', '.join(flags) or 'no flags'}" + \
                            (f"; max penetration {float(pen.episode_max[e]) * 1000:.1f} mm" if pen.available else "")
                        episode_step[e] = 0
                    pen.reset(torch.as_tensor(np.flatnonzero(ended), device=inner.device))
                    b.player.reset_rnn(torch.as_tensor(np.flatnonzero(ended), device=inner.device))
            with lock:
                update_poses(w)
            fold_err = float(inner.fold_error()[w])
            status.content = (("▶ playing" if requests["playing"] else "⏸ paused") + f"  \n**{label(w)}**  \nepisode step {episode_step[w]} / {int(inner.max_episode_length)}  \n"
                              f"fold error {fold_err * 1000:.1f} mm (fold < "
                              f"{inner.cfg.task.keypoint_tolerance * 1000:.0f} mm)  \n"
                              + (lambda d: f"point cloud: matched {float(d[0][w]) * 1000:.1f} mm, "
                                 f"chamfer {float(d[1][w]) * 1000:.1f} mm  \n" if d else "")(inner.point_distances())
                              + (f"robot penetration now {float(pen.now[w]) * 1000:.1f} mm, "
                                 f"episode max {float(pen.episode_max[w]) * 1000:.1f} mm  \n" if pen.available
                                 else "robot penetration unavailable (contact reporting off)  \n")
                              + f"last episode: {last_outcome[w]}")
            total = int(episodes.sum())
            tally.content = (f"**All {n} envs:** {total} episodes, {int(successes.sum())} successes"
                             + (f" ({successes.sum() / total:.0%})" if total else ""))
            if stepping:
                steps_run += 1
                if steps_run % 500 == 0:
                    print(f"[viser_eval_cloth] step {steps_run}: {tally.content}", flush=True)
                time.sleep(max(0.0, inner.step_dt / speed.value - (time.perf_counter() - start)))
            else:
                time.sleep(0.05)  # paused: never advance the sim outside env.step


if __name__ == "__main__":
    main()
