"""RGB video of the pretrained policy running on the Newton backend.

    xvfb-run -a scripts/newton_py -m isaacsimenvs.eval.render_newton --steps 900

Output defaults under ``videos/``, which is gitignored -- renders are reproducible from the
command above, so there is no reason to carry them in history.

Renders through Newton's own ``ViewerGL``, which draws the Newton ``Model``/``State`` directly.
No Kit, and no Newton-state-to-USD sync -- the kit-less backend has neither.

Two environment notes, both load-bearing:

* **Needs a display.** ``ViewerGL`` is pyglet-based and raises ``NoSuchDisplayException`` with no
  X server, so it runs under ``xvfb-run``. ``ViewerRTX`` would avoid that but needs ``ovrtx``,
  which is not installed.
* **``show_collision`` must be on.** The robot's shapes carry the collide flag but not the
  visible one -- they are collision meshes that ``push_collision_api_to_meshes`` authored onto
  mesh prims under Xform wrappers during import. With the default settings the viewer hides them
  and the robot renders as an invisible arm holding a floating tool.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from isaacsimenvs.eval.protocol import disable_randomization, use_single_object_variant


#: Per-role render colours. The scene is otherwise uniformly blue/white, which makes a thin cable
#: on a blue table nearly impossible to follow.
SHAPE_COLORS = {
    "cable": (0.95, 0.42, 0.10),   # orange -- the manipuland, the thing to watch
    "table": (0.30, 0.42, 0.58),   # slate  -- the work surface
    "robot": (0.88, 0.89, 0.92),   # near-white
    "goal": (0.25, 0.75, 0.55),    # green  -- the goal marker
}


#: Overlay colours. Goal keypoints green, object keypoints amber, the error lines between them
#: fading from amber to green -- so the thing the success test measures is what you see.
GOAL_KP_COLOR = (0.25, 0.80, 0.45)
OBJ_KP_COLOR = (1.00, 0.65, 0.10)


def _goal_overlay(viewer, inner, world: int):
    """Draw the keypoints and per-keypoint error that define success (`--goal_keypoints`).

    Secondary to `_reveal_goal_viz`, which renders the task's own goal marker and is the default.
    This adds what the marker cannot show: success is the *max* over per-keypoint distances held
    for `success_steps`, so the quantity actually being thresholded is these segment lengths.

    """
    import warp as wp

    from isaacsimenvs.tasks.play.utils.obs_utils import _keypoints_world

    rew_cfg = inner.cfg.reward
    offsets = (
        inner._keypoint_offsets_fixed
        if rew_cfg.fixed_size_keypoint_reward
        else inner._keypoint_offsets
    )
    # World frame: the viewer draws in world coordinates, and `root_pos_w` is already there.
    obj_kp = _keypoints_world(
        inner.object.data.root_pos_w, inner.object.data.root_quat_w, offsets
    )[world]
    goal_kp = _keypoints_world(
        inner.goal_viz.data.root_pos_w, inner.goal_viz.data.root_quat_w, offsets
    )[world]

    g = goal_kp.detach().cpu().numpy().astype("float32")
    o = obj_kp.detach().cpu().numpy().astype("float32")
    dev = getattr(viewer, "device", None)

    # Colours must be per-element arrays here, not a single tuple -- the GL backend calls
    # `.numpy()` on whatever it is given and a tuple raises inside `_update_vbo`.
    def _col(rgb, n):
        return wp.array([rgb] * n, dtype=wp.vec3, device=dev)

    def _pts(arr):
        return wp.array(arr, dtype=wp.vec3, device=dev)

    viewer.log_points("goal_kp", _pts(g), radii=0.012, colors=_col(GOAL_KP_COLOR, len(g)))
    viewer.log_points("obj_kp", _pts(o), radii=0.009, colors=_col(OBJ_KP_COLOR, len(o)))
    viewer.log_lines(
        "goal_err", _pts(o), _pts(g), colors=_col(GOAL_KP_COLOR, len(g)), width=0.003
    )
    # The goal frame itself, as a wireframe quad through the four keypoints.
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    viewer.log_lines(
        "goal_frame",
        _pts([g[a] for a, _ in edges]),
        _pts([g[b] for _, b in edges]),
        colors=_col(GOAL_KP_COLOR, len(edges)),
        width=0.004,
    )


def _reveal_goal_viz(model) -> int:
    """Make the goal marker's own shapes render, instead of drawing a stand-in over the top.

    The marker is already in the Newton model -- `_colorize` finds and colours its shapes -- but
    they import with `shape_flags == 0`: neither VISIBLE nor COLLIDE_SHAPES. The viewer draws
    collide-flagged shapes (`show_collision=True`) and visible ones, so a shape with neither is
    silently skipped. Setting VISIBLE is the whole fix; collision is already off, which is exactly
    what a goal marker wants -- it must not push the manipuland around.

    Preferred over an immediate-mode overlay because it is the marker the task itself maintains:
    it tracks the goal in every world without the render path re-deriving anything, so what is on
    screen cannot drift from what the env believes the goal is.
    """
    import warp as wp

    if getattr(model, "shape_flags", None) is None:
        return 0
    labels = [str(x) for x in model.body_label]
    shape_body = model.shape_body.numpy()
    flags = model.shape_flags.numpy().copy()
    visible = int(newton_shape_flags().VISIBLE)
    n = 0
    for shape_idx, body_idx in enumerate(shape_body):
        label = labels[body_idx] if 0 <= body_idx < len(labels) else ""
        if "GoalViz" in label:
            flags[shape_idx] = int(flags[shape_idx]) | visible
            n += 1
    model.shape_flags = wp.array(
        flags, dtype=model.shape_flags.dtype, device=model.shape_flags.device
    )
    print(f"[render] revealed {n} goal-marker shapes", flush=True)
    return n


def _restrict_triangles_to_world(viewer, model, world: int) -> None:
    """Draw only the filmed world's deformable surface.

    ``set_visible_worlds`` filters *rigid shapes* only -- the viewer's ``_log_triangles`` logs
    ``state.particle_q`` and ``model.tri_indices`` wholesale, with no world mask anywhere in that
    path (`newton/_src/viewer/viewer.py:2725`). So on a deformable task every environment's sheet is
    drawn while every environment's table and robot is hidden, and the result reads as one robot
    surrounded by loose cloths floating over the floor. That is a viewer limitation, not a physics
    fault -- but it is indistinguishable from a physics fault in a video.

    The fix keeps all particle positions (the mesh indexes into that array, so it must stay whole)
    and narrows the drawn triangles to those whose vertices belong to the filmed world. Membership
    comes from ``model.particle_world``, the per-particle world index Newton already builds
    (`newton/_src/sim/builder.py:1226`) -- not from ``world * particle_count // world_count``, which
    would silently mis-slice if the worlds ever held different particle counts.

    Patched on the **viewer**, not the model: ``model.tri_indices`` is what the VBD solver reads for
    its elastic forces, so rewriting it there would silently change the physics to make a picture
    look right -- the exact trade this project keeps refusing to make.
    """
    import types

    import numpy as np
    import warp as wp

    tri = getattr(model, "tri_indices", None)
    owner = getattr(model, "particle_world", None)
    if tri is None or model.tri_count == 0 or model.world_count <= 1 or owner is None:
        return

    mine = np.asarray(owner.numpy()) == world
    idx = tri.numpy()
    keep = mine[idx].all(axis=1)
    if not keep.any():
        print("[render] triangle world filter matched nothing; leaving all worlds drawn", flush=True)
        return
    kept = wp.array(idx[keep].flatten(), dtype=wp.int32, device=tri.device)

    def _log_triangles(self, state, _indices=kept):
        points = self._apply_layer_transform_to_points(state.particle_q)
        self.log_mesh(
            self._qualify("/model/triangles"),
            points,
            _indices,
            hidden=not self.show_triangles or self._layer_force_hidden(),
            backface_culling=False,
        )

    viewer._log_triangles = types.MethodType(_log_triangles, viewer)
    print(
        f"[render] deformable surface limited to world {world}: "
        f"{int(keep.sum())}/{len(keep)} triangles",
        flush=True,
    )


def newton_shape_flags():
    import newton

    return newton.ShapeFlags


def _segment_color(index: int, total: int):
    """Hue sweep along the cable: segment 0 red, running through to violet at the far end."""
    import colorsys

    hue = 0.02 + 0.80 * (index / max(total - 1, 1))
    return colorsys.hsv_to_rgb(hue, 0.85, 0.95)


def _colorize(model, args) -> None:
    """Colour Newton shapes by role, so table, cable and robot are distinguishable.

    `ViewerGL` renders *collision* shapes (we run with `show_collision=True`, without which the
    robot is invisible) and takes their colour from `model.shape_color`, re-syncing it from the
    model every frame. A USD `visual_material` on the spawn does **not** reach it -- that colours
    the visual prim, which this render path never draws.

    Shapes carry no useful labels of their own (Newton names them `shape_0`, `shape_1`, ...), so
    each is classified through the body that owns it: `shape_body` gives the body index and
    `body_label` carries the USD path.
    """
    import numpy as np
    import warp as wp

    if getattr(model, "shape_color", None) is None:
        print("[render] model has no shape_color; skipping colorization", flush=True)
        return

    labels = [str(x) for x in model.body_label]
    shape_body = model.shape_body.numpy()
    colors = model.shape_color.numpy().copy()

    # Cable bodies are labelled `.../mesh_edge_body_<i>`, so segments can be coloured by their
    # true index rather than by shape ordering. A hue sweep along the cable makes bending, twist
    # and which end is which readable at a glance -- a single colour renders as one orange blob.
    import re

    seg_ids = [
        int(m.group(1))
        for lab in labels
        if (m := re.search(r"mesh_edge_body_(\d+)$", lab))
    ]
    n_seg = (max(seg_ids) + 1) if seg_ids else 1

    counts = dict.fromkeys(SHAPE_COLORS, 0)
    for shape_idx, body_idx in enumerate(shape_body):
        label = labels[body_idx] if 0 <= body_idx < len(labels) else ""
        if "/Rod" in label:
            # The rigid-rod control lives at its own prim path so the coupler's VBD entry cannot
            # claim it; it still wants the manipuland colour.
            colors[shape_idx] = SHAPE_COLORS["cable"]
            counts["cable"] += 1
            continue
        if "Cable" in label:
            role = "cable"
            match = re.search(r"mesh_edge_body_(\d+)$", label)
            if match and n_seg > 1 and not args.flat_cable:
                colors[shape_idx] = _segment_color(int(match.group(1)), n_seg)
                counts[role] += 1
                continue
        elif "Table" in label:
            role = "table"
        elif "GoalViz" in label:
            role = "goal"
        elif "Robot" in label:
            role = "robot"
        else:
            continue
        colors[shape_idx] = SHAPE_COLORS[role]
        counts[role] += 1

    model.shape_color = wp.array(colors, dtype=model.shape_color.dtype, device=model.shape_color.device)
    print(f"[render] colorized shapes: {counts}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the policy on Newton to a video.")
    parser.add_argument("--task", default="Isaacsimenvs-PlayNewton-Direct-v0")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--stride", type=int, default=2, help="capture every Nth policy step")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--world", type=int, default=0, help="which env to film")
    parser.add_argument(
        "--cam_offset",
        type=float,
        nargs=3,
        default=(1.6, -1.6, 1.15),
        help="camera position relative to the filmed env's origin",
    )
    parser.add_argument(
        "--look_at",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.62),
        help="aim point relative to the filmed env's origin. Default is roughly the object's "
        "start height (table_reset_z 0.38 + table_object_z_offset 0.25).",
    )
    parser.add_argument("--checkpoint", default="/share/portal/kk837/simtoolreal/pretrained_policy/model.pth")
    parser.add_argument("--policy_config", default="/share/portal/kk837/simtoolreal/pretrained_policy/config.yaml")
    parser.add_argument("--success_tolerance", type=float, default=0.01)
    parser.add_argument(
        "--zero_action",
        action="store_true",
        help="Hold zero actions instead of running the policy. The diagnostic mode: whatever "
        "the object does here is the scene's own dynamics, not the controller's.",
    )
    parser.add_argument(
        "--flat_cable",
        action="store_true",
        help="Colour the cable one solid colour instead of a per-segment hue sweep.",
    )
    parser.add_argument("--num_assets_per_type", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, default=Path("videos/play_newton.mp4"))
    parser.add_argument(
        "--no_goal", action="store_true", help="do not render the goal marker at all"
    )
    parser.add_argument(
        "--goal_keypoints",
        action="store_true",
        help="additionally draw the keypoints and per-keypoint error that define success",
    )
    args, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args

    import gymnasium as gym
    import imageio.v2 as imageio
    import numpy as np
    import torch
    import warp as wp

    import isaacsimenvs  # noqa: F401  gym.register side effects
    from isaacsimenvs.eval.player import PretrainedPlayer
    from isaacsimenvs.utils.hydra_utils import hydra_task_config_with_yaml

    @hydra_task_config_with_yaml(args.task, "")
    def run(env_cfg, agent_cfg) -> None:
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.assets.num_assets_per_type = args.num_assets_per_type
        env_cfg.assets.handle_head_types = ("hammer",)
        env_cfg.sim.device = args.device
        env_cfg.seed = args.seed
        env_cfg.termination.eval_success_tolerance = args.success_tolerance
        disable_randomization(env_cfg)
        use_single_object_variant()

        env = gym.make(args.task, cfg=env_cfg)
        inner = env.unwrapped

        player = None if args.zero_action else PretrainedPlayer(
            config_path=args.policy_config,
            checkpoint_path=args.checkpoint,
            num_envs=inner.num_envs,
            device=args.device,
            num_observations=int(inner.cfg.observation_space),
            num_actions=int(inner.cfg.action_space),
        )

        from isaaclab_newton.physics.newton_manager import NewtonManager
        from newton.viewer import ViewerGL

        viewer = ViewerGL(width=args.width, height=args.height, headless=True)
        # See the module docstring: the robot's colliders are not flagged visible.
        viewer.show_collision = True
        viewer.show_static = True
        if not args.no_goal:
            _reveal_goal_viz(NewtonManager.get_model())
        viewer.set_model(NewtonManager.get_model())
        viewer.set_visible_worlds([args.world])
        _restrict_triangles_to_world(viewer, NewtonManager.get_model(), args.world)
        _colorize(NewtonManager.get_model(), args)

        origin = inner.scene.env_origins[args.world].detach().cpu().numpy()
        cam = origin + np.asarray(args.cam_offset, dtype=float)
        target = origin + np.asarray(args.look_at, dtype=float)

        # `set_camera(pos, pitch, yaw)` takes angles, not a target, so derive them. Z-up, yaw
        # measured from +X about Z. Confirmed against the reference's known-good framing: camera
        # (1.62, -0.78, 0.86) aimed near (1.0, 0, 0.53) gives yaw 128.5 / pitch -18.3, against
        # the 126.43 / -19.03 it used.
        d = target - cam
        yaw = float(np.degrees(np.arctan2(d[1], d[0])))
        pitch = float(np.degrees(np.arctan2(d[2], np.hypot(d[0], d[1]))))
        viewer.set_camera(wp.vec3(*(float(v) for v in cam)), pitch, yaw)
        print(
            f"[render] env_{args.world} origin {origin.round(3)} camera {cam.round(3)} "
            f"-> target {target.round(3)}  (pitch {pitch:.1f}, yaw {yaw:.1f})",
            flush=True,
        )

        obs, _ = env.reset()
        obs, *_ = env.step(torch.zeros((inner.num_envs, inner.cfg.action_space), device=inner.device))

        args.out.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(args.out, fps=args.fps, macro_block_size=None)
        frames = 0
        # `_successes` is zeroed when an episode terminates, so an end-minus-start read reports
        # ~0 for any env whose episode ended mid-render -- which is every *scoring* env, since
        # scoring extends episodes past a short render. That made six consecutive renders of
        # 2-to-6-goal episodes all caption "scored ~0 goals". Accumulate positive deltas instead,
        # which survives the reset. (`episodes.py` avoids this by snapshotting
        # `extras["episode_final"]` at the termination step.)
        goals_seen = 0
        prev_successes = int(inner._successes[args.world].item())

        for step in range(args.steps):
            if player is None:
                action = torch.zeros(
                    (inner.num_envs, inner.cfg.action_space), device=inner.device
                )
            else:
                action = player.get_action(obs["policy"], deterministic=True)
            obs, _rew, terminated, truncated, _extras = env.step(action.to(inner.device))

            now = int(inner._successes[args.world].item())
            if now > prev_successes:
                goals_seen += now - prev_successes
            prev_successes = now

            done = (terminated.bool() | truncated.bool())
            if player is not None and bool(done.any()):
                player.reset_rnn(done.nonzero(as_tuple=True)[0])

            if step % args.stride:
                continue

            viewer.begin_frame(step * float(inner.step_dt))
            viewer.log_state(NewtonManager._state_0)
            if args.goal_keypoints:
                _goal_overlay(viewer, inner, args.world)
            viewer.end_frame()

            image = wp.to_torch(viewer.get_frame()).detach().cpu().numpy()
            if image.dtype != np.uint8:
                image = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
            # ViewerGL already returns top-down rows, so no vertical flip -- adding one puts the
            # ground plane at the top of the frame. Just drop the alpha channel.
            writer.append_data(np.ascontiguousarray(image[:, :, :3]))
            frames += 1

            if frames % 50 == 0:
                print(
                    f"[render] step {step}/{args.steps}  frames {frames}  "
                    f"goals(env {args.world}) {int(inner._successes[args.world].item())}",
                    flush=True,
                )

        writer.close()
        goals = goals_seen
        # Tolerate the file having been moved or removed between the write and this stat: the
        # size is a nicety, and crashing here discards the goal count -- which is the one number
        # the render exists to report.
        size_mb = args.out.stat().st_size / 1e6 if args.out.exists() else float("nan")
        print(
            f"\n[render] wrote {args.out} ({frames} frames, {size_mb:.1f} MB, "
            f"{frames / args.fps:.1f}s)  env_{args.world} scored ~{goals} goals",
            flush=True,
        )
        env.close()

    run()

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
