"""Play the Flexiv-Wuji checkpoint on a live Newton/VBD cloth in Viser.

Run with .venv/isaaclab3/bin/python, then open http://localhost:8080.
The default matches the 10 cm Yibo-branch base evaluation; --preset materials
matches the 15 cm material sweep. Material/geometry changes restart this process.
"""

from __future__ import annotations

import argparse
from functools import partial
import json
import os
from pathlib import Path
import queue
import sys
import time

import numpy as np
import torch

# Support running this file directly without installing the repository package.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from isaacsimenvs import CONFIG_DIR, REPO_ROOT
from isaacsimenvs.tasks.cloth.backends.vbd import fabrics


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--material', choices=fabrics.names(), default='base')
    parser.add_argument('--preset', choices=('yibo', 'materials'), default='yibo')
    parser.add_argument('--size', type=float, help='Square sheet side length [m].')
    parser.add_argument('--resolution', type=int, help='Vertices per side (default 7).')
    parser.add_argument('--vbd-iterations', type=int, help='Cloth solver iterations per substep (default 12).')
    parser.add_argument('--drop-height', type=float, help='Cloth center above tabletop [m].')
    parser.add_argument('--experiment', type=Path, help='Optional recipe with env overrides.')
    parser.add_argument('--checkpoint', type=Path,
                        default=REPO_ROOT/'pretrained_policy/flexiv_wuji_finetuned/flat7RW_fricition.pth')
    parser.add_argument('--policy-config', '--policy_config', type=Path,
                        default=REPO_ROOT/'pretrained_policy/flexiv_wuji/config.yaml')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--paused', action='store_true')
    parser.add_argument('--steps', type=int, default=0, help='Exit after N policy steps; 0 runs interactively.')
    args = parser.parse_args(argv)
    if args.steps < 0 or (args.resolution is not None and args.resolution < 5):
        parser.error('Require --steps >= 0 and --resolution >= 5.')
    if args.vbd_iterations is not None and args.vbd_iterations < 1:
        parser.error('--vbd-iterations must be positive.')
    if args.size is not None and (not np.isfinite(args.size) or args.size <= 0):
        parser.error('--size must be finite and positive.')
    if args.drop_height is not None and (not np.isfinite(args.drop_height) or args.drop_height < 0):
        parser.error('--drop-height must be finite and nonnegative.')
    return args


def build_config(args):
    """Fresh material config, with the checkpoint's robot, controller and actor layout."""
    import yaml
    from isaacsimenvs.tasks.cloth.backends.vbd.newton_support import compat

    compat.install_isaaclab3_compat()
    from isaacsimenvs.tasks.cloth.backends.vbd.config import DeformableClothEnvCfg
    from isaacsimenvs.scripts.evaluation.player import read_policy_cfg
    from isaacsimenvs.scripts.visualization.rigid_cloth.policy import validate_policy_config

    saved = read_policy_cfg(args.policy_config, args.device)
    validate_policy_config(saved)
    cfg = DeformableClothEnvCfg()
    cfg.from_dict(yaml.safe_load((CONFIG_DIR/'task/Cloth.yaml').read_text()))
    cfg.robot_preset = 'yibo'
    cfg.robot_root_pos = (0., .51, .20)
    cfg.assets.robot_urdf = 'assets/urdf/flexiv_wuji/rizon4s_right_wuji_real_limits.urdf'
    cfg.assets.table_urdf = 'assets/urdf/table_fold_pool.urdf'
    cfg.assets.robot_friction, cfg.assets.finger_tip_friction, cfg.assets.table_friction = .5, 1.5, .5
    cfg.action.from_dict(saved['env']['action'])
    cfg.obs.from_dict(saved['env']['obs'])
    cfg.obs.keypoint_mode, cfg.obs.yibo_contact_thickness, cfg.obs.yibo_goal_lift = 'yibo', .016, 0.
    cfg.reward.object_base_size = float(saved['env']['reward']['object_base_size'])
    cfg.sim.dt, cfg.decimation, cfg.episode_length_s = 1/120, 2, 10.
    cfg.task.mode, cfg.task.release.enabled = 'fold', False
    cfg.task.nan_policy = 'raise'
    branch = args.preset == 'yibo'
    cfg.reset.table_reset_xy, cfg.reset.table_reset_z = (0., 0.), .38
    cfg.reset.table_reset_z_range = .01 if branch else 0.
    cfg.reset.reset_position_noise_x = cfg.reset.reset_position_noise_y = .10 if branch else .05
    cfg.reset.random_yaw, cfg.reset.yaw_range_deg = True, [0., 360.]
    dr = cfg.domain_randomization
    dr.use_obs_delay = dr.use_action_delay = dr.use_object_state_delay_noise = False
    dr.joint_velocity_obs_noise_std = dr.joint_pos_obs_noise_std = dr.action_noise_std = 0.
    dr.object_state_xyz_noise_std = dr.object_state_rotation_noise_degrees = 0.
    cloth = cfg.deformable_cloth
    cloth.size, cloth.start_height = (.10, .30) if branch else (.15, .20)
    cloth.from_dict(fabrics.cloth_params(args.material, plastic=True))
    if args.experiment is not None:
        from isaacsimenvs.scripts.configs.loader import load_experiment

        cfg.from_dict(load_experiment(str(args.experiment)).get('env', {}))
    if args.size is not None:
        cloth.size = args.size
    if args.drop_height is not None:
        from isaacsimenvs.tasks.cloth.backends.assets import urdf_box_size

        cloth.start_height = urdf_box_size(cfg.assets.table_urdf)[2]/2 + args.drop_height
    if args.resolution is not None:
        cloth.resolution = args.resolution
    if args.vbd_iterations is not None:
        cloth.vbd_iterations = args.vbd_iterations
    # One visible environment; do not inherit training batch sizes or renderers.
    cfg.scene.num_envs, cfg.sim.device, cfg.seed = 1, args.device, args.seed
    if cfg.backend != 'vbd' or cfg.robot_preset != 'yibo' or cfg.obs.keypoint_mode != 'yibo':
        raise ValueError('This player requires the VBD backend and Yibo robot/actor encoding.')
    cfg.task.check()
    cloth.check_thickness()
    return cfg


def numpy(value):
    from isaacsimenvs.tasks.cloth.backends.base import as_torch

    return as_torch(value).detach().cpu().numpy()


def asset_pose(asset, origin):
    """Newton assets expose xyzw; Viser takes wxyz. Positions are environment-local."""
    position = numpy(asset.data.root_pos_w)[0] - origin
    quaternion = numpy(asset.data.root_quat_w)[0][[3, 0, 1, 2]]
    return position, quaternion


class Playback:
    """Use the environment's actor inputs, action controller, autoresets and fold scoring."""

    def __init__(self, args, cfg):
        from isaacsimenvs.tasks.cloth.cloth_env import ClothEnv
        from isaacsimenvs.scripts.evaluation.player import PretrainedPlayer

        self.env = self.player = None
        self.episodes = self.folds = 0
        self.reward = 0.
        try:
            self.env = ClothEnv(cfg)
            self.player = PretrainedPlayer(args.policy_config, args.checkpoint, 1, device=args.device,
                                           num_observations=134, num_actions=27, expl_coef=0.)
            self.env.seed(args.seed)
            self.reset()
        except BaseException:
            self.close()
            raise

    def reset(self):
        self.obs, _ = self.env.reset()
        self.player.reset_rnn()
        self.reward = 0.

    def step(self):
        # Do not use inference_mode: UI resets must be able to clear the LSTM later.
        with torch.no_grad():
            action = self.player.get_action(self.obs['policy'], deterministic=True)
            if not bool(torch.isfinite(action).all()):
                raise RuntimeError('Nonfinite policy action; reset or rebuild the scene.')
            obs, reward, terminated, truncated, info = self.env.step(action)
        if not all(bool(torch.isfinite(value).all()) for value in obs.values()):
            raise RuntimeError('Nonfinite observation; reset or rebuild the scene.')
        self.obs, self.reward = obs, float(reward[0])
        if bool((terminated | truncated).any()):
            self.episodes += 1
            self.folds += int(info['episode_final']['done_fold'][0])
            self.player.reset_rnn()

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None


def main(argv=None):
    args = parse_args(argv)
    os.chdir(REPO_ROOT)
    import importlib.util

    if importlib.util.find_spec('isaaclab_newton') is None:
        raise SystemExit('Use .venv/isaaclab3/bin/python for the native Newton/VBD viewer.')
    import viser
    from viser.extras import ViserUrdf
    import yourdfpy
    from isaacsimenvs.tasks.cloth.backends.assets import urdf_box_size
    from isaacsimenvs.tasks.cloth.mdp.geometry import grid

    cfg = build_config(args)
    playback = None
    server = viser.ViserServer(host=args.host, port=args.port)
    restart = None
    try:
        playback = Playback(args, cfg)
        env, cloth = playback.env, cfg.deformable_cloth
        origin = numpy(env.scene.env_origins)[0]
        server.scene.set_up_direction('+z')
        server.scene.add_grid('/ground', width=3., height=3., cell_size=.1)
        path = REPO_ROOT/cfg.assets.robot_urdf
        model = yourdfpy.URDF.load(path, load_collision_meshes=False,
                                  filename_handler=partial(yourdfpy.filename_handler_magic, dir=path.parent))
        robot_frame = server.scene.add_frame('/robot', show_axes=False)
        robot = ViserUrdf(server, model, root_node_name='/robot')
        joint_ids = [list(env.backend.robot.data.joint_names).index(name) for name in robot.get_actuated_joint_names()]
        table = server.scene.add_box('/table', dimensions=urdf_box_size(cfg.assets.table_urdf), color=(170, 135, 100))
        points = numpy(env.backend.cloth_points()[0])[0] - origin
        triangles = np.asarray(grid.grid_triangles(cloth.resolution), dtype=np.uint32).reshape(-1, 3)
        moving = numpy(env.backend.cloth_grid.rest)[0, triangles, 0].mean(axis=1) > 0
        meshes = [server.scene.add_mesh_simple(f'/cloth/{name}', points, triangles[mask], color=color, side='double')
                  for name, mask, color in [('moving', moving, (242, 173, 56)),
                                            ('stationary', ~moving, (74, 143, 194))]]
        particles = server.scene.add_point_cloud('/particles', points, (230, 230, 240), point_size=.004, visible=False)
        current = server.scene.add_point_cloud('/keypoints', np.zeros((4, 3)), (255, 100, 80), point_size=.009)
        target = server.scene.add_point_cloud('/targets', np.zeros((4, 3)), (66, 220, 180), point_size=.009)

        @server.on_client_connect
        def camera(client):
            client.camera.position = (.8, -.8, 1.)
            client.camera.look_at = (0., 0., .6)
            client.camera.up_direction = (0., 0., 1.)

        commands = queue.SimpleQueue()
        with server.gui.add_folder('Playback'):
            playing = server.gui.add_checkbox('Play', initial_value=not args.paused or bool(args.steps))
            speed = server.gui.add_slider('Speed', min=.05, max=2., step=.05, initial_value=1.)
            step_button = server.gui.add_button('Step')
            reset_button = server.gui.add_button('Reset episode')
            status = server.gui.add_markdown('')
        with server.gui.add_folder('Display'):
            markers = server.gui.add_dropdown('Keypoints', options=('Policy markers', 'Physical corners', 'Hidden'))
            show_particles = server.gui.add_checkbox('Particles', initial_value=False)
            wireframe = server.gui.add_checkbox('Wireframe', initial_value=False)
            server.gui.add_markdown('Orange/red: moving half/current markers. Green: corresponding targets. '
                                    'Policy targets retain Yibo encoding; physical targets score the fold.')
        with server.gui.add_folder('Rebuild scene'):
            material = server.gui.add_dropdown('Material', options=fabrics.names(), initial_value=args.material)
            size = server.gui.add_number('Side length [m]', initial_value=cloth.size, min=.02, max=.6, step=.01)
            resolution = server.gui.add_number('Vertices per side', initial_value=cloth.resolution, min=5, max=41, step=1)
            iterations = server.gui.add_number('VBD iterations / substep', initial_value=cloth.vbd_iterations, min=1, max=200, step=1)
            drop = server.gui.add_number('Drop above table [m]', initial_value=cloth.start_height-urdf_box_size(cfg.assets.table_urdf)[2]/2,
                                         min=0., max=.5, step=.01)
            rebuild = server.gui.add_button('Apply and restart')
            server.gui.add_markdown('Restarts simulation and policy; the browser reconnects on this port.')

        # Viser callbacks enqueue commands; all simulation and CUDA work stays on this thread.
        step_button.on_click(lambda _: commands.put('step'))
        reset_button.on_click(lambda _: commands.put('reset'))
        rebuild.on_click(lambda _: commands.put('rebuild'))

        def draw():
            state = env.backend.read_state()
            view = env.task.derive(state)
            vertices = numpy(state.points_w)[0] - origin
            with server.atomic():
                robot_frame.position, robot_frame.wxyz = asset_pose(env.backend.robot, origin)
                robot.update_cfg(numpy(state.joint_pos)[0, joint_ids])
                table.position, table.wxyz = asset_pose(env.backend.table, origin)
                for mesh in meshes:
                    mesh.vertices, mesh.wireframe = vertices, wireframe.value
                particles.points, particles.visible = vertices, show_particles.value
                if markers.value == 'Policy markers':
                    kp, goal = env.observations.keypoints_w(view, state)
                else:
                    kp, goal = view.corners_w, view.targets_w
                current.points, target.points = numpy(kp)[0]-origin, numpy(goal)[0]-origin
                current.visible = target.visible = markers.value != 'Hidden'
            status.content = (f'Fold error: **{float(view.fold_error[0])*100:.1f} cm** · reward {playback.reward:.2f}  \n'
                              f'Completed: {playback.episodes} · folds: {playback.folds}  \n'
                              f'`{args.material}` · {cloth.size*100:g} cm · {cloth.resolution}×{cloth.resolution} particles  \n'
                              f'VBD: {cloth.vbd_iterations} iterations × {cloth.substeps} cloth substeps; '
                              f'{cfg.newton.num_substeps} coupled substeps/tick')

        draw()
        print(f'Deformable cloth viewer ready: http://localhost:{args.port}', flush=True)
        total_steps = 0
        last_error = None
        while not args.steps or total_steps < args.steps:
            started = time.monotonic()
            step_once = False
            try:
                while not commands.empty():
                    command = commands.get_nowait()
                    if command == 'reset':
                        playback.reset()
                        last_error = None
                    elif command == 'step':
                        playing.value, step_once = False, True
                    elif command == 'rebuild':
                        restart = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:],
                                   '--material', material.value, '--size', str(size.value),
                                   '--resolution', str(int(resolution.value)), '--drop-height', str(drop.value),
                                   '--vbd-iterations', str(int(iterations.value))]
                if restart is not None:
                    break
                if playing.value or step_once:
                    playback.step()
                    total_steps += 1
                if last_error is None:
                    draw()
            except Exception as exc:
                playing.value = False
                last_error = exc
                status.content = f'**Paused: {type(exc).__name__}: {exc}**  \nReset or apply/restart to recover.'
                if args.steps:
                    raise
            time.sleep(max(.01 if not playing.value else 0., env.step_dt/float(speed.value)-(time.monotonic()-started)))
        print(json.dumps(dict(steps=total_steps, episodes=playback.episodes, folds=playback.folds)), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if playback is not None:
                playback.close()
        finally:
            server.stop()
    if restart is not None:
        os.execv(sys.executable, restart)


if __name__ == '__main__':
    main()
