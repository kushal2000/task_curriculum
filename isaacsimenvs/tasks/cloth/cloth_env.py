"""One DirectRLEnv lifecycle for rigid and VBD cloth manipulation.

Isaac Lab owns physics stepping and autoresets. This class connects the robot
controller, scene adapter, task, and observation builder through its standard hooks.
"""

from __future__ import annotations

import torch
from isaaclab.envs import DirectRLEnv

from .backends.robot import ROBOT
from .mdp.actions import JointController
from .mdp.geometry import fold, grid
from .mdp.observations import ObservationBuilder, obs_dim
from .mdp.task import ClothTask


class ClothEnv(DirectRLEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        cfg.task.check()
        self.robot_spec = ROBOT
        cfg.action_space = ROBOT.num_joints
        cfg.observation_space = obs_dim(cfg.obs.obs_list, ROBOT.num_joints)
        cfg.state_space = obs_dim(cfg.obs.state_list, ROBOT.num_joints)
        if cfg.backend == "rigid":
            from .backends.rigid.backend import RigidBackend

            self.backend = RigidBackend(cfg)
        elif cfg.backend == "vbd":
            from .backends.vbd.backend import VbdBackend

            self.backend = VbdBackend(cfg)
        else:
            raise ValueError(f"unknown cloth backend {cfg.backend!r}; expected rigid or vbd")

        # Newton must install its construction hooks before Isaac creates the scene.
        with self.backend.startup():
            super().__init__(cfg, render_mode, **kwargs)
        self.backend.initialize()
        self.controller = JointController(cfg, *self.backend.robot_defaults(), self.step_dt)
        self.task = ClothTask(
            cfg, self.backend.cloth_grid, self.scene.env_origins, self.controller._arm_ids, self.controller._hand_ids
        )
        self.observations = ObservationBuilder(cfg, self.controller, self.task)
        self.reward_buf = self.task.reward_buf
        self.backend.finish_setup()
        print(
            f"[cloth] {cfg.backend}: {self.num_envs} envs, {cfg.task.mode}, "
            f"obs/state {cfg.observation_space}/{cfg.state_space}",
            flush=True,
        )

    # Isaac Lab calls these hooks in this order for each policy step.

    def _setup_scene(self):
        self.backend.build_scene(self.scene, self.sim)

    def _pre_physics_step(self, actions):
        self.controller.process(actions, self.episode_length_buf)

    def _apply_action(self):
        commands = self.controller.commands(self.backend.read_joint_pos())
        self.backend.apply_joint_commands(commands)

    def _get_dones(self):
        release = self.cfg.task.release
        touching = self.backend.hand_cloth_touching() if release.enabled and release.touch == "contact" else None
        self._transition = self.task.evaluate(
            self.backend.read_state(), self.episode_length_buf, self.max_episode_length, touching
        )
        return self._transition.terminated, self._transition.truncated

    def _get_rewards(self):
        self.extras.update(self.task.extras)
        return self._transition.reward

    def _reset_idx(self, env_ids):
        ids = (
            torch.arange(self.num_envs, device=self.device)
            if env_ids is None
            else torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
        )
        super()._reset_idx(ids)
        initial = self.task.sample_reset(ids, *self.controller.sample_reset(ids))
        self.backend.reset(ids, initial)
        self.task.reset(ids)
        self.observations.reset(ids)

    def _get_observations(self):
        state = self.backend.read_state()
        view = self.task.derive(state)
        obs = self.observations.build(state, view, self.episode_length_buf)
        self.observations.guard_finite(obs, state, self.episode_length_buf)
        self.extras.update(self.task.extras)
        return obs

    # Evaluation and viewer queries use the same state and goals as the task.

    def _now(self):
        state = self.backend.read_state()
        return state, self.task.derive(state)

    def _particles_w(self):
        return self.backend.cloth_points()[0]

    def cloth_keypoints_w(self):
        return self._now()[1].corners_w

    def fold_targets_w(self):
        return self._now()[1].targets_w

    def fold_error(self):
        return self._now()[1].fold_error

    def footprint_ratio(self):
        return self._now()[1].footprint

    def is_folded(self):
        return self.task.is_folded(*self._now())

    def folded_half_w(self):
        _, view = self._now()
        return fold.moving_half_targets(self.backend.cloth_grid, view.frame_r, view.frame_c, self.task._unfold_env)

    def folded_half_topology(self):
        return grid.half_triangles(self.backend.cloth_grid.resolution, reflected=self.cfg.task.mode != "unfold")

    def viewer_plates(self, env_id):
        return self.backend.viewer_plates(env_id)

    def hand_cloth_touching(self):
        return self.backend.hand_cloth_touching()

    # Existing evaluation/viewer accessors. Episode history stays on the task,
    # simulator assets on the adapter, and joint mappings on the controller.

    @property
    def robot(self):
        return self.backend.robot

    @property
    def table(self):
        return self.backend.table

    @property
    def _cloth_grid(self):
        return self.backend.cloth_grid

    @property
    def _perm_lab_to_canon(self):
        return self.controller._perm_lab_to_canon

    @property
    def _successes(self):
        return self.task._successes

    @property
    def _moving_idx(self):
        return self.backend.cloth_grid.moving_idx

    @property
    def _stationary_idx(self):
        return self.backend.cloth_grid.stationary_idx
