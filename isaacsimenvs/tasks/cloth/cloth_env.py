"""``ClothEnv``: the DirectRLEnv hooks of the fold task, once, for both simulators.

Each hook is a call into the utils (cf. Play's ``play_env.py``)::

    _pre_physics_step   actions.apply_actions
    _apply_action       joint position targets + the hand's passive spring
    _get_dones          snapshot.derive -> terminations.compute
    _get_rewards        rewards.compute -> metrics.log_step
    _reset_idx          events draws -> the backend's writes -> snapshot.clear
    _get_observations   snapshot.derive -> observations.build -> observations.guard_finite

A backend (``rigid_cloth_env.py`` PhysX, ``deformable_cloth_env.py`` Newton) builds the scene and implements
the small simulator surface below: ``_read_state`` (fill a ``SimState``) and the reset writes.
Nothing else in the task reads or writes the simulator.
"""

from __future__ import annotations

import torch
from isaaclab.envs import DirectRLEnv

from isaacsimenvs.tasks.cloth.utils.geometry import fold, grid
from isaacsimenvs.tasks.cloth.utils.mdp import actions, events, metrics, observations, rewards, snapshot, terminations
from isaacsimenvs.tasks.cloth.utils.mdp.flexiv_wuji import PLAY_ROBOT_ROOT_POS, ROBOT
from isaacsimenvs.tasks.cloth.utils.mdp.snapshot import SimState, TaskView


class ClothEnv(DirectRLEnv):
    #: Set by the backend before ``super().__init__``: ``(N, 3)`` length / width / thickness [m].
    _dims: torch.Tensor

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        cfg.cloth.check()
        self._spec = self.robot_spec = ROBOT
        cfg.action_space = ROBOT.num_joints
        cfg.observation_space = observations.obs_dim(cfg.obs.obs_list, ROBOT.num_joints)
        cfg.state_space = observations.obs_dim(cfg.obs.state_list, ROBOT.num_joints)
        super().__init__(cfg, render_mode, **kwargs)  # builds the scene (_setup_scene)
        self._init_task()

    # ------------------------------------------------------------------ backend surface

    def _read_state(self) -> SimState:
        raise NotImplementedError

    def _robot_defaults(self) -> tuple[list[str], torch.Tensor, torch.Tensor]:
        """``(Lab joint names, default joint pos (N, J), limits (N, J, 2))``."""
        raise NotImplementedError

    def _body_ids(self, names: list[str]) -> list[int]:
        raise NotImplementedError

    def _write_robot(self, ids, joint_pos, joint_vel) -> None:
        raise NotImplementedError

    def _set_robot_targets(self, pos_target, effort) -> None:
        raise NotImplementedError

    def _write_table(self, ids, table_z) -> None:
        raise NotImplementedError

    def _write_cloth(self, ids, xy, yaw, angle) -> None:
        """Place the cloth flat-at-rest at env-frame ``xy``, heading ``yaw``, the + half turned up by
        ``angle`` (0 = flat), on this episode's table."""
        raise NotImplementedError

    def _cloth_start_xy(self, ids, yaw) -> torch.Tensor:
        return events.place_with_noise(self, ids.numel())

    def _ply_gap(self) -> torch.Tensor:
        return self._dims[:, 2]

    def _randomize(self, ids) -> None:
        """Per-reset physics DR (the backend's materials)."""

    # ------------------------------------------------------------------ setup

    def _init_task(self) -> None:
        dev = self.device
        names, self._default_joint_pos, limits = self._robot_defaults()
        joint_maps = actions.joint_maps(names, ROBOT)
        self._perm_canon_to_lab = torch.tensor(joint_maps["canon_to_lab"], device=dev)
        self._perm_lab_to_canon = torch.tensor(joint_maps["lab_to_canon"], device=dev)
        self._arm_ids, self._hand_ids = joint_maps["arm"], joint_maps["hand"]
        self._joint_limits = limits
        self._joint_lower_canon = limits[0, :, 0][self._perm_lab_to_canon]
        self._joint_upper_canon = limits[0, :, 1][self._perm_lab_to_canon]
        self._arm_lower, self._arm_upper = limits[:, self._arm_ids, 0], limits[:, self._arm_ids, 1]
        self._hand_lower, self._hand_upper = limits[:, self._hand_ids, 0], limits[:, self._hand_ids, 1]
        self._spring_ks, self._spring_ref = actions.spring_constants(names, ROBOT, dev)

        self._palm_id = self._body_ids([ROBOT.palm_body_name])[0]
        self._fingertip_ids = self._body_ids(list(ROBOT.fingertip_link_names))
        contact_bodies = self.cfg.cloth.release.contact_bodies
        self._hand_body_ids = (
            self._body_ids_regex(contact_bodies) if contact_bodies else self._fingertip_ids + [self._palm_id]
        )
        self._palm_offset = torch.tensor(ROBOT.palm_center_offset, device=dev)
        self._fingertip_offsets = torch.tensor(ROBOT.fingertip_offsets, device=dev)
        self._palm_obs_shift = torch.tensor(self.cfg.robot_root_pos, device=dev) - torch.tensor(PLAY_ROBOT_ROOT_POS, device=dev)

        self._cloth_grid = grid.make_grid(self._dims, self._point_resolution(), self._ply_gap())
        self._object_scales = self._dims / float(self.cfg.reward.object_base_size)
        snapshot.allocate(self)
        self.reward_buf = torch.zeros(self.num_envs, device=dev)
        print(f"[cloth] {type(self).__name__}: {self.num_envs} envs, task {self.cfg.cloth.task}, obs "
              f"{self.cfg.observation_space} / state {self.cfg.state_space}, {self._cloth_grid.num_points} "
              f"points per cloth", flush=True)

    def _point_resolution(self) -> int:
        raise NotImplementedError

    def _body_ids_regex(self, regex: str) -> list[int]:
        raise NotImplementedError

    # ------------------------------------------------------------------ hooks

    def _pre_physics_step(self, action: torch.Tensor) -> None:
        actions.apply_actions(self, action)

    def _apply_action(self) -> None:
        q = self._read_joint_pos()
        self._set_robot_targets(self._cur_targets, actions.spring_effort(self, q))

    def _read_joint_pos(self) -> torch.Tensor:
        return self._read_state().joint_pos

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._step_state = self._read_state()
        self._step_view = snapshot.derive(self, self._step_state)
        return terminations.compute(self, self._step_state, self._step_view)

    def _get_rewards(self) -> torch.Tensor:
        reward = rewards.compute(self, self._step_state, self._step_view)
        metrics.log_step(self)
        return reward

    def _reset_idx(self, env_ids) -> None:
        if env_ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
        super()._reset_idx(ids)
        pos, vel = events.robot_start(
            self, ids, self._default_joint_pos[ids],
            self._joint_limits[ids, :, 0], self._joint_limits[ids, :, 1],
        )
        self._write_robot(ids, pos, vel)
        self._write_table(ids, events.table_height(self, ids))
        angle = events.start_task(self, ids)
        yaw = events.start_yaw(self, ids.numel())
        self._write_cloth(ids, self._cloth_start_xy(ids, yaw), yaw, angle)
        self._randomize(ids)
        snapshot.clear(self, ids)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        state = self._read_state()
        view = snapshot.derive(self, state)
        obs = observations.build(self, state, view)
        observations.guard_finite(self, obs, state)
        return obs

    # ------------------------------------------------------------------ queries (eval, viewers, probes)

    def _now(self) -> tuple[SimState, TaskView]:
        state = self._read_state()
        return state, snapshot.derive(self, state)

    def _particles_w(self) -> torch.Tensor:
        return self._read_state().points_w

    def cloth_keypoints_w(self) -> torch.Tensor:
        return self._now()[1].corners_w

    def fold_targets_w(self) -> torch.Tensor:
        return self._now()[1].targets_w

    def fold_error(self) -> torch.Tensor:
        return self._now()[1].fold_error

    def footprint_ratio(self) -> torch.Tensor:
        return self._now()[1].footprint

    def is_folded(self) -> torch.Tensor:
        state, view = self._now()
        return terminations.is_folded(self, state, view)

    def folded_half_w(self) -> torch.Tensor:
        """``(N, M, 3)``: where every moving-half point belongs (the goal as a ghost cloth)."""
        _, view = self._now()
        return fold.moving_half_targets(self._cloth_grid, view.frame_r, view.frame_c, self._unfold_env)

    def folded_half_topology(self) -> list[int]:
        return grid.half_triangles(self._cloth_grid.resolution, reflected=self.cfg.cloth.task != "unfold")

    @property
    def _moving_idx(self) -> torch.Tensor:
        return self._cloth_grid.moving_idx

    @property
    def _stationary_idx(self) -> torch.Tensor:
        return self._cloth_grid.stationary_idx
