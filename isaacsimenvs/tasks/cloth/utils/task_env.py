"""``ClothTaskEnv``: the DirectRLEnv hooks of the fold task, once, for both simulators.

Each hook is a call into the utils (cf. Play's ``play_env.py``)::

    _pre_physics_step   action_utils.apply_actions
    _apply_action       joint position targets + the hand's passive spring
    _get_dones          state.derive -> termination_utils.compute
    _get_rewards        reward_utils.compute -> logging_utils.log_step
    _reset_idx          reset_utils draws -> the backend's writes -> state.clear
    _get_observations   state.derive -> obs_utils.build -> obs_utils.guard_finite

A backend (``rigid_body_env.py`` PhysX, ``deformable_env.py`` Newton) builds the scene and implements
the small simulator surface below: ``_read_state`` (fill a ``ClothState``) and the reset writes.
Nothing else in the task reads or writes the simulator.
"""

from __future__ import annotations

import torch
from isaaclab.envs import DirectRLEnv

from . import action_utils, geometry, logging_utils, obs_utils, reset_utils, reward_utils, state, termination_utils
from .robot import PLAY_ROBOT_ROOT_POS, ROBOT
from .state import ClothState, TaskView


class ClothTaskEnv(DirectRLEnv):
    #: Set by the backend before ``super().__init__``: ``(N, 3)`` length / width / thickness [m].
    _dims: torch.Tensor

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        cfg.fold.check()
        self._spec = self.robot_spec = ROBOT
        cfg.action_space = ROBOT.num_joints
        cfg.observation_space = obs_utils.obs_dim(cfg.obs.obs_list, ROBOT.num_joints)
        cfg.state_space = obs_utils.obs_dim(cfg.obs.state_list, ROBOT.num_joints)
        super().__init__(cfg, render_mode, **kwargs)  # builds the scene (_setup_scene)
        self._init_task()

    # ------------------------------------------------------------------ backend surface

    def _read_state(self) -> ClothState:
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

    def _write_sheet(self, ids, xy, yaw, angle) -> None:
        """Place the sheet flat-at-rest at env-frame ``xy``, heading ``yaw``, the + half turned up by
        ``angle`` (0 = flat), on this episode's table."""
        raise NotImplementedError

    def _sheet_start_xy(self, ids, yaw) -> torch.Tensor:
        return reset_utils.place_with_noise(self, ids.numel())

    def _ply_gap(self) -> torch.Tensor:
        return self._dims[:, 2]

    def _randomize(self, ids) -> None:
        """Per-reset physics DR (the backend's materials)."""

    # ------------------------------------------------------------------ setup

    def _init_task(self) -> None:
        dev = self.device
        names, self._default_joint_pos, limits = self._robot_defaults()
        m = action_utils.joint_maps(names, ROBOT)
        self._perm_canon_to_lab = torch.tensor(m["canon_to_lab"], device=dev)
        self._perm_lab_to_canon = torch.tensor(m["lab_to_canon"], device=dev)
        self._arm_ids, self._hand_ids = m["arm"], m["hand"]
        self._joint_limits = limits
        self._joint_lower_canon = limits[0, :, 0][self._perm_lab_to_canon]
        self._joint_upper_canon = limits[0, :, 1][self._perm_lab_to_canon]
        self._arm_lower, self._arm_upper = limits[:, self._arm_ids, 0], limits[:, self._arm_ids, 1]
        self._hand_lower, self._hand_upper = limits[:, self._hand_ids, 0], limits[:, self._hand_ids, 1]
        self._spring_ks, self._spring_ref = action_utils.spring_constants(names, ROBOT, dev)

        self._palm_id = self._body_ids([ROBOT.palm_body_name])[0]
        self._fingertip_ids = self._body_ids(list(ROBOT.fingertip_link_names))
        rx = self.cfg.fold.release.contact_bodies
        self._hand_body_ids = self._body_ids_regex(rx) if rx else self._fingertip_ids + [self._palm_id]
        self._palm_offset = torch.tensor(ROBOT.palm_center_offset, device=dev)
        self._fingertip_offsets = torch.tensor(ROBOT.fingertip_offsets, device=dev)
        self._palm_obs_shift = torch.tensor(self.cfg.robot_root_pos, device=dev) - torch.tensor(PLAY_ROBOT_ROOT_POS, device=dev)

        self._sheet_geom = geometry.make_sheet(self._dims, self._point_resolution(), self._ply_gap())
        self._object_scales = self._dims / float(self.cfg.reward.object_base_size)
        state.allocate(self)
        self.reward_buf = torch.zeros(self.num_envs, device=dev)
        print(f"[cloth] {type(self).__name__}: {self.num_envs} envs, task {self.cfg.fold.task}, obs "
              f"{self.cfg.observation_space} / state {self.cfg.state_space}, {self._sheet_geom.num_points} "
              f"points per sheet", flush=True)

    def _point_resolution(self) -> int:
        raise NotImplementedError

    def _body_ids_regex(self, regex: str) -> list[int]:
        raise NotImplementedError

    # ------------------------------------------------------------------ hooks

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        action_utils.apply_actions(self, actions)

    def _apply_action(self) -> None:
        q = self._read_joint_pos()
        self._set_robot_targets(self._cur_targets, action_utils.spring_effort(self, q))

    def _read_joint_pos(self) -> torch.Tensor:
        return self._read_state().joint_pos

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._step_state = self._read_state()
        self._step_view = state.derive(self, self._step_state)
        return termination_utils.compute(self, self._step_state, self._step_view)

    def _get_rewards(self) -> torch.Tensor:
        reward = reward_utils.compute(self, self._step_state, self._step_view)
        logging_utils.log_step(self)
        return reward

    def _reset_idx(self, env_ids) -> None:
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else \
            torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
        super()._reset_idx(ids)
        pos, vel = reset_utils.robot_start(self, ids, self._default_joint_pos[ids],
                                           self._joint_limits[ids, :, 0], self._joint_limits[ids, :, 1])
        self._write_robot(ids, pos, vel)
        self._write_table(ids, reset_utils.table_height(self, ids))
        angle = reset_utils.start_task(self, ids)
        yaw = reset_utils.start_yaw(self, ids.numel())
        self._write_sheet(ids, self._sheet_start_xy(ids, yaw), yaw, angle)
        self._randomize(ids)
        state.clear(self, ids)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        s = self._read_state()
        obs = obs_utils.build(self, s, state.derive(self, s))
        obs_utils.guard_finite(self, obs, s)
        return obs

    # ------------------------------------------------------------------ queries (eval, viewers, probes)

    def _now(self) -> tuple[ClothState, TaskView]:
        s = self._read_state()
        return s, state.derive(self, s)

    def _particles_w(self) -> torch.Tensor:
        return self._read_state().points_w

    def cloth_keypoints_w(self) -> torch.Tensor:
        return self._now()[1].keypoints_w

    def fold_targets_w(self) -> torch.Tensor:
        return self._now()[1].targets_w

    def fold_error(self) -> torch.Tensor:
        return self._now()[1].fold_error

    def footprint_ratio(self) -> torch.Tensor:
        return self._now()[1].footprint

    def is_folded(self) -> torch.Tensor:
        s, v = self._now()
        return termination_utils.is_folded(self, s, v)

    def folded_half_w(self) -> torch.Tensor:
        """``(N, M, 3)``: where every moving-half point belongs (the goal as a ghost sheet)."""
        _, v = self._now()
        return geometry.moving_half_targets(self._sheet_geom, v.frame_r, v.frame_c, self._unfold_env)

    def folded_half_topology(self) -> list[int]:
        return geometry.half_triangles(self._sheet_geom.resolution, reflected=self.cfg.fold.task != "unfold")

    @property
    def _moving_idx(self) -> torch.Tensor:
        return self._sheet_geom.moving_idx

    @property
    def _stationary_idx(self) -> torch.Tensor:
        return self._sheet_geom.stationary_idx
