"""``ClothEnv``: the fold task's lifecycle, shared by both backends.

Isaac Lab's ``DirectRLEnv`` steps the physics and resets finished envs; this class wires its hooks
to the task code in ``mdp/``:

    _pre_physics_step   actions.process_actions          policy action -> joint targets
    _apply_action       (here)                           targets + the hand's passive spring
    _get_dones          terminations.compute_dones
    _get_rewards        rewards.compute_reward
    _reset_idx          robot / reset / randomization    the episode's start and DR draws
    _get_observations   observations.build_observations

The fold itself (targets, error, criterion, goal marker) is ``mdp/task.ClothTaskMixin``. A backend
subclasses this env and supplies the sheet: ``backends/rigid/env.RigidClothEnv`` (PhysX, hinged
plates) and ``backends/vbd/env.VbdClothEnv`` (Newton, a VBD cloth). The scene (robot, table, goal
marker) is ``backends/scene.py``; it ends with the backend's ``_install_cloth``, which must set
``self._cloth``, ``self.object`` (the sheet presented as a rigid object, ``backends/adapter.py``),
``self._cloth_rest_local`` and ``self._cloth_kp_idx``, and fill ``self._object_scale_per_env``.
"""

from __future__ import annotations

import torch
from isaaclab.envs import DirectRLEnv

from isaacsimenvs.robots import get_robot_spec

from .backends import scene
from .mdp.actions import process_actions
from .mdp.observations import build_observations, guard_finite, obs_dim
from .mdp.randomization import init_dr, reset_dr
from .mdp.reset import reset_sheet, reset_table, reset_task_trackers
from .mdp.rewards import compute_reward
from .mdp.robot import NUM_FINGERTIPS, init_robot, reset_robot, spring_effort
from .mdp.terminations import compute_dones

__all__ = ["ClothEnv"]


class ClothEnv(DirectRLEnv):
    """Fold half a sheet onto the other half, with the robot of ``cfg.robot``."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs) -> None:
        self.robot_spec = get_robot_spec(cfg.robot)
        n_joints = self.robot_spec.num_joints
        cfg.action_space = n_joints
        cfg.observation_space = obs_dim(cfg.obs.obs_list, n_joints)
        cfg.state_space = obs_dim(cfg.obs.state_list, n_joints)
        super().__init__(cfg, render_mode, **kwargs)  # builds the scene
        self._check_env_order()
        init_robot(self)
        self._init_task_state()
        init_dr(self)
        self._finish_backend_setup()
        self._init_fold_targets()
        self._sync_observed_keypoints()
        # Fold error of a flat sheet: the stand-in for a non-finite one (rewards.compute_reward).
        self._flat_fold_err = float(self.fold_error().nan_to_num(nan=0.0).max().item())
        self._update_goal()

    def _check_env_order(self) -> None:
        """Row i of every PhysX asset view must be env i: Isaac Lab indexes asset data by env id."""
        for name, asset in (("robot", self.robot), ("table", self.table), ("goal_viz", self.goal_viz)):
            view = getattr(asset, "root_physx_view", None)
            paths = getattr(view, "prim_paths", None)
            if paths is None:  # Newton: environments are separate worlds, indexed by env id
                continue
            wrong = [(i, p) for i, p in enumerate(paths) if f"/env_{i}/" not in p]
            if wrong:
                raise RuntimeError(f"{name}: PhysX view rows are not in env order, e.g. row {wrong[0][0]} is {wrong[0][1]}")

    def _finish_backend_setup(self) -> None:
        """Backend hook: runs once the assets are live, before the task first reads the sheet."""

    def _init_task_state(self) -> None:
        n, dev = self.num_envs, self.device
        reset_cfg = self.cfg.reset
        # Folds this episode, and in the finished one (the logger's `successes`).
        self._successes = torch.zeros(n, dtype=torch.long, device=dev)
        self._prev_episode_successes = torch.zeros(n, dtype=torch.long, device=dev)
        # The fold: steps the criterion has held, the bonus mask, best fold error (-1 = unset).
        self._fold_hold = torch.zeros(n, dtype=torch.long, device=dev)
        self._fold_bonus_mask = torch.zeros(n, dtype=torch.bool, device=dev)
        self._closest_fold_err = torch.full((n,), -1.0, device=dev)
        # The hand and goal measurements (rewards.update_hand_and_goal_metrics).
        self._curr_fingertip_distances = torch.zeros(n, NUM_FINGERTIPS, device=dev)
        self._closest_fingertip_dist = torch.full((n, NUM_FINGERTIPS), -1.0, device=dev)
        self._keypoints_max_dist = torch.zeros(n, device=dev)
        self._closest_keypoint_max_dist = torch.full((n,), -1.0, device=dev)
        self._lifted_object = torch.zeros(n, dtype=torch.bool, device=dev)
        self._object_init_z = torch.full((n,), reset_cfg.table_reset_z + reset_cfg.table_object_z_offset, device=dev)
        self._table_z_per_env = torch.full((n,), float(reset_cfg.table_reset_z), device=dev)
        # The observed keypoints' offsets in the sheet frame (ClothTaskMixin._sync_observed_keypoints).
        self._keypoint_offsets = torch.zeros(n, len(self._cloth_kp_idx), 3, device=dev)
        # Observed by the critic (`reward`) from the first reset on, before any step has paid one.
        self.reward_buf = torch.zeros(n, device=dev)
        # Divergence guards: envs flagged this step (reset by compute_dones), and running counts.
        self._nonfinite_envs = torch.zeros(n, dtype=torch.bool, device=dev)
        self._nonfinite_count = 0
        self._insane_count = 0

    # ------------------------------------------------------------------ Isaac Lab hooks

    def _setup_scene(self) -> None:
        # Through the module: the Newton backend wraps it with the scene replication.
        scene.setup_scene(self)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        process_actions(self, actions)

    def _apply_action(self) -> None:
        # Every physics step (decimation times per policy step).
        self.robot.set_joint_position_target(self._cur_targets)
        spring = spring_effort(self)
        if spring is not None:
            self.robot.set_joint_effort_target(spring)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        return compute_dones(self)

    def _get_rewards(self) -> torch.Tensor:
        reward = compute_reward(self)
        self._publish_metrics()
        return reward

    def _reset_idx(self, env_ids) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        reset_robot(self, ids)
        reset_table(self, ids)
        reset_sheet(self, ids)
        reset_task_trackers(self, ids)
        reset_dr(self, ids)
        self._update_goal()

    def _get_observations(self) -> dict[str, torch.Tensor]:
        # Both follow the sheet, so both are refreshed before it is observed.
        self._sync_observed_keypoints()
        self._update_goal()
        obs = build_observations(self)
        guard_finite(self, obs)
        return obs

    def _publish_metrics(self) -> None:
        """Step extras for the RL-Games observer and the evaluation harness: per-step reward terms
        (``episode_cumulative``), per-episode outcomes (``episode_final``, one ``done_<reason>``
        per termination reason)."""
        final = {"successes": self._successes.float()}
        final.update({f"done_{name}": value.float() for name, value in self._termination_reasons.items()})
        self.extras["episode_cumulative"] = self._reward_terms
        self.extras["episode_final"] = final
        self.extras["successes"] = self._prev_episode_successes.float()
