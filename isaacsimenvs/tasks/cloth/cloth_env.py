"""``ClothEnv``: the fold task's lifecycle, shared by both backends.

Isaac Lab's ``DirectRLEnv`` steps the physics and resets finished envs; this class connects its
hooks to four parts, which hand each other the dataclasses of ``mdp/state.py``:

    controller     mdp/actions.JointController         policy action -> joint targets (+ passive spring)
    task           mdp/task.ClothTask                  episode start, dones, reward, fold goal
    observations   mdp/observations.ObservationBuilder policy and critic vectors, observation DR
    dr             mdp/randomization.DomainRandomization  per-episode DR draws

    _pre_physics_step   controller.process
    _apply_action       targets + spring effort to the robot (every physics step)
    _get_dones          task.compute_dones(state, view)
    _get_rewards        task.compute_reward(state, view)
    _reset_idx          controller / task draws, written to the scene; DR draws
    _get_observations   task.update_goal, observations.build

A backend subclasses this env and supplies the sheet: ``backends/rigid/env.RigidClothEnv`` (PhysX,
hinged plates) and ``backends/vbd/env.VbdClothEnv`` (Newton, a VBD cloth). The scene (robot, table,
goal marker) is ``backends/scene.py``; it ends with the backend's ``_install_cloth``.
"""

from __future__ import annotations

import torch
from isaaclab.envs import DirectRLEnv

from isaacsimenvs.robots import get_robot_spec

from .backends import scene
from .mdp.actions import JointController
from .mdp.observations import ObservationBuilder, obs_dim
from .mdp.randomization import DomainRandomization
from .mdp.robot import RobotLayout, to_torch
from .mdp.state import FoldView, SheetLayout, SimState
from .mdp.task import ClothTask

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
        dev, n = self.device, self.num_envs
        self.robot_layout = RobotLayout(self.robot, self.robot_spec, num_joints=n_joints, robot_urdf=cfg.assets.robot_urdf, device=dev)
        layout = self._sheet_layout()
        self.task = ClothTask(
            cfg, layout, self.robot_layout, num_envs=n, device=dev, env_origins=self.scene.env_origins,
            extras=self.extras, observed_goal_quat=self._observed_goal_quat,
        )
        # Observed by the critic (`reward`) from the first reset on, before any step has paid one.
        self.reward_buf = torch.zeros(n, device=dev)
        self.dr = DomainRandomization(cfg, self.robot_layout, self.robot, num_envs=n, num_keypoints=len(layout.keypoint_idx), device=dev)
        self.controller = JointController(cfg, self.robot_layout, self.dr, step_dt=self.step_dt, num_envs=n, device=dev)
        self.observations = ObservationBuilder(
            cfg, self.robot_layout, self.controller, self.task, self.dr, num_envs=n, device=dev,
            env_origins=self.scene.env_origins, extras=self.extras,
        )
        self._finish_backend_setup()
        view = self._derive_sheet()
        # Fold error of a flat sheet: the stand-in for a non-finite one (ClothTask.compute_reward).
        self.task.flat_fold_err = float(view.fold_error.nan_to_num(nan=0.0).max().item())
        self._write_goal(view)
        self._step_state: tuple[SimState, FoldView] | None = None

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

    # ------------------------------------------------------------------ backend hooks

    def _sheet_layout(self) -> SheetLayout:
        raise NotImplementedError

    def _read_points(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``(points_w, points_vel_w)``, each (N, P, 3): the sheet's grid points now."""
        raise NotImplementedError

    def _observed_goal_quat(self, rot: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _finish_backend_setup(self) -> None:
        """Runs once the assets are live, before the task first reads the sheet."""

    def _begin_reset(self, env_ids: torch.Tensor) -> None:
        """First thing in a reset, before any of the task's draws."""

    def _end_reset(self, env_ids: torch.Tensor) -> None:
        """Last thing in a reset."""

    def _write_sheet(self, env_ids: torch.Tensor, pose_w: torch.Tensor) -> None:
        """Lay the sheet flat at ``pose_w`` (n, 7): its xy and its yaw; at rest."""
        self.object.write_root_pose_to_sim(pose_w, env_ids=env_ids)
        self.object.write_root_velocity_to_sim(torch.zeros(env_ids.numel(), 6, device=self.device), env_ids=env_ids)

    # ------------------------------------------------------------------ state reads

    def _read_state(self) -> SimState:
        data = self.robot.data
        points, velocities = self._read_points()
        return SimState(
            joint_pos=to_torch(data.joint_pos),
            joint_vel=to_torch(data.joint_vel),
            body_state_w=to_torch(data.body_state_w),
            body_pos_w=to_torch(data.body_pos_w),
            body_quat_w=to_torch(data.body_quat_w),
            points_w=points,
            points_vel_w=velocities,
        )

    def _to_world(self, env_ids: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        """An env-frame pose (n, 7) in the world frame."""
        return torch.cat([pose[:, :3] + self.scene.env_origins[env_ids], pose[:, 3:]], dim=-1)

    def _reset_table(self, env_ids: torch.Tensor) -> None:
        """Draw the table's pose and move it there."""
        self.table.write_root_pose_to_sim(self._to_world(env_ids, self.task.sample_table(env_ids)), env_ids=env_ids)

    def _derive_sheet(self) -> FoldView:
        """The fold from the sheet alone (no robot read): at startup and after a reset's writes."""
        return self.task.geometry.derive(*self._read_points())

    def _write_goal(self, view: FoldView) -> None:
        """Refresh the goal pose and move the goal marker onto it (all envs)."""
        self.goal_viz.write_root_pose_to_sim(self.task.update_goal(view))

    # ------------------------------------------------------------------ Isaac Lab hooks

    def _setup_scene(self) -> None:
        # Through the module: the Newton backend wraps it with the scene replication.
        scene.setup_scene(self)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.controller.process(actions, self.task.episode_start(self.episode_length_buf))

    def _apply_action(self) -> None:
        # Every physics step (decimation times per policy step).
        self.robot.set_joint_position_target(self.controller.targets)
        spring = self.controller.spring_effort(to_torch(self.robot.data.joint_pos))
        if spring is not None:
            self.robot.set_joint_effort_target(spring)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # The physics state of this step, read once for the dones and the reward that follows.
        state = self._read_state()
        view = self.task.derive(state)
        self._step_state = (state, view)
        return self.task.compute_dones(state, view, self.episode_length_buf, self.max_episode_length)

    def _get_rewards(self) -> torch.Tensor:
        state, view = self._step_state
        reward = self.task.compute_reward(state, view)
        self._publish_metrics()
        return reward

    def _reset_idx(self, env_ids) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
        self._begin_reset(ids)
        super()._reset_idx(env_ids)
        # Draws in a fixed order (it sets the random stream): robot, table, sheet, then DR.
        joint_pos, joint_vel = self.controller.sample_reset(ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=ids)
        self.controller.reset(ids, joint_pos)
        self._reset_table(ids)
        self._write_sheet(ids, self._to_world(ids, self.task.sample_sheet(ids)))
        self.task.reset(ids)
        self.dr.reset(ids)
        self.observations.reset(ids)
        self._write_goal(self._derive_sheet())
        self._end_reset(ids)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        state = self._read_state()
        view = self.task.derive(state)
        # The goal rides on the stationary half's live frame: refreshed before it is observed.
        self._write_goal(view)
        start = self.task.episode_start(self.episode_length_buf)
        obs = self.observations.build(state, view, self.episode_length_buf, self.reward_buf, start)
        self.observations.guard_finite(obs, state, view, self.episode_length_buf)
        return obs

    def _publish_metrics(self) -> None:
        """Step extras for the RL-Games observer and the evaluation harness: per-step reward terms
        (``episode_cumulative``), per-episode outcomes (``episode_final``, one ``done_<reason>``
        per termination reason)."""
        task = self.task
        final = {"successes": task.successes.float()}
        final.update({f"done_{name}": value.float() for name, value in task.termination_reasons.items()})
        self.extras["episode_cumulative"] = task.reward_terms
        self.extras["episode_final"] = final
        self.extras["successes"] = task.prev_episode_successes.float()

    # ------------------------------------------------------------------ queries (eval, viewers)

    def _now(self) -> FoldView:
        return self._derive_sheet()

    def fold_error(self) -> torch.Tensor:
        """``(N,)`` max distance of a moving-half corner from its fold target."""
        return self._now().fold_error

    def fold_targets_w(self) -> torch.Tensor:
        """``(N, 4, 3)`` where the moving half's corners belong when folded."""
        return self._now().targets_w

    def cloth_keypoints_w(self) -> torch.Tensor:
        """``(N, 4, 3)`` the moving half's corners."""
        return self._now().keypoints_w

    def footprint_ratio(self) -> torch.Tensor:
        """``(N,)`` the sheet's extent along its fold axis over its size."""
        return self._now().footprint

    def is_folded(self) -> torch.Tensor:
        return self.task.is_folded(self._now())

    def folded_half_w(self) -> torch.Tensor:
        """``(N, M, 3)`` where every point of the moving half belongs when folded (the ghost goal)."""
        return self.task.geometry.folded_half_w(self._now())

    def folded_half_topology(self) -> list[int]:
        return self.task.geometry.folded_half_topology()

    def cloth_points_w(self) -> torch.Tensor:
        """``(N, P, 3)`` the sheet's grid points."""
        return self._read_points()[0]

    # ------------------------------------------------------------------ old names (scripts; removed in S6)

    def _particles_w(self) -> torch.Tensor:
        return self.cloth_points_w()

    def _stationary_frame(self, parts: torch.Tensor):
        return self.task.geometry.stationary_frame(parts)

    @property
    def _successes(self) -> torch.Tensor:
        return self.task.successes

    @property
    def _lifted_object(self) -> torch.Tensor:
        return self.task.lifted_object

    @property
    def _keypoint_offsets(self) -> torch.Tensor:
        return self.task.keypoint_offsets

    @property
    def _moving_idx(self) -> torch.Tensor:
        return self.task.geometry.moving_idx

    @property
    def _stationary_idx(self) -> torch.Tensor:
        return self.task.geometry.stationary_idx

    @property
    def _palm_body_id(self) -> int:
        return self.robot_layout.palm_body_id

    @property
    def _fingertip_body_ids(self) -> list[int]:
        return self.robot_layout.fingertip_body_ids

    @property
    def _perm_lab_to_canon(self) -> torch.Tensor:
        return self.robot_layout.perm_lab_to_canon
