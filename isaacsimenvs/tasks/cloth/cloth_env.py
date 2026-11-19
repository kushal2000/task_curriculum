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
    _reset_idx          backend.begin_reset, controller / task draws written to the scene, DR draws,
                        backend.end_reset
    _get_observations   task.update_goal, observations.build

The simulator side is a backend object (``backends/base.ClothBackend``) the env owns:
``backends/rigid/backend.RigidBackend`` (PhysX, the fold-shape pool of hinged plates) or
``backends/vbd/backend.VbdBackend`` (Newton, a VBD cloth). It builds the scene (robot, table, goal
marker, sheet), reads the sheet's points and writes resets; the task never sees which one it is.
"""

from __future__ import annotations

import torch
from isaaclab.envs import DirectRLEnv

from isaacsimenvs.robots import get_robot_spec

from .backends.base import ClothBackend
from .mdp.actions import JointController
from .mdp.observations import ObservationBuilder, obs_dim
from .mdp.randomization import DomainRandomization
from .mdp.robot import RobotLayout, to_torch
from .mdp.state import FoldView, SimState
from .mdp.task import ClothTask

__all__ = ["ClothEnv"]


class ClothEnv(DirectRLEnv):
    """Fold half a sheet onto the other half, with the robot of ``cfg.robot``."""

    def __init__(self, cfg, render_mode: str | None = None, *, backend: type[ClothBackend], **kwargs) -> None:
        self.robot_spec = get_robot_spec(cfg.robot)
        n_joints = self.robot_spec.num_joints
        cfg.action_space = n_joints
        cfg.observation_space = obs_dim(cfg.obs.obs_list, n_joints)
        cfg.state_space = obs_dim(cfg.obs.state_list, n_joints)
        self.backend = backend(cfg, self.robot_spec)
        super().__init__(cfg, render_mode, **kwargs)  # builds the scene (_setup_scene)
        self._check_env_order()
        dev, n = self.device, self.num_envs
        self.robot_layout = RobotLayout(
            self.robot, self.robot_spec, num_joints=n_joints, robot_urdf=cfg.assets.robot_urdf, device=dev
        )
        layout = self.backend.sheet_layout()
        self.task = ClothTask(
            cfg, layout, self.robot_layout, num_envs=n, device=dev, env_origins=self.scene.env_origins,
            extras=self.extras, observed_goal_quat=self.backend.observed_goal_quat,
        )
        # Observed by the critic (`reward`) from the first reset on, before any step has paid one.
        self.reward_buf = torch.zeros(n, device=dev)
        self.dr = DomainRandomization(
            cfg, self.robot_layout, self.robot, num_envs=n, num_keypoints=len(layout.keypoint_idx), device=dev
        )
        self.controller = JointController(cfg, self.robot_layout, self.dr, step_dt=self.step_dt, num_envs=n, device=dev)
        self.observations = ObservationBuilder(
            cfg, self.robot_layout, self.controller, self.task, self.dr, num_envs=n, device=dev,
            env_origins=self.scene.env_origins, extras=self.extras,
        )
        self.backend.finish_setup()
        view = self._derive_sheet()
        # Fold error of a flat sheet: the stand-in for a non-finite one (ClothTask.compute_reward).
        self.task.flat_fold_err = float(view.fold_error.nan_to_num(nan=0.0).max().item())
        self._write_goal(view)
        self._step_state: tuple[SimState, FoldView] | None = None
        self.backend.start(self._reset_table)

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

    # ------------------------------------------------------------------ the scene, as the backend built it

    @property
    def robot(self):
        return self.backend.robot

    @property
    def table(self):
        return self.backend.table

    @property
    def goal_viz(self):
        return self.backend.goal_viz

    # ------------------------------------------------------------------ state reads

    def sim_state(self) -> SimState:
        """The robot and the sheet now (``mdp/state.SimState``)."""
        data = self.robot.data
        points, velocities = self.backend.read_points()
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
        return self.task.geometry.derive(*self.backend.read_points())

    def _write_goal(self, view: FoldView) -> None:
        """Refresh the goal pose and move the goal marker onto it (all envs)."""
        self.goal_viz.write_root_pose_to_sim(self.task.update_goal(view))

    # ------------------------------------------------------------------ Isaac Lab hooks

    def _setup_scene(self) -> None:
        self.backend.build_scene(self.scene, self.sim)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.controller.process(actions, self.task.episode_start(self.episode_length_buf))
        self.backend.pre_physics_step()

    def _apply_action(self) -> None:
        # Every physics step (decimation times per policy step).
        self.robot.set_joint_position_target(self.controller.targets)
        spring = self.controller.spring_effort(to_torch(self.robot.data.joint_pos))
        if spring is not None:
            self.robot.set_joint_effort_target(spring)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # The physics state of this step, read once for the dones and the reward that follows.
        state = self.sim_state()
        view = self.task.derive(state)
        self._step_state = (state, view)
        dones = self.task.compute_dones(state, view, self.episode_length_buf, self.max_episode_length)
        self.backend.add_termination_reasons(self.task.termination_reasons)
        return dones

    def _get_rewards(self) -> torch.Tensor:
        state, view = self._step_state
        reward = self.task.compute_reward(state, view)
        self._publish_metrics()
        self.backend.add_episode_final(
            self.extras["episode_final"], self.task.termination_reasons, self.episode_length_buf,
            self.observations.tracker.episode_stats(),
        )
        return reward

    def _reset_idx(self, env_ids) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
        self.backend.begin_reset(ids)
        super()._reset_idx(env_ids)
        # Draws in a fixed order (it sets the random stream): robot, table, sheet, then DR.
        joint_pos, joint_vel = self.controller.sample_reset(ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=ids)
        self.controller.reset(ids, joint_pos)
        self._reset_table(ids)
        self.backend.write_sheet(ids, self._to_world(ids, self.task.sample_sheet(ids)))
        self.task.reset(ids)
        self.dr.reset(ids)
        self.observations.reset(ids)
        self._write_goal(self._derive_sheet())
        self.backend.end_reset(ids)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        state = self.sim_state()
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

    def fold_view(self, state: SimState | None = None) -> FoldView:
        """The fold now, or of ``state`` (``mdp/state.FoldView``): the sheet's points, the
        stationary half's frame, the corners and their targets, fold error, footprint."""
        return self._derive_sheet() if state is None else self.task.derive(state)

    def fold_error(self) -> torch.Tensor:
        """``(N,)`` max distance of a moving-half corner from its fold target."""
        return self.fold_view().fold_error

    def fold_targets_w(self) -> torch.Tensor:
        """``(N, 4, 3)`` where the moving half's corners belong when folded."""
        return self.fold_view().targets_w

    def cloth_keypoints_w(self) -> torch.Tensor:
        """``(N, 4, 3)`` the moving half's corners."""
        return self.fold_view().keypoints_w

    def footprint_ratio(self) -> torch.Tensor:
        """``(N,)`` the sheet's extent along its fold axis over its size."""
        return self.fold_view().footprint

    def is_folded(self) -> torch.Tensor:
        return self.task.is_folded(self.fold_view())

    def folded_half_w(self) -> torch.Tensor:
        """``(N, M, 3)`` where every point of the moving half belongs when folded (the ghost goal)."""
        return self.task.geometry.folded_half_w(self.fold_view())

    def folded_half_topology(self) -> list[int]:
        return self.task.geometry.folded_half_topology()

    def object_pos_w(self) -> torch.Tensor:
        """``(N, 3)`` the moving half's centre: the sheet as the policy's rigid object."""
        return self.fold_view().object_pos_w

    def cloth_points_w(self) -> torch.Tensor:
        """``(N, P, 3)`` the sheet's grid points."""
        return self.backend.read_points()[0]
