"""One ClothEnv and its pretrained policy behind the rigid-cloth viewer interface.

The environment owns observations, actions, rewards, and automatic resets. The
viewer only pins the reset pose and selects the training or real robot profile.
Simulator imports stay inside construction so profile checks work without Kit.
"""
from __future__ import annotations

from dataclasses import fields, replace
import copy
import gc
import math

import numpy as np
import torch

from isaacsimenvs import REPO_ROOT
from isaacsimenvs.tasks.cloth.backends.robot import ROBOT, YIBO_ROBOT, RobotSpec
from isaacsimenvs.scripts.visualization.rigid_cloth.reset import load_training_env_config, sample_reset_pose, validate_reset_settings


POLICY_FIELDS = (
    "joint_pos", "joint_vel", "prev_action_targets", "palm_pos", "palm_rot", "object_rot",
    "fingertip_pos_rel_palm", "keypoints_rel_palm", "keypoints_rel_goal", "object_scales",
)


def robot_profile(profile: str):
    """Return the complete robot spec, URDF, table URDF and root position."""
    if profile in ("training", "yibo"):
        from isaacsimenvs.tasks.play.robots import FLEXIV_WUJI_RIGHT_HUMAN_LIKE

        source = FLEXIV_WUJI_RIGHT_HUMAN_LIKE
        spec = RobotSpec(**{field.name: getattr(source, field.name) for field in fields(RobotSpec)})
        if profile == "yibo":
            return (
                YIBO_ROBOT, "assets/urdf/flexiv_wuji/rizon4s_right_wuji_real_limits.urdf",
                "assets/urdf/table_fold_pool.urdf", (0.0, 0.51, 0.20),
            )
        return (
            spec, "assets/urdf/flexiv_wuji/rizon4s_right_wuji_human_like_joint_limits.urdf",
            "assets/urdf/table_narrow_flexiv.urdf", (0.0, 0.8, 0.0),
        )
    if profile == "real":
        return (
            ROBOT, "assets/urdf/flexiv_wuji/rizon4s_right_wuji_real_limits.urdf",
            "assets/urdf/table_fold_pool.urdf", (0.0, 0.51, 0.20),
        )
    raise ValueError(f"Unknown policy robot profile {profile!r}; expected training, real or yibo")


def table_position(profile: str, saved_env: dict) -> tuple[float, float, float]:
    """Nominal table center shared by policy and static preview setups."""
    if profile == "training":
        reset = saved_env["reset"]
        return (*map(float, reset["table_reset_xy"]), float(reset["table_reset_z"]))
    if profile in ("real", "yibo"):
        return (0.0, 0.0, 0.38)
    raise ValueError(f"Unknown robot/table setup {profile!r}")


def validate_policy_config(saved: dict) -> None:
    """Reject another robot or reordered actor layout even when tensor widths match."""
    env = saved["env"]
    if env.get("robot") != "flexiv_wuji_right_human_like":
        raise ValueError("This viewer supports checkpoints trained with flexiv_wuji_right_human_like")
    if tuple(env["obs"]["obs_list"]) != POLICY_FIELDS:
        raise ValueError("Checkpoint observation order differs from the supported 134-value Flexiv/Wuji layout")


def build_policy_env_cfg(settings, saved: dict, profile: str, device: str, training_config=None):
    """Pin geometry and noise while retaining checkpoint observation/action scaling."""
    from isaacsimenvs.tasks.cloth.cloth_cfg import ClothCfg
    from isaacsimenvs.tasks.cloth.backends.assets import urdf_box_size
    from isaacsimenvs.scripts.evaluation.utils import disable_randomization

    validate_policy_config(saved)
    spec, robot_urdf, table_urdf, root = robot_profile(profile)
    source = saved["env"]
    validate_reset_settings(settings)
    cfg = ClothCfg.for_backend("rigid")
    cfg.from_dict(load_training_env_config(training_config))
    cfg.scene.num_envs = 1
    cfg.sim.device = device
    cfg.seed = 42
    cfg.sim.dt = float(source["sim"]["dt"])
    cfg.decimation = int(source["decimation"])
    cfg.sim.render_interval = cfg.decimation
    cfg.episode_length_s = float(source["episode_length_s"])
    cfg.assets.robot_urdf = robot_urdf
    cfg.assets.table_urdf = table_urdf
    cfg.robot_root_pos = root
    cfg.robot_self_collision = bool(source.get("robot_self_collision", True))
    cfg.action.from_dict(source["action"])
    cfg.obs.from_dict(source["obs"])
    cfg.obs.keypoint_mode = settings.keypoint_mode
    cfg.obs.keypoint_scale = float(source["reward"]["keypoint_scale"])
    cfg.reward.object_base_size = float(source["reward"]["object_base_size"])
    cfg.task.mode = "fold"
    center = table_position(profile, source)
    cfg.reset.table_reset_xy, cfg.reset.table_reset_z = center[:2], center[2]
    cloth = cfg.rigid_cloth
    cloth.designs = [settings.design]
    cloth.geometries = [[float(settings.length), float(settings.width)]]
    cloth.thickness = float(settings.thickness)
    cloth.density = float(settings.density)
    cloth.bending_rigidity = float(settings.bending_rigidity)
    cloth.bending_viscosity = float(settings.bending_viscosity)
    cloth.hinge_armature = float(settings.armature)
    cloth.hinge_stiffness_override = settings.hinge_stiffness_override
    cloth.hinge_damping_override = settings.hinge_damping_override
    cloth.rest_offset = settings.rest_offset
    cloth.contact_offset = settings.contact_offset
    cloth.table_static_friction = float(settings.table_static_friction)
    cloth.hand_static_friction = float(settings.hand_static_friction)
    cloth.dynamic_friction_ratio = float(settings.dynamic_friction_ratio)
    cloth.spawn_clearance = float(settings.drop_height)
    # The ordinary reset creates a valid temporary state. The viewer reset hook
    # then applies its manual position or sampled region before observations.
    table_size = urdf_box_size(str(REPO_ROOT / table_urdf))
    cloth.place_x = [-table_size[0] / 2 + .005, table_size[0] / 2 - .005]
    cloth.place_y = [-table_size[1] / 2 + .005, table_size[1] / 2 - .005]
    sensor_dr = copy.deepcopy(cfg.domain_randomization)
    disable_randomization(cfg)
    if settings.sample_sensor:
        cfg.domain_randomization = sensor_dr
    cfg.rigid_cloth.randomization.enabled = settings.sample_material
    return cfg, spec


class PolicyPhysics:
    """Policy-rate stepping; manual reset retains the model and clears its LSTM."""

    def __init__(self, settings, *, profile="training", config_path, checkpoint_path,
                 device="cuda:0", expl_coef=0.0, deterministic=True, training_config=None):
        from isaaclab.sim import SimulationContext
        from isaaclab.sim.utils import create_new_stage
        from isaacsimenvs.tasks.cloth.cloth_env import ClothEnv
        from isaacsimenvs.tasks.cloth.backends.assets import urdf_box_size
        from isaacsimenvs.scripts.evaluation.utils import PretrainedPlayer, read_policy_cfg

        self.env = self.sim = self.cloth = None
        self.player = None
        self.deterministic = deterministic
        self.episodes = self.successes = 0
        self.time = self.reward = 0.0
        saved = read_policy_cfg(config_path, device)
        cfg, robot_spec = build_policy_env_cfg(settings, saved, profile, device, training_config)
        self.robot_urdf = REPO_ROOT / cfg.assets.robot_urdf
        self.robot_root_pos = tuple(cfg.robot_root_pos)
        self.table_size = urdf_box_size(str(REPO_ROOT / cfg.assets.table_urdf))
        self.table_position = (*cfg.reset.table_reset_xy, cfg.reset.table_reset_z)

        validate_reset_settings(settings, torch.tensor([[settings.length, settings.width, settings.thickness]]), self.table_size)

        class ViewerClothEnv(ClothEnv):
            viewer_settings = replace(settings)
            viewer_table_size = self.table_size

            def _reset_idx(inner, env_ids):
                super()._reset_idx(env_ids)
                ids = (torch.arange(inner.num_envs, device=inner.device) if env_ids is None
                       else torch.as_tensor(env_ids, device=inner.device, dtype=torch.long).reshape(-1))
                pose = inner.viewer_settings
                xy, yaw = sample_reset_pose(pose, inner.backend._dims[ids], table_size=inner.viewer_table_size)
                inner.backend._write_cloth(
                    ids, xy, yaw,
                    torch.full((len(ids),), math.radians(pose.bend_deg), device=inner.device),
                )
                # Auto-reset returns observations immediately, unlike public reset().
                # Forward here ensures those observations contain the configured bend.
                inner.scene.write_data_to_sim()
                inner.sim.forward()

        try:
            create_new_stage()
            self.env = ViewerClothEnv(cfg, robot_spec=robot_spec)
            self.env.viewer_settings = replace(settings)
            self.sim = self.env.sim
            self._robot_materials_base = self.env.robot.root_physx_view.get_material_properties().clone()
            group = self.env.backend.groups[0]
            self.cloth = group.articulation
            self.spec = group.design.resolve(settings.length, settings.width, settings.thickness)
            self.body_ids, self.joint_ids = group.body_ids, group.joint_ids
            self.dt = self.env.step_dt
            self.player = PretrainedPlayer(
                config_path, checkpoint_path, num_envs=1, device=device,
                num_observations=134, num_actions=27, expl_coef=expl_coef,
            )
            self.reset(settings)
        except BaseException:
            self.sim = self.sim or SimulationContext.instance()
            self.close()
            raise

    def reset(self, settings):
        validate_reset_settings(settings, self.env.backend._dims, self.table_size)
        self.env.viewer_settings = replace(settings)
        cfg = self.env.cfg.rigid_cloth
        cfg.spawn_clearance = float(settings.drop_height)
        self.update_material(settings)
        with torch.no_grad():
            self.obs, _ = self.env.reset()
        self.player.reset_rnn()
        self.time = self.reward = 0.0

    def update_material(self, settings):
        from isaacsimenvs.tasks.cloth.backends.rigid.bending import hinge_geometry_factors
        from isaacsimenvs.tasks.cloth.backends.rigid.inertia import hinge_inertias
        from isaacsimenvs.tasks.cloth.backends.rigid.scene import pair_materials
        cfg = self.env.cfg.rigid_cloth
        if getattr(getattr(cfg, 'randomization', None), 'enabled', False):
            # The backend owns sampled materials. In particular Yibo friction is
            # a startup draw, so a manual reset must not replace it with sliders.
            return
        factors = hinge_geometry_factors(self.spec)
        for parameter, method in (("bending_rigidity", "stiffness"), ("bending_viscosity", "damping")):
            value = float(getattr(settings, parameter))
            setattr(cfg, parameter, value)
            override = getattr(cfg, f"hinge_{method}_override", None)
            values = torch.as_tensor(value * factors if override is None else np.full_like(factors, override),
                                     dtype=torch.float32, device=self.cloth.device)[None]
            getattr(self.cloth, f"write_joint_{method}_to_sim")(values, joint_ids=self.joint_ids)
        cfg.hinge_armature = float(settings.armature)
        armature = cfg.hinge_armature + float(settings.armature_scale) * hinge_inertias(self.spec, settings.density)
        self.cloth.write_joint_armature_to_sim(
            torch.as_tensor(armature, dtype=torch.float32, device=self.cloth.device)[None], joint_ids=self.joint_ids,
        )
        for name in ("table_static_friction", "hand_static_friction", "dynamic_friction_ratio"):
            setattr(cfg, name, float(getattr(settings, name)))
        indices = torch.tensor([0], dtype=torch.int32, device="cpu")
        coefficients = pair_materials(cfg.table_static_friction, cfg.hand_static_friction, cfg.dynamic_friction_ratio)
        for name, asset in (("cloth", self.cloth), ("table", self.env.table), ("robot", self.env.robot)):
            view = asset.root_physx_view
            materials = view.get_material_properties()
            materials[...] = torch.as_tensor(coefficients[name], dtype=materials.dtype, device=materials.device)
            view.set_material_properties(materials, indices)

    def step(self):
        with torch.no_grad():
            action = self.player.get_action(self.obs["policy"], deterministic=self.deterministic)
            self.obs, reward, terminated, truncated, extras = self.env.step(action)
        self.reward = float(reward[0])
        self.time += self.dt
        if bool((terminated | truncated)[0]):
            self.episodes += 1
            final = extras.get("episode_final", {})
            self.successes += int(float(final.get("successes", [0])[0]) > 0)
            self.player.reset_rnn()
            self.time = 0.0

    def read_poses(self):
        pose = self.cloth.data.body_link_pose_w[0, self.body_ids].detach().cpu().numpy()
        return pose[:, :3].copy(), pose[:, 3:7].copy()

    def robot_joint_positions(self, names):
        robot = self.env.robot
        indices = [robot.joint_names.index(name) for name in names]
        return robot.data.joint_pos[0, indices].detach().cpu().numpy().copy()

    def keypoints(self, *, scaled=True):
        """World-space current/target pair: physical landmarks or policy markers."""
        state, view = self.env._now()
        points = (self.env.observations.keypoints_w(view, state) if scaled else
                  (view.corners_w, view.targets_w))
        return tuple(value[0].detach().cpu().numpy().copy() for value in points)

    def current_points(self):
        return self.keypoints()[0]

    def goal_points(self):
        return self.keypoints()[1]

    def metrics(self):
        return {
            "time": self.time,
            "mass": float(self.cloth.root_physx_view.get_masses().sum()),
            "hinge_angles_deg": np.degrees(self.cloth.data.joint_pos[0, self.joint_ids].detach().cpu().numpy()),
            "fold_error": float(self.env.fold_error()[0]),
            "reward": self.reward, "episodes": self.episodes, "successes": self.successes,
            "robot_clearance": (float(self.env._robot_cloth_clearance.distance()[0])
                                if self.env._robot_cloth_clearance is not None else None),
        }

    def close(self):
        sim, env = self.sim, self.env
        self.env = self.sim = self.cloth = self.player = None
        self.obs = None
        if sim is not None:
            # Remove standalone STOP callback before stopping or closing the env.
            type(sim).clear_instance()
        try:
            if env is not None:
                env.close()
        finally:
            if sim is not None:
                sim.stop()
            gc.collect()
