"""Shared DirectRLEnv configuration; backend configs supply physical parameters."""

from __future__ import annotations

from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass

from .backends import robot as flexiv_wuji
from .mdp.config import (
    ActionCfg,
    AssetsCfg,
    ClothTaskCfg,
    DomainRandomizationCfg,
    ObsCfg,
    ResetCfg,
    RewardCfg,
    TerminationCfg,
)

__all__ = ["ClothCfg"]


@configclass
class ClothCfg(DirectRLEnvCfg):
    """Shared settings. Use ``for_backend`` to include the chosen physics configuration."""

    @classmethod
    def for_backend(cls, backend: str = "rigid") -> ClothCfg:
        """Construct a complete rigid or VBD config in its corresponding Isaac runtime."""
        if backend == "rigid":
            from .backends.rigid.config import RigidClothEnvCfg

            return RigidClothEnvCfg()
        if backend == "vbd":
            from .backends.vbd.config import DeformableClothEnvCfg

            return DeformableClothEnvCfg()
        raise ValueError(f"unknown cloth backend {backend!r}; expected rigid or vbd")

    backend: str = "rigid"
    """Scene adapter: rigid plates or vbd particle cloth."""

    robot_root_pos: tuple[float, float, float] = flexiv_wuji.ROOT_POS
    robot_self_collision: bool = True
    decimation: int = 2
    episode_length_s: float = 10.0
    action_space: int = flexiv_wuji.ROBOT.num_joints
    observation_space: int = 134
    state_space: int = 156
    viewer: ViewerCfg = ViewerCfg(eye=(0.5, -1.5, 1.2), lookat=(0.0, 0.4, 0.5), resolution=(640, 480))
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=768, env_spacing=1.2, replicate_physics=False, clone_in_fabric=False
    )
    assets: AssetsCfg = AssetsCfg()
    action: ActionCfg = ActionCfg()
    obs: ObsCfg = ObsCfg()
    reward: RewardCfg = RewardCfg()
    reset: ResetCfg = ResetCfg()
    termination: TerminationCfg = TerminationCfg()
    domain_randomization: DomainRandomizationCfg = DomainRandomizationCfg()
    task: ClothTaskCfg = ClothTaskCfg()
