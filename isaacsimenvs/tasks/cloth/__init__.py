"""Cloth task registration; simulator imports stay lazy until configuration is selected."""

from __future__ import annotations

import gymnasium as gym

from isaacsimenvs import CONFIG_DIR

# The deformable VBD sheet on Newton (coupled MJWarp + VBD), the same robot built in.
gym.register(
    id="Isaacsimenvs-Cloth-Direct-v0",
    entry_point="isaacsimenvs.tasks.cloth:make_env",
    order_enforce=False,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaacsimenvs.tasks.cloth.backends.vbd.config:DeformableClothEnvCfg",
        "env_cfg_yaml_entry_point": str(CONFIG_DIR / "task" / "Cloth.yaml"),
        "rl_games_cfg_entry_point": str(CONFIG_DIR / "train" / "PlayPPO.yaml"),
        "rl_games_sapg_cfg_entry_point": str(CONFIG_DIR / "train" / "ClothSAPG.yaml"),
    },
)


# Rigid2Deform: the fold task on a list of plate/hinge designs, one size per env, on PhysX. One
# robot (Flexiv Rizon 4s + right Wuji Hand 2, identified hand: robot.py, built into the cfg).
# Not `Cloth*`-prefixed: experiments/_common.sh sends `Isaacsimenvs-Cloth*` to Newton.
gym.register(
    id="Isaacsimenvs-RigidBodyCloth-Direct-v0",
    entry_point="isaacsimenvs.tasks.cloth:make_env",
    order_enforce=False,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaacsimenvs.tasks.cloth.backends.rigid.config:RigidClothEnvCfg",
        "env_cfg_yaml_entry_point": str(CONFIG_DIR / "task" / "RigidBodyCloth.yaml"),
        "rl_games_cfg_entry_point": str(CONFIG_DIR / "train" / "PlayPPO.yaml"),
        "rl_games_sapg_cfg_entry_point": str(CONFIG_DIR / "train" / "ClothSAPG.yaml"),
    },
)


def make_env(cfg, **kwargs):
    """Choose compatibility setup before importing the shared DirectRLEnv class."""
    if cfg.backend == "vbd":
        from isaacsimenvs.tasks.cloth.backends.vbd.newton_support import compat

        compat.install_isaaclab3_compat()
    from .cloth_env import ClothEnv

    return ClothEnv(cfg, **kwargs)


__all__ = ["ClothEnv", "ClothCfg"]


def __getattr__(name):
    # Keep Gym registration and MDP imports independent of simulator startup.
    if name == "ClothEnv":
        from .cloth_env import ClothEnv

        return ClothEnv
    if name == "ClothCfg":
        from .cloth_cfg import ClothCfg

        return ClothCfg
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
