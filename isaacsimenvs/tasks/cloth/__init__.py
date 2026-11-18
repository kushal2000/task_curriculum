"""Cloth folding tasks: one fold task (``mdp/``) on two backends (``backends/``).

    Isaacsimenvs-Cloth-Direct-v0          VBD cloth sheet, Newton (``.venv_isaaclab3``)
    Isaacsimenvs-FoldPoolPhysx-Direct-v0  fold-shape pool of hinged plates, PhysX (``.venv_isaacsim``)

plus one id per robot overlay (``cfg/robot/``), below. Start with ``README.md``.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym

_CFG_DIR = Path(__file__).resolve().parents[2] / "cfg"


gym.register(
    id="Isaacsimenvs-Cloth-Direct-v0",
    entry_point="isaacsimenvs.tasks.cloth.backends.vbd.env:ClothEnv",
    order_enforce=False,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaacsimenvs.tasks.cloth.backends.vbd.config:ClothEnvCfg",
        "env_cfg_yaml_entry_point": str(_CFG_DIR / "task" / "Cloth.yaml"),
        "rl_games_cfg_entry_point": str(_CFG_DIR / "train" / "PlayPPO.yaml"),
        "rl_games_sapg_cfg_entry_point": str(_CFG_DIR / "train" / "ClothSAPG.yaml"),
    },
)


gym.register(
    id="Isaacsimenvs-FoldPoolPhysx-Direct-v0",
    entry_point="isaacsimenvs.tasks.cloth.backends.rigid.env:RigidClothEnv",
    order_enforce=False,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaacsimenvs.tasks.cloth.backends.rigid.config:RigidClothEnvCfg",
        "env_cfg_yaml_entry_point": str(_CFG_DIR / "task" / "FoldPoolPhysx.yaml"),
        "rl_games_cfg_entry_point": str(_CFG_DIR / "train" / "PlayPPO.yaml"),
        "rl_games_sapg_cfg_entry_point": str(_CFG_DIR / "train" / "ClothSAPG.yaml"),
    },
)


# The same two tasks on the Flexiv Rizon 4s + Wuji Hand 2 or the Kuka's Sharpa, e.g.
# `Isaacsimenvs-FoldPoolPhysxFlexivWujiHumanLike-Direct-v0`: the task's own cfg and YAML, then a
# robot overlay from `cfg/robot/` (robot spec, URDF, self-collision as the Flexiv Play tasks train).
_ROBOT_VARIANTS = {
    "FlexivWuji": "flexiv_wuji",
    "FlexivWujiHumanLike": "flexiv_wuji_human_like",
    "FlexivWujiHumanLikeS2R": "flexiv_wuji_human_like_s2r",
    "FlexivSharpa": "flexiv_sharpa",
    "FlexivWujiHumanLikeSharpaGain": "flexiv_wuji_human_like_sharpagain",
    "FlexivSharpaWujiGain": "flexiv_sharpa_wujigain",
    "FlexivWujiRightHumanLike": "flexiv_wuji_right_human_like",
    "FlexivWujiRightReal": "flexiv_wuji_right_real",
    "FlexivWujiRightS2REff": "flexiv_wuji_right_s2r_eff",
    "FlexivWujiRightSharpaGain": "flexiv_wuji_right_sharpagain",
    "FlexivWujiRightHLRL": "flexiv_wuji_right_hl_reallimits",
}
for _task in ("Cloth", "FoldPoolPhysx"):
    _base = gym.spec(f"Isaacsimenvs-{_task}-Direct-v0")
    for _suffix, _robot in _ROBOT_VARIANTS.items():
        gym.register(
            id=f"Isaacsimenvs-{_task}{_suffix}-Direct-v0",
            entry_point=_base.entry_point,
            order_enforce=False,
            disable_env_checker=True,
            kwargs={
                **_base.kwargs,
                "env_cfg_robot_yaml_entry_point": str(_CFG_DIR / "robot" / f"{_robot}.yaml"),
            },
        )
del _task, _base, _suffix, _robot
