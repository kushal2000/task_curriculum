"""Cloth folding tasks: one fold task (``mdp/``) on two backends (``backends/``).

    Isaacsimenvs-Cloth-Direct-v0       VBD cloth sheet, Newton (``.venv_isaaclab3``)
    Isaacsimenvs-RigidCloth-Direct-v0  fold-shape pool of hinged plates, PhysX (``.venv_isaacsim``)

The robot is a config field, ``env.robot_setup`` (a file stem in ``cfg/robot/``), set in the run
recipe or on the command line. The older ids -- ``FoldPoolPhysx`` and one per robot, e.g.
``Isaacsimenvs-FoldPoolPhysxFlexivWujiRightReal-Direct-v0`` -- stay registered as aliases: each
names its canonical task and robot setup (``utils/hydra_utils.resolve_task``). Start with
``README.md``.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym

_CFG_DIR = Path(__file__).resolve().parents[2] / "cfg"

_TASKS = {
    "Cloth": ("backends.vbd.env:VbdClothEnv", "backends.vbd.config:ClothEnvCfg", "Cloth.yaml"),
    "RigidCloth": ("backends.rigid.env:RigidClothEnv", "backends.rigid.config:RigidClothEnvCfg", "FoldPoolPhysx.yaml"),
}


def _register(task_id: str, base: str, **extra) -> None:
    env, cfg, task_yaml = _TASKS[base]
    gym.register(
        id=task_id,
        entry_point=f"isaacsimenvs.tasks.cloth.{env}",
        order_enforce=False,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"isaacsimenvs.tasks.cloth.{cfg}",
            "env_cfg_yaml_entry_point": str(_CFG_DIR / "task" / task_yaml),
            "rl_games_cfg_entry_point": str(_CFG_DIR / "train" / "PlayPPO.yaml"),
            "rl_games_sapg_cfg_entry_point": str(_CFG_DIR / "train" / "ClothSAPG.yaml"),
            **extra,
        },
    )


for _base in _TASKS:
    _register(f"Isaacsimenvs-{_base}-Direct-v0", _base)

# --- aliases: the ids before the robot became a config field (old recipes and launchers use them).
_ALIAS_BASE = {"Cloth": "Cloth", "FoldPoolPhysx": "RigidCloth"}
_ALIAS_SETUP = {
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
_register("Isaacsimenvs-FoldPoolPhysx-Direct-v0", "RigidCloth", canonical_task="Isaacsimenvs-RigidCloth-Direct-v0")
for _old, _base in _ALIAS_BASE.items():
    for _suffix, _setup in _ALIAS_SETUP.items():
        _register(
            f"Isaacsimenvs-{_old}{_suffix}-Direct-v0", _base,
            canonical_task=f"Isaacsimenvs-{_base}-Direct-v0", robot_setup=_setup,
        )
del _base, _old, _suffix, _setup
