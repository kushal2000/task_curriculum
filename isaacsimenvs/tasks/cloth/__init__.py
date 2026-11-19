"""Cloth folding tasks: one fold task (``ClothEnv`` + ``mdp/``) on two backends (``backends/``).

    Isaacsimenvs-Cloth-Direct-v0       VBD cloth sheet, Newton (``.venv_isaaclab3``)
    Isaacsimenvs-RigidCloth-Direct-v0  fold-shape pool of hinged plates, PhysX (``.venv_isaacsim``)

The robot is a config field, ``env.robot_setup`` (a file stem in ``configs/robot/``), set in the run
recipe or on the command line. The older ids -- ``FoldPoolPhysx`` and one per robot, e.g.
``Isaacsimenvs-FoldPoolPhysxFlexivWujiRightReal-Direct-v0`` -- stay registered as aliases: each
names its canonical task and robot setup (``scripts/configs/loader.resolve_task``). Start with
``README.md``.
"""

from __future__ import annotations

import gymnasium as gym

from isaacsimenvs import CONFIG_DIR as _CFG_DIR

# task -> (backend, its config class: the task's defaults)
_TASKS = {
    "Cloth": ("vbd", "backends.vbd.config:VbdClothEnvCfg"),
    "RigidCloth": ("rigid", "backends.rigid.config:RigidClothEnvCfg"),
}


def make_env(cfg, render_mode: str | None = None, *, backend: str, **kwargs):
    """The gym entry point: ``ClothEnv`` with the named backend. Imports stay lazy until here (the
    Newton backend's Isaac Lab 3.0 shims must load before any other Isaac Lab module)."""
    if backend == "vbd":
        from .backends.vbd.backend import VbdBackend as Backend
    elif backend == "rigid":
        from .backends.rigid.backend import RigidBackend as Backend
    else:
        raise ValueError(f"unknown cloth backend {backend!r}; expected rigid or vbd")
    from .cloth_env import ClothEnv

    return ClothEnv(cfg, render_mode, backend=Backend, **kwargs)


def _register(task_id: str, base: str, **extra) -> None:
    backend, cfg = _TASKS[base]
    gym.register(
        id=task_id,
        entry_point="isaacsimenvs.tasks.cloth:make_env",
        order_enforce=False,
        disable_env_checker=True,
        kwargs={
            "backend": backend,
            "env_cfg_entry_point": f"isaacsimenvs.tasks.cloth.{cfg}",
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
