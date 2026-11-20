"""Cloth folding tasks: one fold task (``ClothEnv`` + ``mdp/``) on two backends (``backends/``).

    Isaacsimenvs-Cloth-Direct-v0       VBD cloth sheet, Newton (``.venv_isaaclab3``)
    Isaacsimenvs-RigidCloth-Direct-v0  fold-shape pool of hinged plates, PhysX (``.venv_isaacsim``)

Both run the Flexiv + right Wuji as the real robot is set up. Their configuration is ``config/``
(the fields in ``config/schema/``, the values in ``config/task.yaml``). Start with ``README.md``.
"""

from __future__ import annotations

import gymnasium as gym

from isaacsimenvs import CLOTH_CONFIG_DIR, PLAY_CONFIG_DIR

# task -> (backend, its config: the configclass with config/task.yaml applied)
_TASKS = {
    "Cloth": ("vbd", "config:vbd_cloth_env_cfg"),
    "RigidCloth": ("rigid", "config:rigid_cloth_env_cfg"),
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


def _register(task_id: str, base: str) -> None:
    backend, cfg = _TASKS[base]
    gym.register(
        id=task_id,
        entry_point="isaacsimenvs.tasks.cloth:make_env",
        order_enforce=False,
        disable_env_checker=True,
        kwargs={
            "backend": backend,
            "env_cfg_entry_point": f"isaacsimenvs.tasks.cloth.{cfg}",
            "rl_games_cfg_entry_point": str(PLAY_CONFIG_DIR / "PlayPPO.yaml"),  # Play's PPO, as the prior
            "rl_games_sapg_cfg_entry_point": str(CLOTH_CONFIG_DIR / "ClothSAPG.yaml"),
        },
    )


for _base in _TASKS:
    _register(f"Isaacsimenvs-{_base}-Direct-v0", _base)
del _base
