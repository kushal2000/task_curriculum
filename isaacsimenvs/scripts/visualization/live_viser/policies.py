"""The policy menu (``policies.yaml`` or ``--policies``) and the players behind it.

Players are made as ``scripts/evaluation/eval.py`` makes them: a point-cloud student (recognised from
its checkpoint) reads the env's cloud, which must be the one it trained on; an rl_games policy reads
the keypoint vector (with a point-cloud env, the env's clean ``teacher`` vector), SAPG coefficient 0.
"""

from __future__ import annotations

from pathlib import Path

import yaml

DEFAULT_MENU = Path(__file__).with_name("policies.yaml")


def load_menu(path: str | Path) -> dict[str, tuple[str, str | None]]:
    """``{menu name: (checkpoint, training recipe or None)}``; an entry is ``<checkpoint>`` or
    ``{checkpoint: <path>, recipe: <recipe>}`` (the recipe for a checkpoint with no run.json beside it)."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return {str(k): (str(v), None) if isinstance(v, str) else (str(v["checkpoint"]), v.get("recipe"))
            for k, v in data.items()}


def student_checkpoint(checkpoint: str) -> dict | None:
    """The student in ``checkpoint`` (with its cloud and DR, ``env``), or None for an rl_games policy."""
    from isaacsimenvs.scripts.evaluation.eval import _student_checkpoint

    return _student_checkpoint(checkpoint)


class Players:
    """The menu's players, each loaded on first use and kept."""

    def __init__(self, menu: dict[str, tuple[str, str | None]], inner, policy_config: str) -> None:
        self.menu, self.inner, self.policy_config = menu, inner, policy_config
        self._loaded: dict = {}

    def get(self, name: str):
        """The player of menu entry ``name``; ValueError if a student's cloud is not the env's."""
        if name not in self._loaded:
            inner, ckpt = self.inner, self.menu[name][0]
            student = student_checkpoint(ckpt)
            if student is not None:
                from isaacsimenvs.scripts.distillation.student_player import EnvCloudStudentPlayer
                from isaacsimenvs.scripts.evaluation.eval import _check_student_cloud

                if inner.observations.pointcloud is None:
                    raise ValueError(f"{name}: this env builds no point cloud (start the viewer with a student)")
                _check_student_cloud(inner.cfg, student["env"])
                self._loaded[name] = EnvCloudStudentPlayer(inner, student, inner.device)
            else:
                from isaacsimenvs.scripts.evaluation.player import PretrainedPlayer
                from isaacsimenvs.tasks.cloth.mdp.observations import obs_dim

                num_actions = int(inner.cfg.action_space)
                self._loaded[name] = PretrainedPlayer(
                    config_path=self.policy_config, checkpoint_path=ckpt, num_envs=inner.num_envs,
                    device=str(inner.device), num_observations=obs_dim(list(inner.cfg.obs.obs_list), num_actions),
                    num_actions=num_actions, expl_coef=0.0,
                )
        return self._loaded[name]
