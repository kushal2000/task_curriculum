"""config.yaml's `mode`: which perception runs and, unless policy.checkpoint is given, which network.

| mode           | perception (perception/main.py)                 | network (policy.checkpoint null)       |
|----------------|-------------------------------------------------|----------------------------------------|
| pc_label       | cloud_camera.py, the reference (label)          | policy.default_checkpoint.pc_label     |
| pc_goal        | cloud_camera.py, the goal                       | policy.default_checkpoint.pc_goal      |
| foundationpose | deploy/nodes/perception.py, object.name plates  | policy.default_checkpoint.foundationpose |
| cotracker      | cloth_tracker.py, object.name=[cloth]           | policy.default_checkpoint.cotracker    |

perception.source=fake keeps the mode and sends a still sheet instead (fake_cloud.py / fake_sheet.py).
Every rigid2deform entry point calls resolve(cfg) right after loading the config; policy.py's check
refuses a checkpoint of another kind (or point-cloud mode) than the mode says.
"""

from __future__ import annotations

from omegaconf import DictConfig

#: mode -> (network kind (deploy/utils/policy_net.py), point-cloud mode (cloud_obs.mode_of) or None)
MODES = {"pc_label": ("pointcloud", "label"), "pc_goal": ("pointcloud", "goal"),
         "foundationpose": ("keypoints", None), "cotracker": ("keypoints", None)}


def resolve(cfg: DictConfig) -> str:
    """cfg.mode, checked; a null policy.checkpoint becomes policy.default_checkpoint[mode] (in place)."""
    mode = str(cfg.mode)
    if mode not in MODES:
        raise SystemExit(f"[rigid2deform] mode={mode}: one of {' | '.join(MODES)}")
    if cfg.policy.checkpoint is None:
        cfg.policy.checkpoint = cfg.policy.default_checkpoint[mode]
    return mode


def kind_of(mode: str) -> str:
    """'pointcloud' | 'keypoints'."""
    return MODES[mode][0]


def cloud_mode(mode: str) -> str | None:
    """'label' | 'goal' for a point-cloud mode, else None."""
    return MODES[mode][1]
