"""SimToolReal Play: the one entry point. The shared nodes (deploy/nodes/) + policy.py on config.yaml;
deploy/utils/runner.py has what starts when and how it all stops.

    PY=.venv_deploy/bin/python; R=deploy/simtoolreal/run.py; ARM="robot.arm=<ROBOT_SN> robot.hand=<HAND_SN>"
    $PY $R robot.mode=home       $ARM    # 1. move to home (or free_drive: move it there by hand)
    (start perception)                   # 2. deploy/simtoolreal/perception.py (FoundationPose env)
    $PY $R $ARM                          # 3. robot.mode=policy (the default): the policy run
        record.name=eraser_try1          record it (recordings/<date>/<date>_<time>_eraser_try1.jsonl)
        record.camera=true               ... with the camera video (<name>.camera.mp4)
        run.viewer=false                 no viewer
        run.duration_s=60, ...           any other config key
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))                     # repo root, for deploy.*
from deploy.utils.config import repo_path                     # noqa: E402
from deploy.utils.runner import run                           # noqa: E402

# config.yaml's policy defaults are the right hand's policy. Both hands take the same 134 obs and 27
# actions, so nothing downstream would notice a right policy driving the left hand.
RIGHT_DEFAULTS = {"checkpoint": "pretrained_policy/nov6_dr.pth", "config": "pretrained_policy/config.yaml"}


def check_policy_side(cfg) -> None:
    """Refuse a missing checkpoint, or a policy trained for the other hand than robot.side."""
    side, pol = cfg.robot.side, cfg.policy
    if cfg.robot.mode != "policy":          # only policy mode loads it (runner.py)
        return
    # rl_games' restore() skips a missing file silently: the run would go on with random weights
    for k in ("checkpoint", "config"):
        if not repo_path(pol[k]).is_file():
            raise SystemExit(f"[run] policy.{k}={pol[k]}: no such file ({repo_path(pol[k])})")
    if side != "right":
        kept = [k for k, v in RIGHT_DEFAULTS.items() if str(pol[k]) == v]
        if kept:
            raise SystemExit(f"[run] robot.side={side} with the right hand's default "
                             f"{' and '.join(f'policy.{k}={RIGHT_DEFAULTS[k]}' for k in kept)}: "
                             f"give the {side} policy's policy.checkpoint=... policy.config=...")
    # rl_games' experiment name, e.g. 0_play_flexiv_left_dr: if it names a side, it must be robot.side
    from omegaconf import OmegaConf
    path = repo_path(pol.config)
    name = str(OmegaConf.select(OmegaConf.load(path), "train.params.config.full_experiment_name") or "") \
        if path.is_file() else ""
    other = "left" if side == "right" else "right"
    if other in name.lower().split("_"):
        raise SystemExit(f"[run] robot.side={side} but policy.config {pol.config} is from run {name!r} "
                         f"(the {other} hand)")


if __name__ == "__main__":
    run(HERE / "config.yaml", HERE / "policy.py", check=check_policy_side)
