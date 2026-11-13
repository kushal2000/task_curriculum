"""SimToolReal Play: the one entry point. The shared nodes (deploy/nodes/) + policy.py on config.yaml;
deploy/utils/runner.py has what starts when and how it all stops.

    PY=.venv_deploy/bin/python; R=deploy/simtoolreal/run.py; ARM="robot.arm=<ROBOT_SN> robot.hand=<HAND_SN>"
    $PY $R robot.mode=home       $ARM    # 1. move to home (or free_drive: move it there by hand)
    (start perception)                   # 2. deploy/simtoolreal/perception.py (FoundationPose env)
    $PY $R $ARM                          # 3. robot.mode=policy (the default): the policy run
        record.name=eraser_try1          record it (recordings/<date>_<time>_eraser_try1.jsonl)
        record.camera=true               ... with the camera video (<name>.camera.mp4)
        run.viewer=false                 no viewer
        run.duration_s=60, ...           any other config key
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))                     # repo root, for deploy.*
from deploy.utils.runner import run                           # noqa: E402

if __name__ == "__main__":
    run(HERE / "config.yaml", HERE / "policy.py")
