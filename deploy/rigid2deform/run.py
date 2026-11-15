"""rigid2deform: the one entry point. The shared nodes (deploy/nodes/) + policy.py on config.yaml;
config.yaml's `task` picks what the policy does (fold, so far), `object.name` the sheet, whose
`perception` says which perception node perception.py starts (foundationpose: deploy/nodes/perception.py).
deploy/utils/runner.py has what starts when and how it all stops.

    PY=.venv_deploy/bin/python; R=deploy/rigid2deform/run.py; ARM="robot.arm=<ROBOT_SN> robot.hand=<HAND_SN>"
    $PY $R robot.mode=home       $ARM    # 1. move to home (or free_drive: move it there by hand)
    (start perception)                   # 2. deploy/rigid2deform/perception.py, the same object.name
                                         #    (perception.source=fake: no camera)
    $PY $R $ARM                          # 3. robot.mode=policy (the default): task=fold
        policy.checkpoint=deploy/rigid2deform/policy/box2RW_allDR.pth    another of the five runs
        record.name=fold_try1            record it (recordings/<date>/<date>_<time>_fold_try1.jsonl)
        run.viewer=false, ...            any other config key
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))                     # repo root, for deploy.*
from deploy.rigid2deform.policy import check                  # noqa: E402
from deploy.utils.runner import run                           # noqa: E402


if __name__ == "__main__":
    run(HERE / "config.yaml", HERE / "policy.py", check)
