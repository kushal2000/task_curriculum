"""rigid2deform's perception: the node object.name's perception needs, on config.yaml.

    ~/miniconda3/envs/foundationpose/bin/python deploy/rigid2deform/perception/main.py [overrides]
    ~/miniconda3/envs/cotracker/bin/python deploy/rigid2deform/perception/main.py object.name=[cloth] fold.sheet_size=0.15
    .venv_deploy/bin/python deploy/rigid2deform/perception/main.py perception.source=fake [fold.fake_fold_deg=90]
    ~/miniconda3/envs/foundationstereo/bin/python deploy/rigid2deform/perception/main.py \
        policy.checkpoint=deploy/rigid2deform/policy/pc/<name>.pth [perception.pointcloud.depth=raw]

| perception.source | object.name's perception | runs                                                        |
|-------------------|--------------------------|-------------------------------------------------------------|
| camera            | foundationpose (plates)  | deploy/nodes/perception.py: one box to click per plate, the |
|                   |                          | moving one first (needs the FoundationPose environment)     |
| camera            | cotracker (cloth)        | cloth_tracker.py: click the points in its window, then it   |
|                   |                          | tracks without one (needs the cotracker environment)        |
| fake              | any implemented          | fake_sheet.py: a still sheet, no camera (any env)           |

That table is for a keypoint policy. The network decides the perception: a point-cloud policy
(policy.checkpoint, deploy/utils/policy_net.py checkpoint_kind) needs the two-camera cloud: cloud_camera.py
(perception.pointcloud; object.name is not read; needs the foundationstereo environment: its window to
click the cloth once), or with perception.source=fake fake_cloud.py (a still, flat sheet). So this script reads policy.checkpoint
too: give it the same one as run.py (both read config.yaml).

For a keypoint policy the config is checked first (../sheets/: two matching plates, ...), so a wrong
object.name stops here and not at the first click. Give run.py the same object.name.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))    # repo root, for deploy.*
from deploy.rigid2deform.sheets import sheet_perception, sheet_source   # noqa: E402
from deploy.utils.config import REPO, load_config, repo_path            # noqa: E402
from deploy.utils.policy_net import checkpoint_kind                      # noqa: E402

CONFIG = HERE.parent / "config.yaml"


def main() -> None:
    if any(a.startswith("config=") for a in sys.argv[1:]):
        raise SystemExit(f"[perception] this script reads {CONFIG.relative_to(REPO)}; drop config=")
    cfg = load_config(path=CONFIG)
    if checkpoint_kind(repo_path(cfg.policy.checkpoint)) == "pointcloud":
        if cfg.perception.source == "fake":
            from deploy.rigid2deform.perception import fake_cloud
            fake_cloud.main()
        else:
            from deploy.rigid2deform.perception import cloud_camera
            cloud_camera.main()
        return
    sheet_source(cfg).check(cfg)
    if cfg.perception.source == "fake":
        from deploy.rigid2deform.perception import fake_sheet
        fake_sheet.main()
    elif sheet_perception(cfg) == "foundationpose":
        from deploy.nodes import perception
        sys.argv.insert(1, f"config={CONFIG.relative_to(REPO)}")   # the shared node reads its config from argv
        perception.main()
    elif sheet_perception(cfg) == "cotracker":
        from deploy.rigid2deform.perception import cloth_tracker
        cloth_tracker.main()
    else:
        raise SystemExit(f"[perception] perception={sheet_perception(cfg)}: no camera node yet")


if __name__ == "__main__":
    main()
