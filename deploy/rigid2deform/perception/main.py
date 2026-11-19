"""rigid2deform's perception: the node config.yaml's `mode` needs (mode.py).

    ~/miniconda3/envs/foundationstereo/bin/python deploy/rigid2deform/perception/main.py [mode=pc_goal] [perception.pointcloud.depth=raw]
    ~/miniconda3/envs/foundationpose/bin/python deploy/rigid2deform/perception/main.py mode=foundationpose
    ~/miniconda3/envs/cotracker/bin/python deploy/rigid2deform/perception/main.py mode=cotracker object.name=[cloth] fold.sheet_size=0.15
    .venv_deploy/bin/python deploy/rigid2deform/perception/main.py perception.source=fake [mode=...]

| mode           | perception.source=camera                                      | fake (no camera, any env)  |
|----------------|---------------------------------------------------------------|----------------------------|
| pc_label       | cloud_camera.py: both D415s -> point_cloud; at its start the  | fake_cloud.py: a still     |
| pc_goal        |   flat cloth, one click (the reference | the goal); the        |   flat sheet's cloud       |
|                |   foundationstereo environment (its window)                   |                            |
| foundationpose | deploy/nodes/perception.py: one box to click per plate, the   | fake_sheet.py: a still     |
|                |   moving one first (the FoundationPose environment)           |   sheet                    |
| cotracker      | cloth_tracker.py: click the points in its window, then it     | fake_sheet.py              |
|                |   tracks without one (the cotracker environment)              |                            |

Give run.py the same mode (both read config.yaml; policy.py refuses a network of another kind). For
foundationpose / cotracker the config is checked first (../sheets/: two matching plates, ...), so a
wrong object.name stops here and not at the first click; object.name's objects must be tracked by the mode.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))    # repo root, for deploy.*
from deploy.rigid2deform.mode import kind_of, resolve                   # noqa: E402
from deploy.rigid2deform.sheets import sheet_perception, sheet_source   # noqa: E402
from deploy.utils.config import REPO, load_config                       # noqa: E402

CONFIG = HERE.parent / "config.yaml"


def main() -> None:
    if any(a.startswith("config=") for a in sys.argv[1:]):
        raise SystemExit(f"[perception] this script reads {CONFIG.relative_to(REPO)}; drop config=")
    cfg = load_config(path=CONFIG)
    mode = resolve(cfg)
    fake = cfg.perception.source == "fake"
    print(f"[perception] mode={mode}" + (" (fake)" if fake else ""), flush=True)
    if kind_of(mode) == "pointcloud":
        if fake:
            from deploy.rigid2deform.perception import fake_cloud
            fake_cloud.main()
        else:
            from deploy.rigid2deform.perception import cloud_camera
            cloud_camera.main()
        return
    sheet_source(cfg).check(cfg)
    if sheet_perception(cfg) != mode:
        raise SystemExit(f"[perception] mode={mode}, but object.name={list(cfg.object.name)} is tracked by "
                         f"{sheet_perception(cfg)}: give mode={sheet_perception(cfg)} or other objects")
    if fake:
        from deploy.rigid2deform.perception import fake_sheet
        fake_sheet.main()
    elif mode == "foundationpose":
        from deploy.nodes import perception
        sys.argv.insert(1, f"config={CONFIG.relative_to(REPO)}")   # the shared node reads its config from argv
        perception.main()
    else:
        from deploy.rigid2deform.perception import cloth_tracker
        cloth_tracker.main()

if __name__ == "__main__":
    main()
