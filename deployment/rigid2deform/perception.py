"""rigid2deform's perception: the node object.name's perception needs, on config.yaml.

    ~/miniconda3/envs/foundationpose/bin/python deployment/rigid2deform/perception.py [overrides]
    ~/miniconda3/envs/cotracker/bin/python deployment/rigid2deform/perception.py object.name=[cloth] fold.sheet_size=0.15
    .venv_deploy/bin/python deployment/rigid2deform/perception.py perception.source=fake [fold.fake_fold_deg=90]

| perception.source | object.name's perception | runs                                                        |
|-------------------|--------------------------|-------------------------------------------------------------|
| camera            | foundationpose (plates)  | deployment/nodes/perception.py: one box to click per plate, the |
|                   |                          | moving one first (needs the FoundationPose environment)     |
| camera            | cotracker (cloth)        | cloth_tracker.py: click the points in its window, then it   |
|                   |                          | tracks without one (needs the cotracker environment)        |
| fake              | any implemented          | fake_perception.py: a still sheet, no camera (any env)      |

The config is checked first (sheets/: two matching plates, ...), so a wrong object.name stops here and
not at the first click. Give run.py the same object.name.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))    # repo root, for deploy.*
from deployment.rigid2deform.sheets import sheet_perception, sheet_source   # noqa: E402
from deployment.utils.config import REPO, load_config                       # noqa: E402

CONFIG = HERE / "config.yaml"


def main() -> None:
    if any(a.startswith("config=") for a in sys.argv[1:]):
        raise SystemExit(f"[perception] this script reads {CONFIG.relative_to(REPO)}; drop config=")
    cfg = load_config(path=CONFIG)
    sheet_source(cfg).check(cfg)
    if cfg.perception.source == "fake":
        from deployment.rigid2deform import fake_perception
        fake_perception.main()
    elif sheet_perception(cfg) == "foundationpose":
        from deployment.nodes import perception
        sys.argv.insert(1, f"config={CONFIG.relative_to(REPO)}")   # the shared node reads its config from argv
        perception.main()
    elif sheet_perception(cfg) == "cotracker":
        from deployment.rigid2deform import cloth_tracker
        cloth_tracker.main()
    else:
        raise SystemExit(f"[perception] perception={sheet_perception(cfg)}: no camera node yet")


if __name__ == "__main__":
    main()
