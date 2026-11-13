"""SimToolReal's perception: deploy/nodes/perception.py on config.yaml (its docstring has the details).

    ~/miniconda3/envs/foundationpose/bin/python deploy/simtoolreal/perception.py [overrides]
    .venv_deploy/bin/python deploy/simtoolreal/perception.py perception.source=fake
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))    # repo root, for deploy.*
from deploy.utils.config import REPO        # noqa: E402

CONFIG = HERE / "config.yaml"

if __name__ == "__main__":
    if any(a.startswith("config=") for a in sys.argv[1:]):
        raise SystemExit(f"[perception] this script reads {CONFIG.relative_to(REPO)}; drop config=")
    from deploy.nodes import perception
    sys.argv.insert(1, f"config={CONFIG.relative_to(REPO)}")   # the shared node reads its config from argv
    perception.main()
