# Installation

Two venvs on one source tree, both editable-installing it:

| venv | stack | runs | built by |
|---|---|---|---|
| `.venv/isaacsim` | Python 3.11, Isaac Sim 5.1 / Isaac Lab 2.3 | Play, rigid fold (PhysX) | `install_isaacsim.sh` |
| `.venv/isaaclab3` | Python 3.12, Isaac Lab 3.0 + Newton (kit-less) | VBD fold | `install_isaaclab3.sh` |

From the repo root:

```bash
docs/installation/install_isaacsim.sh
docs/installation/install_isaaclab3.sh     # Isaac Lab from the vendored third_party/IsaacLab
# The URDF -> USD cache the Newton envs read (.cache/usd/), baked with Isaac Sim:
OMNI_KIT_ACCEPT_EULA=YES .venv/isaacsim/bin/python -m isaacsimenvs.scripts.assets.prepare_usd_cache --populate \
    --num_assets_per_type 1 'env.assets.handle_head_types=[hammer]'
```

Newton needs every robot URDF it loads in `.cache/usd/`; for the right Wuji, run the populate step again
with `--task Isaacsimenvs-PlayFlexivWujiRightHumanLike-Direct-v0 env.assets.table_urdf=assets/urdf/table_narrow.urdf`.
Newton code runs on `.venv/isaaclab3/bin/python` directly: the venv is built on uv's own Python, not anaconda's,
whose old `libstdc++` once needed a launcher workaround.

The robot models (`assets/urdf/`) and third-party source (`third_party/`) are committed, so there is
nothing else to fetch. The Play priors (`pretrained_policy/`) are not: see the main README.

**Engaging cluster:** the host glibc is too old for Isaac Sim's wheels, so `engaging/` builds the same
venvs inside an Apptainer container (`.venv/isaacsim_ct`). Submit each
from the repo root, e.g. `experiments/submit.sh docs/installation/engaging/install_isaacsim.sh`.

## The Isaac Sim venv by hand

What `install_isaacsim.sh` does, step by step.

### Prerequisites

- Python 3.11
- NVIDIA GPU with driver >= 525.60
- CUDA 12+
- `uv` for package management ([install instructions](https://docs.astral.sh/uv/getting-started/installation/))

### Install

```bash
uv venv .venv/isaacsim --python 3.11

# PyTorch for CUDA 12.6 (Isaac Sim 5.x is built against this)
uv pip install --python .venv/isaacsim/bin/python torch --index-url https://download.pytorch.org/whl/cu126

# Vendored rl_games (PPO + SAPG)
uv pip install --python .venv/isaacsim/bin/python -e ./rl_games/

# Inference / tooling deps
uv pip install --python .venv/isaacsim/bin/python \
  omegaconf hydra-core "gym==0.23.1" scipy numpy yourdfpy trimesh viser \
  requests tqdm tyro huggingface_hub "imageio[ffmpeg]" wandb termcolor

# Isaac Lab + Isaac Sim (~15 GB download; first launch builds RTX shaders, ~2-5 min)
uv pip install --python .venv/isaacsim/bin/python \
  "isaaclab[isaacsim,all]==2.3.2.post1" --extra-index-url https://pypi.nvidia.com

# typing_extensions fix: install AFTER isaaclab so it wins the resolution
# (tyro CLIs need NoExtraItems from typing_extensions>=4.13; isaaclab pulls 4.12.2).
uv pip install --python .venv/isaacsim/bin/python "typing_extensions>=4.13"

# websockets for viser (the live viewer needs >=13.1; isaacsim-kernel pins 12.0, which nothing a
# training run loads uses). --no-deps so the resolver does not reject Isaac's pin.
uv pip install --python .venv/isaacsim/bin/python --no-deps "websockets==16.0"

# Register the repo-local package (isaacsimenvs)
uv pip install --python .venv/isaacsim/bin/python -e . --no-deps
```

`--no-deps` on the last line is required so the repo packages install cleanly without
re-resolving the heavy Isaac Sim stack.

Keep the version pins exact (`isaaclab==2.3.2.post1`, torch 2.7.x+cu126) — newer Isaac Lab
releases change `DirectRLEnv` / `UrdfConverter` APIs that `isaacsimenvs` depends on.

Verify:

```bash
.venv/isaacsim/bin/python -c "
import torch, isaaclab, isaacsimenvs
print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())
print('isaaclab:', isaaclab.__file__)
"
```

## Running

Non-interactive launches need the EULA env var:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
```

Optional speedup — point the Omniverse shader cache at a local SSD instead of NFS:

```bash
export OMNI_KIT_CACHE_PATH=/scratch/$USER/ov_cache
mkdir -p "$OMNI_KIT_CACHE_PATH"
```

## Gotchas

- **AppLauncher before isaaclab imports.** Isaac Lab sub-namespaces (`isaaclab.sim`,
  `isaaclab.envs`, ...) only resolve after `AppLauncher(args)` runs. Any script importing
  `isaaclab.*` must instantiate `AppLauncher` first.
- **Kit shutdown can hang** after the work is done. Scripts flush stdout/stderr and call
  `os._exit(0)` rather than waiting for a clean Kit teardown.
- **One Isaac Sim instance per GPU.** Booting a second Kit process while another is running
  on the same GPU can crash the booting one mid-startup. Wait for training/eval/demo
  processes to finish before launching another.

To prepare the default VBD robot and table, use the rigid cloth task with the
VBD table asset (the rigid task otherwise defaults to a different table):

```sh
OMNI_KIT_ACCEPT_EULA=YES .venv/isaacsim/bin/python -m isaacsimenvs.scripts.assets.prepare_usd_cache --populate \
    --task Isaacsimenvs-RigidBodyCloth-Direct-v0 --num_envs 1 \
    env.assets.table_urdf=assets/urdf/table_narrow.urdf
```
