# Installation

Two venvs on one source tree, both editable-installing it:

| venv | stack | runs | built by |
|---|---|---|---|
| `.venv_isaacsim` | Python 3.11, Isaac Sim 5.1 / Isaac Lab 2.3 | Play, rigid fold pool (PhysX) | `install_isaacsim.sh` |
| `.venv_isaaclab3` | Python 3.12, Isaac Lab 3.0 + Newton (kit-less) | VBD cloth | `install_isaaclab3.sh` |

From the repo root:

```bash
docs/installation/install_isaacsim.sh
docs/installation/install_isaaclab3.sh        # Isaac Lab 3.0 from source into third_party/ (pinned SHA)
isaacsimenvs/scripts/robots/fetch_assets.sh   # Rizon 4s + Wuji Hand 2 vendor meshes (assets/flexiv_wuji/)
```

Run Newton code through `experiments/newton_py`, not `.venv_isaaclab3/bin/python`: the venv was built
on an anaconda interpreter whose old `libstdc++` lacks a symbol `omni.client` needs, and the wrapper
preloads the system one (see its header).

Newton cannot convert URDFs (no Kit), so it reads robots from a USD cache (`.usd_cache/`) baked with
Isaac Sim, once per URDF and self-collision value:

```bash
SC=false experiments/bake_robot.sh
```

The live viewer on PhysX needs websockets >= 13, which Isaac Sim pins to 12; `install_isaacsim.sh`
installs websockets 16 into `third_party/viser_websockets`, outside the venv, and the viewers put it
first on the path.

The policies (`pretrained_policy/`) are not in git: the Play prior goes in
`pretrained_policy/flexiv_right_wuji/nov8_dr_friction_0p5.pth`, the network config in
`pretrained_policy/config.yaml` (`PRETRAINED_POLICY_DIR` moves the folder).

## The Isaac Sim venv by hand

What `install_isaacsim.sh` does, step by step.

### Prerequisites

- Python 3.11
- NVIDIA GPU with driver >= 525.60
- CUDA 12+
- `uv` for package management ([install instructions](https://docs.astral.sh/uv/getting-started/installation/))

### Install

```bash
uv venv .venv_isaacsim --python 3.11

# PyTorch for CUDA 12.6 (Isaac Sim 5.x is built against this)
uv pip install --python .venv_isaacsim/bin/python torch --index-url https://download.pytorch.org/whl/cu126

# Vendored rl_games (PPO + SAPG)
uv pip install --python .venv_isaacsim/bin/python -e ./rl_games/

# Inference / tooling deps
uv pip install --python .venv_isaacsim/bin/python \
  omegaconf hydra-core "gym==0.23.1" scipy numpy yourdfpy trimesh viser \
  requests tqdm tyro huggingface_hub "imageio[ffmpeg]" wandb termcolor

# Isaac Lab + Isaac Sim (~15 GB download; first launch builds RTX shaders, ~2-5 min)
uv pip install --python .venv_isaacsim/bin/python \
  "isaaclab[isaacsim,all]==2.3.2.post1" --extra-index-url https://pypi.nvidia.com

# typing_extensions fix: install AFTER isaaclab so it wins the resolution
# (tyro CLIs need NoExtraItems from typing_extensions>=4.13; isaaclab pulls 4.12.2).
uv pip install --python .venv_isaacsim/bin/python "typing_extensions>=4.13"

# Register the repo-local package (isaacsimenvs)
uv pip install --python .venv_isaacsim/bin/python -e . --no-deps
```

`--no-deps` on the last line is required so the repo packages install cleanly without
re-resolving the heavy Isaac Sim stack.

Keep the version pins exact (`isaaclab==2.3.2.post1`, torch 2.7.x+cu126) — newer Isaac Lab
releases change `DirectRLEnv` / `UrdfConverter` APIs that `isaacsimenvs` depends on.

Verify:

```bash
.venv_isaacsim/bin/python -c "
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
