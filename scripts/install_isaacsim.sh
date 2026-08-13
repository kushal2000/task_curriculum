#!/bin/bash
# Create .venv_isaacsim exactly as docs/isaacsim_installation.md prescribes.
# Order matters: torch before isaaclab, typing_extensions after it, repo last with
# --no-deps so it does not re-resolve the Isaac stack.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
VENV=".venv_isaacsim"
PY="$VENV/bin/python"

step () { echo; echo "=== $* ==="; date '+%H:%M:%S'; }

step "uv venv (python 3.11)"
uv venv "$VENV" --python 3.11

step "torch (cu126)"
uv pip install --python "$PY" torch --index-url https://download.pytorch.org/whl/cu126

step "vendored rl_games (editable)"
uv pip install --python "$PY" -e ./rl_games/

step "tooling deps"
uv pip install --python "$PY" \
  omegaconf hydra-core "gym==0.23.1" scipy numpy yourdfpy trimesh viser \
  requests tqdm tyro huggingface_hub "imageio[ffmpeg]" wandb termcolor \
  tensorboard pytest pyyaml

step "isaaclab + isaacsim (~15 GB)"
uv pip install --python "$PY" \
  "isaaclab[isaacsim,all]==2.3.2.post1" --extra-index-url https://pypi.nvidia.com

step "typing_extensions re-pin (must come after isaaclab)"
uv pip install --python "$PY" "typing_extensions>=4.13"

step "repo package (no deps)"
uv pip install --python "$PY" -e . --no-deps

step "verify"
"$PY" -c "
import torch, isaaclab
print('torch:', torch.__version__, 'cuda avail:', torch.cuda.is_available())
print('isaaclab:', isaaclab.__file__)
"
echo
echo "INSTALL COMPLETE"
date '+%H:%M:%S'
