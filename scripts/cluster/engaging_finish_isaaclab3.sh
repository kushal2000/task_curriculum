#!/bin/bash
# Finish .venv_isaaclab3 on Engaging after `isaaclab.sh -i` stops at isaaclab_ov.
#   sbatch scripts/cluster/engaging_finish_isaaclab3.sh   (submit from the repo root)
#
# isaaclab_ov needs `ovstage`, which ships only manylinux_2_35 wheels; every Engaging node is
# Rocky 8 (glibc 2.28), so it can never install here. Nothing on the kit-less Newton path needs it
# (its one import, in newton/patches.py, is the OvPhysx backend). This installs the rest of what
# the eval / train code imports, with torch pinned to what isaaclab.sh put there, then checks that
# a policy checkpoint actually loads (CKPT / POLICY_CFG, optional).
#SBATCH --job-name=finish_il3
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/orcd/scratch/orcd/011/hglin/.uv_cache}"
export PATH="$HOME/.local/bin:$PATH"
PY=.venv_isaaclab3/bin/python
SRC=third_party/IsaacLab/source
echo "[finish_il3] node $(hostname -s)"

PIN="$(mktemp)"
"$PY" -c "
import importlib.metadata as md
for p in ('torch', 'torchvision', 'newton', 'warp-lang', 'newton-usd-schemas'):
    print(f'{p}=={md.version(p)}')" >"$PIN"
echo "==> keeping:"; cat "$PIN"

# isaaclab.sh installs the root pyproject's third-party list only after every submodule, so the
# isaaclab_ov failure skipped it (lazy_loader, gymnasium, usd-core, ...). The list is that one minus
# the workspace members and the two entries with glibc >= 2.35 wheels only (see the file).
echo "==> Isaac Lab root dependencies"
uv pip install --python "$PY" -c "$PIN" -r scripts/cluster/engaging_isaaclab3_deps.txt

echo "==> Isaac Lab parts after isaaclab_ov (skipped: no glibc 2.28 wheels)"
uv pip install --python "$PY" -c "$PIN" -e "$SRC/isaaclab_physx" -e "$SRC/isaaclab_tasks" -e "$SRC/isaaclab_rl"

echo "==> Repo packages"
uv pip install --python "$PY" -c "$PIN" -e ./rl_games
uv pip install --python "$PY" -c "$PIN" -e . gymnasium pytest

echo "==> Verifying"
"$PY" - <<'PY'
import torch
print("torch   :", torch.__version__)
for mod in ("isaaclab", "isaaclab_newton", "isaaclab_contrib", "isaaclab_tasks", "isaaclab_rl",
            "newton", "warp", "rl_games", "gymnasium", "isaacsimenvs"):
    m = __import__(mod)
    print(f"{mod:17}", getattr(m, "__version__", "ok"))
PY

if [[ -n "${CKPT:-}" ]]; then
  echo "==> Loading ${CKPT} with ${POLICY_CFG:-checkpoints/config.yaml} (CPU)"
  scripts/newton_py - "$CKPT" "${POLICY_CFG:-checkpoints/config.yaml}" <<'PY'
import sys, torch
from isaacsimenvs.eval.player import PretrainedPlayer
p = PretrainedPlayer(sys.argv[2], sys.argv[1], num_envs=2, device="cpu", expl_coef=0.0)
a = p.get_action(torch.zeros(2, 140))
print("checkpoint LOADED, action", tuple(a.shape), "finite:", bool(torch.isfinite(a).all()))
PY
fi
echo "[finish_il3] done"
