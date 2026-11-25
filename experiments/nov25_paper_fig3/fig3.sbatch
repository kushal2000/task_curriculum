#!/bin/bash
#SBATCH --job-name=fig3_sim
#SBATCH --partition=portal
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --chdir=/share/portal/kk837/task_curriculum
#SBATCH --output=debug_outputs/train_logs/nov/fig3/slurm_%j.out
# The paper's Fig. 3 sim images: VBD states (Newton) -> fitted proxies -> Isaac Sim renders.  sbatch fig3.sbatch [vbd frame] [--sweep]
set -euo pipefail
F="${1:-53}"; shift || true
D=experiments/nov25_paper_fig3; O=debug_outputs/train_logs/nov/fig3
export PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH="$PWD" TMPDIR="/tmp/${USER}_${SLURM_JOB_ID}"; mkdir -p "$TMPDIR" "$O"
[ -n "${SKIP_VBD:-}" ] || LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 .venv/isaaclab3/bin/python $D/vbd_states.py $O/states
.venv/isaaclab3/bin/python $D/fit_proxies.py $O/states "$F"
.venv/isaacsim/bin/python $D/render.py $O/states $O/${IMAGES:-images} "$F" "$@"
