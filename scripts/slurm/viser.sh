#!/bin/bash
# Live policy viewer in the browser: scripts/slurm/submit.sh [sbatch options] scripts/slurm/viser.sh \
#     --task <id> [--port 8080] [live_viser.py args]
#
# The job log prints the ssh tunnel. viser needs websockets >= 13.1; docs/installation/install_isaacsim.sh
# installs it into .venv/isaacsim over Isaac Sim's 12.0 pin.
#
#SBATCH --job-name=viser
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=portal
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
source scripts/slurm/_common.sh

PORT=8080
_prev=""
for _a in "$@"; do [[ "$_prev" == "--port" ]] && PORT="$_a"; _prev="$_a"; done
echo "[slurm] tunnel: ssh -N -L ${PORT}:$(hostname -s):${PORT} $USER@<login node>   then http://localhost:${PORT}"
"${PY[@]}" -m isaacsimenvs.eval.live_viser "$@"
