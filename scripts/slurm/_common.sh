# Sourced by the job scripts here. The working directory is the repo root (submit.sh --chdir).
#
# Picks the interpreter from the --task in the job's arguments, so the command names the backend:
#   Isaacsimenvs-Cloth*  -> Newton / VBD     scripts/newton_py (.venv_isaaclab3)
#   everything else      -> PhysX (Kit)      .venv_isaacsim, or scripts/isaacsim_py (the Isaac Sim
#                                            container) where .venv_isaacsim_ct exists (Engaging)
# and logs the job's provenance and its GPU memory to outputs/slurm/ next to the job log.

set -uo pipefail
export PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES

TASK=""
_prev=""
for _a in "$@"; do
    [[ "$_prev" == "--task" ]] && TASK="$_a"
    [[ "$_a" == --task=* ]] && TASK="${_a#--task=}"
    _prev="$_a"
done
[[ -n "$TASK" ]] || { echo "[slurm] pass --task <gym id>" >&2; exit 2; }

if [[ "$TASK" == Isaacsimenvs-Cloth* ]]; then
    BACKEND=newton
    PY=(scripts/newton_py)
    HEADLESS=()
else
    BACKEND=physx
    if [[ -d .venv_isaacsim_ct ]]; then PY=(scripts/isaacsim_py); else PY=(.venv_isaacsim/bin/python); fi
    HEADLESS=(--headless)
fi

JOB_TAG="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}${SLURM_ARRAY_TASK_ID:+_${SLURM_ARRAY_TASK_ID}}"
LOG_PREFIX="outputs/slurm/${SLURM_JOB_NAME:-job}_${JOB_TAG}"
mkdir -p outputs/slurm

echo "[slurm] task=$TASK backend=$BACKEND python=${PY[*]}"
echo "[slurm] host=$(hostname -s) job=$JOB_TAG commit=$(git rev-parse --short HEAD) dirty=$(test -n "$(git status --porcelain)" && echo YES || echo no)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

gpu_log_start() {
    nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 30 \
        >"${LOG_PREFIX}.gpumem.csv" 2>/dev/null &
    _SMI_PID=$!
}
gpu_log_stop() {
    kill "${_SMI_PID:-}" 2>/dev/null || true
    local peak
    peak=$(cut -d, -f2 "${LOG_PREFIX}.gpumem.csv" 2>/dev/null | tr -dc '0-9\n' | sort -n | tail -1)
    echo "[slurm] peak_gpu_mib=${peak:-?}"
}
