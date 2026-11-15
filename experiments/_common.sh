# Sourced by the job scripts here. The working directory is the repo root (submit.sh --chdir).
#
# Picks the interpreter from the --task in the job's arguments, so the command names the backend:
#   Isaacsimenvs-Cloth*  -> Newton / VBD     .venv/isaaclab3
#   everything else      -> PhysX (Kit)      .venv/isaacsim
# and logs the job's provenance and its GPU memory next to the job log (debug_outputs/train_logs/.slurm/
# until the job moves both into its own folder: train.py does it for training, adopt_log for eval).

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
    PY=(.venv/isaaclab3/bin/python)
    HEADLESS=()
else
    BACKEND=physx
    PY=(.venv/isaacsim/bin/python)
    HEADLESS=(--headless)
fi

JOB_TAG="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}${SLURM_ARRAY_TASK_ID:+_${SLURM_ARRAY_TASK_ID}}"
LOG_PREFIX="debug_outputs/train_logs/.slurm/${SLURM_JOB_NAME:-job}_${JOB_TAG}"
mkdir -p debug_outputs/train_logs/.slurm

# Move this job's log from .slurm/ into <dir> as slurm_<job>.out, leaving a symlink (Slurm keeps
# writing to the moved file). The shell twin of isaacsimenvs/utils/paths.py:adopt_slurm_log.
adopt_log() {
    local dest="$1" src="${LOG_PREFIX}.out"
    mkdir -p "$dest"
    dest="$(cd "$dest" && pwd)"
    if [[ -f "$src" && ! -L "$src" ]]; then
        mv "$src" "$dest/slurm_${JOB_TAG}.out" && ln -s "$dest/slurm_${JOB_TAG}.out" "$src"
    fi
    LOG_PREFIX="$dest/gpumem_${JOB_TAG}"   # the GPU-memory log, started after this, goes there too
}

echo "[slurm] task=$TASK backend=$BACKEND python=${PY[*]}"
echo "[slurm] host=$(hostname -s) job=$JOB_TAG commit=$(git rev-parse --short HEAD) dirty=$(test -n "$(git status --porcelain)" && echo YES || echo no)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

gpu_log_start() {
    nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 30 \
        >"${LOG_PREFIX}.csv" 2>/dev/null &
    _SMI_PID=$!
}
gpu_log_stop() {
    kill "${_SMI_PID:-}" 2>/dev/null || true
    local peak
    peak=$(cut -d, -f2 "${LOG_PREFIX}.csv" 2>/dev/null | tr -dc '0-9\n' | sort -n | tail -1)
    echo "[slurm] peak_gpu_mib=${peak:-?}"
}
