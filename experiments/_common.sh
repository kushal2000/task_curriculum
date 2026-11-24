# Sourced by the scripts in experiments/ with the job's arguments: "source experiments/_common.sh "$@"".
# Scheduler-agnostic: no partitions, nodes or paths of a particular cluster. A batch system only has to
# run one of these scripts on a GPU machine (see experiments/README.md).
#
# Sets:
#   REPO        the repository root (the scripts cd there)
#   TASK        the --task in the arguments ("" if none)
#   BACKEND     physx (Isaac Sim 5.1, .venv_isaacsim) or newton (Isaac Lab 3, .venv_isaaclab3), from TASK:
#               Isaacsimenvs-Cloth* -> newton, every other id -> physx
#   PY          the interpreter for BACKEND (an array); HEADLESS (--headless for Kit, empty for Newton)
#   INDEX       this job's index in an array: $INDEX, else $SLURM_ARRAY_TASK_ID, else 0
#   JOB         a job id for names: $JOB_ID, else $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID, else local<pid>
#   LOG_DIR     where per-attempt logs and memory logs go (default slurm_logs, as the run record reads)
# Functions: provenance, mem_log_start / mem_log_stop, run_watched.

set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

TASK=""
_prev=""
for _a in "$@"; do
    [[ "$_prev" == "--task" ]] && TASK="$_a"
    [[ "$_a" == --task=* ]] && TASK="${_a#--task=}"
    _prev="$_a"
done
unset _a _prev

if [[ "$TASK" == Isaacsimenvs-Cloth* ]]; then
    BACKEND=newton
    PY=("$REPO/experiments/newton_py")
    HEADLESS=()
else
    BACKEND=physx
    PY=("$REPO/.venv_isaacsim/bin/python")
    HEADLESS=(--headless)
fi

INDEX="${INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
JOB="${JOB_ID:-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local$$}}}"
LOG_DIR="${LOG_DIR:-slurm_logs}"
mkdir -p "$LOG_DIR"
# Isaac Lab writes to $TMPDIR/isaaclab; a shared /tmp/isaaclab may belong to another user.
export TMPDIR="${TMPDIR_BASE:-/tmp}/${USER}_${JOB}_${INDEX}"
mkdir -p "$TMPDIR"
# Warp's kernel cache per job too: jobs that compile the same kernel into a shared ~/.cache/warp at
# the same time can read each other's half-written files (2026-11-24, visibility_warp: "No such file
# ... .meta").
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$TMPDIR/warp}"

# One line of where and what: host, commit, uncommitted changes, GPU.
provenance() {  # <log prefix>
    echo "[$1] host=$(hostname -s) job=$JOB index=$INDEX backend=$BACKEND commit=$(git rev-parse --short HEAD)" \
         "dirty=$(test -n "$(git status --porcelain --untracked-files=no)" && echo YES || echo no)"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
}

# GPU memory / utilisation and the user's host RSS every 30 s, into $LOG_DIR/<stem>.{gpumem,hostmem}.csv.
# Host RAM matters on PhysX: it pins host memory for its GPU buffers, and a failed pinned allocation shows
# up only as a bare "PxgCudaMemoryAllocator.cpp LINE 68" warning before the CUDA context dies.
mem_log_start() {  # <file stem>
    _MEM_STEM="$LOG_DIR/$1"
    nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 30 \
        >"$_MEM_STEM.gpumem.csv" 2>/dev/null &
    _SMI_PID=$!
    ( while sleep 30; do echo "$(date +%T),$(ps -u "$USER" -o rss= | awk '{s+=$1} END {printf "%d", s/1024}') MiB"; done ) \
        >"$_MEM_STEM.hostmem.csv" &
    _HOST_PID=$!
}
mem_log_stop() {  # prints peak_gpu_mib=.. peak_host_mib=..
    kill "${_SMI_PID:-}" "${_HOST_PID:-}" 2>/dev/null || true
    echo "peak_gpu_mib=$(cut -d, -f2 "$_MEM_STEM.gpumem.csv" 2>/dev/null | tr -dc '0-9\n' | sort -n | tail -1)" \
         "peak_host_mib=$(cut -d, -f2 "$_MEM_STEM.hostmem.csv" 2>/dev/null | tr -dc '0-9\n' | sort -n | tail -1)"
}

# run_watched <attempt log stem> <command...>: run the command in its own process group, tee its output
# to $LOG_DIR/<stem>.try<k>.log, and return its exit code.
#
# * PhysX's GPU pipeline sometimes dies at simulator start ("Fetching GPU Narrowphase failed! 700",
#   "PxgCudaMemoryAllocator" warnings) -- intermittently: the same config and machine also start fine --
#   and the process then HANGS instead of exiting. On PhysX the log is watched for those messages; the
#   attempt is killed and the command retried from scratch, up to PX_TRIES (default 3) attempts.
# * A TERM / INT to this script kills the whole process group. A batch system that tracks jobs by
#   process tree rather than a cgroup would otherwise leave the re-parented python (and its GPU)
#   running after the job is cancelled.
run_watched() {
    local stem=$1; shift
    local fatal='Fetching GPU Narrowphase failed|Cuda context manager error|PxgCudaMemoryAllocator'
    local tries=1 rc=1 try alog pid
    [ "$BACKEND" = physx ] && tries="${PX_TRIES:-3}"
    for try in $(seq 1 "$tries"); do
        alog="$LOG_DIR/${stem}.try${try}.log"
        setsid "$@" > >(tee "$alog") 2>&1 &
        pid=$!
        trap 'kill -9 -- "-'"$pid"'" 2>/dev/null; kill "${_SMI_PID:-}" "${_HOST_PID:-}" 2>/dev/null; exit 143' TERM INT
        rc=""
        while kill -0 "$pid" 2>/dev/null; do
            if [ "$BACKEND" = physx ] && grep -qE "$fatal" "$alog" 2>/dev/null; then
                echo "[watch] try $try: PhysX GPU failure ($(grep -m1 -oE "$fatal" "$alog")); killing and retrying"
                kill -9 -- "-$pid" 2>/dev/null
                rc=86
                break
            fi
            sleep 15 & wait $!
        done
        if [ -z "$rc" ]; then wait "$pid"; rc=$?; else wait "$pid" 2>/dev/null; fi
        [ "$rc" -ne 86 ] && break
    done
    trap - TERM INT
    return "$rc"
}
