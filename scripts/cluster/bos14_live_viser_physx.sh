#!/bin/bash
# Live, interactive policy viewer on PHYSX (isaacsimenvs/eval/live_viser.py, PhysX backend), port 8081.
#
#   sbatch scripts/cluster/bos14_live_viser_physx.sh
#   TASK=Isaacsimenvs-RigidClothPhysx-Direct-v0 EXTRA_ARGS="env.rigid_cloth.variant=box3-mid" sbatch scripts/cluster/bos14_live_viser_physx.sh
#   ENVS=20 POLICY="run 1: pool + folded starts (s1; cloth 56/50%)" \
#     EXTRA_ARGS="env.fold_pool.init_fold_prob=0.8 env.rigid_cloth.joint_damping=0.02" sbatch scripts/cluster/bos14_live_viser_physx.sh
#       (run 1's folded starts; 20 envs fit all 19 start folds of the pool, one per env)
#
# Then read slurm_logs/live_viser_px_<job>.out for the node, and from your laptop:
#   ssh -N -L 8081:<node>:8081 <you>@<bos14 login>      then open http://localhost:8081
#
# Runs in .venv_isaacsim (Isaac Sim 5.1). viser needs websockets >= 13 and Isaac Sim ships 12, so a
# copy of websockets 16 in .venv_isaacsim/viser_overlay goes first on PYTHONPATH for this process
# only (nothing Kit loads headless imports websockets). The Newton viewer (bos14_live_viser.sh, 8080)
# can run at the same time.

#SBATCH --job-name=live_viser_px
#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --exclude=bos14-node-[092,104,105]
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out

set -uo pipefail
cd /home/yiboc/mit/task_curriculum
mkdir -p slurm_logs
PORT="${PORT:-8081}"
export PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="$PWD/.venv_isaacsim/viser_overlay${PYTHONPATH:+:$PYTHONPATH}"
[ -d .venv_isaacsim/viser_overlay/websockets ] || { echo "[live_px] missing .venv_isaacsim/viser_overlay/websockets (copy websockets>=13 there)"; exit 2; }
echo "[live] node $(hostname -s) ($(hostname -I | awk '{print $1}')) port $PORT commit $(git rev-parse --short HEAD)"
echo "[live] tunnel: ssh -N -L ${PORT}:$(hostname -s):${PORT} $USER@<bos14 login>   then http://localhost:${PORT}"
read -r -a EXTRA <<<"${EXTRA_ARGS:-}"
exec .venv_isaacsim/bin/python -m isaacsimenvs.eval.live_viser \
    --task "${TASK:-Isaacsimenvs-FoldPoolPhysx-Direct-v0}" \
    --num_envs "${ENVS:-8}" \
    --port "$PORT" \
    --headless \
    ${POLICY:+--policy "$POLICY"} \
    "${EXTRA[@]}"
