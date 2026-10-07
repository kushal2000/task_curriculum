#!/bin/bash
# Profile PhysX fold-task scene construction (cProfile), to find what grows faster than the env count.
#
#   ENVS=4096 sbatch scripts/cluster/bos14_physx_build_profile.sh
#   less slurm_logs/build_prof_chain_4096.txt   (raw stats: .prof next to it)

#SBATCH --job-name=px_prof
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --exclude=bos14-node-057,bos14-node-062,bos14-node-092
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out

set -uo pipefail
cd /home/yiboc/mit/task_curriculum
export PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES
ENVS="${ENVS:-4096}"
KIND="${KIND:-chain}"
TASK=Isaacsimenvs-RigidClothPhysx-Direct-v0
[ "$KIND" = pool ] && TASK=Isaacsimenvs-FoldPoolPhysx-Direct-v0
.venv_isaacsim/bin/python scripts/analysis/physx_build_profile.py --headless --task "$TASK" \
    --num_envs "$ENVS" --out "slurm_logs/build_prof_${KIND}_${ENVS}${OUT_TAG:-}.txt" env.rigid_cloth.randomization.enabled=false \
    ${EXTRA_ARGS:-}
echo "[px_prof] exit=$?"
