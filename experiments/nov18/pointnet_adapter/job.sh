#!/bin/bash
#SBATCH --job-name=pn_adapter
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=portal
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=01:00:00
source experiments/_common.sh --task Isaacsimenvs-RigidBodyCloth-Direct-v0
adopt_log experiments/nov18/pointnet_adapter
"${PY[@]}" experiments/nov18/pointnet_adapter/collect_train.py "${HEADLESS[@]}" "$@"
