#!/bin/bash
#SBATCH --job-name=pc_check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=portal
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=00:30:00
source experiments/_common.sh --task Isaacsimenvs-RigidBodyCloth-Direct-v0
adopt_log experiments/nov18/point_cloud
"${PY[@]}" experiments/nov18/point_cloud/check.py "${HEADLESS[@]}" "$@"
