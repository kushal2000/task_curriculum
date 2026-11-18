#!/bin/bash
# Hand self-penetration probe (scripts/analysis/finger_self_pen_probe.py) on the s12 VBD policies,
# self-collision off as trained, VBD ns2, pinned start. One array task per policy.
#   sbatch --array=0-3 scripts/cluster/bos14_finger_pen.sh
#SBATCH --job-name=finger_pen
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --exclude=bos14-node-[040,047,051,057,062,086,092,104,105,109,110,112,121,123]
#SBATCH --output=slurm_logs/%x_%A_%a.out

cd /home/yiboc/mit/task_curriculum_flexiv
export PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD
O=outputs/2026-11-11
BASE="env.cloth.start_height=0.20 env.robot_self_collision=false env.newton.njmax=32768 env.newton.num_substeps=2 env.cloth.substeps=6 env.cloth.nan_policy=reset"
ALLDR_PHYS="env.cloth.soft_contact_mu=0.265 env.assets.finger_tip_friction=0.5 env.assets.table_friction=1.0"
case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) NAME=vbdRW_sharpa_ep3300; R=FlexivWujiRightSharpaGain; EX=""
     CK=$O/08-33-16/0_pxvbd_vbdRW_sharpa_s1_278506/nn/snap_0_pxvbd_vbdRW_sharpa_s1_278506_ep_3300.pth ;;
  1) NAME=vbdRW_sharpaAB_ep2700; R=FlexivWujiRightSharpaGain; EX=""
     CK=$O/02-32-56/0_pxvbd_vbdRW_sharpaAB_s1_277598/nn/snap_0_pxvbd_vbdRW_sharpaAB_s1_277598_ep_2700.pth ;;
  2) NAME=vbdRW_s2rEff_ep3750; R=FlexivWujiRightS2REff; EX=""
     CK=$O/06-35-45/0_pxvbd_vbdRW_s2rEff_s1_278458/nn/last_0_pxvbd_vbdRW_s2rEff_s1_278458_ep_3750_rew__88.49689_.pth ;;
  3) NAME=vbdRW_allDR_ep3750; R=FlexivWujiRightReal; EX="$ALLDR_PHYS"
     CK=$O/02-33-13/0_pxvbd_vbdRW_allDR_s1_277599/nn/last_0_pxvbd_vbdRW_allDR_s1_277599_ep_3750_rew__92.731636_.pth ;;
esac
echo "[finger_pen] $NAME node $(hostname -s)"
scripts/newton_py scripts/analysis/finger_self_pen_probe.py \
    --task "Isaacsimenvs-Cloth${R}-Direct-v0" --checkpoint "$CK" \
    --num_envs "${ENVS:-32}" --steps "${STEPS:-600}" \
    --out "docs/results/finger_self_pen/${NAME}.json" \
    $BASE $EX ${EXTRA_ARGS:-}
echo "[finger_pen] exit=$?"
