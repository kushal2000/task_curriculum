#!/bin/bash
# Evaluate the snapshots of a point-cloud RL run (bos14_pc_rl_train.sh) while it trains.
#   nohup scripts/cluster/pc_rl_trend_driver.sh <RUN_TAG> <job id> [EVERY=300] [SEEDS="1 2 3"] &
# Every snapshot nn/snap_*_ep_<N>.pth with N a multiple of EVERY (and the last one) gets two jobs on the
# RTX 5090 partition: without noise (@clean) and with the full cloud noise + delays (@noise); each is PhysX
# pinned + rand and VBD ns6 pinned + rand x SEEDS. Results: docs/results/distill/rl_<RUN_TAG>_ep<N>@<cond>__*.json
# Stops when the training job has left the queue and every snapshot has been submitted.
set -uo pipefail
cd /home/yiboc/mit/task_curriculum_flexiv
tag=$1 job=$2 every=${3:-300} seeds=${4:-"1 2 3"}
DL="--pc_delay 6 --proprio_delay 3 --action_delay 3"
log="slurm_logs/distill/trend_${tag}_${job}.log"
done_list="slurm_logs/distill/.trend_${tag}_${job}.done"
touch "$done_list"
submit() {  # snapshot epoch
    local snap=$1 ep=$2 cond eargs e
    for cond in clean noise; do
        [ "$cond" = clean ] && eargs="--eval_clean" || eargs="--eval_noise $DL"
        e=$(MODE=evalx STUDENT="$snap" TAG="rl_${tag}_ep${ep}@${cond}" SIMS="physx vbd" SEEDS="$seeds" STARTS="pinned rand" \
            PX_STARTS="pinned rand" EVAL_ARGS="$eargs" \
            sbatch --parsable --time=04:00:00 --partition=gpu_5090 --gres=gpu:rtx5090:1 --mem=60G \
            '--exclude=bos14-node-[105,110,121]' --job-name="rle_${tag}_${ep}_${cond}" scripts/cluster/bos14_distill_pc.sh)
        echo "$(date +%m-%d_%H:%M) $tag ep$ep $cond eval=$e $snap" >>"$log"
    done
    echo "$ep" >>"$done_list"
}
while true; do
    alive=$(squeue -h -j "$job" 2>/dev/null | wc -l)
    last=0
    for snap in $(ls outputs/*/*/0_pcrl_${tag}_s1_${job}/nn/snap_*_ep_*.pth 2>/dev/null); do
        ep=${snap##*_ep_}; ep=${ep%.pth}
        [ "$ep" -gt "$last" ] && last=$ep
        [ $(( ep % every )) -eq 0 ] || continue
        grep -qx "$ep" "$done_list" || submit "$snap" "$ep"
    done
    if [ "$alive" -eq 0 ]; then
        # the final snapshot, whatever its epoch
        if [ "$last" -gt 0 ] && ! grep -qx "$last" "$done_list"; then
            submit "$(ls outputs/*/*/0_pcrl_${tag}_s1_${job}/nn/snap_*_ep_${last}.pth | head -1)" "$last"
        fi
        echo "$(date +%m-%d_%H:%M) $tag: training job $job gone, driver done" >>"$log"
        break
    fi
    sleep 300
done
