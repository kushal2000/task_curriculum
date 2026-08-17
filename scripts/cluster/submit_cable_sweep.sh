#!/bin/bash
# Submit one SLURM job per cable-eval configuration.
#
#   scripts/cluster/submit_cable_sweep.sh
#
# Edit CONFIGS below. Each line is "<tag> <hydra overrides...>". Results land in
# docs/results/<tag>.json and full logs in slurm_logs/, warnings included -- the eval harness
# raises on a contact-buffer overflow, so a non-zero exit means the scene was dropping contacts.
#
# Keep the fleet around 8-10 jobs.

set -euo pipefail
cd /share/portal/kk837/task_curriculum

# Shared baseline: flat table, no supports/clearance anywhere.
BASE="env.cable.thickness=0.03"

# Damping 5.0 + 60 VBD iterations is the calm configuration: 5.40 m/s peak against 8.61 for the
# stock settings and 39.9 with a raised contact ramp, where a healthy cable sits near 4.81.
# Substeps are deliberately absent -- 8 substeps measured 15.5 m/s, three times worse.
CONFIGS=(
  "s_d5i60          env.cable.angular_damping=5.0  env.cable.vbd_iterations=60"
  "s_d5i120         env.cable.angular_damping=5.0  env.cable.vbd_iterations=120"
  "s_d10i60         env.cable.angular_damping=10.0 env.cable.vbd_iterations=60"
  "s_d10i120        env.cable.angular_damping=10.0 env.cable.vbd_iterations=120"
  "s_d20i120        env.cable.angular_damping=20.0 env.cable.vbd_iterations=120"
  "s_d5i60_lin1     env.cable.angular_damping=5.0  env.cable.vbd_iterations=60 env.cable.linear_damping=1.0"
  "s_d5i60_th40     env.cable.angular_damping=5.0  env.cable.vbd_iterations=60 env.cable.thickness=0.04"
  "s_d5i60_seg24    env.cable.angular_damping=5.0  env.cable.vbd_iterations=60 env.cable.segments=24"
  "s_d5i60_k1e3     env.cable.angular_damping=5.0  env.cable.vbd_iterations=60 env.cable.rigid_contact_k_start=1.0e3"
  "s_d5i60_mu40     env.cable.angular_damping=5.0  env.cable.vbd_iterations=60 env.cable.soft_contact_mu=40.0"
)

for line in "${CONFIGS[@]}"; do
  read -r tag rest <<< "$line"
  # shellcheck disable=SC2086
  jid=$(sbatch --job-name="ce_$tag" --parsable \
        scripts/cluster/sbatch_cable_eval.sh "$tag" $BASE $rest)
  echo "submitted $jid  $tag"
done

echo
echo "watch:   squeue -u \$USER"
echo "results: python3 scripts/cluster/collect_cable_results.py"
