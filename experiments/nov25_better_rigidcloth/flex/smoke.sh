#!/bin/bash
# smoke.sh NAME NUM_ENVS STEPS [extra overrides / KEY=VAL ...]
cd /share/portal/kk837/task_curriculum_nov25
N=$1; E=$2; S=$3; shift 3; OVR=(); for t in "$@"; do if [[ "$t" =~ ^[A-Z_][A-Z0-9_]*= ]]; then export "$t"; else OVR+=("$t"); fi; done
export PYGLET_HEADLESS=1 PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD:$PWD/rl_games LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 WARP_CACHE_PATH=${WARP_CACHE_PATH:-$PWD/.cache/warp_vbd} VBD_FLEX=${VBD_FLEX:-1} NEWTON_PUSH_APPROXIMATION=${NEWTON_PUSH_APPROXIMATION:-1}
.venv/isaaclab3/bin/python -m isaacsimenvs.scripts.evaluation.eval --task Isaacsimenvs-Cloth-Direct-v0 --checkpoint ${CKPT:-pretrained_policy/play_pc_rigidclothRL_64live.pth} --policy_config pretrained_policy/config.yaml --sapg_expl_coef 0 --max_steps $S --success_tolerance 0.01 --num_assets_per_type 1 --single_variant --randomize --protocol isaacsimenvs/tasks/cloth/config/eval/vbd.yaml --num_envs $E --seed ${SEED:-1} --out outputs/flex/$N.json env.hand_sysid=false env.cloth.thickness=0.016 env.newton.use_mujoco_contacts=true env.newton.solver=cg "${OVR[@]}" > outputs/flex/$N.log 2>&1
echo "$N rc=$?"; grep -E "\[flex\]|Traceback|Error" outputs/flex/$N.log | head -5
python3 -c "
import json;d=json.load(open('outputs/flex/$N.json'));print({k:v for k,v in d['termination_reasons'].items() if v}, 'len', d['per_env_length'], 'fold_err', [round(x,3) for x in d['fold']['best_fold_err']])"
