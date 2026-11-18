"""Is the cloth state read right after an auto-reset the NEW episode's, or the old one's (stale buffers)?
For envs that end: sheet centroid before the step (old episode), in the observation the reset step
returns, and one step later. PhysX: .venv_isaacsim/bin/python scripts/distill/probe_reset_state.py --headless ..."""
import argparse, sys
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--task", required=True); p.add_argument("--experiment", default=None); p.add_argument("--teacher", required=True)
AppLauncher.add_app_launcher_args(p)
args, hydra_args = p.parse_known_args(); sys.argv = [sys.argv[0]] + hydra_args
try:
    app = AppLauncher(args).app
except ImportError:
    app = None
import gymnasium as gym, torch, os
import isaacsimenvs  # noqa
from isaacsimenvs.scripts.distillation.student import TeacherNet
from isaacsimenvs.scripts.evaluation.protocol import use_single_object_variant
from isaacsimenvs.scripts.configs.loader import hydra_task_config_with_yaml

@hydra_task_config_with_yaml(args.task, "rl_games_sapg_cfg_entry_point", experiment=args.experiment)
def run(env_cfg, _a):
    env_cfg.scene.num_envs = 128; env_cfg.termination.eval_success_tolerance = 0.01
    use_single_object_variant()
    env = gym.make(args.task, cfg=env_cfg); inner = env.unwrapped; n, dev = inner.num_envs, inner.device
    ck = torch.load(args.teacher, map_location="cpu", weights_only=False); ck = ck[0] if 0 in ck else ck
    teacher = TeacherNet(ck["model"]).to(dev)
    obs, _ = env.reset(); state = teacher.initial_state(n, dev); start = torch.ones(n, dtype=torch.bool, device=dev)
    def cen():
        return (inner._particles_w() - inner.scene.env_origins.unsqueeze(1)).mean(dim=1).clone()
    def kp():
        return obs["policy"][:, 107:119].view(n, 4, 3).mean(dim=1).clone()
    pending, shown = None, 0
    for step in range(700):
        before, kp_before = cen(), kp()
        act, state = teacher.step(obs["policy"], state, start)
        obs, _r, term, trunc, _e = env.step(act)
        after, kp_after = cen(), kp()
        if pending is not None:
            ids, c_old, c_reset, k_old, k_reset = pending
            for i in ids[:3].tolist():
                print(f"[probe] env {i}: old-final {[round(float(v),3) for v in c_old[i]]} at-reset-obs {[round(float(v),3) for v in c_reset[i]]} "
                      f"one-step-later {[round(float(v),3) for v in after[i]]} | kp_rel_palm mean old {[round(float(v),3) for v in k_old[i]]} "
                      f"reset {[round(float(v),3) for v in k_reset[i]]} later {[round(float(v),3) for v in kp_after[i]]} | ep_len at reset obs {int(len_reset[i])}", flush=True)
            shown += len(ids[:3]); pending = None
        ended = term.bool() | trunc.bool()
        if bool(ended.any()) and shown < 12:
            pending = (ended.nonzero(as_tuple=True)[0], before, after, kp_before, kp_after); len_reset = inner.episode_length_buf.clone()
        start = ended
        if shown >= 12: break
    env.close()
run()
sys.stdout.flush(); os._exit(0)
