import time, statistics, collections, sys
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import gymnasium as gym, torch
import isaacsimenvs  # noqa
T = sys.argv[1]
mod, cls = gym.spec(T).kwargs["env_cfg_entry_point"].split(":")
cfg = getattr(__import__(mod, fromlist=[cls]), cls)(); NE = int(sys.argv[2]) if len(sys.argv) > 2 else 12; cfg.scene.num_envs = NE; cfg.assets.num_assets_per_type = 1
if hasattr(cfg, "pointcloud"): cfg.pointcloud.extra_obs = ()
env = gym.make(T, cfg=cfg); e = env.unwrapped
TT = collections.Counter()
def wrap(owner, name, key):
    f = getattr(owner, name)
    def g(*a, **k):
        torch.cuda.synchronize(); t = time.perf_counter(); r = f(*a, **k); torch.cuda.synchronize(); TT[key] += time.perf_counter() - t; return r
    setattr(owner, name, g)
wrap(e.sim, "step", "physx step")
for nm in ("robot", "object", "goal_viz", "table"):
    o = getattr(e, nm, None)
    if o is not None:
        wrap(o, "write_data_to_sim", f"write {nm}"); wrap(o, "update", f"update {nm}")
for n in ("_pre_physics_step", "_apply_action", "_get_dones", "_get_rewards", "_get_observations", "_reset_idx"):
    wrap(e, n, n)
env.reset(); a = torch.zeros(NE, e.cfg.action_space, device=e.device)
for _ in range(40): env.step(a)
TT.clear(); N = 300
torch.cuda.synchronize(); t = time.perf_counter()
for _ in range(N): env.step(a)
torch.cuda.synchronize(); tot = (time.perf_counter() - t) / N * 1e3
print(f"SPLIT {T} x{NE} envs: step {tot:.2f} ms (with per-region syncs)")
for k, v in sorted(TT.items(), key=lambda kv: -kv[1]): print(f"SPLIT   {k:22s} {v / N * 1e3:6.2f} ms")
print(f"SPLIT   unaccounted            {tot - sum(TT.values()) / N * 1e3:6.2f} ms")
app.close()
