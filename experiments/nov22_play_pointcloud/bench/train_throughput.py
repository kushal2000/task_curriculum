"""Env steps per second of the point-cloud Play task at training scale (all observation groups on),
random actions: ``python train_throughput.py <num envs> <steps>``, run from the repo (or worktree) root."""
import sys, time, statistics
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import gymnasium as gym, torch
import isaacsimenvs  # noqa
n, steps = int(sys.argv[1]), int(sys.argv[2])
T = "Isaacsimenvs-PlayFlexivWujiRightHumanLikePC-Direct-v0"
mod, cls = gym.spec(T).kwargs["env_cfg_entry_point"].split(":")
cfg = getattr(__import__(mod, fromlist=[cls]), cls)(); cfg.scene.num_envs = n
env = gym.make(T, cfg=cfg); e = env.unwrapped
env.reset()
g = torch.Generator(device=e.device).manual_seed(0)
act = lambda: torch.rand(n, e.cfg.action_space, device=e.device, generator=g) * 2 - 1
for _ in range(30): env.step(act())
blocks = []
for _ in range(5):
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(steps): env.step(act())
    torch.cuda.synchronize(); blocks.append(n * steps / (time.perf_counter() - t))
print(f"THROUGHPUT {n} envs: {statistics.median(blocks):,.0f} env steps/s (median of 5 x {steps} steps; "
      f"{[round(b) for b in blocks]}), {torch.cuda.max_memory_allocated() / 2**30:.1f} GB peak")
app.close()
