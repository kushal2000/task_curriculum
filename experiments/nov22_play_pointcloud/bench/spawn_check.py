"""Do tools spawn inside the table? One env per pool tool: the lowest point of each tool's surface
(object_surface scored samples) relative to the table top, right after reset and after settling.
    python spawn_check.py <task id> <num envs>"""
import sys
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import gymnasium as gym, torch
import isaacsimenvs  # noqa
T, n = sys.argv[1], int(sys.argv[2])
mod, cls = gym.spec(T).kwargs["env_cfg_entry_point"].split(":")
cfg = getattr(__import__(mod, fromlist=[cls]), cls)(); cfg.scene.num_envs = n; cfg.pointcloud.extra_obs = ()
env = gym.make(T, cfg=cfg); e = env.unwrapped
top = float(e.cfg.reset.table_reset_z) + float(e._table_half_height)
def lowest():
    pos = e.object.data.root_pos_w - e.scene.env_origins
    return e.env_shapes.points_w(pos, e.object.data.root_quat_w, "fixed")[..., 2].min(dim=1).values - (e._table_z_per_env + e._table_half_height if hasattr(e, "_table_z_per_env") else top)
env.reset()
a = torch.zeros(n, e.cfg.action_space, device=e.device)
env.step(a)                                     # the first step after reset: the spawn pose
z0 = lowest()
for _ in range(60): env.step(a)
z1 = lowest()
for name, z in (("after reset (1 step)", z0), ("after 1 s", z1)):
    cm = z * 100
    print(f"SPAWN {name}: lowest tool point vs table top [cm]: min {cm.min():.1f}, 1% {cm.quantile(0.01):.1f}, "
          f"median {cm.median():.1f}; tools > 0.5 cm into the table: {(cm < -0.5).sum().item()} / {n}, > 2 cm: {(cm < -2).sum().item()}", flush=True)
bad = (z1 * 100 < -0.5).nonzero().flatten()[:8].tolist()
for i in bad:
    print(f"SPAWN stuck env {i}: {e._object_urdf_paths[int(e._object_asset_index_per_env[i])].rsplit('/', 1)[-1][:60]}  lowest {z1[i] * 100:.1f} cm", flush=True)
app.close()
