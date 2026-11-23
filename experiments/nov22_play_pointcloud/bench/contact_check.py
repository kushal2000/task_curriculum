"""Does the hand collide with the tool? Replays render_play_pc's setup (student, tool, envs) and logs
per step the fingertips' signed distance to the tool surface (object_surface SDF) and the tool's speed:
    python contact_check.py <student .pth> <tool> <keep_robot_visuals 0|1> <steps>"""
import sys
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True, enable_cameras=False).app
import gymnasium as gym, torch
import isaacsimenvs  # noqa
from isaacsimenvs.scripts.distillation.student import SplitStudentNet
from isaacsimenvs.tasks.play.utils.object_surface import matrix_from_quat
ck_path, tool, keep_vis, steps = sys.argv[1], int(sys.argv[2]), bool(int(sys.argv[3])), int(sys.argv[4])
ck = torch.load(ck_path, map_location="cpu", weights_only=False)
mod, cls = gym.spec(ck["env"]["task"]).kwargs["env_cfg_entry_point"].split(":")
cfg = getattr(__import__(mod, fromlist=[cls]), cls)()
cfg.scene.num_envs = tool + 1; cfg.scene.env_spacing = 3.0; cfg.seed = 0
cfg.assets.keep_robot_visuals = keep_vis
cfg.termination.eval_success_tolerance = 0.01
cfg.pointcloud.extra_obs = ()
env = gym.make(ck["env"]["task"], cfg=cfg); e = env.unwrapped
lay = e.layout; n_p, m = lay["n_proprio"], lay["num_points"]
net = SplitStudentNet(n_p, int(e.cfg.action_space), 3, int(ck["model"]["aux.2.weight"].shape[0]),
                      **{**ck["net_args"], "compile": False}).to(e.device)
net.load_state_dict(ck["model"]); net.eval()
obs, _ = env.reset(); state = net.initial_state(e.num_envs, e.device)
start = torch.ones(e.num_envs, dtype=torch.bool, device=e.device)
print(f"CHECK keep_robot_visuals={keep_vis} tool={tool} envs={e.num_envs}", flush=True)
worst = []
for t in range(steps):
    pol = obs["policy"]
    with torch.no_grad():
        mu, _a, state = net.step(pol[:, :n_p], pol[:, n_p:n_p + 3 * m].reshape(-1, m, 3), pol[:, n_p + 3 * m:n_p + 4 * m] > 0.5, state, start)
    obs, _r, te, tr, _ = env.step(mu.float().clamp(-1, 1))
    start = te.bool() | tr.bool()
    org = e.scene.env_origins
    ft = e.robot.data.body_pos_w[:, e._fingertip_body_ids_t] - org.unsqueeze(1)          # (N,5,3) fingertip bodies
    o_pos = e.object.data.root_pos_w - org; o_q = e.object.data.root_quat_w
    local = torch.matmul(ft - o_pos.unsqueeze(1), matrix_from_quat(o_q))                  # tool frame
    sd = e.env_shapes.sdf(local) * 100.0                                                   # cm, <0 inside
    v = e.object.data.root_lin_vel_w.norm(dim=-1) * 100.0
    k = tool
    worst.append(float(sd[k].min()))
    if t % 25 == 0:
        print(f"CHECK t={t:3d} env{k}: fingertip sdf cm {[round(x, 1) for x in sd[k].tolist()]}  tool speed {v[k]:.1f} cm/s"
              f"  tool z {o_pos[k, 2]:.3f}", flush=True)
w = torch.tensor(worst)
print(f"CHECK summary: deepest fingertip {w.min():.1f} cm; steps with a fingertip > 0.5 cm inside: {(w < -0.5).sum().item()} / {len(w)}", flush=True)
app.close()
