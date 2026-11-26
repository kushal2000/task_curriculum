"""Dump every robot collision shape on Newton (as imported for the training recipe): vertices in the
body frame, plus body poses at the reset pose. -> outputs/hulls/newton.npz"""
import os, sys, argparse, pickle
sys.path.insert(0, os.getcwd())
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p); a, extra = p.parse_known_args()
sys.argv = [sys.argv[0], *extra]
try: AppLauncher(a)
except ImportError: pass
import gymnasium as gym, numpy as np, torch
import isaacsimenvs  # noqa
from isaacsimenvs.tasks.loader import hydra_task_config_with_yaml
T = "Isaacsimenvs-RigidClothNewton-Direct-v0"

def qmat(q):  # xyzw
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

@hydra_task_config_with_yaml(T, "", experiment="experiments/nov24_play_pc_rl/play_pc_rl_A.yaml")
def run(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = 1
    env = gym.make(T, cfg=env_cfg); inner = env.unwrapped
    env.reset()
    for _ in range(30):
        env.step(torch.zeros(env.action_space.shape, device=inner.device))
    from isaaclab_newton.physics import NewtonManager
    import newton
    m = NewtonManager._model; st = NewtonManager._state_0
    bl = [str(x).rsplit("/", 1)[-1] for x in m.body_label]; sl = [str(x) for x in m.shape_label]
    sb = m.shape_body.numpy(); stype = m.shape_type.numpy(); ssc = m.shape_scale.numpy(); stf = m.shape_transform.numpy()
    bq = st.body_q.numpy()
    out = {"shapes": [], "body_pose": {}}
    for s, b in enumerate(sb):
        if b < 0 or bl[b].startswith("slat") or "plate" in sl[s].lower():
            continue
        src = m.shape_source[s]
        if src is None or not hasattr(src, "vertices"):
            print("non-mesh", sl[s], int(stype[s]), ssc[s]); continue
        v = np.asarray(src.vertices, dtype=np.float64) * ssc[s]
        R = qmat(stf[s][3:7]); v = v @ R.T + stf[s][:3]          # -> body frame
        out["shapes"].append(dict(body=bl[b], label=sl[s], verts=v, faces=np.asarray(src.indices).reshape(-1, 3),
                                  type=int(stype[s]), maxhullvert=getattr(src, "maxhullvert", None),
                                  margin=float(m.shape_margin.numpy()[s]), gap=float(m.shape_gap.numpy()[s])))
        out["body_pose"][bl[b]] = (bq[b][:3].copy(), qmat(bq[b][3:7]))
    print("type ids:", {k: v for k, v in vars(newton.GeoType).items() if not k.startswith("_")} if hasattr(newton, "GeoType") else "")
    print(f"dumped {len(out['shapes'])} robot shapes on {len(out['body_pose'])} bodies")
    pickle.dump(out, open(os.environ.get("HULL_OUT", "outputs/hulls/newton.pkl"), "wb"))
    env.close()
run()
