"""Dump PhysX's cooked convex hulls for every robot collider in the training env (IL 2.3 / Isaac Sim 5.1),
in each rigid body's frame, plus the raw authored mesh points. -> outputs/hulls/physx.pkl"""
import os, sys, argparse, pickle
sys.path.insert(0, os.getcwd())
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p); a, extra = p.parse_known_args()
sys.argv = [sys.argv[0], *extra]
app = AppLauncher(a).app
import gymnasium as gym, numpy as np, torch
import isaacsimenvs  # noqa
from isaacsimenvs.tasks.loader import hydra_task_config_with_yaml
T = "Isaacsimenvs-RigidCloth-Direct-v0"

@hydra_task_config_with_yaml(T, "", experiment="experiments/nov24_play_pc_rl/play_pc_rl_A.yaml")
def run(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = 1
    env = gym.make(T, cfg=env_cfg); inner = env.unwrapped
    env.reset()
    for _ in range(30):
        env.step(torch.zeros(env.action_space.shape, device=inner.device))
    from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils, PhysicsSchemaTools
    import omni.usd
    from omni.physx import get_physx_cooking_interface
    stage = omni.usd.get_context().get_stage()
    sid = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
    ci = get_physx_cooking_interface()
    xc = UsdGeom.XformCache()
    root = stage.GetPrimAtPath("/World/envs/env_0/Robot")
    out = {"shapes": [], "body_pose": {}}
    robot = inner.robot
    names = list(robot.body_names)
    pose = robot.data.body_link_pose_w[0].cpu().numpy()  # wxyz
    def qm(q):
        w, x, y, z = q
        return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    for i, n in enumerate(names):
        out["body_pose"][n] = (pose[i, :3].copy(), qm(pose[i, 3:7]))
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        body = prim
        while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
            body = body.GetParent()
        if not body:
            continue
        bname = body.GetName()
        Mb = np.array(xc.GetLocalToWorldTransform(body), dtype=np.float64)       # row-vector convention
        Mp = np.array(xc.GetLocalToWorldTransform(prim), dtype=np.float64)
        rel = Mp @ np.linalg.inv(Mb)                                              # prim -> body
        approx = prim.GetAttribute("physics:approximation").Get() if prim.GetAttribute("physics:approximation") else None
        res = {}
        def cb(r, convexes):
            res["r"] = r; res["c"] = [(np.array([[v.x, v.y, v.z] for v in c.vertices]), list(c.indices),
                                     [(p.index_base, p.num_vertices) for p in c.polygons]) for c in convexes]
        cands = [prim] + [m for m in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()) if m.IsA(UsdGeom.Mesh)]
        cands += [c.GetPrimInPrototype() for c in list(cands) if c.IsInstanceProxy()]
        for c in cands:
            res.clear()
            ci.request_convex_collision_representation(stage_id=sid, collision_prim_id=PhysicsSchemaTools.sdfPathToInt(c.GetPath()),
                                                       run_asynchronously=False, on_result=cb)
            if res.get("c"):
                res["via"] = str(c.GetPath()); break
        raw = []
        for m in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
            if m.IsA(UsdGeom.Mesh):
                pts = np.array(m.GetAttribute("points").Get(), dtype=np.float64)
                Mm = np.array(xc.GetLocalToWorldTransform(m), dtype=np.float64) @ np.linalg.inv(Mb)
                raw.append(np.c_[pts, np.ones(len(pts))] @ Mm)
        raw = np.concatenate(raw)[:, :3] if raw else None
        hulls = []
        for v, idx, polys in res.get("c", []):
            hulls.append(dict(verts_prim=v, verts=(np.c_[v, np.ones(len(v))] @ rel)[:, :3], indices=idx, polys=polys))
        out["shapes"].append(dict(body=bname, path=str(prim.GetPath()), approx=approx, result=str(res.get("r")), hulls=hulls, raw=raw))
        print(f"{bname:30s} {approx} result={res.get('r')} via={res.get('via')} hulls={len(hulls)} hullverts={[len(h['verts']) for h in hulls]} raw={None if raw is None else len(raw)}", flush=True)
    pickle.dump(out, open("outputs/hulls/physx.pkl", "wb"))
    env.close()
run()
app.close()
