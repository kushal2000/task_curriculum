import sys, numpy as np, mujoco, mujoco_warp as mjw, warp as wp
sys.path.insert(0, "experiments/nov25_better_rigidcloth/flex"); from probe_cpu import build
solver = sys.argv[1] if len(sys.argv) > 1 else "Newton"; nworld = int(sys.argv[2]) if len(sys.argv) > 2 else 1
m = build(res=13, solver=solver); d = mujoco.MjData(m); d.ctrl[0] = -0.035
wm = mjw.put_model(m); wd = mjw.put_data(m, d, nworld=nworld, nconmax=2000, njmax=6000)
for i in range(400):
    mjw.step(wm, wd)
    if i % 50 == 0 or i < 5:
        q = wd.qpos.numpy(); p = wd.flexvert_xpos.numpy()[0]
        print(i, "finite", np.isfinite(q).all(), "ncon", int(wd.nacon.numpy()[0]), "zmin %.4f" % np.nanmin(p[:, 2]), "nefc", int(wd.nefc.numpy()[0]))
        if not np.isfinite(q).all(): break
for i in range(400): mujoco.mj_step(m, d)
print("cpu after 400: ncon", d.ncon, "zmin %.4f" % d.flexvert_xpos[:, 2].min())
