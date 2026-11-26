import sys, time, numpy as np, mujoco, mujoco_warp as mjw, warp as wp
sys.path.insert(0, "experiments/nov25_better_rigidcloth/flex"); from probe_warp_m import xml
res, nworld = int(sys.argv[1]), int(sys.argv[2])
m = mujoco.MjModel.from_xml_string(xml(res=res, solver="CG")); d = mujoco.MjData(m); d.ctrl[0] = -0.035
wm = mjw.put_model(m); wd = mjw.put_data(m, d, nworld=nworld, nconmax=400 * nworld, njmax=1600 * nworld)
for _ in range(5): mjw.step(wm, wd)
wp.synchronize(); t = time.time(); n = 720
for i in range(n):
    if i == 240:
        c = wd.ctrl.numpy(); c[:, 1] = -0.08; wd.ctrl = wp.array(c, dtype=float)
    mjw.step(wm, wd)
wp.synchronize(); dt = (time.time() - t) / n
p = wd.flexvert_xpos.numpy()
print(f"res {res} nworld {nworld} nv {m.nv}: {dt*1e3:.2f} ms/step = {nworld/dt:.0f} world-steps/s; finite {np.isfinite(p).all()}; zmin {np.nanmin(p[...,2])*1e3:.1f} mm; x {np.nanmin(p[0,:,0])*1e3:.0f}..{np.nanmax(p[0,:,0])*1e3:.0f} mm")
