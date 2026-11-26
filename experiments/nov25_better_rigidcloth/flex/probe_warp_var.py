import sys, numpy as np, mujoco, mujoco_warp as mjw, warp as wp
sys.path.insert(0, "experiments/nov25_better_rigidcloth/flex"); from probe_cpu import build
for kw in [dict(elastic2d="none"), dict(elastic2d="stretch"), dict(elastic2d="bend"), dict(young=0.0)]:
    try:
        m = build(res=13, **kw); d = mujoco.MjData(m); d.ctrl[0] = -0.035
        wm = mjw.put_model(m); wd = mjw.put_data(m, d, nworld=1, nconmax=2000, njmax=6000)
        out = "stable 600"
        maxc = 0
        for i in range(600):
            mjw.step(wm, wd); 
            if i % 20 == 0:
                maxc = max(maxc, int(wd.nacon.numpy()[0]))
                if not np.isfinite(wd.qpos.numpy()).all(): out = f"NaN by {i}"; break
        p = wd.flexvert_xpos.numpy()[0]
        print(kw, out, "max ncon", maxc, "zmin %.4f" % np.nanmin(p[:, 2]) if np.isfinite(p).any() else "")
    except Exception as e: print(kw, "ERR", repr(e)[:300])
