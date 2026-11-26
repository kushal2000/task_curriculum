import sys, numpy as np, mujoco, mujoco_warp as mjw, warp as wp
from isaacsimenvs.tasks.cloth.backends.vbd.flex import add_sheet_flex
from isaacsimenvs.tasks.cloth.mdp.geometry.grid import grid_mesh
res = int(sys.argv[1]) if len(sys.argv) > 1 else 7
v, t = grid_mesh(0.10, res, "alternate"); v = np.asarray(v, float); t = np.asarray(t).reshape(-1, 3); v[:, 2] += 0.006
def spec(solver):
    s = mujoco.MjSpec.from_string(f"""<mujoco><option timestep="0.000694" solver="{solver}" iterations="50" integrator="implicitfast" cone="elliptic"/>
  <worldbody><geom name="table" type="box" size="0.3 0.3 0.01" pos="0 0 -0.01"/>
   <body name="pusher" pos="0.03 0 0.03"><joint name="pz" type="slide" axis="0 0 1" damping="5"/><joint name="px" type="slide" axis="1 0 0" damping="5"/>
   <geom type="box" size="0.01 0.04 0.01" mass="0.05"/></body></worldbody>
   <actuator><position joint="pz" kp="20"/><position joint="px" kp="20"/></actuator></mujoco>""")
    young = 2 * 100 * 1.25 / 0.0035
    add_sheet_flex(s, v, t, density=2.0, radius=0.004, young=young, poisson=0.25, thickness=0.0035, damping=1e-4, edge_damping=1e-3,
                   friction=0.612, solref=0.005, selfcollide=sys.argv[2] if len(sys.argv) > 2 else "auto", contype=1, conaffinity=1)
    return s.compile()
for solver in ["Newton", "CG"]:
    m = spec(solver); d = mujoco.MjData(m); d.ctrl[0] = -0.035
    ok = "ok"
    for i in range(1440):
        if i == 480: d.ctrl[1] = -0.08
        mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos).all(): ok = f"NaN {i}"; break
    p = d.flexvert_xpos; print(f"CPU {solver}: {ok} nv {m.nv} z {p[:,2].min()*1e3:.1f}..{p[:,2].max()*1e3:.1f} x {p[:,0].min()*1e3:.0f}..{p[:,0].max()*1e3:.0f} ncon {d.ncon}", flush=True)
    m = spec(solver); d = mujoco.MjData(m); d.ctrl[0] = -0.035
    wm = mjw.put_model(m); wd = mjw.put_data(m, d, nworld=4, nconmax=2000, njmax=8000); ok = "ok"
    for i in range(1440):
        if i == 480: c = wd.ctrl.numpy(); c[:, 1] = -0.08; wd.ctrl = wp.array(c, dtype=float)
        mjw.step(wm, wd)
        if i % 60 == 0 and not np.isfinite(wd.qpos.numpy()).all(): ok = f"NaN by {i}"; break
    p = wd.flexvert_xpos.numpy()[0]; print(f"GPU {solver}: {ok} z {np.nanmin(p[:,2])*1e3:.1f}..{np.nanmax(p[:,2])*1e3:.1f} x {np.nanmin(p[:,0])*1e3:.0f}..{np.nanmax(p[:,0])*1e3:.0f} ncon {int(wd.nacon.numpy()[0])}", flush=True)
