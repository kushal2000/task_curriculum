import sys, numpy as np, mujoco, mujoco_warp as mjw, warp as wp
from isaacsimenvs.tasks.cloth.backends.vbd.flex import add_sheet_flex
from isaacsimenvs.tasks.cloth.mdp.geometry.grid import grid_mesh
v, t = grid_mesh(0.10, 7, "alternate"); v = np.asarray(v, float); t = np.asarray(t).reshape(-1, 3); v[:, 2] += 0.006
def spec(solver, selfc, damping, it=50):
    s = mujoco.MjSpec.from_string(f"""<mujoco><option timestep="0.000694" solver="{solver}" iterations="{it}" integrator="implicitfast" cone="elliptic"/>
  <worldbody><geom name="table" type="box" size="0.3 0.3 0.01" pos="0 0 -0.01"/>
   <body name="pusher" pos="0.03 0 0.03"><joint name="pz" type="slide" axis="0 0 1" damping="5"/><joint name="px" type="slide" axis="1 0 0" damping="5"/>
   <geom type="box" size="0.01 0.04 0.01" mass="0.05"/></body></worldbody>
   <actuator><position joint="pz" kp="20"/><position joint="px" kp="20"/></actuator></mujoco>""")
    add_sheet_flex(s, v, t, density=2.0, radius=0.004, young=2*100*1.25/0.0035, poisson=0.25, thickness=0.0035, damping=damping, edge_damping=1e-3,
                   friction=0.612, solref=0.005, selfcollide=selfc, contype=1, conaffinity=1)
    return s.compile()
for solver, selfc, damp in [("CG","none",1e-4),("CG","none",0.0),("CG","auto",0.0),("Newton","none",0.0),("Newton","narrow",0.0)]:
    m = spec(solver, selfc, damp); d = mujoco.MjData(m); d.ctrl[0] = -0.035
    wm = mjw.put_model(m); wd = mjw.put_data(m, d, nworld=4, nconmax=2000, njmax=8000); ok = "ok"
    for i in range(1440):
        if i == 480: c = wd.ctrl.numpy(); c[:, 1] = -0.08; wd.ctrl = wp.array(c, dtype=float)
        mjw.step(wm, wd)
        if i % 60 == 0 and not np.isfinite(wd.qpos.numpy()).all(): ok = f"NaN by {i}"; break
    p = wd.flexvert_xpos.numpy()[0]; print(f"GPU {solver} self={selfc} damp={damp}: {ok} z {np.nanmin(p[:,2])*1e3:.1f}..{np.nanmax(p[:,2])*1e3:.1f} x {np.nanmin(p[:,0])*1e3:.0f}..{np.nanmax(p[:,0])*1e3:.0f} ncon {int(wd.nacon.numpy()[0])}", flush=True)
