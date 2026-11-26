import numpy as np, mujoco, mujoco_warp as mjw, warp as wp
import isaacsimenvs.tasks.cloth.backends.vbd.flex as F
from isaacsimenvs.tasks.cloth.mdp.geometry.grid import grid_mesh
v, t = grid_mesh(0.10, 7, "alternate"); v = np.asarray(v, float); t = np.asarray(t).reshape(-1, 3); v[:, 2] += 0.006
base = """<mujoco><option timestep="0.000694" solver="CG" iterations="50" integrator="implicitfast" cone="elliptic"/>
  <worldbody><geom name="table" type="box" size="0.3 0.3 0.01" pos="0 0 -0.01"/></worldbody></mujoco>"""
def run(tag, m):
    d = mujoco.MjData(m); wm = mjw.put_model(m); wd = mjw.put_data(m, d, nworld=2, nconmax=2000, njmax=8000)
    for i in range(300): mjw.step(wm, wd)
    q = wd.qpos.numpy()[0]; p = wd.flexvert_xpos.numpy()[0]
    print(f"{tag}: |qpos| max {np.abs(q).max()*1e3:.2f} mm, xpos z {p[:,2].min()*1e3:.2f}..{p[:,2].max()*1e3:.2f}, ncon {int(wd.nacon.numpy()[0])}", flush=True)
s = mujoco.MjSpec.from_string(base)
F.add_sheet_flex(s, v, t, density=2.0, radius=0.004, young=7e4, poisson=0.25, thickness=0.0035, damping=0.0, edge_damping=1e-3, friction=0.6, solref=0.005, selfcollide="none", contype=1, conaffinity=1)
run("mine", s.compile())
orig = F._lumped_masses
F._lumped_masses = lambda vv, tt, dd: np.full(len(vv), 0.02 / len(vv))
s = mujoco.MjSpec.from_string(base)
F.add_sheet_flex(s, v, t, density=2.0, radius=0.004, young=7e4, poisson=0.25, thickness=0.0035, damping=0.0, edge_damping=1e-3, friction=0.6, solref=0.005, selfcollide="none", contype=1, conaffinity=1)
for b in s.bodies[1:]: b.inertia = [6.8e-9]*3
run("uniform mass", s.compile())
xml = base.replace("</worldbody>", f"""<flexcomp name="sheet" type="grid" count="7 7 1" spacing="{0.1/6} {0.1/6} {0.1/6}" pos="0 0 0.006" radius="0.004" dim="2" mass="0.02">
      <edge equality="false" damping="0.001"/><elasticity young="7e4" poisson="0.25" thickness="0.0035" elastic2d="both"/>
      <contact condim="3" solref="0.005 1" selfcollide="none"/></flexcomp></worldbody>""")
run("flexcomp", mujoco.MjModel.from_xml_string(xml))
