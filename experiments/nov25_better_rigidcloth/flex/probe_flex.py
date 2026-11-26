"""Feasibility: a 2D flex sheet in MuJoCo Warp (shell bending + membrane), pushed by a box, on a table, N worlds."""
import sys, time
import numpy as np, mujoco, mujoco_warp as mjw, warp as wp

res = int(sys.argv[1]) if len(sys.argv) > 1 else 13
nworld = int(sys.argv[2]) if len(sys.argv) > 2 else 32
spacing = 0.10 / (res - 1)
xml = f"""
<mujoco>
  <option timestep="0.000694" solver="Newton" iterations="20" integrator="implicitfast" cone="elliptic"/>
  <worldbody>
    <geom name="table" type="box" size="0.3 0.3 0.01" pos="0 0 -0.01" friction="0.354"/>
    <body name="pusher" pos="0.03 0 0.03">
      <joint name="pz" type="slide" axis="0 0 1" damping="5"/>
      <joint name="px" type="slide" axis="1 0 0" damping="5"/>
      <geom type="box" size="0.01 0.04 0.01" mass="0.05" friction="0.612"/>
    </body>
    <flexcomp name="sheet" type="grid" count="{res} {res} 1" spacing="{spacing} {spacing} {spacing}"
              pos="0 0 0.006" radius="0.004" dim="2" mass="0.02">
      <edge equality="false" damping="0.001"/>
      <elasticity young="5e4" poisson="0.2" thickness="0.001" elastic2d="both"/>
      <contact condim="3" solref="0.005 1" selfcollide="none" friction="0.6"/>
    </flexcomp>
  </worldbody>
  <actuator>
    <position joint="pz" kp="20" ctrlrange="-0.1 0.1"/>
    <position joint="px" kp="20" ctrlrange="-0.1 0.1"/>
  </actuator>
</mujoco>"""
m = mujoco.MjModel.from_xml_string(xml)
print(f"res {res}: nq {m.nq} nv {m.nv} nflex {m.nflex} flexvert {m.nflexvert} nbody {m.nbody} flex_dim {m.flex_dim}")
d = mujoco.MjData(m)
wm = mjw.put_model(m)
wd = mjw.put_data(m, d, nworld=nworld, nconmax=4000 * nworld, njmax=8000 * nworld)
ctrl = np.zeros((nworld, m.nu), np.float32); ctrl[:, 0] = -0.035   # press down
wd.ctrl = wp.array(ctrl, dtype=float)
mjw.step(wm, wd); wp.synchronize()
t = time.time(); steps = 1440
for i in range(steps):
    if i == 480:
        ctrl[:, 1] = -0.08; wd.ctrl = wp.array(ctrl, dtype=float)  # drag toward -x
    mjw.step(wm, wd)
wp.synchronize(); dt = time.time() - t
xpos = wd.flexvert_xpos.numpy() if hasattr(wd, "flexvert_xpos") else None
print(f"{steps} steps x {nworld} worlds: {dt:.1f} s  ({steps * nworld / dt:.0f} world-steps/s, {dt / steps * 1e3:.2f} ms/step)")
if xpos is not None:
    p = xpos[0]
    print("sheet z min/max", p[:, 2].min(), p[:, 2].max(), " x extent", p[:, 0].min(), p[:, 0].max(), " nan", np.isnan(xpos).any())
print("ncon world0", int(wd.nacon.numpy()[0]) if hasattr(wd, "nacon") else "?")
