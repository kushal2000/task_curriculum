import sys, time, numpy as np, mujoco
def build(res=13, young=5e4, thick=1e-3, dt=0.000694, solver="Newton", damping=0.001, elastic2d="both", flexdof="full"):
    sp = 0.10 / (res - 1)
    xml = f"""<mujoco><option timestep="{dt}" solver="{solver}" iterations="30" integrator="implicitfast" cone="elliptic"/>
  <worldbody><geom name="table" type="box" size="0.3 0.3 0.01" pos="0 0 -0.01" friction="0.354"/>
    <body name="pusher" pos="0.03 0 0.03"><joint name="pz" type="slide" axis="0 0 1" damping="5"/><joint name="px" type="slide" axis="1 0 0" damping="5"/>
      <geom type="box" size="0.01 0.04 0.01" mass="0.05" friction="0.612"/></body>
    <flexcomp name="sheet" type="grid" count="{res} {res} 1" spacing="{sp} {sp} {sp}" pos="0 0 0.006" radius="0.004" dim="2" mass="0.02" dof="{flexdof}">
      <edge equality="false" damping="{damping}"/>
      <elasticity young="{young}" poisson="0.2" thickness="{thick}" elastic2d="{elastic2d}"/>
      <contact condim="3" solref="0.005 1" selfcollide="none" friction="0.6"/></flexcomp>
  </worldbody><actuator><position joint="pz" kp="20"/><position joint="px" kp="20"/></actuator></mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)
def run(m, steps=1440):
    d = mujoco.MjData(m); d.ctrl[0] = -0.035; t = time.time()
    for i in range(steps):
        if i == 480: d.ctrl[1] = -0.08
        mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos).all(): return f"NaN at step {i}"
    p = d.flexvert_xpos
    return f"ok {time.time()-t:.1f}s  z {p[:,2].min()*1e3:.1f}..{p[:,2].max()*1e3:.1f} mm  x {p[:,0].min()*1e3:.0f}..{p[:,0].max()*1e3:.0f} mm  ncon {d.ncon}"
if __name__ == "__main__":
    for kw in [dict(), dict(young=1e4), dict(young=5e4, dt=0.0002), dict(elastic2d="stretch"), dict(elastic2d="bend"), dict(flexdof="trilinear"), dict(res=7)]:
        try: m = build(**kw); print(kw, f"nv {m.nv}", run(m))
        except Exception as e: print(kw, "ERR", e)
