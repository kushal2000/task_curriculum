import sys, numpy as np, mujoco, mujoco_warp as mjw, warp as wp
def xml(res=13, solver="Newton", integ="implicitfast", table="box", mass=0.02, pusher=True, elastic="both", dt=0.000694, young=5e4):
    sp = 0.10 / (res - 1)
    tb = '<geom name="table" type="box" size="0.3 0.3 0.01" pos="0 0 -0.01"/>' if table == "box" else '<geom name="table" type="plane" size="0.3 0.3 0.01"/>'
    pu = '''<body name="pusher" pos="0.03 0 0.03"><joint name="pz" type="slide" axis="0 0 1" damping="5"/><joint name="px" type="slide" axis="1 0 0" damping="5"/>
      <geom type="box" size="0.01 0.04 0.01" mass="0.05"/></body>''' if pusher else ""
    el = f'<elasticity young="{young}" poisson="0.2" thickness="0.001" elastic2d="{elastic}"/>' if elastic != "off" else ""
    act = '<actuator><position joint="pz" kp="20"/><position joint="px" kp="20"/></actuator>' if pusher else ""
    return f"""<mujoco><option timestep="{dt}" solver="{solver}" iterations="30" integrator="{integ}" cone="elliptic"/>
  <worldbody>{tb}{pu}
    <flexcomp name="sheet" type="grid" count="{res} {res} 1" spacing="{sp} {sp} {sp}" pos="0 0 0.006" radius="0.004" dim="2" mass="{mass}">
      <edge equality="false" damping="0.001"/>{el}
      <contact condim="3" solref="0.005 1" selfcollide="none"/></flexcomp>
  </worldbody>{act}</mujoco>"""
def test(**kw):
    m = mujoco.MjModel.from_xml_string(xml(**kw)); d = mujoco.MjData(m)
    if m.nu: d.ctrl[0] = -0.035
    wm = mjw.put_model(m); wd = mjw.put_data(m, d, nworld=1, nconmax=2000, njmax=8000)
    for i in range(600):
        mjw.step(wm, wd)
        if i % 30 == 0 and not np.isfinite(wd.qpos.numpy()).all(): return f"NaN by {i}"
    p = wd.flexvert_xpos.numpy()[0]; return f"stable, zmin {p[:,2].min()*1e3:.1f} mm, ncon {int(wd.nacon.numpy()[0])}"
for kw in [dict(elastic="off", pusher=False), dict(elastic="off", pusher=False, table="plane"), dict(elastic="off", pusher=False, solver="CG"),
           dict(elastic="off", pusher=False, integ="Euler"), dict(elastic="off", pusher=False, mass=2.0), dict(pusher=False, table="plane", solver="CG"),
           dict(table="plane", solver="CG"), dict(solver="CG", integ="Euler")]:
    try: print(kw, test(**kw), flush=True)
    except Exception as e: print(kw, "ERR", repr(e)[:200], flush=True)
