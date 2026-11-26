"""PhysX's cooked convex hulls for every robot collider, from a de-instanced flattened copy of the robot
USD (the cooking query refuses instance proxies). Vertices in each rigid body's frame. -> outputs/hulls/physx.pkl"""
import sys, pickle, numpy as np
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils, PhysicsSchemaTools, Sdf
import omni.usd
from omni.physx import get_physx_cooking_interface
src = sys.argv[1]
flat = Usd.Stage.Open(src).Flatten()  # an Sdf.Layer
tmp = "/tmp/" + __import__("getpass").getuser() + "_robot_flat.usda"
flat.Export(tmp)
st = Usd.Stage.Open(tmp)
for p in list(st.Traverse()):
    if p.IsInstanceable():
        p.SetInstanceable(False)
# The cooking query only accepts a Mesh carrying the CollisionAPI; the asset puts the API (and
# physics:approximation=convexHull) on the parent Xform. Copy them onto the child mesh -- the same
# geometry and the same default cooking settings PhysX uses for it in training.
for p in list(st.Traverse()):
    if p.HasAPI(UsdPhysics.CollisionAPI) and not p.IsA(UsdGeom.Mesh):
        ap = p.GetAttribute("physics:approximation").Get()
        for m in Usd.PrimRange(p):
            if m.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(m); UsdPhysics.MeshCollisionAPI.Apply(m).CreateApproximationAttr(ap)
UsdPhysics.Scene.Define(st, "/physicsScene")
st.GetRootLayer().Save()
del st
omni.usd.get_context().open_stage(tmp)
for _ in range(10): app.update()
stage = omni.usd.get_context().get_stage()
sid = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
if sid < 0:
    sid = UsdUtils.StageCache.Get().Insert(stage).ToLongInt()
ci = get_physx_cooking_interface()
xc = UsdGeom.XformCache()
out = {"shapes": []}
for prim in stage.Traverse():
    if not prim.HasAPI(UsdPhysics.CollisionAPI) or not prim.IsA(UsdGeom.Mesh) or str(prim.GetPath()).startswith("/colliders"):
        continue
    body = prim
    while body and body.GetPath() != Sdf.Path("/") and not body.HasAPI(UsdPhysics.RigidBodyAPI):
        body = body.GetParent()
    if not body or body.GetPath() == Sdf.Path("/"):
        print("no body for", prim.GetPath()); continue
    Mb = np.array(xc.GetLocalToWorldTransform(body)); Mp = np.array(xc.GetLocalToWorldTransform(prim))
    rel = Mp @ np.linalg.inv(Mb)
    res = {}
    def cb(r, convexes):
        res["r"] = r; res["c"] = [(np.array([[v.x, v.y, v.z] for v in c.vertices]), list(c.indices), [(q.index_base, q.num_vertices) for q in c.polygons]) for c in convexes]
    for c in [prim] + [m for m in Usd.PrimRange(prim) if m.IsA(UsdGeom.Mesh)]:
        ci.request_convex_collision_representation(stage_id=sid, collision_prim_id=PhysicsSchemaTools.sdfPathToInt(c.GetPath()), run_asynchronously=False, on_result=cb)
        if res.get("c"): break
    raw = []
    for m in Usd.PrimRange(prim):
        if m.IsA(UsdGeom.Mesh):
            pts = np.array(m.GetAttribute("points").Get(), dtype=np.float64)
            Mm = np.array(xc.GetLocalToWorldTransform(m)) @ np.linalg.inv(Mb)
            raw.append((np.c_[pts, np.ones(len(pts))] @ Mm)[:, :3])
    hl = [dict(verts=(np.c_[v, np.ones(len(v))] @ rel)[:, :3], verts_prim=v, indices=i, polys=pl) for v, i, pl in res.get("c", [])]
    attrs = {a.GetName(): a.Get() for a in prim.GetAttributes() if "physx" in a.GetName().lower() or "approx" in a.GetName()}
    out["shapes"].append(dict(body=body.GetName(), path=str(prim.GetPath()), hulls=hl, raw=np.concatenate(raw) if raw else None, attrs=attrs))
    print(f"{body.GetName():30s} {res.get('r')} hulls={[len(h['verts']) for h in hl]} raw={sum(len(r) for r in raw)} {attrs}", flush=True)
pickle.dump(out, open("outputs/hulls/physx.pkl", "wb"))
app.close()
