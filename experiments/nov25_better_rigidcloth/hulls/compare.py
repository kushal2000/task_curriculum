"""Compare robot colliders: PhysX (training, cooked convex hulls) vs Newton (raw triangle meshes).
Inputs outputs/hulls/{physx,newton,newton_nsc}.pkl -> outputs/hulls/*.png, summary.json"""
import pickle, json, numpy as np, trimesh
from scipy.spatial import ConvexHull
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
O = "outputs/hulls/"
px = pickle.load(open(O + "physx.pkl", "rb")); nw = pickle.load(open(O + "newton.pkl", "rb")); nsc = pickle.load(open(O + "newton_nsc.pkl", "rb"))
PH, NE = "#2a6fdb", "#e8590c"

# pair PhysX and Newton shapes per body by raw-vertex count
pairs = []
used = set()
for s in px["shapes"]:
    for j, t in enumerate(nw["shapes"]):
        if j not in used and t["body"] == s["body"] and abs(len(t["verts"]) - len(s["raw"])) == 0:
            used.add(j); pairs.append((s, t)); break
print("paired", len(pairs), "of", len(px["shapes"]))

def hull_mesh(v):
    h = ConvexHull(v); return trimesh.Trimesh(v, h.simplices, process=True).convex_hull
rows = []
for s, t in pairs:
    raw = trimesh.Trimesh(t["verts"], t["faces"], process=False)
    frame_err = np.abs(np.sort(t["verts"], 0) - np.sort(s["raw"], 0)).max()  # same points in the same body frame?
    ph = hull_mesh(s["hulls"][0]["verts"]); exact = hull_mesh(t["verts"])
    # how far training's collider surface lies outside the real surface (filled concavities, hull simplification)
    samp = ph.sample(4000)
    _, d_out, _ = trimesh.proximity.closest_point(raw, samp)
    # raw vertices cut off by the 64-vertex-limited hull
    eq = ConvexHull(s["hulls"][0]["verts"]).equations
    cut = (t["verts"] @ eq[:, :3].T + eq[:, 3]).max(1)
    rows.append(dict(body=s["body"], raw_verts=len(t["verts"]), physx_hull_verts=len(s["hulls"][0]["verts"]),
                     frame_err_mm=1e3 * frame_err, vol_physx_hull_cm3=ph.volume * 1e6, vol_exact_hull_cm3=exact.volume * 1e6,
                     vol_raw_cm3=(raw.volume * 1e6 if raw.is_watertight else None), raw_watertight=bool(raw.is_watertight),
                     hull_beyond_mesh_mean_mm=1e3 * d_out.mean(), hull_beyond_mesh_p95_mm=1e3 * np.percentile(d_out, 95),
                     hull_beyond_mesh_max_mm=1e3 * d_out.max(), mesh_outside_hull_max_mm=1e3 * max(0, cut.max())))
for r in rows:
    print(f"{r['body']:28s} raw {r['raw_verts']:6d} hull {r['physx_hull_verts']:3d}  frame {r['frame_err_mm']:.3f}mm  V hull {r['vol_physx_hull_cm3']:.2f} exact {r['vol_exact_hull_cm3']:.2f} raw {r['vol_raw_cm3']}  beyond mean {r['hull_beyond_mesh_mean_mm']:.2f} p95 {r['hull_beyond_mesh_p95_mm']:.2f} max {r['hull_beyond_mesh_max_mm']:.2f}  cut {r['mesh_outside_hull_max_mm']:.2f}")

def world(v, body, poses):
    p, R = poses[body]; return v @ R.T + p

# ---------------- ring/pinky at the commanded rest pose
poses = nsc["body_pose"]
def get(body):
    for s, t in pairs:
        if s["body"] == body:
            return world(t["verts"], body, poses), t["faces"], world(s["hulls"][0]["verts"], body, poses)
rr, rf, rh = get("r_ring_finger_proximal"); pr, pf, phh = get("r_pinky_proximal")
ring_raw = trimesh.Trimesh(rr, rf, process=False); pinky_raw = trimesh.Trimesh(pr, pf, process=False)
ring_h, pinky_h = hull_mesh(rh), hull_mesh(phh)
def min_dist(a, b, n=20000):
    pa = a.sample(n); _, d, _ = trimesh.proximity.closest_point(b, pa); return d.min()
def overlap(a_hull, b_pts):
    eq = ConvexHull(a_hull.vertices).equations; sd = (b_pts @ eq[:, :3].T + eq[:, 3]).max(1); return -sd.min()  # >0 = depth inside
pair = dict(raw_min_gap_mm=1e3 * min_dist(ring_raw, pinky_raw), physx_hull_min_gap_mm=1e3 * min_dist(ring_h, pinky_h),
            physx_hull_overlap_mm=1e3 * max(overlap(ring_h, pinky_h.sample(20000)), overlap(pinky_h, ring_h.sample(20000))),
            raw_pinky_pts_inside_ring_raw=int(ring_raw.contains(pinky_raw.sample(5000)).sum()) if ring_raw.is_watertight else None)
print("ring/pinky proximal at rest:", pair)
json.dump(dict(shapes=rows, ring_pinky_rest=pair), open(O + "summary.json", "w"), indent=1)

# ---------------- figures
def draw(ax, v, f, color, alpha, lw=0):
    pc = Poly3DCollection(v[f], facecolor=color, alpha=alpha, edgecolor=color if lw else "none", linewidth=lw); ax.add_collection3d(pc)
def hull_faces(m): return m.vertices, m.faces
def setlim(ax, pts, pad=0.002):
    c = (pts.max(0) + pts.min(0)) / 2; r = (pts.max(0) - pts.min(0)).max() / 2 + pad
    ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r); ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

# Fig 1: per-link overlay for the hand links, body frame
hand = [(s, t) for s, t in pairs if s["body"].startswith("r_") and "_abd" not in s["body"]]
fig = plt.figure(figsize=(18, 4.2 * ((len(hand) + 5) // 6)))
for i, (s, t) in enumerate(hand):
    ax = fig.add_subplot((len(hand) + 5) // 6, 6, i + 1, projection="3d")
    ph = hull_mesh(s["hulls"][0]["verts"])
    sub = t["faces"][:: max(1, len(t["faces"]) // 3000)]
    draw(ax, t["verts"], sub, NE, 0.55)
    draw(ax, ph.vertices, ph.faces, PH, 0.12, lw=0.6)
    setlim(ax, t["verts"]); ax.view_init(20, 35)
    r = [x for x in rows if x["body"] == s["body"] and x["raw_verts"] == len(t["verts"])][0]
    ax.set_title(f"{s['body'][2:]}\nhull beyond surface: mean {r['hull_beyond_mesh_mean_mm']:.1f}, max {r['hull_beyond_mesh_max_mm']:.1f} mm", fontsize=8)
fig.suptitle("Robot colliders, body frame.  Blue wireframe: PhysX convex hull (training).  Orange solid: Newton collider = the raw triangle mesh", fontsize=12)
plt.tight_layout(); plt.savefig(O + "hull_overlay_links.png", dpi=110); plt.close()

# Fig 2: ring + pinky proximal at the rest pose, both representations, 3 views
fig = plt.figure(figsize=(15, 5))
allp = np.vstack([rr, pr])
for k, (el, az, name) in enumerate([(90, -90, "top"), (0, 0, "front (along x)"), (25, 40, "3/4")]):
    ax = fig.add_subplot(1, 3, k + 1, projection="3d")
    for m, c in ((ring_raw, NE), (pinky_raw, "#c92a2a")):
        draw(ax, m.vertices, m.faces[:: max(1, len(m.faces) // 4000)], c, 0.6)
    for m in (ring_h, pinky_h):
        draw(ax, m.vertices, m.faces, PH, 0.1, lw=0.6)
    setlim(ax, allp); ax.view_init(el, az); ax.set_title(name)
fig.suptitle(f"Ring (orange) and pinky (red) proximal links at the commanded rest pose, Newton's raw meshes; blue = PhysX hulls\n"
             f"raw-mesh gap {pair['raw_min_gap_mm']:.2f} mm, PhysX-hull gap {pair['physx_hull_min_gap_mm']:.2f} mm (overlap {pair['physx_hull_overlap_mm']:.2f} mm)", fontsize=11)
plt.tight_layout(); plt.savefig(O + "ring_pinky_rest.png", dpi=110); plt.close()

# Fig 3: bar chart of how much fatter training's hull is
hb = [r for r in rows if r["body"].startswith("r_")]
fig, ax = plt.subplots(figsize=(14, 4.5))
x = np.arange(len(hb)); ax.bar(x - 0.2, [r["hull_beyond_mesh_mean_mm"] for r in hb], 0.4, color=PH, label="mean")
ax.bar(x + 0.2, [r["hull_beyond_mesh_max_mm"] for r in hb], 0.4, color="#9ec5fe", label="max")
ax.set_xticks(x); ax.set_xticklabels([r["body"][2:] + ("" if r["raw_verts"] > 20000 or "distal" not in r["body"] else " (pad)") for r in hb], rotation=40, ha="right", fontsize=8)
ax.set_ylabel("PhysX hull surface outside the real surface [mm]"); ax.legend(); ax.spines[["top", "right"]].set_visible(False)
ax.set_title("How far training's collision surface sits outside Newton's (raw mesh) surface, per hand collider")
plt.tight_layout(); plt.savefig(O + "hull_excess_bar.png", dpi=110); plt.close()
print("wrote figures")
