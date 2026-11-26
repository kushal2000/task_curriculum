import sys, torch, math
d = torch.load(sys.argv[1], weights_only=False)
m, rec = d["meta"], d["rec"]
for k in ("robot_root_pos","hand_sysid","table_reset_z","table_half","start_height","fixed_start_pose","success","surface_layered","tol","keypoint_tolerance","fold_lift","centre","table_z_cloud","object_scale","keypoint_idx","decimation","dt"):
    print(k, m[k])
print("goal_tasks", m["goal_tasks"])
for s in (0, 1, 5, 11, 12, 13, 19, 20, 40, 70, 119):
    r = rec[s]; e = 0
    live = r["xyz"][e, :64][r["mask"][e, :64]]; goal = r["xyz"][e, 64:][r["mask"][e, 64:]]
    P = r["points"][e]
    mv = P[m["moving_idx"]].mean(0); st = P[m["stationary_idx"]].mean(0)
    f = lambda t: " ".join(f"{v:+.4f}" for v in t.tolist())
    print(f"step {r['step']:3d} live n={len(live):2d} c=({f(live.mean(0)) if len(live) else '-'}) goal n={len(goal):2d} c=({f(goal.mean(0)) if len(goal) else '-'})"
          f" | sheet c=({f(P.mean(0))}) mov=({f(mv)}) stat=({f(st)}) yaw={math.degrees(r['frame_yaw'][e]):+.1f} fe={r['fold_error'][e]:.4f} lay={r['layered'][e] if r['layered'] is not None else None:.4f} frozen={bool(r['frozen'][e])}")
