"""One line per distillation run: latest training row. python3 scripts/analysis/distill_status.py [min_job]"""
import glob, json, sys
lo = int(sys.argv[1]) if len(sys.argv) > 1 else 0
for f in sorted(glob.glob("outputs/distill/*_s1_*/log.jsonl")):
    name = f.split("/")[2]
    if not name.rsplit("_", 1)[-1].isdigit() or int(name.rsplit("_", 1)[-1]) < lo:
        continue
    rows = [json.loads(l) for l in open(f)]
    if not rows:
        continue
    r = rows[-1]
    best = max((x["student_success"] or 0) for x in rows if x["beta"] == 0) if any(x["beta"] == 0 for x in rows) else None
    print(name.ljust(30), f"it {r['iter']:5d} beta {r['beta']:.2f} sps {int(r['sps']):6d} act {r['loss_act']:.4f} "
          f"succ {r['student_success']} (n {r['student_episodes']}, best {best}) goal_cm {r['aux_goal_cm']} "
          f"{list(r['aux_goal_cm_by_drag'].values())} palm_cm {r['aux_palm_cm']} skip {r.get('skipped_minibatches')} gpu {r['gpu_gb']}")
