"""Success rates of eval_old_new.sbatch's JSONs. Denominator: all envs (a censored env is a failure)."""
import collections, glob, json, math, os, sys
R = collections.defaultdict(lambda: collections.Counter())
S = collections.defaultdict(lambda: collections.Counter())
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*__s*.json"))):
    pol, cell, _ = os.path.basename(f).rsplit(".", 1)[0].split("__")
    d = json.load(open(f)); ok = [g > 0 for g in d["per_env_goals"]]; x = d["per_env_extra_done"]
    for key in [(pol, cell)] + ([(pol, "ALL NEW (pooled)")] if cell != "old" else []):
        c = R[key]; c["n"] += len(ok); c["ok"] += sum(ok); c["cens"] += d["censored"]; c["files"] += 1
        c["pen"] += sum(x.get("finger_penetration", [])); c["fall"] += d["termination_reasons"]["fall"]
        c["timeout"] += d["termination_reasons"]["timeout"]; c["hand_far"] += d["termination_reasons"]["hand_far"]
    for k, v in x.items():
        if k.startswith("shape_"):
            for e, on in enumerate(v):
                if on: S[(pol, cell, k[6:])]["n"] += 1; S[(pol, cell, k[6:])]["ok"] += ok[e]
for (p, c), v in sorted(R.items()):
    q = v["ok"] / v["n"]
    print(f"{p:16s} {c:18s} {100*q:5.1f}% +-{196*math.sqrt(q*(1-q)/v['n']):.1f}  n={v['n']} files={v['files']} "
          f"pen={100*v['pen']/v['n']:.1f}% timeout={100*v['timeout']/v['n']:.1f}% fall={100*v['fall']/v['n']:.1f}% hand_far={100*v['hand_far']/v['n']:.1f}% censored={v['cens']}")
print("\nper shape (finished episodes; min / max by cell)")
for (p, c) in sorted({k[:2] for k in S}):
    r = sorted((v["ok"] / v["n"], k[2], v["n"]) for k, v in S.items() if k[:2] == (p, c))
    print(p, c, "worst", [(f"{100*a:.0f}%", b, n) for a, b, n in r[:3]], "best", [(f"{100*a:.0f}%", b, n) for a, b, n in r[-3:]])
