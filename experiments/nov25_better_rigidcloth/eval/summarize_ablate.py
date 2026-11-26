"""ablate.sbatch's results: success over all envs (censored = failure), per config; new-shape runs also by group."""
import collections, glob, json, math, os, sys
R = collections.defaultdict(collections.Counter)
for f in sorted(g for g in glob.glob(os.path.join(sys.argv[1], "64live__*__s*.json")) if not g.endswith(".depth.json")):
    name = os.path.basename(f).split("__")[1]
    d = json.load(open(f)); ok = [g > 0 for g in d["per_env_goals"]]; x = d["per_env_extra_done"]
    pen = x.get("finger_penetration", [False] * len(ok)); tr = d["termination_reasons"]
    groups = {k[6:]: v for k, v in x.items() if k.startswith("shape_")}
    for e in range(len(ok)):
        sh = next((s for s, v in groups.items() if v[e]), "?")
        g = "plate2" if sh == "plate2" else "2vert" if sh.startswith("2vert") else "2flaps" if sh.startswith("2flaps") else "old7"
        for key in ((name, "all"), (name, g)):
            c = R[key]; c["n"] += 1; c["ok"] += ok[e]; c["pen"] += pen[e]
    R[(name, "all")]["timeout"] += tr["timeout"]; R[(name, "all")]["fall"] += tr["fall"]; R[(name, "all")]["files"] += 1
def pct(c): q = c["ok"] / c["n"]; return f"{100*q:5.1f} +-{196*math.sqrt(q*(1-q)/c['n']):4.1f}"
order = [l.split("|")[0] for l in open(sys.argv[2] if len(sys.argv) > 2 else "experiments/nov25_better_rigidcloth/eval/ablate_configs.txt")]
print(f"{'config':16s} {'all':>12s}  files  pen%  timeout%  fall%   | plate2        2vert         2flaps")
for n in order:
    a = R.get((n, "all"))
    if not a: print(f"{n:16s}   (pending)"); continue
    sub = "  ".join(pct(R[(n, g)]) if (n, g) in R else "      -     " for g in ("plate2", "2vert", "2flaps"))
    print(f"{n:16s} {pct(a)}  {a['files']}    {100*a['pen']/a['n']:4.1f}  {100*a['timeout']/a['n']:6.1f}  {100*a['fall']/a['n']:5.1f}   | {sub}")
