"""Success per config of vbd_transfer.sbatch (pooled over seeds; censored = failure)."""
import glob, json, math, collections, sys
d = sys.argv[1] if len(sys.argv) > 1 else "outputs/vbd_transfer"
R = collections.defaultdict(lambda: [0, 0, collections.Counter(), 0])
for f in glob.glob(f"{d}/*__s*.json"):
    name = f.split("/")[-1].rsplit("__s", 1)[0]
    j = json.load(open(f)); g = j["per_env_goals"]
    r = R[name]; r[0] += sum(x > 0 for x in g); r[1] += len(g); r[3] += 1
    for k, v in j["termination_reasons"].items():
        r[2][k] += v
order = [l.split("|")[0] for l in open(f"{d}/configs.txt")]
for n in order + sorted(set(R) - set(order)):
    if n not in R:
        print(f"{n:28s} pending"); continue
    ok, tot, tr, files = R[n]; q = ok / tot
    print(f"{n:28s} {100*q:5.1f}% +-{196*math.sqrt(q*(1-q)/tot):4.1f}  ({ok}/{tot}, {files} seeds)  fall {tr['fall']} timeout {tr['timeout']} nonfinite {tr['nonfinite']}")
