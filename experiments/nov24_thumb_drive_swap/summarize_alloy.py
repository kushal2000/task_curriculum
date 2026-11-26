"""Summarize alloy_guards.sh: per variant, both seeds pooled (sums over 2 x 32 envs).
    python experiments/nov24_thumb_drive_swap/summarize_alloy.py [outputs/alloy_guards] [outputs/thumb_kstart_sweep ...]"""
import collections, glob, json, os, sys

dirs = sys.argv[1:] or ["outputs/alloy_guards"]
rows = collections.defaultdict(lambda: collections.Counter())
extra = collections.defaultdict(dict)
for d in dirs:
    for f in sorted(glob.glob(os.path.join(d, "*_track.json"))):
        name = os.path.basename(f).rsplit("_s", 1)[0]
        t = json.load(open(f)); r = rows[name]; r["seeds"] += 1
        for j in t["joints"].values():
            r["uncmd"] += j["jumps_uncommanded"]
            r["uncmd_cloth"] += j.get("uncommanded_jumps_by_location", {}).get("cloth", 0)
        r["nonfinite"] += t.get("nonfinite_envs_dropped", 0)
        fl = t.get("forces", {}).get("links", {})
        if fl:
            extra[name]["link_force_max_n"] = max(extra[name].get("link_force_max_n", 0), max(v["force_max_n"] for v in fl.values()))
        h = t.get("cloth_health", {})
        for k in ("speed", "kick", "strain"):
            if k in h:
                unit = "frac" if k == "strain" else "mps"
                extra[name][f"{k}_max"] = max(extra[name].get(f"{k}_max", 0), h[k][f"max_{unit}"])
                extra[name].setdefault(f"{k}_p99", []).append(round(h[k][f"p99_{unit}"], 3))
        if "kick" in h:
            r["jump_steps_near_cloth"] += h["kick"].get("uncmd_near_cloth_jump_steps", 0)
            r["jumps_with_kick>1"] += h["kick"].get("uncmd_near_cloth_jumps_with_kick_over_1", 0)
            extra[name].setdefault("kick>1_base_rate", []).append(round(h["kick"].get("base_rate_kick_over_1", float("nan")), 4))
        if "speed" in h:
            r["speed>20_envs"] += h["speed"].get("envs_ever_over_20", 0)
        r["cuts"] += h.get("exit_cap_cuts", 0)
        r["sc_overflow_env_steps"] += h.get("self_contact_overflow", {}).get("env_steps_with_overflow", 0)
    for f in sorted(glob.glob(os.path.join(d, "*_pierce.json"))):
        name = os.path.basename(f).rsplit("_s", 1)[0]
        p = json.load(open(f)); r = rows[name]
        for o in p["outcome"]:
            r[f"ep_{o}"] += 1
        r["ep_pierced"] += p["episodes_with_pierce"]
keys = ["seeds", "uncmd", "uncmd_cloth", "jump_steps_near_cloth", "jumps_with_kick>1", "nonfinite", "speed>20_envs",
        "cuts", "sc_overflow_env_steps", "ep_fold", "ep_fall", "ep_timeout", "ep_hand_far", "ep_nonfinite", "ep_pierced"]
print("variant".ljust(14) + "".join(k[:12].rjust(13) for k in keys))
for name, r in rows.items():
    print(name.ljust(14) + "".join(str(r.get(k, 0)).rjust(13) for k in keys))
print()
for name, e in extra.items():
    print(name.ljust(14), {k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.items()})
