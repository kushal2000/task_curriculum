"""outputs/obs_audit/obs_audit_summary.json: the audit's headline numbers from the captures and evals."""
import glob, json, torch
O = "outputs/obs_audit"
out = {"evals_32env_rand_900steps": {}, "captures": {}, "newton_plates_table_tunnelling": {}, "bodyquat_diag": {}}
for f in sorted(glob.glob(f"{O}/eval_*_s*.json")):
    d = json.load(open(f))
    out["evals_32env_rand_900steps"][f.split("/")[-1][5:-5]] = {
        "fold": d["termination_reasons"]["fold"], "n": d["num_envs"], "fall": d["termination_reasons"]["fall"],
        "surface_within_tol": d.get("surface", {}).get("envs_within_tolerance")}
for tag in ("pinned", "rand"):
    out["captures"][tag] = json.load(open(f"{O}/audit_{tag}.json"))["summaries"]
    out["captures"][tag + "_proprio_max_abs_diff_record0"] = json.load(open(f"{O}/audit_{tag}.json"))["proprio_max_abs_diff_record0"]
for w in ("plates", "vbd"):
    for tag in ("pinned", "rand"):
        d = torch.load(f"{O}/{w}_{tag}.pt", weights_only=False)
        m, rec = d["meta"], d["rec"]
        tz = torch.stack([r["table_z"] for r in rec]) + float(m["table_half"])          # (T, N) table top
        zmin = torch.stack([r["points"][:, :, 2].min(1).values for r in rec]) - tz         # lowest grid point
        xy = torch.stack([r["points"][:, :, :2].abs().amax(1) for r in rec])               # (T, N, 2)
        half_x = 0.3 if w == "plates" else 0.2375   # table_fold_pool 0.6 x 0.4, table_narrow 0.475 x 0.4
        on_table = (xy[..., 0] < half_x) & (xy[..., 1] < 0.2)
        under = (zmin < -0.005) & on_table
        out["newton_plates_table_tunnelling" if w == "plates" else "newton_plates_table_tunnelling"][f"{w}_{tag}"] = {
            "envs_with_sheet_5mm_below_table_top_while_over_table": int(under.any(0).sum()), "n": int(under.shape[1]),
            "deepest_mm": round(float(zmin.min()) * 1e3, 1),
            "folded_first_episode": int(m["folded_first_episode"].sum())}
        if "diag" in d:
            out["bodyquat_diag"][f"{w}_{tag}"] = [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()
                                                   if k in ("step", "max_abs_raw_minus_state", "max_abs_raw_minus_state_as_xyzw",
                                                            "robot_surface_point_shift_mm_median")} |
                                                  {"visibility_disagree_frac_mean": round(sum(x["visibility_disagree_frac"]) / len(x["visibility_disagree_frac"]), 4)}
                                                  for x in d["diag"]]
json.dump(out, open(f"{O}/obs_audit_summary.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ("evals_32env_rand_900steps", "newton_plates_table_tunnelling")}, indent=1))
