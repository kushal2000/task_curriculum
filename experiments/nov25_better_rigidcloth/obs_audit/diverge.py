"""Pinned plates vs pinned VBD, env 0..: where the palm, the sheet and the clouds part ways."""
import sys, torch
a = torch.load(sys.argv[1], weights_only=False); b = torch.load(sys.argv[2], weights_only=False)
ma = a["meta"]; n_p = 103
def palm(r): return r["policy"][:, 30*0 + 81:84]  # placeholder, replaced below
off = 0
for f, w in ma["proprio_fields"]:
    if f == "palm_pos": po = off
    if f == "fingertip_pos_rel_palm": fo = off
    off += w
tz = float(ma["table_z_cloud"])
mv = ma["moving_idx"]
for s in list(range(18, 270, 12)):
    ra, rb = a["rec"][s], b["rec"][s]
    pa, pb = ra["policy"][:, po:po+3], rb["policy"][:, po:po+3]
    la = ra["points"][:, mv, 2].max(1).values - tz; lb = rb["points"][:, mv, 2].max(1).values - tz
    print(f"step {ra['step']:3d} palm diff mm (env0..3) {((pa-pb).norm(dim=-1)[:4]*1e3).numpy().round(1)} "
          f"palm z-table mm P {((pa[:2,2]-tz)*1e3).numpy().round(0)} V {((pb[:2,2]-tz)*1e3).numpy().round(0)} | "
          f"moving-half max lift mm P {(la[:4]*1e3).numpy().round(0)} V {(lb[:4]*1e3).numpy().round(0)} | "
          f"fe P {ra['fold_error'][:4].numpy().round(3)} V {rb['fold_error'][:4].numpy().round(3)}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
steps = [r["step"] for r in a["rec"]]
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
for d, name, c in ((a, "plates (Newton)", "#2b6cb0"), (b, "VBD", "#dd6b20")):
    lift = torch.stack([r["points"][:, mv, 2].max(1).values - tz for r in d["rec"]]) * 1e3
    fe = torch.stack([r["fold_error"] for r in d["rec"]])
    for e in range(lift.shape[1]):
        ax[0].plot(steps, lift[:, e].clamp(-20, 80), color=c, alpha=0.5, lw=1, label=name if e == 0 else None)
        ax[1].plot(steps, fe[:, e].clamp(max=0.12), color=c, alpha=0.5, lw=1, label=name if e == 0 else None)
pd = torch.stack([(ra["policy"][:, po:po+3] - rb["policy"][:, po:po+3]).norm(dim=-1) for ra, rb in zip(a["rec"], b["rec"])]) * 1e3
ax[2].plot(steps, pd.mean(1), color="k")
ax[0].set_title("moving half: highest grid point above table [mm]"); ax[1].set_title("fold error [m] (designated)")
ax[2].set_title("|palm plates - palm VBD| [mm], env mean (same pinned start)")
for x in ax:
    x.axvline(20, color="grey", ls=":", lw=1); x.set_xlabel("episode step (policy from step 21)"); x.grid(alpha=0.3)
ax[0].legend(); ax[1].legend()
fig.tight_layout(); fig.savefig(sys.argv[3] if len(sys.argv) > 3 else "outputs/obs_audit/divergence_pinned.png", dpi=130)
