"""Assemble the cloth-realism page from outputs/cloth_probe/final and outputs/vbd_eval.

    .venv_isaaclab3/bin/python experiments/nov16_vbd_cloth/build_page.py --out <page.html>
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "outputs/cloth_probe/final"
E = ROOT / "outputs/vbd_eval"
D17 = ROOT / "outputs/cloth_probe/coarse17_page"  # the 17 mm irregular sheet (*_vbd_coarse presets)
B = ROOT / "outputs/cloth_bench"  # scripts/evaluation/throughput.py results
#: (sheet-key variant, long name, short name, probe dir)
MODELS = [("legacy", "today (7×7 grid)", "today", D), ("new-final", "new (10 mm irregular)", "new 10 mm", D),
          ("new-c17", "new (17 mm irregular)", "new 17 mm", D17)]
NM = len(MODELS)
FABS = ["chiffon", "silk", "napkin", "cotton", "big_napkin", "towel", "denim"]
NAME = dict(chiffon="chiffon", silk="silk", napkin="napkin", cotton="cotton", big_napkin="big napkin", towel="towel",
            denim="denim")
SWATCH = dict(chiffon="rarely stays, even pressed", silk="springs back; pressed holds", napkin="springs back; pressed holds",
              cotton="stays (slight gap)", big_napkin="stays", towel="stays", denim="stays")
COLOR = dict(denim="#3e5f8a", towel="#c9a46a", cotton="#d9d4c5", big_napkin="#b85c5c", napkin="#d88d7a", silk="#9b7bb5",
             chiffon="#7fb6a8")


def j(name, d=D):
    return json.loads((d / name).read_text())


def res(name):
    """{sheet key: result} merged over every model's probe dir."""
    out = {}
    for d in dict.fromkeys(m[3] for m in MODELS):
        if (d / name).is_file():
            out.update(j(name, d)["results"])
    return out


def chip(ok, text):
    return f'<span class="chip {"ok" if ok is True else "no" if ok is False else "mid"}">{text}</span>'


def sw(f):
    return f'<span class="sw" style="background:{COLOR[f]}"></span>{NAME[f]}'


def seq_table():
    seq = res("sequence_tight.json")
    can = res("cantilever.json")
    rows = ['<table><thead><tr><th>Fabric</th><th>Model</th><th>Real swatch</th><th>Cantilever (meas.)</th>'
            '<th>Fold, no press</th><th>Fold + 0.93 N press</th><th>Opened after press</th></tr></thead><tbody>']
    for f in FABS:
        for i, (v, long, _short, _d) in enumerate(MODELS):
            key = f"{f}__{v}__10cm"
            r = {x["phase"]: x for x in seq[key]}
            a, c, e = r["folded, no press"], r["folded + pressed"], r["opened (after pressed fold)"]
            c_ = can[key]

            def fold(x):
                if x["fold_err_cm"] < 4.0 and x["open_deg"] < 45:
                    return chip(True, "holds") + f' <span class="sub mono">flap {x["open_deg"]:.0f}°</span>'
                return chip(False, f'opens {x["open_deg"]:.0f}°')

            rows.append(
                f'<tr class="{"grp" if i == 0 else ""}">'
                + (f'<td rowspan="{NM}">{sw(f)}</td>' if i == 0 else "")
                + f"<td>{long}</td>"
                + (f'<td rowspan="{NM}">{SWATCH[f]}</td>' if i == 0 else "")
                + f'<td class="n">{c_["tip_angle_deg"]:.0f}° <span class="sub">({c_["measured_deg"]:.0f}°)</span></td>'
                + f"<td>{fold(a)}</td><td>{fold(c)}</td>"
                + f'<td class="n">{e["max_height_mm"]:.0f} mm high <span class="sub">crease {e["set_deg_max"]:.0f}°</span></td></tr>')
    return "".join(rows) + "</tbody></table>"


def hold_table():
    seq = res("sequence_tight.json")
    rows = ['<table><thead><tr><th>Fabric</th><th>Real swatch</th><th>Model</th><th>No press: 180 steps untouched</th>'
            '<th>Pressed: 180 steps untouched</th></tr></thead><tbody>']
    for f in FABS:
        for i, (v, _long, short, _d) in enumerate(MODELS):
            r = {x["phase"]: x for x in seq[f"{f}__{v}__10cm"]}

            def cell(x):
                w = x["worst_err_180_cm"]
                if x["held_180_4cm"]:
                    return chip(True, "holds 180") + f' <span class="sub mono">worst {w:.1f} cm{" · 1.5 cm too" if x["held_180_1p5cm"] else ""}</span>'
                return chip(False, "opens") + f' <span class="sub mono">worst {min(w, 10):.1f} cm</span>'

            rows.append(f'<tr class="{"grp" if i == 0 else ""}">' + (f'<td rowspan="{NM}">{sw(f)}</td><td rowspan="{NM}">{SWATCH[f]}</td>' if i == 0 else "")
                        + f"<td>{short}</td><td>{cell(r['folded, no press'])}</td>"
                        + f"<td>{cell(r['folded + pressed'])}</td></tr>")
    return "".join(rows) + "</tbody></table>"


def foldline_table():
    fl = res("foldline.json")
    rows = ['<table><thead><tr><th>Fabric</th><th>Model</th><th>0 mm</th><th>+10 mm</th><th>+20 mm</th>'
            '<th>15°</th><th>45°</th></tr></thead><tbody>']
    for f in ("cotton", "silk", "towel"):
        for i, (v, _long, short, _d) in enumerate(MODELS):
            base = fl[f"{f}__{v}__10cm__c0mm_a0"]["crease_offset_err_mm"]
            cells = []
            for c, a in ((0, 0), (10, 0), (20, 0), (0, 15), (0, 45)):
                r = fl[f"{f}__{v}__10cm__c{c}mm_a{a}"]
                d = r["crease_offset_err_mm"] - base
                cells.append(f'<td class="n">{d:+.1f} mm <span class="sub">±{r["crease_wiggle_mm"]:.1f}</span></td>')
            rows.append(f'<tr class="{"grp" if i == 0 else ""}">' + (f'<td rowspan="{NM}">{sw(f)}</td>' if i == 0 else "")
                        + f"<td>{short}</td>" + "".join(cells) + "</tr>")
    return "".join(rows) + "</tbody></table>"


def wrinkle_table(fname, phases):
    r = res(fname)
    rows = ['<table><thead><tr><th>Fabric</th><th>Size</th><th>Model</th>'
            + "".join(f"<th>{p[1]}</th>" for p in phases) + "</tr></thead><tbody>"]
    for f in ("cotton", "silk", "denim"):
        for size in (10, 20, 30):
            for i, (v, _long, short, _d) in enumerate(MODELS):
                ph = {x["phase"]: x for x in r.get(f"{f}__{v}__{size}cm", [])}
                cells = []
                for key, _lab, fmt in phases:
                    x = ph.get(key, {})
                    cells.append(f'<td class="n">{fmt(x)}</td>')
                rows.append(f'<tr class="{"grp" if (i == 0 and size == 10) else ""}">'
                            + (f'<td rowspan="{3 * NM}">{sw(f)}</td>' if (i == 0 and size == 10) else "")
                            + (f'<td rowspan="{NM}">{size} cm</td>' if i == 0 else "")
                            + f"<td>{short}</td>" + "".join(cells) + "</tr>")
    return "".join(rows) + "</tbody></table>"


def perf_table():
    """Throughput (scripts/evaluation/throughput.py, napkin presets, zero actions) per cloth and env count."""
    runs = [("measured", "today (7×7 grid)"), ("vbd", "new (10 mm irregular)"), ("vbd_coarse", "new (17 mm irregular)"),
            ("vbd_coarse+p512", "new 17 mm, 512 self-contact pairs")]
    ns = (64, 256, 1024)
    rows = ['<table><thead><tr><th>Cloth</th><th>Particles</th>' + "".join(f"<th>{n} envs</th>" for n in ns)
            + "<th>GPU memory / env</th></tr></thead><tbody>"]
    any_ = False
    for tag, name in runs:
        d = {n: json.loads((B / f"{tag}_n{n}.json").read_text()) for n in ns if (B / f"{tag}_n{n}.json").is_file()}
        if not d:
            continue
        any_ = True
        cells = []
        for n in ns:
            x = d.get(n)
            cells.append(f'<td class="n">{x["env_steps_per_s"]:.0f}/s <span class="sub">{x["ms_per_step"]:.0f} ms · {x["gpu_mem_gb"]:.0f} GB</span></td>'
                         if x else '<td class="sub">out of memory (48 GB)</td>')
        lo, hi = d.get(64), d.get(256)
        per = f'{1e3 * (hi["gpu_mem_gb"] - lo["gpu_mem_gb"]) / 192:.0f} MB' if lo and hi else "–"
        np_ = next(iter(d.values()))["particles"]
        rows.append(f"<tr><td>{name}</td><td class=\"n\">{np_}</td>" + "".join(cells) + f'<td class="n">{per}</td></tr>')
    return "".join(rows) + "</tbody></table>", any_


ONLY_4CM = False


def eval_table():
    cols = [("", 0, "Touches keypoints · 4 cm"), ("", 1, "Holds 180 steps · 4 cm")]
    if not ONLY_4CM:
        cols += [("_1p5", 0, "Touches · 1.5 cm"), ("_1p5", 1, "Holds 180 · 1.5 cm")]
    rows = ['<table><thead><tr><th>Fabric</th><th>Cloth</th>' + "".join(f"<th>{c[2]}</th>" for c in cols) + "</tr></thead><tbody>"]
    any_ = False
    tot = {}
    for f in FABS:
        for i, v in enumerate(("legacy", "vbd")):
            cells = []
            for suf, h, _ in cols:
                p = E / f"{v}{suf}" / f"{f}_hold{h}.json"
                preset = ROOT / "isaacsimenvs/cfg/fabric" / f"{f}{'_vbd' if v == 'vbd' else '_measured'}.yaml"
                if not p.is_file() or p.stat().st_mtime < preset.stat().st_mtime:  # missing, or run on an older preset
                    cells.append('<td class="sub">running</td>')
                    continue
                any_ = True
                g = json.loads(p.read_text())["per_env_goals"]  # first episode of every env
                k, n = sum(1 for x in g if x > 0), len(g)
                a_, b_ = tot.get((v, suf, h), (0, 0))
                tot[(v, suf, h)] = (a_ + k, b_ + n)
                cells.append(f'<td class="n">{chip(k / n >= 0.5, f"{100 * k / n:.0f}%")} <span class="sub">{k}/{n}</span></td>')
            rows.append(f'<tr class="{"grp" if i == 0 else ""}">' + (f'<td rowspan="2">{sw(f)}</td>' if i == 0 else "")
                        + f"<td>{'today (7×7 grid)' if v == 'legacy' else 'new (10 mm irregular)'}</td>" + "".join(cells) + "</tr>")
    for i, v in enumerate(("legacy", "vbd")):
        cells = []
        for suf, h, _ in cols:
            k, n = tot.get((v, suf, h), (0, 0))
            cells.append(f'<td class="n"><b>{100 * k / max(n, 1):.0f}%</b> <span class="sub">{k}/{n}</span></td>' if n else '<td class="sub">running</td>')
        rows.append(f'<tr class="{"grp" if i == 0 else ""}">' + ('<td rowspan="2"><b>all seven</b></td>' if i == 0 else "")
                    + f"<td>{'today' if v == 'legacy' else 'new'}</td>" + "".join(cells) + "</tr>")
    return "".join(rows) + "</tbody></table>", any_


def diag_table():
    src = [("today (7×7 grid), plastic", E / "legacy"), ("today, plasticity off", E / "diag/legacy_noplast"),
           ("new (10 mm irregular), 10 N/m membrane", E / "vbd"), ("new, 100 N/m membrane", E / "diag/vbd_tri100")]
    if not all((d / "denim_hold0.json").is_file() for _, d in src):
        return None
    rows = ['<table><thead><tr><th>Cloth</th>' + "".join(f"<th>{NAME[f]}</th>" for f in FABS)
            + "<th>All seven</th><th>Sheet left the table</th><th>Timed out</th></tr></thead><tbody>"]
    for name, d in src:
        ks, n, fall, tout = [], 0, 0, 0
        for f in FABS:
            r = json.loads((d / f"{f}_hold0.json").read_text())
            g = r["per_env_goals"]
            ks.append(sum(1 for x in g if x > 0))
            n += len(g)
            fall += r["termination_reasons"]["fall"]
            tout += r["termination_reasons"]["timeout"]
        rows.append(f"<tr><td>{name}</td>" + "".join(f'<td class="n">{k}/32</td>' for k in ks)
                    + f'<td class="n"><b>{100 * sum(ks) / n:.0f}%</b></td><td class="n">{fall}</td><td class="n">{tout}</td></tr>')
    return "".join(rows) + "</tbody></table>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-evals", action="store_true", help="leave the eval table off (results still running)")
    ap.add_argument("--only-4cm", action="store_true", help="eval table at the task's 4 cm tolerance only")
    a = ap.parse_args()
    global ONLY_4CM
    ONLY_4CM = a.only_4cm
    seq_keys, lab = [], {}
    for v, _long, short, _d in MODELS:
        for f in FABS:
            k = f"{f}__{v}__10cm"
            seq_keys.append(k)
            lab[k] = f"{NAME[f]} · {short}"
    fl_keys, fl_lab = [], {}
    for v, _long, short, _d in MODELS:
        for c, ang in ((0, 0), (10, 0), (20, 0), (0, 15), (0, 45)):
            k = f"cotton__{v}__10cm__c{c}mm_a{ang}"
            fl_keys.append(k)
            fl_lab[k] = f"{short} · {f'+{c} mm' if c else f'{ang}°' if ang else 'midline'}"
    wr_keys, wr_lab = [], {}
    for v, _long, short, _d in MODELS:
        for f in ("cotton", "silk", "denim"):
            for size in (10, 20, 30):
                k = f"{f}__{v}__{size}cm"
                wr_keys.append(k)
                wr_lab[k] = f"{NAME[f]} {size} cm · {short}"
    shear_file = "shear_strong" if (D / "shear_strong.json").is_file() else "shear"
    ev_html, ev_any = eval_table()
    if a.no_evals:
        ev_any = False
    amp = lambda x: f'{x.get("max_height_mm", 0):.0f} mm' if x else "–"  # noqa: E731
    kap = lambda x: f'{x.get("kappa_mean", 0):.1f} /m' if x else "–"  # noqa: E731
    rel = lambda x: f'{x.get("max_height_mm", 0):.1f} mm · set {x.get("set_deg_max", 0):.0f}°' if x else "–"  # noqa: E731
    spec = dict(
        title="Realistic VBD Cloth",
        header=(
            "<h1>Realistic VBD cloth</h1>"
            "<p>The fold task's VBD sheet, rebuilt so it behaves like fabric rather than like a hinged plate, "
            "next to today's sheet. Every test below is scripted, cloth only, at the env's contact settings "
            "(8 mm particle radius against the table, 2 mm self-contact) and timestep (1/1440 s, 12 VBD iterations).</p>"
            "<ul><li><b>No preferred fold line.</b> A 10 mm irregular mesh (rotated, jittered, Delaunay) replaces the "
            "7×7 grid, whose middle row made every fold land on it.</li>"
            "<li><b>Bending from the measured swatch.</b> Each hinge bends against the fabric's measured rigidity B "
            "(per-edge B / hinge width), so the stiffness no longer changes with the mesh. A per-fabric solver factor "
            "(0.9–2.2×) makes up for VBD under-converging soft bending at 12 iterations.</li>"
            "<li><b>Creasing on curvature.</b> A crease forms where the sheet is bent tighter than the fabric's yield "
            "curvature, plus a little internal friction. A tight pressed fold creases; a loose loop or a drape does not. "
            "This replaces the two-stage plasticity.</li>"
            "<li><b>A coarser variant at the old memory cost.</b> The same model on a 17 mm irregular mesh (50 particles on "
            "10 cm, as the 7×7 grid) with its solver factor re-fit on the cantilever: presets "
            "<code>*_vbd_coarse.yaml</code>, shown as the third row everywhere below. It fits 1024 envs per GPU again; "
            "silk and napkin tell an unpressed from a pressed fold less clearly at this resolution.</li>"
            "<li><b>Wrinkles where the physics allows them.</b> Larger sheets buckle into folds; at 10 cm there is too "
            "little room for them.</li></ul>"),
        sections=[
            dict(panel=dict(title="Fold, press, open", json=str(D / "sequence_tight.json"), npz=str(D / "sequence_tight.npz"), extra=[dict(json=str(D17 / "sequence_tight.json"), npz=str(D17 / "sequence_tight.npz"))],
                            select=seq_keys, cols=7, labels=lab, stride=1,
                            lede="The far edge is carried onto its mirror image 4 mm above the other half and let go, the sheet is opened, "
                                 "then folded again with a 0.93 N press (the 213 g swatch weight) and opened. "
                                 "Back row today's sheet, middle the new 10 mm sheet, front the new 17 mm sheet, 10 cm.",
                            jumps=[["Fold, no press", "fold: far edge carried over, no"], ["Open", "open (after unpressed"],
                                   ["Fold + press", "fold: far edge carried over (then"], ["Open", "open (after pressed"]])),
            dict(table=dict(title="Against the real swatches", html=seq_table(),
                            lede="A fold <b>holds</b> when the flap stays down (within 45° of the other half) and its keypoints within the "
                                 "task's 4 cm; the angle is how far the flap lifts off the lower half (0° = flat). The new "
                                 "crease is rounded over one or two 10 mm hinges, so a folded edge sits about 1 cm short of "
                                 "the near edge, where the old grid creases with a zero-radius kink. The cantilever is a 5 cm "
                                 "flap drooping under its own weight; the measured swatch angle is in brackets.")),
            dict(table=dict(title="Does the fold stay? 180 steps, no hand", html=hold_table(),
                            lede="The eval's hold criterion applied to the cloth alone: after the fold is let go, the four "
                                 "keypoints must stay within 4 cm of their targets for 180 policy steps (3 s) with nothing "
                                 "touching the sheet; the worst error over the window is shown. Every new-cloth fold that "
                                 "the real swatch keeps, stays; the ones the swatch lets spring open, open.")),
            dict(panel=dict(title="Fold anywhere", json=str(D / "foldline.json"), npz=str(D / "foldline.npz"), extra=[dict(json=str(D17 / "foldline.json"), npz=str(D17 / "foldline.npz"))],
                            select=fl_keys, cols=5, labels=fl_lab, stride=1,
                            lede="Cotton, pressed folds about the midline, 10 and 20 mm off it, and at 15° and 45°. Back row "
                                 "today, middle new 10 mm, front new 17 mm.")),
            dict(table=dict(title="Where the crease formed", html=foldline_table(),
                            lede="Crease position across the commanded line, relative to the midline fold on the same "
                                 "mesh (± its spread along the crease). Today's grid snaps to its rows, 16.7 mm apart; "
                                 "the new mesh follows the line.")),
            dict(panel=dict(title="Shear and squeeze", json=str(D / f"{shear_file}.json"), npz=str(D / f"{shear_file}.npz"), extra=[dict(json=str(D17 / f"{shear_file}.json"), npz=str(D17 / f"{shear_file}.npz"))],
                            select=wr_keys, cols=9, labels=wr_lab, stride=1,
                            lede="Both sides held; one slides sideways by 25% while they come 15% closer, then both let go. "
                                 "Cotton, silk and denim at 10, 20 and 30 cm.")),
            dict(table=dict(title="Buckling under shear", html=wrinkle_table(f"{shear_file}.json", [
                ("sheared, held", "While held: buckle height", amp),
                ("sheared, held", "Mean curvature", kap),
                ("released", "After release", rel)]),
                lede="Today's sheet takes the whole deformation in-plane and stays flat at every size. The new cotton and "
                     "silk buckle into diagonal ridges that grow with the sheet; silk relaxes flat when let go, cotton "
                     "keeps faint creases (its lower yield). Denim, the floppiest swatch (B = 2.8e-6 N m), stays flat "
                     "here too: its soft membrane absorbs the squeeze, a limit of the 10 N/m membrane VBD needs to "
                     "converge.")),
            dict(panel=dict(title="Lift by two corners, drop", json=str(D / "hang.json"), npz=str(D / "hang.npz"), extra=[dict(json=str(D17 / "hang.json"), npz=str(D17 / "hang.npz"))],
                            select=wr_keys, cols=9, labels=wr_lab, stride=1,
                            lede="The far corners are lifted and brought together, the sheet hangs, then is let go.")),
            dict(table=dict(title="Hanging and dropped", html=wrinkle_table("hang.json", [
                ("hanging", "Hanging: mean curvature", kap),
                ("dropped", "Dropped: pile height and set", rel)]), lede="")),
        ],
        footer=("<p>Sources: <code>isaacsimenvs/tasks/cloth/tools/cloth_probe.py</code> (scenarios), "
                "<code>utils/hinge_rheology.py</code> (bending model), <code>utils/cloth_geometry.py</code> (mesh), "
                "presets <code>isaacsimenvs/cfg/fabric/*_vbd.yaml</code>. Drag a stage to orbit, scroll to zoom.</p>"),
    )
    if ev_any:
        spec["sections"].append(dict(table=dict(
            title="Rigid-body policy on these cloths", html=ev_html,
            lede="The PhysX fold-pool checkpoint (<code>flat7DRABC</code>, never trained on cloth), zero-shot, 32 pinned "
                 "envs per fabric, each scored on its first episode, at the task's 4 cm keypoint tolerance and a tight "
                 "1.5 cm, 8 mm contact radius and 2 mm self-contact on both cloths. <b>Touches</b>: the four keypoints "
                 "reach their fold targets once. <b>Holds</b>: they stay within tolerance for 180 consecutive steps (3 s) "
                 "after reaching them. The rigid policy reaches the new cloth's targets far more often at 4 cm, the two "
                 "cloths tie at 1.5 cm, and on the new cloth the fold rarely survives the next 3 s. The table above shows "
                 "the cloth itself keeps those folds when left alone, so what undoes them is the hand, which never had to "
                 "hold still over a fold that does not lock in place: the part a cloth finetune has to add.")))
    pf_html, pf_any = perf_table()
    if pf_any:
        spec["sections"].insert(0, dict(table=dict(
            title="Simulation cost", html=pf_html,
            lede="Environment steps per second on one L40S (policy steps × envs), the full fold-task env with the napkin "
                 "preset and zero actions, 200 timed steps after 60 warm-up. The step time hardly changes with the env "
                 "count (the GPU is launch-bound), so throughput scales until memory runs out; memory per env is the "
                 "limit, and it is mostly the self-contact buffer (particles × triangle pairs per particle). Both new cloths "
                 "step ~12% slower than today's whatever the mesh: that is the hinge rheology's fixed per-substep work.")))
    dt = diag_table()
    if dt and ev_any:
        spec["sections"].append(dict(table=dict(
            title="Why the new cloth is easier to touch", html=dt,
            lede="The same zero-shot policy, touches at 4 cm, with one thing changed at a time. Turning today's plasticity "
                 "off changes almost nothing (46% to 48%), so the old plasticity is not what held the policy back. A 10x "
                 "stiffer membrane on the new cloth costs 13 points (84% to 71%): stretch does make the corners easier to "
                 "pull onto target. The remaining gap (48% to 71%) comes from the mesh: the 7×7 grid can only fold along "
                 "its rows 16.7 mm apart, so a fold the hand lays between rows misses, and the policy times out or drags "
                 "the sheet off the table trying.")))
    sp = Path(a.out).with_suffix(".spec.json")
    sp.write_text(json.dumps(spec))
    subprocess.run([str(ROOT / ".venv_isaaclab3/bin/python"), str(ROOT / "isaacsimenvs/tasks/cloth/tools/cloth_probe_page.py"),
                    "--spec", str(sp), "--out", a.out], check=True)


if __name__ == "__main__":
    main()
