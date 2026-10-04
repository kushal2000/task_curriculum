"""Measure the hinged-slat rigid cloth in a physics engine, without Isaac Sim.

``rigid_cloth.py`` is arithmetic: it asserts that the chain lies flat, that it folds onto its mirror,
and that slats ``i`` and ``i+2`` only just touch at the joint limit. None of that is a statement
about a *solver*. This probe loads the emitted URDF into MuJoCo -- the same engine family the task
runs on, MJWarp being MuJoCo's Warp port -- adds the free base and a floor, and measures five things
the geometry cannot tell you:

  1. **settle**    -- dropped flat, does it land flat and stay still, or does it buzz?
  2. **fold**      -- released in the folded configuration, does the crease HOLD or spring open?
                      This is the rigid analogue of the measurement behind ``ClothCfg.edge_ke``
                      (5.0 springs open after 1 step; 0.5 holds 7; 0.05 also holds 7).
  3. **drape**     -- overhanging a table edge, how far does the free half droop? 1.0 is a limp
                      cloth hanging vertically, 0.0 is a rigid plate. This is the single number
                      that says whether the hinge damping/stiffness is cloth-like, and it is the
                      knob to calibrate before anything is trained.
  4. **cost**      -- steps/s for one sheet, as a floor on what the rigid model can be.
  5. **rest_load** -- how many constraints the solver is carrying with the sheet just lying there.
                      A surface hinge rests exactly ON its one-sided limit, so every joint is a
                      permanently active constraint; a mid-plane hinge rests in the middle of its
                      range and carries none. Invisible in every other number here and paid on
                      every step of every environment.

MuJoCo only -- no Kit, no Newton, no GPU:

    .venv_isaaclab3/bin/python -m scripts.analysis.rigid_cloth_probe
    .venv_isaaclab3/bin/python -m scripts.analysis.rigid_cloth_probe --variant box3-mid
    .venv_isaaclab3/bin/python -m scripts.analysis.rigid_cloth_probe --all --json out.json

``--stiffness`` is applied to ``model.jnt_stiffness`` after compilation rather than written into the
URDF, because URDF has no spring element at all: on the Isaac Lab side the same number goes on the
articulation's actuator config (``stiffness`` against a zero target). Sweeping it here is how to
pick it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

# The shared by-path loader; `_rigid_cloth_common` explains why these scripts must not import
# `isaacsimenvs` as a package. `sys.path` first, so the import works both as
# `-m scripts.analysis.<name>` (sys.path[0] is the repo root) and as a plain script path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _rigid_cloth_common import load_module  # noqa: E402  (needs the path insert above)

rc = load_module("isaacsimenvs/tasks/cloth/utils/rigid_cloth.py")

GROUND_GEOM = "ground"
TABLE_GEOM = "table"


def _name_slat_geoms(mspec, spec) -> dict[str, list[str]]:
    """Give every slat geom a name, and report them per link.

    MuJoCo's URDF importer produces unnamed geoms (verified: `mj_id2name` returns None for all of
    them), and `add_pair` addresses geoms by name, so nothing can be paired until they have one.
    """
    out: dict[str, list[str]] = {}
    for i in range(spec.num_slats):
        link = rc.slat_link_name(i)
        names = []
        for j, geom in enumerate(mspec.body(link).geoms):
            geom.name = f"{link}_g{j}"
            names.append(geom.name)
        out[link] = names
    return out


def spec_from_args(args):
    """The chain to measure: a named variant, or one assembled from the individual flags."""
    if args.variant:
        return rc.variant(args.variant)
    return rc.uniform_chain(
        size=args.size,
        num_slats=args.num_slats,
        thickness=args.thickness,
        hinge=args.hinge,
        shape=args.shape,
    )


def _build(args, spec, extra_table: bool = False, free_base: bool = True, root_pos=None):
    """Compile the URDF into a MuJoCo model with a floor and, optionally, a clamped root + table."""
    import mujoco

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    urdf = rc.write_chain_urdf(
        spec,
        out / f"{args.variant or 'rigid_cloth'}.urdf",
        density=args.density,
        joint_damping=args.damping,
    )

    mspec = mujoco.MjSpec.from_file(str(urdf))
    mspec.option.timestep = args.dt
    root = mspec.body(rc.slat_link_name(spec.root))
    if root_pos is not None:
        root.pos = root_pos
    # A cloth is not bolted to anything, so the base is free -- MuJoCo's URDF importer welds the
    # root to the world unless told otherwise (verified: nq stays at the joint count). The drape
    # probe is the exception and wants the weld: see `probe_drape`.
    if free_base:
        root.add_freejoint()
    # The floor drops well out of the way for the drape probe. An infinite plane at z = 0 sits
    # directly under the overhang and holds it up, which reads as a perfectly cloth-stiff sheet
    # (drape_fraction 0.001) when nothing is stiff at all.
    mspec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[2.0, 2.0, 0.1],
        pos=[0.0, 0.0, -0.5 if extra_table else 0.0],
        name=GROUND_GEOM,
    )
    if extra_table:
        # A step whose top face is at z = 0 and whose edge is at x = 0, so the +x half overhangs.
        mspec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.25, 0.25, 0.10],
            pos=[-0.25, 0.0, -0.10],
            name=TABLE_GEOM,
        )

    # The URDF importer leaves every geom UNNAMED, and an explicit contact pair is addressed by
    # name, so the slats have to be named before any pair can reference them.
    slat_geoms = _name_slat_geoms(mspec, spec)

    # Explicit pairs are the whole point: MJWarp would otherwise take the element-wise MAXIMUM of
    # the two materials, so a 0.25 sheet on a 1.5 fingertip rubs at 1.5, while the cloth this
    # replaces rubs at sqrt(0.25 x 1.5) = 0.61. A pair overrides that per surface.
    if args.pairs:
        externals = {GROUND_GEOM: args.ground_friction}
        if extra_table:
            externals[TABLE_GEOM] = args.ground_friction
        # With a clamped base the root link is WELDED to the world, and MuJoCo refuses a contact
        # pair between two static bodies ("contact 0 is between two static bodies", a fatal error).
        # Skipping it costs nothing physically: a welded link cannot slide, so its friction against
        # the ground never does any work.
        welded = set() if free_base else {rc.slat_link_name(spec.root)}
        for link, geom, mu in rc.cloth_contact_pairs(spec, externals):
            if link in welded:
                continue
            for gname in slat_geoms[link]:
                # friction is [slide, slide, spin, roll, roll]; only sliding is being corrected.
                mspec.add_pair(
                    geomname1=gname,
                    geomname2=geom,
                    friction=[mu, mu, 0.005, 0.0001, 0.0001],
                )

    model = mspec.compile()

    # URDF cannot express friction for MuJoCo -- `<contact_coefficients mu=...>` is parsed and
    # DISCARDED (verified), exactly the same gap as joint stiffness -- so it is applied here.
    #
    # Left unset, every slat silently inherits MuJoCo's default 1.0 while the sheet this model
    # replaces runs at `soft_contact_mu` 0.25. That mismatch is not cosmetic: at 1.0 the
    # `cyl-mid-odd` crease holds 92.4 deg, and at the cloth's own 0.25 it slips to 77.2 deg.
    #
    # Both materials carry their RAW coefficients here -- the sheet its own `soft_contact_mu`, the
    # ground whatever that surface really is. Correcting the MIXING is the pairs' job above, not
    # something to fake by writing an already-mixed number into a material.
    for g in range(model.ngeom):
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or ""
        model.geom_friction[g, 0] = (
            args.friction if body.startswith(rc.SLAT_LINK_PREFIX) else args.ground_friction
        )

    if args.stiffness:
        for i in range(model.njnt):
            if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE:
                model.jnt_stiffness[i] = args.stiffness
    return model, mujoco.MjData(model)


def _rest_height(spec) -> float:
    """Height of the sheet's mid-surface when lying flat [m].

    Half the collider thickness in every configuration: a mid-plane hinge rests on a cylinder of
    radius ``thickness/2``, and a surface hinge on a slab of that thickness. It is NOT ``pitch/2``
    once ``thickness`` stops defaulting to ``pitch``, which is exactly what the surface hinge is for.
    """
    return 0.5 * spec.thickness


def _returning_ply(spec) -> list[int]:
    """The slats that end up as the upper ply -- the moving half MINUS the wall, if the wall is in it.

    For an odd chain the wall is the crease slat, which is in neither half, so the whole moving half
    returns. For an even chain the wall is taken out of the moving half and only the rest returns.
    Getting this wrong measures the ply height as an average of the ply and a vertical slat.
    """
    return [i for i in spec.moving if i != spec.wall]


def _joint_qpos_adr(model, num_slats: int) -> dict[int, int]:
    import mujoco

    adr = {}
    for i in range(num_slats):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, rc.slat_joint_name(i))
        if jid >= 0:
            adr[i] = model.jnt_qposadr[jid]
    return adr


def _slat_geom_z(model, data, num_slats: int):
    """World z of each slat's collider centre."""
    import mujoco

    out = []
    for i in range(num_slats):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, rc.slat_link_name(i))
        out.append(float(data.xipos[bid][2]))
    return out


def _far_edge(model, data, spec):
    """World position of the moving half's OUTERMOST EDGE.

    Not the last body's centre of mass, which is what an earlier version used. The two coincide only
    when the last slat is narrow: on ``box3-mid`` the last slat is a 49 mm plate, so its COM sits
    24.5 mm short of the edge and the drape of a plate hanging dead vertical read 0.49 instead of
    0.98 -- i.e. the metric reported the two-plate model as twice as stiff as it is, purely because
    its slats are big. Any metric compared ACROSS these variants has to be anchored to the sheet's
    geometry, not to a body's mass distribution.
    """
    import mujoco
    import numpy as np

    i = spec.moving[-1]
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, rc.slat_link_name(i))
    dx, dz = rc.chain_geom_offset(spec, i)
    local = np.array([dx + spec.widths[i] / 2.0, 0.0, dz])
    return np.asarray(data.xpos[bid]) + np.asarray(data.xmat[bid]).reshape(3, 3) @ local


def _harden_limits(model, dmax: float = 0.999, solref: float = 0.002) -> None:
    """Make joint limits nearly rigid, instead of MuJoCo's default soft ones.

    ``solimp = [dmin, dmax, width, midpoint, power]`` -- ``dmax`` is index **1**, and the ceiling on
    how hard any constraint can push. At its default 0.95 a one-sided hinge limit cannot hold a slat
    whose gravitational angular acceleration is ``3g/(2L)``, so the sheet walks straight through its
    own limits: measured 54 degrees of violation on ``box-surface``. Nothing warns; the numbers just
    come out as though the sheet were free to bend both ways.

    **0.999 is inside a narrow window, and 0.9999 is outside it.** Measured on ``box-surface``'s
    cantilever, sampled every 2 s out to 20 s:

        dmax 0.95   (default)  violation 54.7 deg, settles at drape 0.87
        dmax 0.99              violation 10.8 deg, settles at drape 0.46
        dmax 0.999             violation  1.2 deg, settles at drape 0.076
        dmax 0.9999            violation 18.0 deg, NEVER settles -- the tip oscillates between
                               -0.19 and +0.33 with joint speeds to 25 rad/s

    So the surface hinge is correctable, but only between "leaks badly" and "chatters forever", and
    the correct setting is an order of magnitude from each edge rather than a comfortable default.
    A mid-plane hinge is unaffected by the whole sweep -- drape 0.948 and zero violation at every
    value -- because nothing is resting on a limit.

    This is the knob the surface hinge DEPENDS on, and whether it survives the Isaac Lab -> Newton ->
    MJWarp path is unverified.
    """
    model.jnt_solimp[:, 1] = dmax
    model.jnt_solref[:, 0] = solref


def _ply_contacts(model, data) -> int:
    """Slat-on-slat contacts, excluding anything touching the world.

    The number that says whether a fold is held by CONTACT or only by a joint limit. Adjacent links
    in an articulation have their contact pair filtered, so a two-body sheet has no unfiltered pair
    at all and its fold rests entirely on the limit.
    """
    import mujoco

    pairs = set()
    for c in range(data.ncon):
        b1 = model.geom_bodyid[data.contact[c].geom1]
        b2 = model.geom_bodyid[data.contact[c].geom2]
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1) or ""
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2) or ""
        if n1.startswith(rc.SLAT_LINK_PREFIX) and n2.startswith(rc.SLAT_LINK_PREFIX):
            pairs.add(tuple(sorted((n1, n2))))
    return len(pairs)


def _limit_violation(model, data, spec) -> float:
    """Worst amount [rad] by which any hinge is outside its own declared range."""
    import mujoco

    worst = 0.0
    for i, adr in _joint_qpos_adr(model, spec.num_slats).items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, rc.slat_joint_name(i))
        lo, hi = model.jnt_range[jid]
        worst = max(worst, lo - data.qpos[adr], data.qpos[adr] - hi)
    return float(worst)


def _place(model, data, spec, pos, angles=None):
    import mujoco

    mujoco.mj_resetData(model, data)
    if pos is not None:
        data.qpos[0:3] = pos
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    if angles is not None:
        for i, adr in _joint_qpos_adr(model, spec.num_slats).items():
            data.qpos[adr] = angles[i]
    mujoco.mj_forward(model, data)


def _run(model, data, steps: int):
    import mujoco

    for _ in range(steps):
        mujoco.mj_step(model, data)


def _unstable(data) -> bool:
    """Did MuJoCo have to reset a diverging acceleration during this run?

    Read from ``data.warning`` rather than left to the ``MUJOCO_LOG.TXT`` the engine drops in the
    working directory: a probe whose whole job is to report numbers must not report them next to a
    silent "the simulation is unstable" in a file nobody opens. Measured to fire at hinge stiffness
    1e-2 and above (see the results doc); everything inside the usable window is clean.
    """
    import mujoco

    return bool(data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)


# ------------------------------------------------------------------------------------ probes


def probe_settle(args, spec) -> dict:
    """Drop the flat chain and check it lands flat and goes quiet."""
    import numpy as np

    model, data = _build(args, spec)
    _place(model, data, spec, [0.0, 0.0, args.drop_height])
    _run(model, data, args.settle_steps)

    z = np.array(_slat_geom_z(model, data, spec.num_slats))
    adr = _joint_qpos_adr(model, spec.num_slats)
    bend = np.array([data.qpos[a] for a in adr.values()])
    return {
        "rest_height_mm": float(z.mean() * 1e3),
        "expected_rest_height_mm": float(_rest_height(spec) * 1e3),
        "flatness_spread_mm": float((z.max() - z.min()) * 1e3),
        "max_bend_deg": float(np.abs(np.degrees(bend)).max()),
        "max_speed_mm_s": float(np.abs(data.qvel).max() * 1e3),
        "unstable": _unstable(data),
    }


def probe_fold(args, spec) -> dict:
    """Release the chain in the folded configuration and see whether the crease holds."""
    import numpy as np

    model, data = _build(args, spec)
    angles = rc.chain_fold_angles(spec)
    predicted = rc.chain_ply_gap(spec)
    # The crease target differs by hinge mode: two 90-degree joints for a mid-plane hinge, one
    # 180-degree joint for a surface hinge. Read it off the configuration rather than assumed.
    crease_joint = max(range(spec.num_slats), key=lambda i: abs(angles[i]))
    target = abs(math.degrees(angles[crease_joint]))
    # Start the fold just clear of the floor, so the first contact is ply-on-ply, not a drop.
    _place(model, data, spec, [0.0, 0.0, _rest_height(spec)], angles)

    ply = _returning_ply(spec)
    adr = _joint_qpos_adr(model, spec.num_slats)
    trace = []
    for _step in range(args.settle_steps):
        _run(model, data, 1)
        z = _slat_geom_z(model, data, spec.num_slats)
        trace.append(float(np.mean([z[i] for i in ply])))

    z = _slat_geom_z(model, data, spec.num_slats)
    gap = float(np.mean([z[i] for i in ply]) - np.mean([z[i] for i in spec.stationary]))
    crease = float(abs(math.degrees(data.qpos[adr[crease_joint]])))
    # "Sprung open" = the crease has given up more than a third of the angle it was released at.
    return {
        "ply_gap_mm": gap * 1e3,
        "predicted_ply_gap_mm": predicted * 1e3,
        "crease_angle_deg": crease,
        "crease_target_deg": target,
        "held": bool(crease > 0.67 * target and gap > 0.5 * predicted),
        # Is the fold resting on something, or hanging off a soft constraint? Adjacent links are
        # contact-filtered, so a two-body sheet has NO unfiltered pair and reports zero here -- and
        # its plies then sink into each other by however much the joint limit leaks.
        "ply_contacts": _ply_contacts(model, data),
        "limit_violation_deg": math.degrees(_limit_violation(model, data, spec)),
        "ply_height_trace_mm": [round(v * 1e3, 3) for v in trace[:: max(1, len(trace) // 10)]],
        "unstable": _unstable(data),
    }


def probe_drape(args, spec) -> dict:
    """Cantilever test: clamp the sheet at the crease, overhang the moving half, measure the droop.

    ``drape_fraction`` = tip drop / (size / 2): 1.0 is a limp cloth hanging straight down, 0.0 a
    rigid plate sticking out horizontally. This is the number that says whether the chain behaves
    like fabric, and the one ``--damping`` / ``--stiffness`` move.

    **The root is CLAMPED here, unlike every other probe.** A free sheet laid half-on and half-off a
    table edge has its centre of mass exactly over the edge, so it tips off and falls -- which is
    what the first version of this probe measured, reporting ``drape_fraction`` 0.001 and reading
    like a perfectly stiff sheet when nothing was stiff at all. Clamping one end is also what the
    standard fabric measurement (Peirce's cantilever bending length) does, for the same reason.

    Measured TWICE: once with MuJoCo's default constraint impedance and once with joint limits
    hardened (:func:`_harden_limits`). For a mid-plane hinge the two agree, because nothing is resting
    on a limit. For a surface hinge they do not, and neither number is the sheet draping.

    **Every reading carries ``settled``, and a surface hinge with hard limits never earns it.** A
    cantilever is only a measurement once it has stopped moving. Hardening the limits so the surface
    hinge obeys its own geometry leaves 15 stiff one-sided constraints chattering indefinitely --
    sampled every 2 s out to 24 s, the tip swings between -0.19 and +0.33 with joint speeds up to
    57 rad/s and excursions of 26 degrees. Reporting the 2 s snapshot as a drape figure, which an
    earlier version of this probe did, quotes one frame of a permanent oscillation as an equilibrium.
    """
    import numpy as np

    out = {"clamped": True}
    for tag, harden in (("", False), ("_hard_limits", True)):
        # Root welded so that the crease (x = 0) sits at the table edge, at table height. The root's
        # own centre is one half-width inboard of the crease, which is not a pitch when widths differ.
        model, data = _build(
            args, spec, extra_table=True, free_base=False,
            root_pos=[spec.centres[spec.root], 0.0, _rest_height(spec)],
        )
        if harden:
            _harden_limits(model)
        _place(model, data, spec, None)
        # Longer than the other probes, and AVERAGED over the last quarter rather than sampled at
        # the end.
        #
        # At the cloth's own friction the hinges are very nearly frictionless (damping 1e-5), so a
        # hanging chain is a pendulum that keeps swinging: 17 slats still show 1.4 rad/s after 8 s,
        # and that is physical, not numerical. An instantaneous reading of a swinging chain is a
        # phase sample, not a drape. The mean over the last quarter is the equilibrium it is
        # swinging about; `drape_swing` is how far it moves around that, and is the honest measure
        # of "has this settled".
        window = max(1, args.drape_steps // 4)
        _run(model, data, args.drape_steps - window)
        drops = []
        for _ in range(window):
            _run(model, data, 1)
            drops.append(_rest_height(spec) - float(_far_edge(model, data, spec)[2]))
        drop = float(np.mean(drops))
        swing = float(np.max(drops) - np.min(drops))
        adr = list(_joint_qpos_adr(model, spec.num_slats).values())
        bends = [abs(data.qpos[a]) for a in adr]
        speed = float(np.abs(data.qvel).max())
        out[f"drape_fraction{tag}"] = drop / (spec.size / 2.0)
        out[f"tip_drop_mm{tag}"] = drop * 1e3
        out[f"drape_swing{tag}"] = swing / (spec.size / 2.0)
        out[f"max_speed_rad_s{tag}"] = speed
        # Settled = the tip is swinging through less than 2% of the overhang, i.e. 1 mm on a 50 mm
        # half-sheet. Judged on the SWING rather than on the instantaneous speed, because a chain
        # passing through its equilibrium is fast and at rest is slow, and only the amplitude
        # distinguishes "hanging" from "still swinging".
        out[f"settled{tag}"] = bool(swing / (spec.size / 2.0) < 0.02)
        out[f"limit_violation_deg{tag}"] = math.degrees(_limit_violation(model, data, spec))
        # How much of the total turn happens at ONE hinge. A freely-hinged chain hangs straight down
        # from its first joint, so this is ~1 for every variant at zero stiffness; it is the number
        # that shows whether stiffness buys a smooth curve, and a 2-joint model can never get there.
        out[f"bend_concentration{tag}"] = (
            max(bends) / sum(bends) if sum(bends) > 1e-9 else float("nan")
        )
        out[f"unstable{tag}"] = _unstable(data)
    return out


def probe_cost(args, spec) -> dict:
    model, data = _build(args, spec)
    _place(model, data, spec, [0.0, 0.0, _rest_height(spec)])
    _run(model, data, 100)  # warm up contacts
    t0 = time.perf_counter()
    _run(model, data, args.cost_steps)
    dt = time.perf_counter() - t0
    return {
        "steps_per_s": args.cost_steps / dt,
        "sim_x_realtime": args.cost_steps * args.dt / dt,
        "nbody": int(model.nbody),
        "nv": int(model.nv),
    }


def probe_rest_load(args, spec) -> dict:
    """Constraints the solver carries with the sheet just lying flat.

    A surface hinge's rest pose ``qpos = 0`` is exactly ON its one-sided limit, so every joint sits
    against an active limit constraint for the whole episode; a mid-plane hinge rests in the middle
    of a symmetric range and carries none. Nothing else in this probe sees the difference -- the
    sheet lies flat either way -- and it is paid on every step of every environment.
    """
    import mujoco

    model, data = _build(args, spec)
    _place(model, data, spec, [0.0, 0.0, _rest_height(spec)])
    _run(model, data, args.settle_steps)
    limits = sum(
        1
        for i in range(data.nefc)
        if data.efc_type[i] == mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT
    )
    return {
        "nefc": int(data.nefc),
        "active_joint_limits": int(limits),
        "joints": int(spec.num_slats - 1),
    }


PROBES = {
    "settle": probe_settle,
    "fold": probe_fold,
    "drape": probe_drape,
    "cost": probe_cost,
    "rest_load": probe_rest_load,
}


def measure(args, spec) -> dict:
    pitch_note = (
        f"{spec.widths[0] * 1e3:.2f}"
        if len(set(spec.widths)) == 1
        else "+".join(f"{w * 1e3:.1f}" for w in spec.widths[:3])
    )
    dx, dz = rc.chain_fold_residual(spec)
    report = {
        "model": {
            "label": spec.label or "(unnamed)",
            "num_slats": spec.num_slats,
            "widths_mm": pitch_note,
            "thickness_mm": spec.thickness * 1e3,
            "shape": spec.shape,
            "hinge": spec.hinge,
            "wall_slat": spec.wall,
            "crease_slat": spec.crease,
            "total_mass_g": args.density * spec.size * spec.span * 1e3,
            "joint_limit_deg": math.degrees(spec.joint_limit),
            "fold_residual_mm": [dx * 1e3, dz * 1e3],
            "predicted_ply_gap_mm": rc.chain_ply_gap(spec) * 1e3,
            "damping": args.damping,
            "stiffness": args.stiffness,
        }
    }
    for name, fn in PROBES.items():
        report[name] = fn(args, spec)
    return report


def _table(reports: dict[str, dict]) -> str:
    head = (
        f"{'variant':<14} {'bod':>4} {'resid':>6} {'ply gap m/p':>12} {'cont':>5} "
        f"{'drape':>6} {'drape!':>6} {'viol':>6} {'steps/s':>8} {'lim0':>5} {'bidir':>6}"
    )
    lines = [head, "-" * len(head)]
    for name, r in reports.items():
        m, f, d, c, load = r["model"], r["fold"], r["drape"], r["cost"], r["rest_load"]
        gap = f"{f['ply_gap_mm']:.2f}/{f['predicted_ply_gap_mm']:.2f}"

        def drape(tag: str) -> str:
            # Mean over the last quarter of the run. A trailing * means the tip is still swinging
            # through more than 2% of the overhang, so the mean is an equilibrium it passes through
            # rather than one it rests at -- `drape_swing` in the JSON says how far.
            return f"{d['drape_fraction' + tag]:.3f}" + ("" if d["settled" + tag] else "*")

        lines.append(
            f"{name:<14} {c['nbody'] - 1:>4d} {m['fold_residual_mm'][0]:>6.2f} {gap:>12} "
            f"{f['ply_contacts']:>5d} {drape(''):>7} "
            f"{drape('_hard_limits'):>7} {d['limit_violation_deg']:>6.1f} "
            f"{c['steps_per_s']:>8.0f} {load['active_joint_limits']:>5d} "
            f"{('yes' if m['hinge'] == 'mid' else 'no'):>6}"
        )
    lines += [
        "",
        "bod     moving bodies (nbody minus the world)",
        "resid   in-plane fold residual [mm]: how far the folded ply misses its exact mirror",
        "ply gap measured / predicted separation of the two plies when folded [mm]",
        "cont    slat-on-slat contacts holding the fold. ZERO means the fold rests on a joint",
        "        limit alone, because adjacent links are contact-filtered",
        "drape   cantilever tip drop / (size/2) at MuJoCo's default constraint impedance",
        "drape!  the same with joint limits hardened (solimp dmax 0.9999), which is what the",
        "        surface hinge needs to obey its own geometry",
        "        * = STILL MOVING after 2 s, so not an equilibrium and not a drape figure",
        "viol    worst hinge excursion outside its declared range while draping [deg]",
        "lim0    joint-limit constraints active with the sheet lying FLAT (paid on every step)",
        "bidir   can the sheet be folded either way / used upside down",
    ]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--variant",
        default=None,
        choices=sorted(rc.VARIANTS),
        help="named approximation; overrides the individual geometry flags",
    )
    p.add_argument("--all", action="store_true", help="measure every variant and tabulate")
    p.add_argument("--size", type=float, default=0.10, help="sheet side length [m]")
    p.add_argument("--num_slats", type=int, default=rc.DEFAULT_NUM_SLATS)
    p.add_argument("--thickness", type=float, default=None, help="default: one pitch")
    p.add_argument("--density", type=float, default=2.0, help="[kg/m^2], as ClothCfg.density")
    p.add_argument("--shape", default="cylinder", choices=rc.SHAPES)
    p.add_argument(
        "--hinge",
        default="mid",
        choices=rc.HINGE_MODES,
        help="'mid' bends both ways and spends a slat per fold; 'surface' folds exactly but each "
        "hinge is one-directional",
    )
    p.add_argument("--damping", type=float, default=1.0e-5, help="hinge damping [N m s/rad]")
    p.add_argument(
        "--friction",
        type=float,
        default=rc.CLOTH_REFERENCE["soft_contact_mu"],
        help="slat sliding friction; default is the cloth's own soft_contact_mu (0.25)",
    )
    p.add_argument(
        "--ground_friction",
        type=float,
        default=rc.CLOTH_REFERENCE["shape_mu"]["table"],
        help="RAW material coefficient of the floor/table (Cloth.yaml assets.table_friction 0.5). "
        "The cloth-equivalent mixing is applied by --pairs, not by pre-mixing this number",
    )
    p.add_argument(
        "--no-pairs",
        dest="pairs",
        action="store_false",
        help="skip the explicit cloth-equivalent contact pairs and let MJWarp/MuJoCo take the "
        "element-wise MAXIMUM of the two materials, which over-grips against everything",
    )
    p.set_defaults(pairs=True)
    p.add_argument("--stiffness", type=float, default=0.0, help="hinge spring [N m/rad]")
    p.add_argument("--dt", type=float, default=1.0 / 240.0)
    p.add_argument("--drop_height", type=float, default=0.05)
    p.add_argument("--settle_steps", type=int, default=480, help="2 s at the default dt")
    p.add_argument(
        "--drape_steps", type=int, default=1920, help="8 s: a cantilever swings before it hangs"
    )
    p.add_argument("--cost_steps", type=int, default=2000)
    p.add_argument("--out_dir", default="/tmp/rigid_cloth_probe")
    p.add_argument("--json", default=None, help="also write the report here")
    args = p.parse_args()

    if args.all:
        reports = {}
        for name in rc.VARIANTS:
            args.variant = name
            reports[name] = measure(args, rc.variant(name))
        print(_table(reports))
        payload: dict = {"variants": reports}
    else:
        spec = spec_from_args(args)
        print(spec.describe(), "\n")
        payload = measure(args, spec)
        print(json.dumps(payload, indent=2, sort_keys=False))

    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
