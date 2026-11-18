"""Press a scripted probe into a rigid cloth and measure how far it penetrates.

    # one case (optionally with video)
    .venv/isaacsim/bin/python isaacsimenvs/scripts/physics_test/press_test.py --headless \\
        --config <run>/.hydra/config.yaml --design strip_4 --density 0.1 --size 0.2 0.2 \\
        --site "centre panel_1" --forces 5 --video press.mp4 --out press.json [key=value ...]
    # the run's whole randomization: every DR corner + random draws, at the size-range extremes
    ... --design strip_4 --dr_random 64 --thickness 0.002 --forces 50 --probe_mass 0.5 --out sweep.json

The scene is the training scene: ``--config`` (a run's ``.hydra/config.yaml``) plus ``key=value``
overrides, built by the same loader as ``eval_cloth.py`` and the same ``ClothEnv``. The robot's
colliders are disabled (it stays in the scene but touches nothing), and each env gets a probe -- a
sphere the size of a finger pad -- that does the pressing. Reset noise and random yaw are off.

Materials are written per env through the training randomization code (and read back):
* default: one material for every env -- the config's nominal values, overridden by ``--density``
  and ``--material '{"bending_rigidity": 1e-4, ...}'``; one sheet size (``--size``);
* ``--dr_random N``: every corner of the run's randomization box plus N random draws from it
  (``dr_samples.py``), each at every corner of the size grid (``--size_levels`` over the config's
  length/width ranges).
Every (material, size) is pressed at every site (each panel centre and hinge midpoint, or those
matching ``--site``) with every force in ``--forces``.

Script, identical for all envs: settle the cloth; place the probe ``--start_gap`` above its site and
lower it at ``--approach_speed`` (velocity set directly: no force, no impact) until 1 mm above the
face; push straight down with a force ramping 0 -> F over ``--ramp_s`` and held for ``--hold_s``.
The probe is force-controlled (gravity off, a weak x/y spring keeps it over the site).

Measured on the pressed panel every physics step: depth of the probe's lowest point below the
panel's top face (``through`` = deeper than the panel is thick), whether the probe slid off the
panel, panel corners below the table top, and hinge-limit violations. An env FAILS if the probe goes
through or its steady depth (median of the last 0.5 s) exceeds ``--max_depth_mm``.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np

from isaacsimenvs.scripts.evaluation.eval_cloth import launch
from isaacsimenvs.scripts.physics_test import dr_samples
from isaacsimenvs.scripts.physics_test.common import (add_material_args, add_video_args, configure,
                                                      disable_robot, set_camera)


def parse():
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_material_args(p)
    p.add_argument("--site", help="press only the sites whose label contains this, e.g. 'centre panel_1'")
    p.add_argument("--forces", type=float, nargs="+", default=(5,), help="press forces [N]")
    p.add_argument("--probe_radius", type=float, default=0.008)
    p.add_argument("--probe_mass", type=float, default=0.05)
    p.add_argument("--settle_s", type=float, default=0.5)
    p.add_argument("--start_gap", type=float, default=0.02)
    p.add_argument("--approach_speed", type=float, default=0.05)
    p.add_argument("--ramp_s", type=float, default=0.5)
    p.add_argument("--hold_s", type=float, default=2.0)
    p.add_argument("--max_depth_mm", type=float, default=2.0, help="fail above this steady depth")
    p.add_argument("--out", type=Path, help="JSON results path")
    add_video_args(p, slowmo=4.0)
    p.add_argument("overrides", nargs="*", help="key=value edits to --config")
    AppLauncher.add_app_launcher_args(p)
    return p.parse_args()


def make_probe_env(cfg, radius: float, mass: float, render_mode=None):
    """ClothEnv with a probe sphere per env and a robot that collides with nothing."""
    from isaacsimenvs.tasks.cloth.cloth_env import ClothEnv

    class ProbeEnv(ClothEnv):
        def _setup_scene(self):
            super()._setup_scene()
            import isaaclab.sim as sim_utils
            from isaaclab.assets import RigidObject, RigidObjectCfg
            from isaaclab.sim.utils import get_current_stage

            from isaacsimenvs.tasks.cloth.backends.rigid.scene import bind_material_roots

            stage = get_current_stage()
            disabled = disable_robot(stage, self.scene.env_prim_paths, hide=render_mode is not None)
            print(f"[press_test] disabled {disabled} robot colliders", flush=True)

            self.probe = RigidObject(RigidObjectCfg(
                prim_path="/World/envs/env_.*/Probe",
                spawn=sim_utils.SphereCfg(
                    radius=radius,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=True, max_depenetration_velocity=1000.0,  # as the robot links
                        solver_position_iteration_count=8, solver_velocity_iteration_count=0),
                    mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.5)),
            ))
            self.scene.rigid_objects["probe"] = self.probe
            # The robot's friction material, so probe-cloth contact matches finger-cloth contact.
            bind_material_roots(stage, (f"{p}/Probe" for p in self.scene.env_prim_paths),
                                "/World/ClothMaterials/robot")

    return ProbeEnv(cfg, render_mode=render_mode)


def design_sites(spec, site_filter=None):
    """(label, plate index, rest-frame xy) for every panel centre and hinge midpoint."""
    names = [p.name for p in spec.plates]
    sites = [(f"centre {p.name}", k, np.mean(p.vertices, axis=0)) for k, p in enumerate(spec.plates)]
    sites += [(f"hinge {h.parent}|{h.child}", names.index(h.parent), np.asarray(h.anchor[:2])) for h in spec.hinges]
    return [s for s in sites if not site_filter or site_filter in s[0]]


def main() -> None:
    args = parse()
    if args.video:
        args.enable_cameras = True
    app = launch(args)

    import torch

    from isaacsimenvs.tasks.cloth.backends.rigid import designs
    from isaacsimenvs.tasks.cloth.mdp.geometry import rotations

    logging.getLogger("isaaclab.assets.rigid_object.rigid_object").setLevel(logging.ERROR)

    cfg, values, labels, ranges = configure(args)
    rc = cfg.rigid_cloth
    rc.designs = [f"{args.design}.yaml"]
    design = designs.get_design(f"{args.design}.yaml")
    thickness = float(rc.thickness)
    forces = [float(f) for f in args.forces]
    # Sites from a nominal resolve: the labels and their count do not depend on the size.
    S = len(design_sites(design.resolve(0.2, 0.2, thickness), args.site))
    if S == 0:
        raise ValueError(f"no site label contains {args.site!r}")
    G, M, F = len(rc.geometries), len(values), len(forces)
    # env e: geometry e % G (backend: geometry_of with one design), then block e // G = (m, s, f).
    cfg.scene.num_envs = G * M * S * F
    print(f"[press_test] {args.design}: {M} materials x {S} sites x {F} forces x {G} geometries "
          f"= {cfg.scene.num_envs} envs", flush=True)

    if args.video:
        set_camera(cfg, args, (0.30, -0.30, 0.20))
    env = make_probe_env(cfg, args.probe_radius, args.probe_mass, "rgb_array" if args.video else None)
    n, dev, dt = env.num_envs, env.device, env.physics_dt
    probe = env.probe
    (group,) = env.backend.groups
    rows = env.backend._env_local

    e_all = torch.arange(n, device=dev)
    block = e_all // G
    mat_of = block // (S * F)
    site_idx = (block // F) % S
    force_of = torch.tensor(forces, device=dev)[block % F]
    dims = env.backend._dims.cpu().numpy()

    # Per-env geometry from that env's own sheet size.
    sites_e, plate_of, site_rest, polys, corner_list = [], [], [], [], []
    for e in range(n):
        spec = design.resolve(*dims[e])
        label, k, xy = design_sites(spec, args.site)[int(site_idx[e])]
        sites_e.append(label)
        plate_of.append(k)
        site_rest.append(xy)
        verts = [np.asarray(p.vertices) for p in spec.plates]
        V = max(len(v) for v in verts)
        # Pad to equal corner counts by repeating the last corner: harmless for the inside test
        # and for the lowest point.
        verts = [np.concatenate([v, np.repeat(v[-1:], V - len(v), 0)]) for v in verts]
        polys.append(verts[k])
        corner_list.append(np.stack([np.concatenate([np.c_[v, np.full(V, z)] for z in (-thickness / 2, thickness / 2)])
                                     for v in verts]))
    plate_of = torch.tensor(plate_of, device=dev)
    site_rest = torch.tensor(np.array(site_rest), dtype=torch.float32, device=dev)
    poly = torch.tensor(np.array(polys), dtype=torch.float32, device=dev)
    corners = torch.tensor(np.array(corner_list), dtype=torch.float32, device=dev)  # (n, P, 2V, 3)
    spec0 = design.resolve(*dims[0])
    lim = torch.tensor([[math.radians(a) for a in (rc.hinge_limits_deg if h == spec0.fold_hinge else hinge.limits_deg)]
                        for h, hinge in enumerate(spec0.hinges)], device=dev)

    def plates():
        art = group.articulation.data
        return art.body_link_pos_w[rows][:, group.body_ids], art.body_link_quat_w[rows][:, group.body_ids]

    def physics_step(force=None):
        if force is not None:
            probe.set_external_force_and_torque(force[:, None], torch.zeros_like(force)[:, None], is_global=True)
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=dt)

    env.reset()
    with torch.inference_mode():
        dr_samples.apply_materials(env.backend, e_all, values[mat_of.cpu().numpy()])
        for _ in range(int(args.settle_s / dt)):
            physics_step(torch.zeros(n, 3, device=dev))

        pos, quat = plates()
        p_k, q_k = pos[e_all, plate_of], quat[e_all, plate_of]
        top_rest = torch.cat([site_rest, torch.full((n, 1), thickness / 2, device=dev)], 1)
        site_w = p_k + rotations.apply(q_k, top_rest)
        start = site_w + torch.tensor([0.0, 0.0, args.probe_radius + args.start_gap], device=dev)
        probe.write_root_pose_to_sim(torch.cat([start, torch.tensor([1.0, 0, 0, 0], device=dev).expand(n, 4)], 1))
        probe.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev))
        hold_xy = start[:, :2].clone()

        frames = []
        stride = max(1, round(1 / (dt * 30 * args.slowmo)))
        if args.video:
            for _ in range(4):  # first renders are black or grainy while the renderer warms up
                env.render()
            frames.append(env.render())

        # Approach at constant speed, no force, until 1 mm above the face.
        for i in range(int(2 * args.start_gap / args.approach_speed / dt) + 1):
            pos, quat = plates()
            face = (pos[e_all, plate_of] + rotations.apply(quat[e_all, plate_of], top_rest))[:, 2]
            gap = probe.data.root_pos_w[:, 2] - args.probe_radius - face
            moving = gap > 0.001
            if not bool(moving.any()):
                break
            vel = torch.zeros(n, 6, device=dev)
            vel[:, 2] = torch.where(moving, -torch.clamp(gap - 0.001, max=args.approach_speed * dt) / dt, 0.0)
            probe.write_root_velocity_to_sim(vel)
            physics_step(torch.zeros(n, 3, device=dev))
            if args.video and i % stride == 0:
                frames.append(env.render())
        probe.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev))

        steps = int((args.ramp_s + args.hold_s) / dt)
        last = int(0.5 / dt)
        depth_hist, inside_hist = [], []
        table_worst = torch.zeros(n, device=dev)
        limit_worst = torch.zeros(n, device=dev)
        up = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)
        top_local = torch.tensor([0.0, 0.0, thickness / 2], device=dev).expand(n, 3)
        for i in range(steps):
            t = (i + 1) * dt
            c, v = probe.data.root_pos_w, probe.data.root_lin_vel_w
            fz = -force_of * min(1.0, t / args.ramp_s) - 0.2 * v[:, 2]
            fxy = 20.0 * (hold_xy - c[:, :2]) - 0.3 * v[:, :2]
            physics_step(torch.cat([fxy, fz[:, None]], 1))

            pos, quat = plates()
            c = probe.data.root_pos_w
            p_k, q_k = pos[e_all, plate_of], quat[e_all, plate_of]
            normal = rotations.apply(q_k, up)
            depth = ((p_k + rotations.apply(q_k, top_local) - (c - args.probe_radius * normal)) * normal).sum(-1)
            rest = rotations.apply(q_k * q_k.new_tensor([1.0, -1.0, -1.0, -1.0]), c - p_k)[:, :2]
            edge = torch.roll(poly, -1, 1) - poly
            cross = edge[..., 0] * (rest[:, None, 1] - poly[..., 1]) - edge[..., 1] * (rest[:, None, 0] - poly[..., 0])
            depth_hist.append(depth.clamp_min(0))
            inside_hist.append((cross >= 0).all(1) | (cross <= 0).all(1))
            if args.video and i % stride == 0:
                frames.append(env.render())

            state, _ = env._now()
            world_corners = pos[:, :, None] + rotations.apply(quat[:, :, None], corners)
            table_worst = torch.maximum(table_worst, (state.table_top_z.reshape(n, 1) - world_corners[..., 2].amin(2)).clamp_min(0).amax(1))
            q = group.articulation.data.joint_pos[rows][:, group.joint_ids]
            limit_worst = torch.maximum(limit_worst, torch.maximum(lim[:, 0] - q, q - lim[:, 1]).clamp_min(0).amax(1))

    D = torch.stack(depth_hist).cpu()
    I = torch.stack(inside_hist).cpu()
    on = torch.where(I, D, torch.full_like(D, float("nan")))
    steady = on[-last:].nanmedian(0).values
    deepest = on.nan_to_num(0).amax(0)
    through = (on > thickness).any(0)
    slid = ~I[-last:].all(0)

    per_env = []
    for i in range(n):
        m = int(mat_of[i])
        st = float(steady[i]) * 1000
        failed = bool(through[i]) or (math.isfinite(st) and st > args.max_depth_mm)
        per_env.append({"env": i, "site": sites_e[i], "force_N": float(force_of[i]),
                        "kind": labels[m], "material_index": m,
                        "material": {p: float(values[m][j]) for j, p in enumerate(dr_samples.PARAMS)},
                        "length": float(dims[i][0]), "width": float(dims[i][1]),
                        "steady_depth_mm": round(st, 2), "max_depth_mm": round(float(deepest[i]) * 1000, 2),
                        "through": bool(through[i]), "slid_off": bool(slid[i]), "failed": failed,
                        "panel_below_table_mm": round(float(table_worst[i]) * 1000, 2),
                        "hinge_limit_excess_deg": round(math.degrees(float(limit_worst[i])), 1)})
    n_fail = sum(e["failed"] for e in per_env)
    worst = max(per_env, key=lambda e: (e["failed"], e["through"], e["max_depth_mm"]))
    result = {"test": "press", "config": str(args.config), "overrides": list(args.overrides), "design": args.design,
              "thickness_m": thickness, "dr_ranges": ranges, "dr_random": args.dr_random,
              "probe_radius_m": args.probe_radius, "probe_mass_kg": args.probe_mass, "forces_N": forces,
              "max_depth_mm": args.max_depth_mm, "physics_dt": dt, "num_envs": n,
              "failed_envs": n_fail, "failed_corner_envs": sum(e["failed"] for e in per_env if e["kind"] == "corner"),
              "failed_random_envs": sum(e["failed"] for e in per_env if e["kind"] == "random"),
              "random_envs": sum(e["kind"] == "random" for e in per_env),
              "failing_ranges": dr_samples.describe_failures(per_env, "failed"),
              "worst_env": worst, "per_env": per_env}

    print("\n" + "=" * 96)
    print(f"press_test  {args.design}  t={thickness*1000:.1f} mm  probe {args.probe_mass} kg  forces {forces} N  "
          f"fail: through or steady depth > {args.max_depth_mm} mm")
    print(f"failed {n_fail}/{n} envs  (corners {result['failed_corner_envs']}, random {result['failed_random_envs']}"
          f"/{result['random_envs']})")
    dr_samples.print_failing_ranges(result["failing_ranges"])
    print(f"worst: env {worst['env']} {worst['site']} @ {worst['force_N']:.0f} N, "
          f"{'THROUGH' if worst['through'] else str(worst['steady_depth_mm']) + ' mm'}, "
          f"{worst['length']*100:.0f}x{worst['width']*100:.0f} cm, material {worst['material']}")
    if n == 1:
        series = D[:, 0].numpy() * 1000
        marks = [int(k / dt) - 1 for k in (0.1, 0.25, 0.5, 1.0, 2.0, 2.5) if int(k / dt) - 1 < len(series)]
        print("depth over time: " + ", ".join(f"t={(m_ + 1) * dt:.2f}s {series[m_]:.1f} mm" for m_ in marks))
        result["depth_series_mm"] = [round(float(x), 2) for x in series]
    print("=" * 96 + "\n", flush=True)
    if args.video:
        import imageio.v2 as imageio
        args.video.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(args.video, frames, fps=30, macro_block_size=1)
        print(f"[press_test] wrote {args.video} (env {args.video_env}, {args.slowmo:g}x slow motion)", flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1))
        print(f"[press_test] wrote {args.out}", flush=True)

    env.close()
    if app is not None:
        app.close()


if __name__ == "__main__":
    sys.exit(main())
