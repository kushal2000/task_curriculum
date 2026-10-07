"""Measure, on PhysX, the two contact facts the rigid-cloth port depends on.

1. **Friction combine modes survive a tensor-API material write.** The PhysX chain matches the
   cloth's per-pair friction with combine modes (slat ``min``, fingertip ``max``), and its DR rewrites
   friction through ``root_physx_view.set_material_properties``. If that write creates fresh materials
   with the default ``average`` mode, the pairs silently go back to averaging.
2. **A rest offset raises the resting height by exactly itself** (the 7 mm the chain needs to sit
   where the cloth's 8 mm particles do).

Each case slides a 10 x 10 x 0.2 cm plate along a fixed 'table' plate at 1 m/s and reads the
deceleration back as an effective friction coefficient, ``mu = -dv/dt / g``.

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python scripts/analysis/physx_contact_probe.py
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402

DT = 1.0 / 120.0
G = 9.81


def plate(path, *, size, pos, mu, mode, kinematic, rest_offset=0.0, contact_offset=0.002):
    return RigidObject(
        RigidObjectCfg(
            prim_path=path,
            spawn=sim_utils.CuboidCfg(
                size=size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=kinematic),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    rest_offset=rest_offset, contact_offset=contact_offset
                ),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=mu,
                    dynamic_friction=mu,
                    friction_combine_mode=mode,
                    restitution_combine_mode="min",
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
        )
    )


def main() -> None:
    sim = SimulationContext(SimulationCfg(dt=DT, device="cuda:0"))
    # Lanes 1 m apart in y; each lane: a fixed base plate and a sliding plate on top.
    lanes = {
        # name: (base mu, base mode, slider mu, slider mode, slider rest offset)
        "slat_min_on_table_avg": (0.5, "average", 0.354, "min", 0.0),
        "slat_min_on_table_avg_DRwrite": (0.5, "average", 0.354, "min", 0.0),
        "tip_max_on_slat_min": (0.354, "min", 0.612, "max", 0.0),
        "tip_max_on_table_avg": (0.5, "average", 0.612, "max", 0.0),
        "default_avg_avg": (0.5, "average", 0.354, "average", 0.0),
        "slat_restoffset_7mm": (0.5, "average", 0.354, "min", 0.007),
    }
    objs = {}
    for i, (name, (bmu, bmode, smu, smode, ro)) in enumerate(lanes.items()):
        y = float(i)
        base = plate(
            f"/World/base_{i}", size=(2.0, 0.3, 0.02), pos=(0.5, y, 0.0), mu=bmu, mode=bmode,
            kinematic=True,
        )
        slider = plate(
            f"/World/slider_{i}", size=(0.1, 0.1, 0.002), pos=(0.0, y, 0.011 + ro + 0.003),
            mu=smu, mode=smode, kinematic=False, rest_offset=ro, contact_offset=ro + 0.002,
        )
        objs[name] = (base, slider)
    sim.reset()
    for _ in range(60):  # settle
        sim.step()
        for b, s in objs.values():
            s.update(DT)

    # Rewrite the DR lane's slider friction through the tensor API, to the SAME value. If the
    # write resets its combine mode to average, this lane reads (0.354 + 0.5) / 2 = 0.427.
    view = objs["slat_min_on_table_avg_DRwrite"][1].root_physx_view
    mats = view.get_material_properties()
    mats[..., 0] = 0.354
    mats[..., 1] = 0.354
    view.set_material_properties(mats, torch.arange(view.count, dtype=torch.int32))

    for name, (b, s) in objs.items():
        z = float(s.data.root_pos_w[0, 2])
        print(f"[probe] {name:32s} rest z = {z * 1e3:7.3f} mm (no-offset would be 12.000)")

    for name, (b, s) in objs.items():
        vel = torch.zeros((1, 6), device="cuda:0")
        vel[0, 0] = 1.0
        s.write_root_velocity_to_sim(vel)
    v0 = {}
    for k in range(20):
        sim.step()
        for name, (b, s) in objs.items():
            s.update(DT)
            if k == 4:
                v0[name] = float(s.data.root_lin_vel_w[0, 0])
    for name, (b, s) in objs.items():
        v1 = float(s.data.root_lin_vel_w[0, 0])
        mu = (v0[name] - v1) / (15 * DT) / G
        print(f"[probe] {name:32s} effective mu = {mu:.3f}  (v {v0[name]:.3f} -> {v1:.3f} m/s)")


if __name__ == "__main__":
    main()
    import os

    os._exit(0)
