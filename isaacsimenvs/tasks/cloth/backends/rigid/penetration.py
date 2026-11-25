"""How deep the fingers are inside the plates: PhysX's own robot-link / plate contacts, read back.

``termination.max_finger_penetration`` ends an episode as a failure once any finger link is deeper
than that inside a plate (termination reason ``finger_penetration``; a fold entered on that step does
not count, ``ClothTask.compute_dones``).

Penetration depth = ``-separation`` of the link's contacts with the plates, clamped at 0. PhysX
reports the separation between the colliders themselves, without the rest offsets: a fingertip
resting on a plate sits at ``separation`` = the plate's rest offset (the contact shell, the sheet's
half-thickness), and the depth starts only once the link is past that whole shell and inside the
slab. So it measures a finger in the visible plate, not a finger pressing into the shell. Under TGS
with 8 position iterations a finger pressed hard enough sinks through the shell and the 1-2 mm slab
(press tests, 2026-11-24); a cloth never lets it.

The links are the robot's finger segments (``RobotSpec.finger_segment_regex``). One rigid contact
view per pool shape: its sensors are the finger links of the envs carrying that shape, each filtered
to that env's own plates (``create_rigid_contact_view`` takes a list of sensor patterns, one filter
list each). PhysX needs ``PhysxContactReportAPI`` on the sensor bodies, authored before the simulation
starts (:func:`author_contact_report`); the views are built on the first read, once it runs.
"""

from __future__ import annotations

import re

import torch

from isaacsimenvs.tasks.cloth.backends.rigid import assets as fsa
from isaacsimenvs.tasks.cloth.backends.rigid.usd import bulk_edit_layer

__all__ = ["FingerPenetration", "author_contact_report", "finger_link_names"]

#: Contact-data buffer per (finger link, plate) pair.
CONTACTS_PER_PAIR = 8


def _robot_prim(b) -> tuple[str, str]:
    """(the robot prim's name, env_0's robot prim path)."""
    name = b.robot.cfg.prim_path.rsplit("/", 1)[-1]
    return name, f"/World/envs/env_0/{name}"


def finger_link_names(b) -> list[str]:
    """Robot body names (direct children of the robot prim) of the finger segments."""
    from isaaclab.sim.utils import get_current_stage
    from pxr import Usd, UsdPhysics

    regex = "|".join(f"(?:{r})" for r in b.robot_spec.finger_segment_regex.values())
    pattern = re.compile(rf"^(?:{regex})$")
    _name, robot_path = _robot_prim(b)
    stage = get_current_stage()
    names = sorted(
        str(prim.GetPath())[len(robot_path) + 1:]
        for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_path))
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and pattern.match(prim.GetName())
    )
    if not names or any("/" in name for name in names):
        raise RuntimeError(f"[finger_penetration] finger links {regex!r} under {robot_path}: {names}")
    return names


def author_contact_report(b, link_names: list[str]) -> None:
    """``PhysxContactReportAPI`` (threshold 0) on every env's finger links, in one layer."""
    from isaaclab.sim.utils import get_current_stage
    from pxr import Sdf

    robot_name, _path = _robot_prim(b)
    schemas = Sdf.TokenListOp.Create(prependedItems=["PhysxContactReportAPI"])
    with bulk_edit_layer(get_current_stage(), "finger_contact_report") as layer:
        for env in range(b.num_envs):
            for link in link_names:
                spec = Sdf.CreatePrimInLayer(layer, f"/World/envs/env_{env}/{robot_name}/{link}")
                spec.SetInfo("apiSchemas", schemas)
                Sdf.AttributeSpec(spec, "physxContactReport:threshold", Sdf.ValueTypeNames.Float).default = 0.0


class FingerPenetration:
    """``read()``: (N,) the deepest finger-in-plate penetration this physics step [m];
    ``episode_max``: its running maximum since the env's last reset."""

    def __init__(self, b, link_names: list[str]) -> None:
        self.b = b
        self.link_names = link_names
        #: Per pool shape: (its contact view, the env of each sensor row); built on the first read.
        self.views: list[tuple[object, torch.Tensor]] | None = None
        self.episode_max = torch.zeros(b.num_envs, device=b.device)

    def _build_views(self) -> None:
        from isaacsim.core.simulation_manager import SimulationManager

        b = self.b
        physics_view = SimulationManager.get_physics_sim_view()
        robot_name, _path = _robot_prim(b)
        self.views = []
        for shape, art in enumerate(b.arts):
            envs = [int(env) for env in b.pool.members(shape).tolist()]
            if not envs:
                continue
            asset_name = art.cfg.prim_path.rsplit("/", 1)[-1]
            plate_bodies = [name for name in art.body_names if name.startswith(fsa.BODY_PREFIX)]
            sensor_paths, filter_paths = [], []
            for env in envs:
                plate_paths = [f"/World/envs/env_{env}/{asset_name}/{body}" for body in plate_bodies]
                for link in self.link_names:
                    sensor_paths.append(f"/World/envs/env_{env}/{robot_name}/{link}")
                    filter_paths.append(plate_paths)
            view = physics_view.create_rigid_contact_view(
                sensor_paths, filter_patterns=filter_paths,
                max_contact_data_count=CONTACTS_PER_PAIR * len(sensor_paths) * len(plate_bodies),
            )
            if view.sensor_count != len(sensor_paths) or view.filter_count != len(plate_bodies):
                raise RuntimeError(
                    f"[finger_penetration] shape {shape}: contact view has {view.sensor_count} sensors x "
                    f"{view.filter_count} filters, expected {len(sensor_paths)} x {len(plate_bodies)}"
                )
            # The env of every sensor row, from the view's own paths (not assumed from the order).
            sensor_env = torch.tensor(
                [int(re.search(r"/env_(\d+)/", path).group(1)) for path in view.sensor_paths],
                device=b.device, dtype=torch.long,
            )
            self.views.append((view, sensor_env))
        self.physics_dt = float(b.sim.get_physics_dt())
        print(
            f"[finger_penetration] contact readout on {len(self.link_names)} finger links x the plates "
            f"({len(self.views)} view(s))",
            flush=True,
        )

    def read(self) -> torch.Tensor:
        b = self.b
        if self.views is None:
            self._build_views()
        depth = torch.zeros(b.num_envs, device=b.device)
        for view, sensor_env in self.views:
            _force, _point, _normal, separation, pair_counts, pair_starts = view.get_contact_data(dt=self.physics_dt)
            pair_counts = pair_counts.to(b.device).long()             # (sensors, plates)
            num_contacts = int(pair_counts.sum())
            if num_contacts == 0:
                continue
            # Contact k of pair p sits at pair_starts[p] + k in the data buffers.
            counts = pair_counts.flatten()
            pair_of_contact = torch.repeat_interleave(torch.arange(counts.numel(), device=b.device), counts)
            rank_in_pair = torch.arange(num_contacts, device=b.device) - (torch.cumsum(counts, 0) - counts)[pair_of_contact]
            contact_index = pair_starts.to(b.device).long().flatten()[pair_of_contact] + rank_in_pair
            contact_depth = (-separation.to(b.device).reshape(-1)[contact_index]).clamp(min=0.0)
            contact_env = sensor_env[pair_of_contact // pair_counts.shape[1]]
            depth.scatter_reduce_(0, contact_env, contact_depth, "amax")
        self.episode_max = torch.maximum(self.episode_max, depth)
        return depth

    def reset(self, env_ids: torch.Tensor) -> None:
        self.episode_max[env_ids] = 0.0
