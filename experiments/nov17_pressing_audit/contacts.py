"""Robot/table normal contacts on individual rigid cloth links, sampled at physics dt.

Enable reporting after scene authoring and before the first sim.reset(). Construct
RigidClothContacts after ClothEnv initialization, then sample after every
scene.update(dt), including both 120 Hz steps of each 60 Hz policy action. No
materials, sleep thresholds, actions or physics stepping are changed here.
"""
from __future__ import annotations

import re

import torch

from isaacsimenvs.tasks.cloth.mdp.geometry import rotations

_ENV_ID = re.compile(r'/env_(\d+)(?:/|$)')


def _env_id(path):
    match = _ENV_ID.search(str(path))
    if match is None:
        raise ValueError(f'Cannot identify contact environment from {path}')
    return int(match.group(1))


def enable_contact_reporting(backend):
    """Apply contact-report API to cloth BODY prims before PhysX starts; leave sleep unchanged."""
    import carb
    from isaaclab.sim.utils import get_current_stage
    from pxr import PhysxSchema, UsdPhysics

    carb.settings.get_settings().set_bool('/physics/disableContactProcessing', False)
    stage = get_current_stage()
    paths = []
    for group in backend.groups:
        for env_id in group.env_ids.tolist():
            for plate in group.design.plates:
                path = f'/World/envs/env_{env_id}/RigidCloth_{group.index}/{plate.name}'
                prim = stage.GetPrimAtPath(path)
                if not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    raise RuntimeError(f'Contact sensor is not a rigid cloth body: {path}')
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
                paths.append(path)
    backend._press_audit_report_paths = tuple(paths)
    return paths


def unpack_contacts(data, force_matrix, *, capacity):
    """Compact PhysX's ragged buffers; verify their vectors against its pair matrix.

    The matrix reports normal force ON the sensor body. Both possible buffer-normal
    conventions are checked against it; an opposite sign is corrected explicitly.
    No norm of an aggregate vector is used as the scalar pressure measurement.
    """
    force, points, normals, separation, counts, starts = data
    if counts.shape != force_matrix.shape[:2] or starts.shape != counts.shape:
        raise RuntimeError('Contact buffer and force matrix pair layouts disagree')
    counts, starts = counts.long().flatten(), starts.long().flatten()
    total = int(counts.sum())
    if bool((counts < 0).any()) or total >= capacity:
        raise RuntimeError(f'Contact capacity reached ({total}/{capacity}); increase capacity and repeat audit')
    active = counts > 0
    if bool(((starts[active] < 0) | (starts[active]+counts[active] > capacity)).any()):
        raise RuntimeError('Contact buffer spans exceed capacity; audit data would be incomplete')
    pair = torch.repeat_interleave(torch.arange(len(counts), device=counts.device), counts)
    block_start = counts.cumsum(0)-counts
    offset = torch.arange(total, device=counts.device)-torch.repeat_interleave(block_start, counts)
    indices = starts[pair]+offset
    scalar = force.flatten()[indices]
    normal = normals[indices]
    vector = scalar[:, None]*normal
    aggregate = torch.zeros_like(force_matrix).reshape(-1, 3).index_add_(0, pair, vector)
    expected = force_matrix.reshape(-1, 3)
    # Use per-component tolerance, retaining sensitivity to small textile forces.
    positive = torch.allclose(aggregate, expected, rtol=2e-3, atol=2e-5)
    negative = torch.allclose(-aggregate, expected, rtol=2e-3, atol=2e-5)
    if not positive and not negative:
        error = float((aggregate-expected).abs().max())
        raise RuntimeError(f'Contact records do not reproduce pair forces (max error {error:g} N); possible overflow')
    sign = 1. if positive else -1.
    vector, normal = vector*sign, normal*sign
    position = points[indices]
    if not bool(torch.isfinite(vector).all() & torch.isfinite(position).all() & torch.isfinite(scalar).all()):
        raise RuntimeError('Nonfinite contact data; audit cannot classify this frame')
    return {'sensor_index': pair//force_matrix.shape[1], 'partner_index': pair%force_matrix.shape[1],
                'force_magnitude': scalar.abs(), 'force_vector': vector, 'position': position,
                'normal': normal, 'separation': separation.flatten()[indices], 'normal_buffer_sign': sign}


def contact_rest_positions(position, sensor_index, body_position, body_quaternion):
    """World contact points -> cloth-rest coordinates through actual BODY LINK poses."""
    q = body_quaternion[sensor_index]
    conjugate = q*q.new_tensor([1., -1., -1., -1.])
    return rotations.apply(conjugate, position-body_position[sensor_index])


class RigidClothContacts:
    """Explicit one-to-many filtered contacts; never wildcard-match multiple robot bodies.

    cloth_body is the canonical plate index within that environment's design.
    partner_type is 0 for any robot link and 1 for the table. All positions and
    vectors use world coordinates, except cloth_rest_position. Forces are Newtons,
    derived from the current physics-step impulses using cfg.sim.dt.
    """

    def __init__(self, backend, *, capacity=16384):
        from isaacsim.core.simulation_manager import SimulationManager

        self.backend, self.dt, self.capacity = backend, float(backend.cfg.sim.dt), int(capacity)
        if self.capacity < 1 or self.dt <= 0:
            raise ValueError('Contact capacity and physics dt must be positive')
        expected_paths = set(getattr(backend, '_press_audit_report_paths', ()))
        if not expected_paths:
            raise RuntimeError('Call enable_contact_reporting(backend) before simulation initialization')
        robot_paths = {}
        for paths in backend.robot.root_physx_view.link_paths:
            env = _env_id(paths[0])
            if any(_env_id(path) != env for path in paths):
                raise RuntimeError('Robot rigid link row spans multiple environments')
            robot_paths[env] = list(paths)
        table_paths = {_env_id(path): path for path in backend.table.root_physx_view.prim_paths}
        self.num_robot_links = len(next(iter(robot_paths.values())))
        if any(len(paths) != self.num_robot_links for paths in robot_paths.values()):
            raise RuntimeError('Contact filter counts must match across environments')
        requested, filters, by_path = [], [], {}
        for group_index, group in enumerate(backend.groups):
            links = group.articulation.root_physx_view.link_paths
            for row, env in enumerate(group.env_ids.tolist()):
                for plate, body in enumerate(group.body_ids):
                    path = links[row][body]
                    if _env_id(path) != env:
                        raise RuntimeError('Cloth rigid link row disagrees with backend environment mapping')
                    requested.append(path)
                    filters.append(robot_paths[env]+[table_paths[env]])
                    by_path[path] = (env, plate, group_index, row, body)
        if set(requested) != expected_paths or len(requested) != len(expected_paths):
            raise RuntimeError('Initialized cloth body paths differ from enabled contact reporters')
        self.view = SimulationManager.get_physics_sim_view().create_rigid_contact_view(
            requested, filter_patterns=filters, max_contact_data_count=self.capacity)
        self.sensor_paths = list(self.view.sensor_paths)
        if set(self.sensor_paths) != expected_paths or len(self.sensor_paths) != len(expected_paths):
            raise RuntimeError('PhysX contact sensor paths do not match requested cloth bodies')
        if self.view.filter_count != self.num_robot_links+1:
            raise RuntimeError('PhysX contact filters do not match robot links plus table')
        device = backend.device
        rows = torch.tensor([by_path[path] for path in self.sensor_paths], device=device, dtype=torch.long)
        self.env_ids, self.plate_ids = rows[:, 0], rows[:, 1]
        self._pose_rows = [(torch.nonzero(rows[:, 2] == i, as_tuple=False).flatten(),
                            rows[rows[:, 2] == i, 3], rows[rows[:, 2] == i, 4])
                           for i in range(len(backend.groups))]
        self.metadata = {'dt': self.dt, 'capacity': self.capacity, 'sensor_paths': self.sensor_paths,
                             'robot_link_names': list(backend.robot.body_names),
                             'env_id': self.env_ids.cpu().tolist(), 'cloth_body': self.plate_ids.cpu().tolist(),
                             'partner_types': {'robot': 0, 'table': 1}, 'force_components': 'normal only',
                             'sleep_threshold_changed': False}

    def sample(self):
        """Read the just-completed physics step; raise instead of silently losing contacts."""
        matrix = self.view.get_contact_force_matrix(dt=self.dt)
        data = self.view.get_contact_data(dt=self.dt)
        records = unpack_contacts(data, matrix, capacity=self.capacity)
        self.normal_buffer_sign = records.pop('normal_buffer_sign')
        sensor = records['sensor_index']
        pos = matrix.new_empty((len(self.sensor_paths), 3))
        quat = matrix.new_empty((len(self.sensor_paths), 4))
        for group, (indices, rows, bodies) in zip(self.backend.groups, self._pose_rows, strict=True):
            pos[indices] = group.articulation.data.body_link_pos_w[rows, bodies]
            quat[indices] = group.articulation.data.body_link_quat_w[rows, bodies]
        records['env_id'] = self.env_ids[sensor]
        records['cloth_body'] = self.plate_ids[sensor]
        records['partner_type'] = (records['partner_index'] == self.num_robot_links).long()
        records['cloth_rest_position'] = contact_rest_positions(records['position'], sensor, pos, quat)
        # Separate net vectors are useful for table-support calibration; point
        # magnitudes above remain the pressure metric when opposing contacts cancel.
        records['robot_force_by_sensor'] = matrix[:, :self.num_robot_links].sum(dim=1).clone()
        records['table_force_by_sensor'] = matrix[:, self.num_robot_links].clone()
        self.metadata['force_matrix_verified'] = True
        self.metadata['normal_buffer_sign'] = self.normal_buffer_sign
        return records
