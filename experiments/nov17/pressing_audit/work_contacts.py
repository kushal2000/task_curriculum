"""Optional friction and mechanical-state recording for the pressing audit."""
from __future__ import annotations

import torch


def unpack_friction(data, capacity):
    force, point, counts, starts = data
    counts, starts = counts.long().flatten(), starts.long().flatten()
    total = int(counts.sum())
    active = counts > 0
    if bool((counts < 0).any()) or total >= capacity:
        raise RuntimeError('Friction contact capacity reached; repeat with larger buffer')
    if bool(((starts[active] < 0) | (starts[active]+counts[active] > capacity)).any()):
        raise RuntimeError('Friction spans exceed the contact buffer')
    pair = torch.repeat_interleave(torch.arange(len(counts), device=counts.device), counts)
    offset = torch.arange(total, device=counts.device)-torch.repeat_interleave(counts.cumsum(0)-counts, counts)
    index = starts[pair]+offset
    vector, position = force[index].clone(), point[index].clone()
    if not bool(torch.isfinite(vector).all() & torch.isfinite(position).all()):
        raise RuntimeError('Nonfinite friction records')
    return pair, vector, position


def friction_records(contacts):
    """Read AFTER normal records were compacted: PhysX reuses count/start buffers."""
    view = contacts.view
    pair, force, position = unpack_friction(view.get_friction_data(dt=contacts.dt), contacts.capacity)
    sensor, partner = pair//view.filter_count, pair%view.filter_count
    return dict(env_id=contacts.env_ids[sensor], cloth_body=contacts.plate_ids[sensor],
                partner_type=(partner == contacts.num_robot_links).long(), partner_index=partner,
                force_vector=force, position=position)


def mechanical_state(backend, max_plates, max_joints, origins):
    n, dev = backend.num_envs, backend.device
    com_pose = torch.zeros(n, max_plates, 7, device=dev)
    com_pose[..., 3] = 1
    velocity = torch.zeros(n, max_plates, 6, device=dev)
    qd = torch.zeros(n, max_joints, device=dev)
    for group in backend.groups:
        ids, bodies, data = group.env_ids, group.body_ids, group.articulation.data
        com_pose[ids, :len(bodies), :3] = data.body_com_pos_w[:, bodies]-origins[ids, None]
        com_pose[ids, :len(bodies), 3:] = data.body_com_quat_w[:, bodies]
        velocity[ids, :len(bodies)] = data.body_com_vel_w[:, bodies]
        qd[ids, :len(group.joint_ids)] = data.joint_vel[:, group.joint_ids]
    robot = backend.robot.data
    return dict(panel_com_pose=com_pose, panel_com_velocity=velocity, hinge_velocity=qd,
                robot_body_pose=torch.cat((robot.body_link_pos_w-origins[:,None],robot.body_link_quat_w),-1),
                robot_com_pose=torch.cat((robot.body_com_pos_w-origins[:,None],robot.body_com_quat_w),-1),
                robot_com_velocity=robot.body_com_vel_w)


def material_metadata(group, row):
    view, bodies, joints = group.articulation.root_physx_view, group.body_ids, group.joint_ids
    return dict(plate_masses=view.get_masses()[row, bodies].cpu().tolist(),
                plate_inertias=view.get_inertias()[row, bodies].cpu().tolist(),
                plate_coms=view.get_coms()[row, bodies].cpu().tolist(),
                stiffness=view.get_dof_stiffnesses()[row, joints].cpu().tolist(),
                damping=view.get_dof_dampings()[row, joints].cpu().tolist(),
                armature=view.get_dof_armatures()[row, joints].cpu().tolist())
