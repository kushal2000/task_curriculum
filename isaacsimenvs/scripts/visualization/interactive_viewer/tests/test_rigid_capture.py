"""HTML captures contain the closed collision prisms in actual body-link poses."""
import json
import math
import re
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from isaacsimenvs.scripts.visualization.interactive_viewer.pose_viewer import (
    ClothPoseViewerWrapper,
    capture_cloth_viewer_frame,
)
from isaacsimenvs.tasks.cloth.backends.rigid import designs
from isaacsimenvs.tasks.cloth.mdp.geometry import rotations
from isaacsimenvs.tasks.cloth.tests.test_rigid_backend_readout import _backend

NAMES = ('single_hinge', 'strip_4', 'strip_8', 'hinge_band', 'flexible_corners')


def capture_case():
    """One mixed batch with known folded body poses and nonzero environment origins."""
    backend = _backend(NAMES, num_envs=5, resolution=7)
    backend._dims[:] = torch.tensor([.2, .16, .002])
    backend.initialize()
    origins = torch.arange(5)[:, None] * torch.tensor([2., 3., .1])
    scene = NS(env_origins=origins)
    identity = torch.tensor([1., 0., 0., 0.]).expand(5, 4)
    robot = NS(data=NS(joint_pos=torch.zeros(5, 1), joint_names=['unused'],
                      root_pos_w=origins.clone(), root_quat_w=identity))
    table = NS(data=NS(root_pos_w=origins+torch.tensor([0., 0., .38]), root_quat_w=identity))
    backend.scene = scene
    env = NS(num_envs=5, scene=scene, robot=robot, table=table, _cloth_grid=backend.cloth_grid,
             cfg=NS(assets=NS(robot_urdf='assets/flexiv_wuji/rizon4s_right_wuji_real_limits.urdf')),
             viewer_plates=backend.viewer_plates, _particles_w=lambda: backend.cloth_points()[0],
             folded_half_w=lambda: backend.cloth_grid.folded_rest(backend.cloth_grid.moving_idx)
                 + origins[:, None] + torch.tensor([0., 0., .6]),
             fold_error=lambda: torch.zeros(5), footprint_ratio=lambda: torch.ones(5))
    wrapper = object.__new__(ClothPoseViewerWrapper)
    wrapper.env = NS(unwrapped=env)
    wrapper._object_urdf_text, wrapper._object_urdf_path = '', None
    wrapper._table_urdf_text = '<robot name="table"><link name="base"><visual><geometry><box size=".6 .4 .3"/></geometry></visual></link></robot>'
    wrapper._table_urdf_path = wrapper._hole_urdf_path = wrapper._hole_urdf_text = None
    wrapper.github_raw_base = 'https://raw.githubusercontent.com/kushal2000/task_curriculum/main/'
    wrapper.url_check = 'skip'
    frames = [[] for _ in NAMES]
    for angle in (0., math.pi/2, math.pi):
        for group in backend.groups:
            for row, global_id in enumerate(group.env_ids.tolist()):
                spec = group.design.resolve(*backend._dimensions_cpu[global_id])
                for plate, body in zip(spec.plates, group.body_ids, strict=True):
                    q = torch.tensor([math.cos(angle/2), 0., -math.sin(angle/2), 0.]) if plate.center[0] > 0 else identity[0]
                    pivot = torch.tensor([0., 0., .001])
                    offset = pivot-rotations.apply(q, pivot)
                    group.articulation.data.body_link_pos_w[row, body] = origins[global_id]+torch.tensor([0., 0., .6])+offset
                    group.articulation.data.body_link_quat_w[row, body] = q
        for env_id in range(5):
            frames[env_id].append(capture_cloth_viewer_frame(env, env_id))
    return backend, wrapper, frames


def payload(html):
    return json.loads(re.search(r'<script id="scene-json" type="application/json">(.*?)</script>', html, re.S)[1])


@pytest.mark.parametrize('env_id', range(5), ids=NAMES)
def test_closed_prisms_survive_capture_and_html_with_real_thickness_and_fold(env_id):
    backend, wrapper, frames = capture_case()
    recording = frames[env_id]
    channels = payload(wrapper._build_html(recording))['trajectory']['deformables']
    spec = backend.groups[env_id].design.resolve(.2, .16, .002)
    vertices = np.asarray(channels['cloth_moving']['vertices'])
    np.testing.assert_allclose(vertices, [f['plate_vertices'] for f in recording], atol=6e-7)
    faces = np.asarray(recording[0]['plate_tris'])
    moving = np.asarray(recording[0]['plate_moving'])
    assert len(faces) == sum(len(designs.extrusion_mesh(p)[1]) for p in spec.plates)
    assert channels['cloth_moving']['flat_shading'] is True
    np.testing.assert_array_equal(np.asarray(channels['cloth_moving']['indices']).reshape(-1, 3), faces[moving])
    np.testing.assert_array_equal(np.asarray(channels['cloth_stationary']['indices']).reshape(-1, 3), faces[~moving])
    # Every edge has two incident triangles: closed solids, not top/bottom-only or observation grids.
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert np.all(counts == 2)
    # An edge-on camera along +Y sees a nonzero 2 mm side even in the flat frame.
    assert np.ptp(vertices[0, :, 2]) == pytest.approx(.002, abs=1e-6)
    # Quarter fold rises by half the physical sheet length; full fold is two actual plies.
    assert vertices[1, :, 2].max() == pytest.approx(.701, abs=1e-6)
    assert np.ptp(vertices[2, :, 2]) == pytest.approx(.004, abs=1e-6)
    # Closed mesh signed volume equals actual length*width*thickness in every pose.
    for frame in vertices:
        tri = frame[faces]
        volume = np.einsum('ij,ij->i', tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum()/6
        assert volume == pytest.approx(.2*.16*.002, rel=1e-3)  # 1 µm HTML precision against a 2 mm slab


def test_rigid_capture_never_silently_falls_back_to_a_particle_grid():
    _, wrapper, frames = capture_case()
    with pytest.raises(ValueError, match='one environment/design'):
        wrapper._build_html([frames[0][0], frames[1][1]])
    del frames[0][1]['plate_vertices']
    with pytest.raises(ValueError, match='one environment/design'):
        wrapper._build_html(frames[0])
