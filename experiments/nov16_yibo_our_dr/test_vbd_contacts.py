"""Geometry and per-world aggregation checks; no simulator needed."""
import pytest
import torch
from experiments.nov16_yibo_our_dr.vbd_contacts import box_signed_distance, contact_depths, reduce_records, _compose_xyzw


def test_collision_envelope_overlap_is_not_center_penetration():
    points=torch.tensor([[0.,0.,.007],[0.,0.,-.002],[0.,0.,.02]])
    normal=torch.tensor([[0.,0.,1.]]).expand_as(points)
    overlap,cross=contact_depths(points,torch.zeros_like(points),normal,torch.full((3,),.008),torch.full((3,),.001))
    torch.testing.assert_close(overlap,torch.tensor([.002,.011,0.]))
    torch.testing.assert_close(cross,torch.tensor([0.,.002,0.]))


def test_world_reduction_keeps_world10_separate_and_clears_empty_rows():
    world=torch.tensor([10,2,10,0])
    counts,overlap,cross=reduce_records(world,torch.tensor([True,True,True,False]),
        torch.tensor([.01,.02,.03,.9]),torch.tensor([0.,.004,.006,.8]),12)
    assert counts.tolist()==[0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,2.,0.]
    assert overlap[10].item()==pytest.approx(.03)
    assert cross[10].item()==pytest.approx(.006)
    empty=reduce_records(world,torch.zeros(4,dtype=torch.bool),overlap[:4],cross[:4],12)
    assert all(torch.count_nonzero(v)==0 for v in empty)


def test_box_sdf_rejects_outside_xy_points_below_tabletop():
    points=torch.tensor([[[0.,0.,.54],[0.,0.,.52],[.5,0.,.52],[0.,0.,.1]]])
    pose=torch.tensor([[0.,0.,.38,0.,0.,0.,1.]])
    sdf=box_signed_distance(points,pose,torch.tensor([[.3,.2,.15]]))
    torch.testing.assert_close(sdf,torch.tensor([[.01,-.01,.2,.13]]),atol=1e-7,rtol=1e-6)


def test_box_sdf_respects_body_and_local_shape_rotation_translation():
    h=2**-.5
    parent=torch.tensor([[1.,2.,3.,0.,0.,h,h]])
    local=torch.tensor([[.2,0.,0.,0.,0.,0.,1.]])
    pose=_compose_xyzw(parent,local)
    torch.testing.assert_close(pose[:,:3],torch.tensor([[1.,2.2,3.]]))
    # After90degree yaw, the long localX half extent(.3) lies alongworldY.
    points=torch.tensor([[[1.,2.49,3.],[1.21,2.2,3.]]])
    sdf=box_signed_distance(points,pose,torch.tensor([[.3,.2,.15]]))
    torch.testing.assert_close(sdf,torch.tensor([[-.01,.01]]),atol=1e-6,rtol=1e-6)
