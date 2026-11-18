"""Regressions for force impulses, work and symmetric crease moments."""
import numpy as np
import pytest
import torch

from experiments.nov17.pressing_audit.lift_work import record_work
from experiments.nov17.pressing_audit.work_contacts import unpack_friction


def test_friction_records_ignore_empty_offsets_and_do_not_alias_native_buffers():
    forces=torch.zeros(8,3);points=torch.zeros_like(forces)
    forces[2]=torch.tensor([-2.,0.,0.]);points[2]=torch.tensor([1.,2.,3.])
    counts=torch.tensor([[1,0]]);starts=torch.tensor([[2,2**32-1]])
    pair,vector,position=unpack_friction((forces,points,counts,starts),8)
    forces.zero_();points.zero_();counts.zero_();starts.zero_()
    assert pair.tolist()==[0]
    assert vector.tolist()==[[-2.,0.,0.]]
    assert position.tolist()==[[1.,2.,3.]]


@pytest.mark.parametrize('span,capacity',[(8,8),(2,1)])
def test_invalid_friction_spans_or_capacity_raise(span,capacity):
    with pytest.raises(RuntimeError):
        unpack_friction((torch.zeros(8,3),torch.zeros(8,3),torch.tensor([[1]]),torch.tensor([[span]])),capacity)


def test_impulse_work_uses_midpoint_velocity_and_recovers_lost_kinetic_energy():
    # A1kg body is stopped from2m/s by a−2Ns impulse. Semi-implicit
    # positions need not move during that step; F·position_delta would be zero.
    pose=np.zeros((1,1,1,7));pose[...,3]=1
    before=np.zeros((1,1,1,6));before[...,0]=2
    after=np.zeros_like(before)
    record=dict(frame_idx=np.array([0]),env_id=np.array([0]),cloth_body=np.array([0]),
                partner_type=np.array([1]),position=np.zeros((1,3)))
    result=record_work(record,np.array([[-2.,0.,0.]]),pose,pose,np.array([[0]]),
                       np.zeros((1,1,3)),np.array([[[0.,-1.,0.]]]),np.zeros((1,1,2)),
                       pose,pose,after,before,1.)
    assert result['work'][0,0,1,0]==-2
    assert result['displacement_work'][0,0,1,0]==0


def test_crease_moment_work_tracks_material_identity_and_reverses_with_rotation():
    pose=np.zeros((1,1,1,7));pose[...,3]=1
    record=dict(frame_idx=np.array([0]),env_id=np.array([0]),cloth_body=np.array([0]),
                partner_type=np.array([0]),position=np.array([[1.,0.,0.]]))
    for half in (0,1):
        angle=np.zeros((1,1,2));angle[...,half]=.1
        args=(pose,pose,np.array([[half]]),np.zeros((1,1,3)),np.array([[[0.,-1.,0.]]]),angle)
        result=record_work(record,np.array([[0.,0.,1.]]),*args)
        assert result['crease_work'][0,0,0,half]==pytest.approx(.1)
        assert result['crease_work'][0,0,0,1-half]==0
        angle*=-1
        assert record_work(record,np.array([[0.,0.,1.]]),*args)['crease_work'][0,0,0,half]==pytest.approx(-.1)


def rule_fixture():
    from experiments.nov17.pressing_audit.lift_work import detect_without_lift_work
    geom=dict(tilt_deg=np.array([[[0.,0.]],[[40.,0.]],[[60.,0.]],[[100.,0.]]]),
              mean_height=np.array([[[.001,.001]],[[.03,.001]],[[.05,.001]],[[.03,.001]]]))
    active=np.ones((4,1),bool);inside=active.copy();dims=np.array([[.2,.2,.002]])
    gravity=np.array([[[0.,0.]],[[.00058,0.]],[[.00098,0.]],[[.00058,0.]]])
    weight=np.array([[.02,.02]]);credit=np.zeros_like(gravity);press=np.zeros_like(gravity);press[0,0,1]=.1
    return detect_without_lift_work,[geom,active,inside,dims,gravity,weight,credit,press]


def test_rule_detects_unassisted_pressed_base_ascent_and_is_symmetric():
    detect,args=rule_fixture()
    expected=detect(*args)
    assert expected[:,0].tolist()==[False,True,True,False]
    flipped=list(args);flipped[0]={key:value[:,:,::-1] for key,value in args[0].items()}
    for index in (4,6,7):flipped[index]=args[index][:,:,::-1]
    flipped[5]=args[5][:,::-1]
    np.testing.assert_array_equal(detect(*flipped),expected)


def test_real_opening_work_credit_survives_release_and_other_half_work_does_not_count():
    detect,args=rule_fixture()
    args[6][:,0,0]=.002
    assert not detect(*args).any()
    # Same total work, all applied to the base: the rising half has no credit.
    args[6][:,0,0]=0;args[6][:,0,1]=.002
    assert detect(*args)[1,0]


def test_free_ascent_without_supported_press_is_outside_the_pressing_rule():
    detect,args=rule_fixture();args[7].fill(0)
    assert not detect(*args).any()


def test_current_support_rule_does_not_treat_old_pressing_as_current_support():
    detect,args=rule_fixture()
    assert detect(*args).any()
    assert not detect(*args,current_press=True).any()
    args[7][:,0,1]=.1
    assert detect(*args,current_press=True)[:,0].tolist()==[False,True,True,False]
    args[6][:,0,0]=.002
    assert not detect(*args,current_press=True).any()


def test_flat_reference_clears_old_losses_without_erasing_new_lift_work():
    from experiments.nov17.pressing_audit.lift_work import restart_work_account
    work=np.array([[-.002],[-.003],[-.0025],[.0005]])
    gravity=np.array([[0.],[0.],[.0001],[.0008]])
    flat=np.array([[True],[True],[False],[False]])
    credit,rise=restart_work_account(work,gravity,flat)
    np.testing.assert_allclose(credit[:,0],[0,0,.0005,.0035])
    np.testing.assert_allclose(rise,gravity)


def test_returning_flat_discards_old_positive_credit():
    from experiments.nov17.pressing_audit.lift_work import restart_work_account
    work=np.array([[0.],[.002],[.001],[.003]])
    gravity=np.array([[0.],[.002],[.0001],[.001]])
    flat=np.array([[True],[False],[True],[False]])
    credit,rise=restart_work_account(work,gravity,flat)
    np.testing.assert_allclose(credit[:,0],[0,.002,0,.002])
    np.testing.assert_allclose(rise[:,0],[0,.002,0,.0009])


def test_upward_pinch_is_not_misclassified_as_downward_pressing():
    from experiments.nov17.pressing_audit.lift_work import net_supported_load
    # Two fingers exert−2N and+3N: net lift despite downward contact magnitude.
    assert net_supported_load(np.array([-2.+3.]),np.array([.1]))[0]==0
    # Reversing the pair's dominance gives2N of actual downward pressing.
    assert net_supported_load(np.array([-3.+1.]),np.array([3.]))[0]==2


def source_fixture(point,force,velocity):
    from experiments.nov17.pressing_audit.lift_work import robot_lifting_work
    pose=np.zeros((2,1,1,7));pose[...,3]=1
    v=np.zeros((2,1,1,6));v[...,:3]=velocity
    record=dict(frame_idx=np.array([1]),env_id=np.array([0]),cloth_body=np.array([0]),
                partner_index=np.array([0]),partner_type=np.array([0]),position=np.array([point]),force_vector=np.array([force]))
    return robot_lifting_work(record,pose,pose,v,np.array([[1]]),np.zeros((2,1,3)),
                              np.tile([0.,-1.,0.],(2,1,1)),np.array([[-1.,1.]]),.01)


def test_stationary_hand_gets_no_credit_for_contact_correction_moving_the_cloth():
    all_work,lift=source_fixture([1.,0.,0.],[0.,0.,2.],[0.,0.,0.])
    assert not all_work.any()
    assert not lift.any()


def test_hand_lifting_gets_credit_but_downward_closing_work_does_not():
    _,lift=source_fixture([1.,0.,0.],[0.,0.,2.],[0.,0.,2.])
    assert lift[1,0,1]==pytest.approx(.04)
    all_work,lift=source_fixture([1.,0.,0.],[0.,0.,-2.],[0.,0.,-2.])
    assert all_work[1,0,1]==pytest.approx(.04)
    assert lift[1,0,1]==0


def test_side_pull_with_opening_moment_can_supply_lifting_work():
    _,lift=source_fixture([1.,0.,1.],[-2.,0.,0.],[-2.,0.,0.])
    assert lift[1,0,1]==pytest.approx(.04)


def test_source_rule_requires_an_observed_starting_reference():
    detect,args=rule_fixture();args[7][:,0,1]=.1
    assert detect(*args,current_press=True).any()
    assert not detect(*args,current_press=True,ready=np.zeros((4,1,2),bool)).any()
