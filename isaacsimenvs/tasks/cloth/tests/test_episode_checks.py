"""Table overhang, substep latching, work history and invalid-success vetoes."""
from types import SimpleNamespace as NS

import torch

from isaacsimenvs.tasks.cloth.mdp.fold_guard import LiftWorkHistory, robot_opening_work, table_excursion
from isaacsimenvs.tasks.cloth.backends.rigid.episode_guard import RigidClothEpisodeGuard
from isaacsimenvs.tasks.cloth.tests.test_release_bonus import task, step


def cfg():
    return NS(minimum_lifting_work_fraction=.1,maximum_base_tilt_deg=20.,minimum_lift_angle_deg=30.,
              minimum_opposed_load=.001,outside_table=True,table_margin=.00001)


def test_bounds_use_vertices_in_table_frame_and_allow_lifting_above_table():
    q=torch.tensor([[2**-.5,0.,0.,2**-.5]])
    # Table-local x=0.29 becomes world y=0.29 after90deg yaw.
    vertex=torch.tensor([[[[1.,2.29,4.]]]])
    center=torch.tensor([[1.,2.,.53]])
    assert table_excursion(vertex,center,q,torch.tensor([.3,.2])).item()==0
    vertex[...,1]=2.31
    torch.testing.assert_close(table_excursion(vertex,center,q,torch.tensor([.3,.2])),torch.tensor([.01]),atol=2e-7,rtol=1e-5)


def test_outside_latches_at_intermediate_substep_even_if_cloth_returns():
    guard=object.__new__(RigidClothEpisodeGuard)
    guard.n,guard.device,guard.cfg=1,'cpu',cfg()
    guard.backend=NS(table=NS(data=NS(root_pos_w=torch.zeros(1,3),root_quat_w=torch.tensor([[1.,0.,0.,0.]]))),_table_size=[.6,.4,.3])
    position=torch.tensor([[[.31,0.,.53]]]);quat=torch.tensor([[[1.,0.,0.,0.]]])
    guard._plate_state=lambda:(position,quat,position,torch.zeros(1,1,6))
    guard.rest_vertices=torch.zeros(1,1,1,3);guard.half=torch.zeros(1,1,dtype=torch.long)
    guard.outside=torch.zeros(1,dtype=torch.bool);guard.nonfinite=guard.outside.clone();guard.pressing=guard.outside.clone()
    guard.excursion=torch.zeros(1);guard.contacts=None;guard.last_counter=0
    guard.history=LiftWorkHistory(1,'cpu',cfg());guard.fresh=torch.zeros(1,dtype=torch.bool)
    guard.observe(1);snapshot=guard.failures()
    assert guard.outside.item()
    position.zero_();guard.observe(2)
    assert guard.outside.item()
    guard.reset(torch.tensor([0]))
    assert not guard.outside.item() and snapshot['outside_table'].item()


def test_source_work_credits_moving_hand_and_never_a_static_contact():
    q=torch.tensor([[1.,0.,0.,0.]])
    zero=torch.zeros(1,3);velocity=torch.zeros(1,6)
    args=[torch.tensor([[1.,0.,0.]]),torch.tensor([[0.,0.,2.]]),zero,q,zero,velocity,
          zero,q,zero,velocity,zero,torch.tensor([[0.,-1.,0.]]),torch.ones(1),.01]
    assert robot_opening_work(*args).item()==0
    velocity[:,2]=2
    torch.testing.assert_close(robot_opening_work(*args),torch.tensor([.04]))
    args[1]=torch.tensor([[0.,0.,-2.]]);velocity[:,2]=-2
    assert robot_opening_work(*args).item()==0


def test_failed_fold_cannot_get_fold_or_release_bonus():
    obj=task();obj._external_failures={'outside_table':torch.tensor([True])}
    reward,done=step(obj,clearance=.2)
    assert done.item() and reward.item()==0
    assert not obj._entry_paid.item() and not obj._released.item() and obj._successes.item()==0
    obj.reset(torch.tensor([0]))
    assert obj._termination_reasons['outside_table'].item()  # completed-step snapshot
    assert step(obj,clearance=.2)[0].item()==2000


def test_unarmed_start_cannot_pay_rewards_or_consume_progress_history():
    obj=task();obj._success_allowed.zero_()
    reward,done=step(obj,clearance=.2)
    assert reward.item()==0 and not done.item()
    assert obj._closest_fold_err.item()==-1
    assert not obj._entry_paid.item()
    obj._success_allowed.fill_(True)
    assert step(obj,clearance=.2)[0].item()==2000


def test_partial_work_history_reset_does_not_erase_other_env_credit():
    history=LiftWorkHistory(2,'cpu',cfg())
    history.credit.fill_(.2);history.ready.fill_(True);history.initialized.fill_(True)
    history.reset(torch.tensor([1]))
    torch.testing.assert_close(history.credit[0],torch.tensor([.2,.2]))
    assert not history.credit[1].any() and not history.ready[1].any()
    assert history.initialized.tolist()==[True,False]
