"""Contact queries preserve body mappings, gap thresholds, and non-finite handling."""

from types import SimpleNamespace as NS

import pytest
import torch

wp = pytest.importorskip("warp")


def test_hand_contacts_map_view_bodies_to_worlds(monkeypatch):
    from isaaclab_newton.physics import NewtonManager

    from isaacsimenvs.tasks.cloth.backends.vbd import contacts

    def array(values, dtype):
        return wp.array(values, dtype=dtype, device="cpu")

    # The robot is body 0 in the cloth view and body 1 in the global model.
    model = NS(
        body_label=["/World/envs/env_0/Table", "/World/envs/env_0/Robot/Hand"],
        particle_count=4,
        particle_world=array([0, 1, 2, 3], wp.int32),
        particle_radius=array([0.01] * 4, float),
    )
    entry = NS(
        view=NS(shape_body=array([0, 1], wp.int32), particle_count=4,
                shape_margin=array([0.001, 0.001], float)),
        body_local_to_global=array([1, 0], wp.int32),
    )
    state = NS(
        particle_q=array([(0, 0, z) for z in (1.009, 1.012, 1.020, float("nan"))], wp.vec3),
        body_q=array([(0, 0, 0, 0, 0, 0, 1), (0, 0, 1, 0, 0, 0, 1)], wp.transform),
    )
    buffer = NS(
        soft_contact_max=5,
        soft_contact_count=array([5], wp.int32),
        soft_contact_shape=array([0, 0, 0, 0, 1], wp.int32),
        soft_contact_indices=array([(i, -1, -1) for i in (0, 1, 2, 3, 2)], wp.vec3i),
        soft_contact_barycentric=array([(1, 0, 0)] * 5, wp.vec3),
        soft_contact_normal=array([(0, 0, 1)] * 5, wp.vec3),
        soft_contact_body_pos=array([(0, 0, 0)] * 5, wp.vec3),
    )
    monkeypatch.setattr(NewtonManager, "_model", model)
    monkeypatch.setattr(NewtonManager, "_state_0", state)
    monkeypatch.setattr(contacts, "_contact_buffer", lambda: (buffer, entry))
    env = NS(num_envs=4, device="cpu", _touch_cache=None,
             cfg=NS(task=NS(release=NS(touch_gap=0.002))))

    # Positive separation below the threshold and NaN both count as touching.
    expected = [True, True, False, True]
    assert contacts.hand_cloth_touching(env).tolist() == expected
    torch.testing.assert_close(env._touch_last_gap[:3], torch.tensor([-0.002, 0.001, 0.009]))
    assert env._touch_last_env.tolist() == [0, 1, 2, 3]
    # Reusing the cached shape mapping must give the same result.
    assert contacts.hand_cloth_touching(env).tolist() == expected


@pytest.mark.parametrize("buffer", [None, NS(soft_contact_max=0)])
def test_no_contact_buffer_means_no_touch(monkeypatch, buffer):
    from isaacsimenvs.tasks.cloth.backends.vbd import contacts

    monkeypatch.setattr(contacts, "_contact_buffer", lambda: (buffer, None))
    env = NS(num_envs=3, device="cpu")
    assert contacts.hand_cloth_touching(env).tolist() == [False, False, False]
