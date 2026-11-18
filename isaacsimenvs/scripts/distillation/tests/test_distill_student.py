"""Point-cloud student (isaacsimenvs/scripts/distillation/student.py): Kit-free, CPU."""

from __future__ import annotations

import torch

from isaacsimenvs.conftest import load_module

st = load_module("isaacsimenvs/scripts/distillation/student.py", "distill_student")

OBS, ACT, EMB, UNITS = 20, 5, 32, 1024


def _teacher_state() -> dict:
    torch.manual_seed(0)
    g = lambda *s: torch.randn(*s) * 0.05  # noqa: E731
    state = {
        "running_mean_std.running_mean": g(OBS), "running_mean_std.running_var": torch.rand(OBS) + 0.5,
        "a2c_network.extra_params": g(6, EMB), "a2c_network.sigma": g(6, ACT),
        "a2c_network.rnn.rnn.weight_ih_l0": g(4 * UNITS, OBS + EMB), "a2c_network.rnn.rnn.weight_hh_l0": g(4 * UNITS, UNITS),
        "a2c_network.rnn.rnn.bias_ih_l0": g(4 * UNITS), "a2c_network.rnn.rnn.bias_hh_l0": g(4 * UNITS),
        "a2c_network.layer_norm.weight": torch.ones(UNITS), "a2c_network.layer_norm.bias": torch.zeros(UNITS),
        "a2c_network.mu.weight": g(ACT, 512), "a2c_network.mu.bias": g(ACT),
        "a2c_network.value.weight": g(1, 512), "a2c_network.value.bias": g(1),
    }
    for i, (a, b) in zip((0, 2, 4, 6), ((UNITS, 1024), (1024, 1024), (1024, 512), (512, 512))):
        state[f"a2c_network.actor_mlp.{i}.weight"] = g(b, a)
        state[f"a2c_network.actor_mlp.{i}.bias"] = g(b)
    return state


def test_split_columns():
    obs_list = ["joint_pos", "object_rot", "keypoints_rel_palm", "keypoints_rel_goal", "object_scales", "palm_pos"]
    sizes = {"joint_pos": 2, "object_rot": 4, "keypoints_rel_palm": 12, "keypoints_rel_goal": 12, "object_scales": 3, "palm_pos": 3}
    proprio, aux = st.split_columns(obs_list, sizes)
    assert proprio == [0, 1, 33, 34, 35]
    assert aux == list(range(6, 30))


def test_student_copy_matches_teacher_when_the_vision_columns_carry_nothing():
    """With the teacher's vision input weights zeroed, a student loaded from it (PointNet feature
    weights zeroed too) is the same function of the proprio columns."""
    state = _teacher_state()
    cols = [0, 1, 2, 3, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    vision = [c for c in range(OBS) if c not in cols]
    state["a2c_network.rnn.rnn.weight_ih_l0"][:, vision] = 0.0
    teacher = st.TeacherNet(state)
    student = st.StudentNet(len(cols), ACT, point_dim=10, aux_dim=24)
    student.load_teacher(teacher, cols)
    with torch.no_grad():
        student.rnn.weight_ih_l0[:, len(cols):] = 0.0
    n = 4
    ts, ss = teacher.initial_state(n, "cpu"), student.initial_state(n, "cpu")
    for t in range(6):
        obs = torch.randn(n, OBS)
        start = torch.zeros(n, dtype=torch.bool)
        start[0] = t == 3
        act, ts = teacher.step(obs, ts, start)
        with torch.no_grad():
            mu, aux, ss = student.step(obs[:, cols], torch.randn(n, 30, 10), torch.rand(n, 30) > 0.3, ss, start)
        assert aux.shape == (n, 24)
        assert torch.allclose(mu.clamp(-1, 1), act, atol=1e-5)


def test_forward_seq_equals_stepping():
    torch.manual_seed(1)
    student = st.StudentNet(7, ACT, point_dim=10, aux_dim=24)
    t_len, n, m = 5, 3, 20
    proprio, points = torch.randn(t_len, n, 7), torch.randn(t_len, n, m, 10)
    mask = torch.rand(t_len, n, m) > 0.2
    mask[2, 1] = False                                  # an env with no valid point
    start = torch.zeros(t_len, n, dtype=torch.bool)
    start[0, 0] = start[3, 2] = True
    state = (torch.randn(1, n, 1024), torch.randn(1, n, 1024))
    mu_seq, aux_seq = student.forward_seq(proprio, points, mask, state, start)
    assert torch.isfinite(mu_seq).all()
    s = state
    for t in range(t_len):
        mu, aux, s = student.step(proprio[t], points[t], mask[t], s, start[t])
        assert torch.allclose(mu, mu_seq[t], atol=1e-5) and torch.allclose(aux, aux_seq[t], atol=1e-5)


def test_pointnet_ignores_masked_points_and_order():
    torch.manual_seed(2)
    net = st.PointNet(10)
    pts, mask = torch.randn(2, 12, 10), torch.ones(2, 12, dtype=torch.bool)
    mask[:, 8:] = False
    a = net(pts, mask)
    pts2 = pts.clone()
    pts2[:, 8:] = 100.0
    perm = torch.cat((torch.randperm(8), torch.arange(8, 12)))
    assert torch.allclose(a, net(pts2[:, perm], mask), atol=1e-6)
