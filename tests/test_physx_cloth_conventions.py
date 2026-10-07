"""The PhysX fold tasks hand the policy what the Newton ones do.

A PhysX-trained policy is evaluated on Newton's VBD cloth, so every convention that reaches the
observation has to match -- including two Newton quirks every existing policy was trained under
(``isaacsimenvs/tasks/cloth/physx_envs.py``). These pin the pure-torch halves of that contract; the
end-to-end check is ``scripts/analysis/cloth_obs_parity.py`` on a GPU.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def adapters():
    return pytest.importorskip("isaacsimenvs.tasks.cloth.utils.physx_adapters")


@pytest.fixture(scope="module")
def cloth_adapter():
    return pytest.importorskip("isaacsimenvs.tasks.cloth.utils.cloth_adapter")


def _yaw_wxyz(yaw: float) -> torch.Tensor:
    return torch.tensor([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def test_root_pose_writes_go_out_wxyz(adapters):
    pose = torch.tensor([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.9]])  # xyzw
    out = adapters.xyzw_pose_to_wxyz(pose)
    assert out[0, :3].tolist() == [1.0, 2.0, 3.0]
    assert out[0, 3:].tolist() == pytest.approx([0.9, 0.1, 0.2, 0.3])


def _newton_object_write(value: torch.Tensor) -> torch.Tensor:
    """What ``patches.install_pose_write_conversion`` hands ``env.object`` on Newton."""
    out = value.clone()
    out[..., 3:7] = torch.cat((value[..., 4:7], value[..., 3:4]), dim=-1)
    return out


class _Recorder:
    def __init__(self):
        self.seen = None

    def write_root_pose_to_sim(self, value, env_ids=None):
        self.seen = value


def test_object_pose_reorder_matches_newton(cloth_adapter):
    """The decoded reset heading is the same on both simulators, for a draw and for the pin."""
    physx_envs_src = (REPO / "isaacsimenvs/tasks/cloth/physx_envs.py").read_text()
    start = physx_envs_src.index("def _reorder_object_pose_writes")
    end = physx_envs_src.index("class _PhysxClothMixin")
    ns = {"torch": torch}
    exec(physx_envs_src[start:end], ns)

    obj = _Recorder()
    ns["_reorder_object_pose_writes"](obj)
    gen = torch.Generator().manual_seed(0)
    for _ in range(16):
        q = torch.randn(4, generator=gen)
        q = q / q.norm()
        pose = torch.cat((torch.zeros(3), q)).unsqueeze(0)
        obj.write_root_pose_to_sim(pose)
        assert torch.equal(obj.seen, _newton_object_write(pose))
        assert float(cloth_adapter.task_yaw(obj.seen)) == pytest.approx(
            float(cloth_adapter.task_yaw(_newton_object_write(pose)))
        )
    # The pinned start (identity, wxyz) decodes to 180 degrees on Newton, so it must here too.
    pin = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    obj.write_root_pose_to_sim(pin)
    assert abs(float(cloth_adapter.task_yaw(obj.seen))) == pytest.approx(math.pi)


def test_goal_quaternion_is_newtons_relabelling():
    """Newton: Isaac Lab 3.0's quat_from_matrix gives xyzw, the write patch treats it as wxyz and the
    read converts back -- the task reads (x, y, z, w) of the TRUE rotation as (w, x, y, z). PhysX's
    helper gives the true wxyz; the override reorders [1, 2, 3, 0] to land on the same four numbers."""
    true_wxyz = torch.tensor([0.5, 0.5, -0.5, 0.5])
    newton_xyzw = true_wxyz[[1, 2, 3, 0]]                 # IL3 quat_from_matrix output
    stored_xyzw = torch.cat((newton_xyzw[1:4], newton_xyzw[0:1]))  # write patch: "wxyz" -> xyzw
    task_reads = torch.cat((stored_xyzw[3:4], stored_xyzw[0:3]))   # read patch: xyzw -> wxyz
    physx_writes = true_wxyz[[1, 2, 3, 0]]               # _PhysxClothMixin._goal_quat_from_matrix
    assert torch.equal(task_reads, physx_writes)
    src = (REPO / "isaacsimenvs/tasks/cloth/physx_envs.py").read_text()
    assert "quat_from_matrix(rot)[..., [1, 2, 3, 0]]" in src


def _blocks(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


@pytest.mark.parametrize(
    "newton,physx",
    [("RigidCloth.yaml", "RigidClothPhysx.yaml"), ("FoldPool.yaml", "FoldPoolPhysx.yaml")],
)
def test_physx_yaml_is_the_newton_task(newton, physx):
    """Same task blocks; only the solver blocks differ."""
    cfg_dir = REPO / "isaacsimenvs/cfg/task"
    a, b = _blocks(cfg_dir / newton), _blocks(cfg_dir / physx)
    newton_only = {"newton", "use_gravcomp", "meshify"}
    physx_only = {"physx_chain"}
    assert set(a) - newton_only == set(b) - physx_only
    for key in set(a) - newton_only:
        if key == "sim":
            sim_b = dict(b["sim"])
            assert "physx" in sim_b
            sim_b.pop("physx")
            assert a["sim"] == sim_b
        elif key == "rigid_cloth":
            # contact_ke / contact_kd are Newton-only (PhysX refuses them); 0 = Newton's default.
            rc_a = dict(a[key])
            assert rc_a.pop("contact_ke", 0.0) == 0.0 and rc_a.pop("contact_kd", 0.0) == 0.0
            assert rc_a == b[key], key
        else:
            assert a[key] == b[key], key


def test_physx_tasks_are_registered():
    src = (REPO / "isaacsimenvs/tasks/cloth/__init__.py").read_text()
    for task in ("Isaacsimenvs-RigidClothPhysx-Direct-v0", "Isaacsimenvs-FoldPoolPhysx-Direct-v0"):
        assert re.search(rf'id="{task}"', src), task
