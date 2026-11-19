"""The PhysX fold tasks hand the policy what the Newton ones do.

A PhysX-trained policy is evaluated on Newton's VBD cloth, so every convention that reaches the
observation has to match -- including two Newton quirks every existing policy was trained under
(``isaacsimenvs/tasks/cloth/backends/rigid/physx.py``). These pin the pure-torch halves of that
contract.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")

from isaacsimenvs.conftest import REPO_ROOT  # noqa: E402

REPO = REPO_ROOT


@pytest.fixture(scope="module")
def adapters():
    return pytest.importorskip("isaacsimenvs.tasks.cloth.backends.rigid.pool")


@pytest.fixture(scope="module")
def cloth_adapter():
    return pytest.importorskip("isaacsimenvs.tasks.cloth.backends.adapter")


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
    src = (REPO / "isaacsimenvs/tasks/cloth/backends/rigid/physx.py").read_text()
    start = src.index("def reorder_object_pose_writes")
    end = src.index("def size_gpu_buffers")
    ns = {"torch": torch}
    exec(src[start:end], ns)

    obj = _Recorder()
    ns["reorder_object_pose_writes"](obj)
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
    physx_writes = true_wxyz[[1, 2, 3, 0]]               # RigidClothEnv._observed_goal_quat
    assert torch.equal(task_reads, physx_writes)
    src = (REPO / "isaacsimenvs/tasks/cloth/backends/rigid/env.py").read_text()
    assert "quat_from_matrix(rot)[..., [1, 2, 3, 0]]" in src


def _blocks(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def test_tasks_and_aliases_are_registered():
    """Canonical ids, and every old id as an alias naming its canonical task and an existing robot
    setup (configs/robot/<setup>.yaml)."""
    gym = pytest.importorskip("gymnasium")
    from isaacsimenvs.conftest import load_module

    load_module("isaacsimenvs/tasks/cloth/__init__.py", "cloth_registration")
    for task in ("Isaacsimenvs-Cloth-Direct-v0", "Isaacsimenvs-RigidCloth-Direct-v0"):
        assert "canonical_task" not in gym.spec(task).kwargs
    alias = gym.spec("Isaacsimenvs-FoldPoolPhysx-Direct-v0").kwargs
    assert alias["canonical_task"] == "Isaacsimenvs-RigidCloth-Direct-v0" and "robot_setup" not in alias
    for old, base in (("Cloth", "Cloth"), ("FoldPoolPhysx", "RigidCloth")):
        kw = gym.spec(f"Isaacsimenvs-{old}FlexivWujiRightReal-Direct-v0").kwargs
        assert kw["canonical_task"] == f"Isaacsimenvs-{base}-Direct-v0"
        assert kw["robot_setup"] == "flexiv_wuji_right_real"
    setups = {p.stem for p in (REPO / "isaacsimenvs/scripts/configs/robot").glob("*.yaml")}
    for spec in gym.registry.values():
        if spec.id.startswith("Isaacsimenvs-") and "robot_setup" in spec.kwargs:
            assert spec.kwargs["robot_setup"] in setups, spec.id
