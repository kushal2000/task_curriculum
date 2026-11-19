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
def base():
    return pytest.importorskip("isaacsimenvs.tasks.cloth.backends.base")


def _yaw_wxyz(yaw: float) -> torch.Tensor:
    return torch.tensor([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def _newton_object_write(value: torch.Tensor) -> torch.Tensor:
    """What ``patches.install_pose_write_conversion`` handed the sheet's root-pose write on Newton."""
    out = value.clone()
    out[..., 3:7] = torch.cat((value[..., 4:7], value[..., 3:4]), dim=-1)
    return out


def _old_decode(root_pose: torch.Tensor) -> torch.Tensor:
    """The heading both sheets decoded from the root pose they received (the old adapters' task_yaw)."""
    qw, qx, qy, qz = root_pose[:, 3], root_pose[:, 4], root_pose[:, 5], root_pose[:, 6]
    return torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def test_reset_heading_is_the_reordered_decode(base):
    """``reset_yaw`` is, bit for bit, the heading the sheet started at when its reset pose went
    through Newton's write conversion: for a draw, and for the pinned start (180 deg)."""
    gen = torch.Generator().manual_seed(0)
    for _ in range(16):
        q = torch.randn(4, generator=gen)
        q = q / q.norm()
        pose = torch.cat((torch.zeros(3), q)).unsqueeze(0)
        assert torch.equal(base.reset_yaw(pose[:, 3:7]), _old_decode(_newton_object_write(pose)))
    pin = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    assert abs(float(base.reset_yaw(pin[:, 3:7]))) == pytest.approx(math.pi)


def test_sheet_yaw_is_the_plain_decode(base):
    """``sheet_yaw`` decodes a quaternion as it is: what the startup placement proofs use."""
    for deg in (0.0, 30.0, 90.0, -120.0):
        q = _yaw_wxyz(math.radians(deg)).unsqueeze(0)
        pose = torch.cat((torch.zeros(1, 3), q), dim=-1)
        assert torch.equal(base.sheet_yaw(q), _old_decode(pose))
        assert float(base.sheet_yaw(q)) == pytest.approx(math.radians(deg))


def test_goal_quaternion_is_newtons_relabelling():
    """Newton: Isaac Lab 3.0's quat_from_matrix gives xyzw, the write patch treats it as wxyz and the
    read converts back -- the task reads (x, y, z, w) of the TRUE rotation as (w, x, y, z). PhysX's
    helper gives the true wxyz; the override reorders [1, 2, 3, 0] to land on the same four numbers."""
    true_wxyz = torch.tensor([0.5, 0.5, -0.5, 0.5])
    newton_xyzw = true_wxyz[[1, 2, 3, 0]]                 # IL3 quat_from_matrix output
    stored_xyzw = torch.cat((newton_xyzw[1:4], newton_xyzw[0:1]))  # write patch: "wxyz" -> xyzw
    task_reads = torch.cat((stored_xyzw[3:4], stored_xyzw[0:3]))   # read patch: xyzw -> wxyz
    physx_writes = true_wxyz[[1, 2, 3, 0]]               # RigidBackend.observed_goal_quat
    assert torch.equal(task_reads, physx_writes)
    src = (REPO / "isaacsimenvs/tasks/cloth/backends/rigid/backend.py").read_text()
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
