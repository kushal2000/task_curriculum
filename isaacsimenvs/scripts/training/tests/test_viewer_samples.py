"""Mixed-design training captures the real global environment row of each design."""
from types import SimpleNamespace as NS

from isaacsimenvs.scripts.training.train import _pose_viewer_samples


def test_per_design_capture_uses_actual_sparse_global_rows():
    groups = [NS(design=NS(name='single_hinge'), env_ids=[7, 2]),
              NS(design=NS(name='strip_8'), env_ids=[11, 0]),
              NS(design=NS(name='empty'), env_ids=[])]
    env = NS(unwrapped=NS(backend=NS(groups=groups)))
    assert _pose_viewer_samples(env, 3, True) == [('single_hinge', 7), ('strip_8', 11)]
    assert _pose_viewer_samples(env, 3, False) == [('', 3)]
    assert _pose_viewer_samples(NS(unwrapped=NS()), 3, True) == [('', 3)]
