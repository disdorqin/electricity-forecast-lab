import torch
from src.TafM_改进源码.models.task_adapters import TaskAdapters


def test_task_adapters_keep_member_axis_and_two_global_gates():
    m = TaskAdapters(5, 3, 7, mode="A2")
    hdir, hmag = m(torch.randn(2, 4, 24, 5), torch.randn(2, 24, 3))
    assert hdir.shape == (2, 24, 7) and hmag.shape == (2, 4, 24, 7)
    assert m.a_dir.ndim == m.a_mag.ndim == 0
