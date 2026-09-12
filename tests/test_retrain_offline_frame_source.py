"""离线重训的帧源接线：默认吃 `pre_equilibration.dcd`，有多少用多少。

这个脚本不 import OpenMM/mdtraj 直到 `main()` 里面，所以这里只钉住两件在
真机之前就能坏掉的事：CLI 默认值，和把一条轨迹切成三折的那点算术。
"""
import pytest

from tools.retrain_local_residual_offline import _three_way_edges

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("n", [3, 4, 5, 41, 100, 500, 1501])
def test_three_way_split_covers_every_frame_exactly_once(n):
    edges = _three_way_edges(n)
    assert edges[0] == 0 and edges[-1] == n
    sizes = [edges[i + 1] - edges[i] for i in range(3)]
    assert all(s > 0 for s in sizes), sizes   # 空分区会让 ledger 与数据集对不上
    assert sum(sizes) == n                    # 不重叠、不丢帧


@pytest.mark.parametrize("n", [0, 1, 2])
def test_three_way_split_refuses_too_few_frames(n):
    with pytest.raises(ValueError):
        _three_way_edges(n)


def test_trajectory_flag_is_optional_and_defaults_to_pre_equilibration():
    import argparse
    import inspect

    from tools import retrain_local_residual_offline as mod

    src = inspect.getsource(mod.main)
    # 默认帧源写死成固定文件名，不是必填参数
    assert 'args.trajectory or [run_dir / "pre_equilibration.dcd"]' in src
    assert 'parser.add_argument("--trajectory", action="append", default=None' in src
    # 抽稀仍然默认关：有多少帧用多少
    assert '"--max-frames-per-trajectory", type=int, default=None' in src
    assert isinstance(argparse.ArgumentParser, type)
