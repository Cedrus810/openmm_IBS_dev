"""2026-09-20 静态核实批：1 高 + 10 中的回归钉子。

只钉**这次改的那件事**，不重测各自模块已有的行为。原则同
`test_stage_range_kwargs_signature_2026_09_19.py`：能用契约钉住的就钉契约，
因为这一批里有三条的形状都是"同一个参数在 N 个调用点里漏了一个"。
"""
import ast
import inspect
import io
import os

import abfe_preoptimizer
from abfe_pipeline import (
    _PREOPT_DERIVED_PATH_KEYS,
    tail_window_max_states,
)
from ibs_engine import _segment_delta_g_by_window, _segment_physical_window

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _tree(name):
    with io.open(os.path.join(_ROOT, name), encoding="utf-8") as fh:
        return ast.parse(fh.read())


# --------------------------------------------------------------------------
# [高] 末窗上界必须传到**每一个** partitioner 调用点（生产/校验/离线重算）
# --------------------------------------------------------------------------
def test_every_metric_partition_call_passes_both_window_caps():
    """整条 λ 表的分窗，三个上界必须一起传。

    唯一豁免 `repartition_tail_from_anchor`：它切的是 anchor 之后的**尾段**，
    那一段的第一个窗口不是耦合端（win0 在冻结前缀里）⟹ 首窗上界不适用；
    末窗上界也**故意不传** —— 溢出槽是留给后续插 λ 的，一次重分就把它吃满，
    下一次插点立刻被迫拆窗。豁免名单写死在这里，多一个都要先说明理由。
    """
    _EXEMPT = {"repartition_tail_from_anchor"}
    hits, exempt_hits = 0, 0
    for src in ("abfe_pipeline.py", "abfe_preoptimizer.py"):
        tree = _tree(src)
        for fn_def in [n for n in ast.walk(tree)
                       if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            for node in ast.walk(fn_def):
                if not isinstance(node, ast.Call):
                    continue
                name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
                if name != "partition_windows_by_metric_integral":
                    continue
                if fn_def.name in _EXEMPT:
                    exempt_hits += 1
                    continue
                hits += 1
                kw = {k.arg for k in node.keywords}
                for cap in ("first_window_max_states", "last_window_max_states"):
                    assert cap in kw, (
                        f"{src}:{node.lineno}（{fn_def.name}）调 partitioner 没传 {cap}。"
                        "漏传 = 该上界在这条路径上恒为 None：生产布局吃不到 cap、"
                        "而另一条路径吃得到 ⟹ 缓存记的布局与实际跑的布局不是同一个。"
                    )
    assert hits >= 3, f"只找到 {hits} 个整表分窗调用点，测试自己过期了"
    assert exempt_hits == 1, f"豁免名单命中 {exempt_hits} 次，期望 1"


def test_offline_recompute_accepts_last_window_cap():
    """签名不收 ⟹ 调用方想传也传不进来（09-20 之前正是如此）。"""
    params = inspect.signature(
        abfe_preoptimizer.recompute_vanishing_path_from_cached_pilot
    ).parameters
    for cap in ("first_window_max_states", "last_window_max_states"):
        assert cap in params, f"离线重算签名缺 {cap}"


# --------------------------------------------------------------------------
# [中] 派生层键表只留一份：不许再手抄一个 dict 字面量
# --------------------------------------------------------------------------
def test_derived_path_params_is_not_a_hand_copied_dict():
    for node in ast.walk(_tree("abfe_pipeline.py")):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg != "derived_path_params":
                continue
            assert not isinstance(kw.value, ast.Dict), (
                f"abfe_pipeline.py:{node.lineno} 又手抄了一份派生键字典。"
                "键表的唯一权威是 _PREOPT_DERIVED_PATH_KEYS —— 手抄一份就会漏一个，"
                "而漏掉的那个键在指纹里恒为 None（改它不作废任何缓存）。"
            )
    assert "stage2_last_window_max_states" in _PREOPT_DERIVED_PATH_KEYS


# --------------------------------------------------------------------------
# [中] 末窗上限：一个函数，三处共用
# --------------------------------------------------------------------------
def test_tail_window_cap_semantics():
    # 不给 => 逐位等于 max（旧行为）
    assert tail_window_max_states(4, 5, None) == 5
    # 给了 => 生效，且这正是生产配置（min=4 / max=5 / last=8）
    assert tail_window_max_states(4, 5, 8) == 8
    # 夹到 2*max-2：9 是"可拆上限"，不是"可以照跑的上限"
    assert tail_window_max_states(4, 5, 99) == 8
    # 下夹到 min
    assert tail_window_max_states(4, 5, 2) == 4


def test_no_hand_rolled_tail_cap_left():
    """`2 * max_states_per_window - 2` 这个推导不许再出现第二份实现。"""
    src = io.open(os.path.join(_ROOT, "abfe_pipeline.py"), encoding="utf-8").read()
    body = src.split("def tail_window_max_states", 1)[1]
    body = body.split("\n_PREOPT_DERIVED_PATH_KEYS", 1)[0]
    rest = src.replace(body, "")
    for pat in ("2 * int(max_states_per_window) - 2", "2 * _hi - 2"):
        assert pat not in rest, (
            f"{pat!r} 在 tail_window_max_states 之外又出现了一份 —— "
            "同一个不变量的第 N 份实现正是这次要修的形状。"
        )


# --------------------------------------------------------------------------
# [中] split-half 逐窗配对用物理窗口号
# --------------------------------------------------------------------------
def test_segment_window_key_is_physical_not_positional():
    # 有窗口被跳过：位置下标 0/1 对应物理窗口 0/2
    segs = [
        {"window_index": 0, "source_window_index": 0, "delta_G_kJ_mol": 1.0},
        {"window_index": 1, "source_window_index": 2, "delta_G_kJ_mol": 2.0},
    ]
    assert [_segment_physical_window(s) for s in segs] == [0, 2]
    got = _segment_delta_g_by_window({"covariance_chain_segments": segs})
    assert got == {0: 1.0, 2: 2.0}, (
        "按位置下标配对 ⟹ full 的窗口 k 会跟 half 的 k+1 比、σ 下界加在错的窗口上"
    )
    # 老记录没有 source_window_index 时回落（那时没跳窗，两者相等）
    legacy = [{"window_index": 3, "delta_G_kJ_mol": 5.0}]
    assert _segment_delta_g_by_window({"covariance_chain_segments": legacy}) == {3: 5.0}


# --------------------------------------------------------------------------
# [中] attachment 腿的 seed 必须跟着 repeat_seed 走
# --------------------------------------------------------------------------
def test_attachment_seed_follows_repeat_seed():
    import runabfe

    class _Cfg:
        def __init__(self, data):
            self.data = data

        def get(self, key, default=None):
            return self.data.get(key, default)

    base = runabfe._attachment_leg_seed(_Cfg({"attachment_seed": 20260728}))
    assert base == 20260728, "没设 repeat_seed 时必须逐位沿用旧常量"
    a = runabfe._attachment_leg_seed(
        _Cfg({"attachment_seed": 20260728, "repeat_seed": 11})
    )
    b = runabfe._attachment_leg_seed(
        _Cfg({"attachment_seed": 20260728, "repeat_seed": 12})
    )
    assert a != b != base and a != base, (
        "两次重复拿到同一个 attachment seed ⟹ ΔG_attach 逐位相同、跨重复 σ 被低估"
    )
    # 同一个 repeat_seed 必须可复现
    assert a == runabfe._attachment_leg_seed(
        _Cfg({"attachment_seed": 20260728, "repeat_seed": 11})
    )


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_"):
            _fn()
            print("ok", _name)
