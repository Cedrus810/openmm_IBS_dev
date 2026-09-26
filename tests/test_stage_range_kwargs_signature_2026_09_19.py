"""`*_range_kwargs` 字典里的键必须是分窗器真的收得下的形参。

2026-09-19 真机 9/9 全崩（TypeError: got an unexpected keyword argument
'last_window_max_states'）的形状：末窗上界被塞进 `_stage_range_kwargs`，而这个字典
会被 `**` splat 进 `vanishing_subdomain_ranges_from_lambdas`（等边数贪心），那个函数
没有这个形参 —— 有它的是 `partition_windows_by_metric_integral`。崩在 stage 2 入口、
第一帧之前，而当时的单元测试只直接测了 partitioner 和 config 键，没测这段透传。

所以这里不测行为，测**契约**：凡是名字以 `range_kwargs` 结尾、又被整份 splat 进
partitioner 的字典，它的每个键都必须在 partitioner 的签名里。
"""
import ast
import inspect
import io
import os

import abfe_preoptimizer

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "abfe_pipeline.py")
_ALLOWED = set(
    inspect.signature(
        abfe_preoptimizer.vanishing_subdomain_ranges_from_lambdas
    ).parameters
)


def _keys_assigned_into_range_kwargs(tree):
    """`x_range_kwargs["k"] = ...` 里的所有 k，按字典名归组。"""
    out = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if (isinstance(tgt, ast.Subscript)
                    and isinstance(tgt.value, ast.Name)
                    and tgt.value.id.endswith("range_kwargs")
                    and isinstance(tgt.slice, ast.Constant)
                    and isinstance(tgt.slice.value, str)):
                out.setdefault(tgt.value.id, set()).add(tgt.slice.value)
    return out


def test_range_kwargs_keys_are_partitioner_parameters():
    with io.open(_SRC, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    groups = _keys_assigned_into_range_kwargs(tree)
    assert groups, "没找到任何 *_range_kwargs 字典——测试自己过期了，别当绿灯"
    for name, keys in sorted(groups.items()):
        bad = sorted(keys - _ALLOWED)
        assert not bad, (
            f"{name} 里的 {bad} 不是 vanishing_subdomain_ranges_from_lambdas 的形参，"
            f"而这个字典会被整份 `**` splat 进去 ⟹ stage 2 入口 TypeError。"
            f"只给 partitioner 之一用的参数，按形参单独传，别搭这个字典的车。"
        )


if __name__ == "__main__":
    test_range_kwargs_keys_are_partitioner_parameters()
    print("ok")
