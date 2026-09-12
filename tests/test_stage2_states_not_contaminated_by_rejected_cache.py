"""被拒绝的 Stage 2 缓存不得污染 pilot 探针网格密度。

## 2026-09-10 后续：探针是探针，λ 是 λ

下面描述的 bug 之所以可能发生，根因是**一个变量名承担两个量**：
`stage2_states` 一开始是探针网格密度（`stage2_n_states`），到函数中途被重新
赋值成最终生产 λ 节点数。同一形状的 bug 至少长出过两次（提前提交污染探针密度；
拿探针数 17 去校验一份正确的 18 点 λ 缓存并每次 resume 重烧 pilot）。现在拆成
两个名字，`stage2_probe_states` **永不重新赋值** —— 见本文件末尾两条断言。

## 这条钉的是什么

`run_full_pipeline` 里 Stage 2 缓存的接受分两步：先比协议指纹，再校验子区间布局。
2026-09-10 之前，缓存态数在**第一步之后就被提交**：

    optimized_lambdas_2 = cached_lambdas
    if len(cached_lambdas) != stage2_states:
        stage2_states = len(cached_lambdas)      # ← 提交得太早
    ...
    if normalized_cached_ranges != expected_subdomain_ranges:
        optimized_lambdas_2 = None               # ← 只回滚它
        window_ranges_2 = None                   # ← 和它
        # stage2_states 没回滚

于是布局校验失败时缓存整份被拒、却留下一个被改过的 `stage2_states`，
后面 `if optimized_lambdas_2 is None:` 那条重跑分支拿它当
`n_states=` 起 fresh pilot ⇒ **探针网格密度用的是缓存态数（例如 23），
不是配置请求的那个（默认 17）**。探针密度变了 λ 布点就变，而写回的
`protocol_key` 记的仍是请求值 ⇒ 下次 resume 指纹匹配，把一份 23 点 pilot
的结果当 17 点 pilot 复用。

## 为什么是源码结构断言而不是行为测试

`run_full_pipeline` 需要真 System / 真 Context / 真 pilot 才能走到这段，
在离线测试里造不出来。这里退而求其次，钉住**赋值点相对于校验点的位置**——
这正是那个 bug 的形状：不是算错，是提交得太早。
"""
from __future__ import annotations

import pathlib
import re

import pytest

PIPELINE_PATH = pathlib.Path(__file__).resolve().parents[1] / "abfe_pipeline.py"

# 缓存接受块的起止锚点。用语义锚点而不是行号——行号会漂。
_BLOCK_START = "optimized_lambdas_2 = cached_lambdas"
_BLOCK_END = "if optimized_lambdas_2 is None:"

_LAYOUT_CHECK = "if normalized_cached_ranges != expected_subdomain_ranges:"
_COMMIT = "stage2_states = _cached_state_count"
_OLD_BUGGY_COMMIT = "stage2_states = len(cached_lambdas)"


def _acceptance_block() -> str:
    """缓存接受块，**已剥掉注释**。

    必须剥：这段代码的注释里逐字引用了修掉的旧写法（"原来在这里就
    `stage2_states = len(cached_lambdas)`"），不剥的话本测试会抓到那句说明文字，
    而不是真的代码。
    """
    source = PIPELINE_PATH.read_text(encoding="utf-8")
    assert source.count(_BLOCK_START) == 1, (
        f"锚点 {_BLOCK_START!r} 在 abfe_pipeline.py 里不是恰好一处；"
        "缓存接受块被重构了，请同步更新本测试的锚点。"
    )
    start = source.index(_BLOCK_START)
    end = source.index(_BLOCK_END, start)
    raw = source[start:end]
    # 逐行去掉 `#` 之后的内容。这段里没有含 `#` 的字符串字面量，够用。
    return "\n".join(line.split("#", 1)[0] for line in raw.splitlines())


@pytest.mark.cpu_only
def test_cached_state_count_is_committed_only_after_layout_validation():
    block = _acceptance_block()

    assert _LAYOUT_CHECK in block, (
        "缓存接受块里找不到子区间布局校验；本测试的前提没了。"
    )
    assert _COMMIT in block, (
        f"找不到 {_COMMIT!r}。缓存态数的提交点被改名或删掉了——"
        "如果是重构，请确认新写法仍然满足'校验全过之后才提交'，并更新本测试。"
    )

    layout_at = block.index(_LAYOUT_CHECK)
    commit_at = block.index(_COMMIT)
    assert commit_at > layout_at, (
        "`stage2_states` 的提交出现在子区间布局校验**之前**。\n"
        "  布局校验失败时缓存整份被拒，但被改过的 stage2_states 会留下来，\n"
        "  后面的 fresh pilot 就用缓存态数当探针网格密度跑，而 protocol_key\n"
        "  记的还是请求值 —— 下次 resume 会把它当成同一份 pilot 复用。"
    )


@pytest.mark.cpu_only
def test_old_early_commit_pattern_is_gone():
    block = _acceptance_block()
    assert _OLD_BUGGY_COMMIT not in block, (
        f"缓存接受块里又出现了 {_OLD_BUGGY_COMMIT!r}。\n"
        "  这是 2026-09-10 修掉的那个提前提交写法。要用缓存态数，"
        "先存进一个候选变量，等布局校验通过再赋给 stage2_states。"
    )


@pytest.mark.cpu_only
def test_layout_mismatch_branch_does_not_touch_stage2_states():
    """布局失败分支只回滚 λ 表和窗口，绝不去"顺手修正" stage2_states。

    反过来"在失败分支里把 stage2_states 复原成请求值"也是错的写法——那说明
    它已经被改过了。正确做法是压根不提前改，所以这个分支里不该出现这个名字。
    """
    block = _acceptance_block()
    layout_at = block.index(_LAYOUT_CHECK)
    else_at = block.index("else:", layout_at)
    failure_branch = block[layout_at:else_at]

    assert not any(
        "stage2_states" in line and "=" in line
        for line in failure_branch.splitlines()
    ), (
        "布局校验失败分支里出现了对 stage2_states 的赋值。\n"
        "  这个分支应当**什么都不做**——请求值从头到尾没被改过，不需要复原。\n"
        "  出现赋值意味着上游又提前提交了一次。"
    )


# --- 2026-09-10: 探针密度与 λ 节点数必须是两个独立变量 ---

_PROBE = "stage2_probe_states"


@pytest.mark.cpu_only
def test_probe_density_variable_is_never_reassigned():
    """探针网格密度只允许被赋值一次（在 run_full_pipeline 开头解析 config）。

    再出现第二处赋值就意味着又有人把 λ 节点数塞回了探针变量里。
    """
    source = PIPELINE_PATH.read_text(encoding="utf-8")
    assignments = [
        line.strip()
        for line in source.splitlines()
        if re.match(rf"\s*{_PROBE}\s*=[^=]", line)
    ]
    assert len(assignments) == 1, (
        f"{_PROBE} 被赋值 {len(assignments)} 次：{assignments}\n"
        "  探针网格密度是 config 的请求值，从头到尾不该变。"
    )


@pytest.mark.cpu_only
def test_fresh_pilot_uses_probe_density_not_lambda_node_count():
    """重新优化时起 pilot 用的是探针密度，缓存校验用的是最终生产态数。"""
    source = PIPELINE_PATH.read_text(encoding="utf-8")
    fresh_at = source.index(_BLOCK_END)
    fresh_call = source[fresh_at:fresh_at + 4000]
    assert f"n_states={_PROBE}," in fresh_call, (
        f"fresh pilot 的 n_states= 不是 {_PROBE}；探针密度又被别的态数顶掉了。"
    )
    assert '_expected_final_states = _resolve(' in source, (
        "缓存态数校验不再走最终生产态数（stage2_final_n_states）——"
        "拿探针密度比会在每次 resume 上误判不匹配并重烧一整轮 pilot。"
    )
