"""状态快照文档的过期契约（2026-08-24 新增，防再次静默过期）。

## 现状说明——这份文件里的两类测试分别代表什么

跟 `tests/test_open_issue_fail_closed_contracts.py` 用的是同一套约定：

* `test_staleness_checker_finds_the_known_stale_docs` 是**普通测试**，描述"过期检测机制
  本身能不能正确工作"——它必须一直通过，不通过说明检测器坏了。
* `test_snapshot_docs_are_not_stale` 用 `xfail(strict=True)` 标记——它描述"文档内容
  本身是不是新鲜"，**这次新增时就是预期失败的**：`README*`/`docs/README.md`/
  `curated_project/00_从这里开始/CURRENT_STATUS.md` 全部还停在 2026-08-12，而仓库实际
  活跃到 08-24。等有人把这几份文档的日期戳刷新到位，这个测试会意外通过（XPASS），
  `strict=True` 会把 XPASS 当成错误报出来，提醒维护者把 `xfail` 标记摘掉——这正是这个
  标记要做的事：逼着"文档已经刷新"这件事被显式确认一次，而不是让检测器从红变绿却没人
  注意到已经不需要再盯着它了。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "diagnostics"))

import check_doc_staleness as staleness  # noqa: E402


def test_staleness_checker_finds_the_known_stale_docs():
    """检测器本身必须一直能跑、一直能正确解析日期戳——这个测试不允许红。"""
    result = staleness.run(ROOT, threshold_days=3)

    # 前沿日期必须来自仓库里真实存在的一个候选源，不能是空的/回退值。
    assert result.frontier_date is not None
    assert result.frontier_source

    # 追踪表里每份文档都必须被解析出一个声明日期——解析失败会在 run() 内部直接
    # raise StalenessCheckError（措辞被改了要去更新正则），不会静默漏掉。
    tracked_paths = {d.relative_path for d in result.docs}
    assert tracked_paths == set(staleness.TRACKED_DOCS)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "README.md/README_cn.md/README_en.md 的科学状态日期戳"
        "仍停在 2026-08-12。2026-08-31 的发布整理只做了目录与文档结构的合并，"
        "**没有**替维护者改写这几份文档里的科学结论和候选数值——那需要人来定。"
        "刷新日期戳后这个测试会 XPASS，strict=True 会报错提醒摘掉这个标记。"
    ),
)
def test_snapshot_docs_are_not_stale():
    result = staleness.run(ROOT, threshold_days=3)
    assert result.all_fresh, "\n" + result.render_report()
