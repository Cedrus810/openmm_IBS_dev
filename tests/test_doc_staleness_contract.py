"""状态快照文档的过期契约（2026-08-24 新增，防再次静默过期）。

## 这份文件里的两条测试分别代表什么

* `test_staleness_checker_finds_the_known_stale_docs` —— **检测机制本身**能不能
  正确工作。它必须一直通过；不通过说明检测器坏了（通常是文档措辞改了、
  `check_doc_staleness.TRACKED_DOCS` 的正则没跟着改）。
* `test_snapshot_docs_are_not_stale` —— **文档内容本身**是不是新鲜。

## 2026-09-02：`xfail(strict=True)` 已摘

第二条测试从 2026-08-24 新增起一直挂着 `xfail(strict=True)`，因为
`README.md`/`README_cn.md`/`README_en.md` 的科学状态全部停在 2026-08-12，
而 2026-08-31 的发布整理只合并了目录结构、**没有**替维护者定科学结论。

2026-09-02 那三份文档的科学状态被真正改写了（主线体系 Atenolol → 4W53、
旧 `output_lrc_fix` 的 `−23.1622` 标为已作废、协议版本号从转述改为直接读源码
常量、日期戳刷到 09-02），检测器转全绿，于是标记按它自己 reason 里写的约定摘掉。

**这两条现在都是普通测试。** 第二条红了就是三份 README 又落后于仓库前沿超过
阈值——去更新文档，不要改测试、不要放宽 `threshold_days`。
"""

from __future__ import annotations

import sys
from pathlib import Path

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


def test_snapshot_docs_are_not_stale():
    """三份 README 的科学状态日期戳必须跟得上仓库前沿。

    2026-09-02：`xfail(strict=True)` 标记已摘掉。原因是那三份文档的科学状态
    在这一天被真正改写了——主线体系从 Atenolol 换成 4W53、旧
    `output_lrc_fix` 的 `−23.1622` 标为已作废、协议版本号从转述改为直接
    读源码常量、日期戳刷到 2026-09-02。此前这个测试挂 `xfail` 是因为
    2026-08-31 的发布整理只合并了目录结构，**没有**替维护者定科学结论。

    从现在起这是一条**普通测试**：它红了就说明三份 README 又落后于仓库前沿
    超过阈值，去更新文档，不要来改这个测试或阈值。
    """
    result = staleness.run(ROOT, threshold_days=3)
    assert result.all_fresh, "\n" + result.render_report()
