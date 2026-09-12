"""状态快照文档的过期契约（2026-08-24 新增，防再次静默过期）。

## 这份文件里的两条测试分别代表什么

* `test_staleness_checker_finds_the_known_stale_docs` —— **检测机制本身**能不能
  正确工作。它必须一直通过；不通过说明检测器坏了（通常是文档措辞改了、
  `check_doc_staleness.TRACKED_DOCS` 的正则没跟着改）。
* `test_snapshot_docs_are_not_stale` —— **文档内容本身**是不是新鲜。

* `test_status_doc_protocol_table_matches_source` —— `docs/STATUS.md` 的协议
  版本表和源码常量是不是还对得上。

## 2026-09-05：追踪对象从三份 README 换成 `docs/STATUS.md`

科学状态、结果登记和协议版本表原来在 `README.md` / `README_cn.md` /
`README_en.md` / `docs/README.md` 各存一份。四份手工同步的代价照例没人付：
`THERMODYNAMIC_PATH_PROTOCOL_VERSION` 早已经是 22，四份里有三份还写着 21。

现在唯一声明科学状态的是 `docs/STATUS.md`，四份文档只留一行指向它。追踪表跟着
换成那一份（阈值仍是 3 天，没放宽）；同时新增第三条测试，把那张协议版本表直接
钉在源码常量上——**日期戳只能抓"整份文档忘了刷"，抓不到"表里某个数字烂了"。**

**三条现在都是普通测试。** 红了就去更新文档，不要改测试、不要放宽阈值。
"""

from __future__ import annotations

import re
import sys

import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "diagnostics"))

import check_doc_staleness as staleness  # noqa: E402



pytestmark = pytest.mark.cpu_only

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
    """`docs/STATUS.md` 的整理日期戳必须跟得上仓库前沿。

    红了就说明状态文档又落后于仓库前沿超过阈值，去更新文档，不要来改这个
    测试或阈值。

    🔑 [2026-09-09] 前沿日期改成 **git 提交日期**（见
    `check_doc_staleness.frontier_activity_date`）。原来用文件 mtime：mtime 不是
    内容派生的，`git clone` / checkout / 解压 tarball 会把所有文件盖成当次操作
    时间，于是这条测试在任何一次全新 checkout 上都**必然**失败（前沿 = clone 当天，
    文档日期是几天前），而仓库内容一字未改 —— CI 每次都是新 clone，等于定时炸弹。
    拿不到 git 元数据时**跳过**而不是拿 mtime 硬判：那种环境里这个判据没有意义。
    """
    result = staleness.run(ROOT, threshold_days=3)
    if result.frontier_source.startswith("mtime:"):
        pytest.skip(
            "拿不到 git 提交日期（不是 git 仓库 / 无 git / 导出的 tarball），"
            f"只能退回 mtime（{result.frontier_source}）。mtime 会被 clone/checkout "
            "整体盖写，用它判过期只会误报，故跳过。"
        )
    assert result.all_fresh, "\n" + result.render_report()


def test_status_doc_protocol_table_matches_source():
    """`docs/STATUS.md` 的协议版本表必须等于源码常量。

    日期戳只能抓"整份文档忘了刷新"，抓不到"表里某个数字悄悄烂了"——2026-09-05
    去重时就发现热力学路径版本在四份文档里有三份还写着 21，源码早是 22。
    这条测试把那张表钉死：改了常量而没改表，这里红。
    """
    import abfe_preoptimizer
    import ibs_engine

    expected = {
        "ibs_engine.IBS_BIAS_PROTOCOL_VERSION": ibs_engine.IBS_BIAS_PROTOCOL_VERSION,
        "abfe_preoptimizer.THERMODYNAMIC_PATH_PROTOCOL_VERSION": (
            abfe_preoptimizer.THERMODYNAMIC_PATH_PROTOCOL_VERSION
        ),
        "ibs_engine.TRADITIONAL_LJ_LRC_PROTOCOL_VERSION": (
            ibs_engine.TRADITIONAL_LJ_LRC_PROTOCOL_VERSION
        ),
        "ibs_engine.WCA_ACCOUNTING_VERSION": ibs_engine.WCA_ACCOUNTING_VERSION,
        "ibs_engine.ESS_GATE_PROTOCOL_VERSION": ibs_engine.ESS_GATE_PROTOCOL_VERSION,
        "ibs_engine.LIGAND_COM_RESTRAINT_PROTOCOL_VERSION": (
            ibs_engine.LIGAND_COM_RESTRAINT_PROTOCOL_VERSION
        ),
    }

    text = (ROOT / "docs" / "STATUS.md").read_text(encoding="utf-8")
    # 表格行形如：| IBS 偏置 | `ibs_engine.IBS_BIAS_PROTOCOL_VERSION` | 32 |
    rows = dict(re.findall(r"\|\s*`([\w.]+)`\s*\|\s*(\d+)\s*\|", text))

    assert set(rows) == set(expected), (
        "docs/STATUS.md 的协议表和这条测试的清单对不上——"
        f"文档里有 {sorted(rows)}，测试期望 {sorted(expected)}。"
        "加了新协议常量就把两边一起加上。"
    )
    for name, value in expected.items():
        assert int(rows[name]) == value, (
            f"docs/STATUS.md 写 {name} = {rows[name]}，源码是 {value}——去更新文档。"
        )


def test_docs_internal_links_resolve():
    """`docs/` 里所有相对 Markdown 链接必须指得到真实文件。

    2026-09-09 加：那次把 `BUG_LOCATION_…` 移进 `archive/` 时要同步改 32 处引用，
    漏一处就是坏链。`docs/README.md`《归档前必查》记着同类的病已经发生过一次
    （`.py` 里 28 处指向 `docs/status/` 等不存在路径）。
    **本测试只覆盖 `docs/` 内部链接**；代码注释里的路径不在范围内（那批是刻意留的债）。
    """
    import re

    broken = []
    for md in sorted((ROOT / "docs").rglob("*.md")):
        for match in re.finditer(r"\[[^\]]*\]\(([^)]+)\)", md.read_text(encoding="utf-8")):
            target = match.group(1).split("#")[0].strip()
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            if not (md.parent / target).resolve().exists():
                broken.append(f"{md.relative_to(ROOT)} -> {target}")
    assert not broken, "docs/ 里有坏链：\n  " + "\n  ".join(broken)
