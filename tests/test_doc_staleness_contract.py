"""文档契约：协议版本表钉在源码常量上，`docs/` 内部链接必须解析得到。

## 2026-10-27：公开发布整理

`docs/STATUS.md` 移进 `docs/archive/`（内部材料，不随公开版发布）。随之：

* 删掉 `test_staleness_checker_finds_the_known_stale_docs` / `test_snapshot_docs_are_not_stale`
  两条「状态快照日期戳要跟上仓库前沿」的测试 —— 被追踪的文档不在公开版里，
  公开版 CI 上它们只会因为文件不存在而红。检测器 `tools/diagnostics/check_doc_staleness.py`
  留着，本地对 `docs/archive/STATUS.md` 手动跑。
* 协议版本表搬进 `docs/METHODS.md`「协议版本」，`test_protocol_table_matches_source`
  改钉那一份。**日期戳只能抓"整份文档忘了刷"，抓不到"表里某个数字烂了"**
  （2026-09-05 去重时热力学路径版本在四份文档里有三份还写着 21，源码早是 22）。

红了就去更新文档，不要改测试。
"""

from __future__ import annotations

import re
import sys

import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))



pytestmark = pytest.mark.cpu_only

def test_protocol_table_matches_source():
    """`docs/METHODS.md` 的协议版本表必须等于源码常量。

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

    text = (ROOT / "docs" / "METHODS.md").read_text(encoding="utf-8")
    # 表格行形如：| IBS 偏置 | `ibs_engine.IBS_BIAS_PROTOCOL_VERSION` | 32 |
    rows = dict(re.findall(r"\|\s*`([\w.]+)`\s*\|\s*(\d+)\s*\|", text))

    assert set(rows) == set(expected), (
        "docs/METHODS.md 的协议表和这条测试的清单对不上——"
        f"文档里有 {sorted(rows)}，测试期望 {sorted(expected)}。"
        "加了新协议常量就把两边一起加上。"
    )
    for name, value in expected.items():
        assert int(rows[name]) == value, (
            f"docs/METHODS.md 写 {name} = {rows[name]}，源码是 {value}——去更新文档。"
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
