#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""状态快照文档的过期契约：文档自称"截至/证据截至/整理日期"某天，仓库实际活跃到哪天。

## 为什么需要它

2026-08-24 这次会话里发现：`README.md`/`README_cn.md`/`README_en.md` 全部停在"2026-08-12"，而 EXP-025~030
那条线已经推进到 08-24——过期 12 天没人发现，纯靠这次手动翻文件才暴露。这些文档不会自动
失效、也不会报错，只会安静地过期，直到有人凑巧去核对。

本脚本把"文档自己写的截止日期"和"仓库里真正活跃到哪天"做成两个可比较的数字，超过阈值
就报错，点名是哪个具体文件把"活跃日期"顶上去的——不需要人再去翻十几个文件猜。

## 用法

    python tools/diagnostics/check_doc_staleness.py --root .
    python tools/diagnostics/check_doc_staleness.py --root . --threshold-days 3 --json

只读，不修改任何文件。exit 0 = 全部在阈值内；exit 1 = 至少一个文档过期。

## 阈值为什么是 3 天，不是 0

这些文档是人工维护的进度快照，不是每次改代码就该跟着重写的产物——0 天的阈值会把"今天
刚写完一半"也判成过期，制造噪音。3 天的宽限只是防止"写完当天忘了改日期戳"这种情况被
误报，但足够抓住现在这种 12 天的真实腐烂。
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
# 计算"仓库活跃到哪天"时扫描的文件。只列**发布后仍然长期存在**的东西：
# 生产模块、生产包和生产回归测试。不含实验脚本和过程文档——那一层已经不在本分支。
_FRONTIER_GLOBS = [
    "*.py",
    "local_residual/*.py",
    "tests/test_*.py",
]


class StalenessCheckError(RuntimeError):
    """声明的日期戳解析失败——措辞被改了，正则要跟着改，不能默默跳过。"""


# 每份追踪文档自己的措辞不一样（中文"截至"/"证据截至"/"整理日期"，英文
# "evidence through"），所以必须逐文件配一条正则，不能猜一个通用模式。
# 2026-08-24 直接读过每份文件确认过原文，见下方注释。
_CN_ZHIQI = re.compile(r"截至\s*(\d{4}-\d{2}-\d{2})")
_EN_THROUGH = re.compile(r"evidence through\s*(\d{4}-\d{2}-\d{2})")

TRACKED_DOCS: dict[str, re.Pattern] = {
    # "> 当前科学边界：截至 2026-08-12，软件与方法开发已经形成系统证据……"
    "README.md": _CN_ZHIQI,
    # "## 当前科学状态（证据截至 2026-08-12）"
    "README_cn.md": _CN_ZHIQI,
    # "## Scientific status (evidence through 2026-08-12)"
    "README_en.md": _EN_THROUGH,
    # 2026-08-31 发布整理，追踪表移除了两份：
    #   * curated_project/00_从这里开始/CURRENT_STATUS.md —— 整理版知识库整体
    #     移出工程区分支（原文在 Atenolol-rank11，登记在 docs/HISTORY_LOG.md）。
    #   * docs/README.md —— 重写成纯导航页后不再声明"截至某日的综合判断"，
    #     没有科学状态断言就没有会腐烂的日期戳。追踪一份不声明日期的文档只会
    #     让 _extract_declared_date 硬报错，那是误报不是发现。
    # 剩下三份 README 仍然带 2026-08-12 的科学状态断言，是真正需要盯的对象。
}


@dataclass
class DocStatus:
    relative_path: str
    declared_date: date
    gap_days: int
    ok: bool


@dataclass
class StalenessResult:
    frontier_date: date
    frontier_source: str
    threshold_days: int
    docs: list[DocStatus] = field(default_factory=list)

    @property
    def all_fresh(self) -> bool:
        return all(d.ok for d in self.docs)

    def render_report(self) -> str:
        lines = [
            f"frontier activity date = {self.frontier_date} "
            f"(from {self.frontier_source})",
            f"threshold = {self.threshold_days} day(s)",
            "",
        ]
        width = max((len(d.relative_path) for d in self.docs), default=20)
        for d in self.docs:
            status = "OK   " if d.ok else "STALE"
            lines.append(
                f"[{status}] {d.relative_path:<{width}}  "
                f"declared={d.declared_date}  gap={d.gap_days}d"
            )
        return "\n".join(lines)


def _extract_declared_date(path: Path, pattern: re.Pattern) -> date:
    text = path.read_text(encoding="utf-8")
    match = pattern.search(text)
    if match is None:
        raise StalenessCheckError(
            f"{path}: 找不到日期戳（正则 {pattern.pattern!r} 没匹配到）——"
            "措辞被改了，去更新 TRACKED_DOCS 里对应的正则，不要静默跳过这份文档。"
        )
    return datetime.strptime(match.group(1), "%Y-%m-%d").date()


def frontier_activity_date(root: Path) -> tuple[date, str]:
    """仓库里"当前活跃到哪天"的信号：生产源码与生产回归测试的最新 mtime。

    刻意不用 `output_*/run_provenance.json` 这类正在跑的产物目录做信号——活跃 run
    目录的内容和 mtime 在几小时内就会被跑着的进程改写，拿它当"文档该不该更新"的判据
    不稳定，而且这类目录本来就要在发布时清掉，不是长期存在的文档。

    🔑 [2026-08-31 发布整理] 原来的信号源是根目录/`docs/experiments/` 下的 EXP-0XX
    文档加 `reports/`。那一整层开发期过程材料已经移出工程区分支（原文在
    Atenolol-rank11，登记在 `docs/HISTORY_LOG.md`），于是"一个候选文件都没找到"
    直接把这个检查打成硬错误。现在改用**发布后仍然长期存在**的东西做信号：生产
    源码和 `tests/`。文档该不该刷新，本来就该跟着代码动，而不是跟着实验记录动。
    """
    candidates: list[tuple[float, Path]] = []

    for pattern in _FRONTIER_GLOBS:
        for p in root.glob(pattern):
            if p.is_file():
                candidates.append((p.stat().st_mtime, p))

    if not candidates:
        raise StalenessCheckError(
            "frontier_activity_date: 一个候选文件都没找到——_FRONTIER_GLOBS "
            "可能配错了，这不该发生。"
        )

    mtime, path = max(candidates, key=lambda item: item[0])
    return datetime.fromtimestamp(mtime).date(), str(path.relative_to(root))


def run(root: Path, threshold_days: int = 3) -> StalenessResult:
    frontier_date, frontier_source = frontier_activity_date(root)
    result = StalenessResult(
        frontier_date=frontier_date,
        frontier_source=frontier_source,
        threshold_days=threshold_days,
    )
    for relative_path, pattern in TRACKED_DOCS.items():
        doc_path = root / relative_path
        if not doc_path.exists():
            raise StalenessCheckError(f"追踪的文档不存在: {relative_path}")
        declared = _extract_declared_date(doc_path, pattern)
        gap_days = (frontier_date - declared).days
        result.docs.append(
            DocStatus(
                relative_path=relative_path,
                declared_date=declared,
                gap_days=gap_days,
                ok=gap_days <= threshold_days,
            )
        )
    return result


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="仓库根目录")
    parser.add_argument(
        "--threshold-days", type=int, default=3, help="超过多少天算过期（默认 3）"
    )
    parser.add_argument("--json", action="store_true", help="输出 JSON 而不是表格")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    result = run(root, threshold_days=args.threshold_days)

    if args.json:
        import json

        payload = {
            "frontier_date": str(result.frontier_date),
            "frontier_source": result.frontier_source,
            "threshold_days": result.threshold_days,
            "all_fresh": result.all_fresh,
            "docs": [
                {
                    "relative_path": d.relative_path,
                    "declared_date": str(d.declared_date),
                    "gap_days": d.gap_days,
                    "ok": d.ok,
                }
                for d in result.docs
            ],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(result.render_report())

    return 0 if result.all_fresh else 1


if __name__ == "__main__":
    raise SystemExit(main())
