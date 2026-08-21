from __future__ import annotations

import re
import sys
from pathlib import Path


def table_blocks(lines):
    blocks = []
    i = 0
    while i < len(lines):
        if lines[i].lstrip().startswith("|"):
            start = i
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                i += 1
            blocks.append((start, i, lines[start:i]))
        else:
            i += 1
    return blocks


def extract_language(source: str, lang: str) -> str:
    lines = source.splitlines()
    zh_start = lines.index("## 第一部分　中文正文") + 1
    en_start = lines.index("## Part II — English Version") + 1
    appendix_start = lines.index("## 附录 / Appendix")

    appendix = lines[appendix_start + 1 :]
    blocks = table_blocks(appendix)
    keep_prefixes = ("| 中文模块", "| 中文术语") if lang == "zh" else ("| English module", "| English term")
    kept_tables = [block for _, _, block in blocks if block and block[0].startswith(keep_prefixes)]

    if lang == "zh":
        title = "# ABFE-IBS 技术报告"
        subtitle = "# 从物理路径、采样分布到当前证据边界"
        notes = [
            "> 整理日期：2026-08-20。本文围绕科学问题、实际数学对象、数值证据与结论边界重写。",
            "> 原始 JSON、日志、checkpoint、预注册协议与源代码中的可复核事实优先于叙述性文字。",
        ]
        body = lines[zh_start:en_start - 1]
        appendix_lines = ["## 附录", "", "### A. 主线代码职责", ""] + kept_tables[0] + ["", "### B. 术语边界", ""] + kept_tables[1]
    else:
        title = "# ABFE-IBS Technical Report"
        subtitle = "# From the Physical Path and Sampling Distribution to the Current Evidence Boundary"
        notes = [
            "> Consolidation date: 2026-08-20. This report is organized around the scientific question, implemented mathematical objects, numerical evidence, and claim boundaries.",
            "> Auditable facts in raw JSON artifacts, logs, checkpoints, preregistration protocols, and source code take precedence over narrative text.",
        ]
        body = lines[en_start:appendix_start]
        renumbered = []
        for line in body:
            match = re.match(r"^(#{2,6}\s+)(1[1-9]|20)(\.\s+.*)$", line)
            if match:
                number = int(match.group(2)) - 10
                line = f"{match.group(1)}{number}{match.group(3)}"
            renumbered.append(line)
        body = renumbered
        appendix_lines = ["## Appendix", "", "### A. Mainline code responsibilities", ""] + kept_tables[0] + ["", "### B. Terminology boundary", ""] + kept_tables[1]

    output = [title, "", subtitle, ""] + notes + ["", "---", ""] + body + [""] + appendix_lines + [""]
    return "\n".join(output)


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: split_story_sources.py INPUT.md ZH.md EN.md")
    src = Path(sys.argv[1]).read_text(encoding="utf-8")
    zh = Path(sys.argv[2])
    en = Path(sys.argv[3])
    zh.write_text(extract_language(src, "zh"), encoding="utf-8")
    en.write_text(extract_language(src, "en"), encoding="utf-8")
    print(f"wrote {zh}")
    print(f"wrote {en}")


if __name__ == "__main__":
    main()
