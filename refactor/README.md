# refactor/ — 全量重构工作台

**2026-09-20 重组**：顶层 52 条目 → 6。主线（仓库根的 .py / tests/）只读，一切写操作只在本目录内。

## 目录结构

| 目录 | 内容 | 性质 |
|---|---|---|
| `domains/` | 17 个功能域（切分产物 = 代码） | 与主线逐字对照的副本，跑不起来（缺 import） |
| `tools/` | 8 个长期工具 | **门**：verify_extract / verify_rename / verify_dispatch / verify_a1 / compound_gate_knife1；**体检**：deps.py / layer_check.py；a1_tool（A1 落地待用） |
| `cuts/` | 16 个 `_*.py` 一次性执行器 + `_shadow_files*.txt` | 存档，不再运行；每个对应一刀，审计留档用 |
| `docs/` | HOW_TO_CONTINUE / CUTTING_PLAN / REFACTORING_GUIDE / WIRING / b3_edge_decisions | 文档 |
| `work/` | PROGRESS.md、留底 `*.before.py`、名单 `*.txt`、`verify/*.verify.sh` ×16、a1_proposals/ | 工作数据 |

## ★ 2026-09-23：主线重放已通

`python3 refactor/tools/replay.py history|mainline [--shadow]` —— 19 刀对**当前主线**整链重放，
每刀过门（含新门 `verify_bound.py`）。`domains/` 降为死档（含 4 个历史 bug）。
现状、挂起项与历史 bug 表见 `work/PROGRESS.md` 顶部「★ 2026-09-23 现状」。

## 新会话阅读顺序

1. `docs/HOW_TO_CONTINUE.md` —— 路线（A→B→C）与授权边界
2. `docs/CUTTING_PLAN.md` —— 切什么
3. `docs/REFACTORING_GUIDE.md` —— 怎么切（门/留底/执行器纪律）
4. `work/PROGRESS.md` —— 现在到哪了、下一刀

## 工具用法（一律在仓库根执行）

```bash
# 门（每次下刀后必须 PASS，命令见 work/verify/<刀名>.verify.sh）
python3 refactor/tools/verify_extract.py <before> <after> --moved <fn> --src <fn> ...
python3 refactor/tools/verify_rename.py  <before> <after> --fn <fn> --bind <obj> --names ... [--in-place] [--expand-kwarg]
python3 refactor/tools/verify_dispatch.py <before> <after> --src <fn> --pos <i>

# 体检
python3 refactor/tools/layer_check.py   # 17 域口径基准：1357 次 / 107 边，刀前后必须不变
python3 refactor/tools/deps.py          # 噪音口径（含 tools/cuts/work），看环与叶子

# 复验（整链逐刀重放，16/16 必须全绿）
for f in refactor/work/verify/*.verify.sh; do bash "$f"; done
```

## 指标基准（刀前后必须不变）

- **layer_check（17 域口径）：跨域一方引用 1357 次 / 107 条边**；文件 81 / 域 17
- 变了 = 动到了符号归属，按 REFACTORING_GUIDE §八处理，不许解释
