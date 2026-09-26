# 全量重构路线图 —— 范围、操作规范、A→B→C

**写于 2026-09-19，配 `WIRING.md` 读。**

**终点：整个仓库重构成分层包。** 本文不讨论要不要做，只讨论**按什么顺序做**、
**动什么不动什么**、**每一刀怎么走**。

数字全部由 `refactor/tools/deps.py` 复现（纯 AST，不 import 被测代码）：

```
python3 refactor/tools/deps.py      # 仓库根执行
```

---

# 第 0 章　授权边界（先读这一章）

## 0.1 处方 ≠ 授权

**本文列出的任何操作都不构成对主线文件的改动授权。**
每一个阶段开工前必须单独拿到用户明确的「动手」指令。
「文档里写了」「上一刀批过了」「这步零风险」——**都不是授权**。
一次授权只覆盖它点名的那一刀，不向后延伸。

## 0.2 绝对禁区（任何授权都不覆盖）

| 禁区 | 说明 |
|---|---|
| `benchmark/` 树、run 目录、`output/` | **实验数据，只读。** 不改不删不「顺手清理」 |
| 任何 `.json` / `.dcd` / `.chk` / checkpoint 产物 | 同上 |
| 缓存身份路径（`*_key`、`*_fingerprint`、进 key 的 sha256、落盘 JSON 的字节形态） | 改动作废存量 GPU 结果；动之前先算代价摆给用户 |
| `docs/CHANGELOG.md` 已结案条目、`docs/archive/` | 只增不改 |
| 已否决方案（`no_ghost_ramp`、`warmup_profile`、f_k 负号、ESS 门进验收） | 不碰、不重提 |
| `git` 操作（commit / push / branch / stash） | 用户自己管，改完不问「要不要提交」 |

## 0.3 越界的定义

命中任一条即为越界，立刻停手并报告：

1. 改了当次授权点名之外的文件；
2. **一刀里同时改了文件边界和代码行为**；
3. 为让 import 通过而新增 lazy import / 函数内 import / `try: import except:`；
4. 为让测试变绿而改**断言内容**（只允许改定位方式，见 3.4）；
5. 动了 0.2 禁区；
6. 没跑 `./tests/run_offline_tests.sh` 就宣布某步完成。

---

# 第 1 章　现状

| 事实 | 验证 | 结果 |
|---|---|---|
| `refactor/` 切分逐行无损 | 两侧 `cat \| sort \| md5sum` | 同为 `814c252c3652c9b60349d46329b57ccc` |
| 净删行数 | 99983 → 99983 | 0 |
| 主线改动 | `git status` | 零 |
| 文件数 / 平均行数 | 17 → 80 / 5881 → 1250 | 量级未变 |

`refactor/` 的价值是**全量重构的域划分草稿 + 一份可对账的地图**，不是可运行的包。
它是 freeze 点，不是要长期跟主线并行的副本——主线一动它就是死档。

**4 个巨定义整块未切**，它们才是最终要下刀的地方：
`ABFEPipeline` 15446 行、`IBSWindowManagerDualLambda` 5407、
`run_full_pipeline` 3376、`Stage2RepairController._decide_once` 2791。

---

# 第 2 章　全量重构的真阻塞是什么

## 2.1 不是那 30 个环

现状：1336 次跨域引用、97 个有向域对、**30 个循环域对**。

> **勘误（2026-09-20）**：上行为 2026-09-19 **infra 抽取前**的基线。抽取后实测
> （`layer_check.py`，17 域口径）：跨域一方引用 **1357 次 / 107 条有向域对边**，
> 合法(严格高→低) 913/62、违反(低→高) 130/23、同层 314/22，
> 文档口径非合法(同层+违反) **444 次 / 45 边**。本文下文出现的 1336/707/629/44
> 均为抽取前基线，读时按此折算；infra 名单与 B3 逐边决策的**相对结构**不受影响。

增量重构里环是阻塞（每拆一个都要造临时 shim，最后全扔）。
**一次性重构里环不是阻塞**——先定目标分层，环由构造解决（见 4.B）。
所以上一版文档里「环降到 0 之前不具备接线条件」那句，在全量重构前提下**不成立，已撤回**。

## 2.2 是那 177 处源码文本断言

`tests/` 共 215 个测试。扒开之后（**口径 = `refactor/tools/a1_tool.py --check` 的 AST 统计，
2026-09-19 实测；本表旧版是正则上界 431，作废**）：

| 类别 | 处数 | 形式 | 全量重构后会怎样 |
|---|---|---|---|
| **C2 好写法必须存在** | **119** | `assert '<实现原文>' in src` | 文件一拆即失效 |
| **C1 坏写法不得出现** | **35** | `assert '<旧的错写法>' not in src` | 同上，**而且失效时是静默的**：断言恒真 |
| **C3 唯一性计数** | **18** | `assert src.count("np.savez(handle, **arrays)") == 1` | 拆文件后**语义无法表达** |
| **C4 顺序/定位** | **1** | `assert fn.index("if k == 0:") < fn.index(...)` | 同 C1 |
| C5 其他 | 4 | — | 基本不受影响 |
| （另有 9 处非 assert 的 `.index/.find/.count` 定位辅助） | 9 | — | 参与收窄安全性判断 |

涉及 61 个测试文件；另有 7 处 `spec_from_file_location`（A 验收②口径：7 → 0）。
断言定位状态实测：auto=74、ambiguous=60、module_level=4、c1_no_target=35、
**not_found=0、c1_exposed=0**——守卫没有钉错地方的。

**C1 那 62 条是最危险的。** 它们是这个仓库把「别再犯」钉死的方式——
f_k 负号、ESS 门进验收、帧数门、缓存身份、覆盖式写法……守卫全在这里。
`assert "abs(e) > 1e5" not in body` 这种断言，文件一改名 `body` 取不到，
**测试不报错，它只是永远通过**。全量重构会一次性废掉这 62 条守卫，
而且你不会知道废了哪几条。

在 10 万行、唯一能判断 ΔG 有没有被改坏的东西就是这套测试的情况下，
**重构期间没有绿灯信号 = 不可接受。**

## 2.3 好消息：仓库已有 AST 先例

**66 个测试已经在用 `ast.parse` 做结构断言**（`test_att27_dead_code_removed.py`、
`test_no_undefined_globals.py`、`test_stage2_controller_module_boundary.py` …）。
第 4 章的 A 不是发明新范式，是把剩下的那批也搬过去。

---

# 第 3 章　操作规范（每一刀都照走）

## 3.1 一刀的定义

**一刀 = 一个能独立跑完整测试并全绿的最小改动。**
不能独立跑绿的不叫一刀，拆到能为止。

## 3.2 六步流程（缺一步作废）

```
① 授权   说明「这一刀改哪些文件的哪些行、为什么」，拿到明确同意
② 留底   记下待改区间 md5：sed -n 'A,Bp' FILE | md5sum
③ 改     只改授权区间。纯搬运与改行为绝不同刀
④ 自检   python3 refactor/tools/deps.py     违反分层的边数必须单调下降或持平
         纯搬运刀额外：搬前搬后 sort|md5sum 相同
⑤ 验收   ./tests/run_offline_tests.sh  （远程 kasuga02，ssh 后先 cd 仓库根）
         0 failed 才算过。本地 pytest 只做快速反馈，不作数
⑥ 报告   报实测结果。失败就贴输出，不修饰、不说「应该没问题」
```

## 3.3 纯搬运 vs 改行为

| 刀型 | 内容 | 验收 |
|---|---|---|
| **纯搬运** | 移动整块定义、加 import、加 `__init__.py` | 行内容逐字不变，`sort \| md5sum` 前后相同 |
| **改行为** | 合并实现、删死代码、改签名 | 等价性需逐份 diff 证明；测试全绿 |

**两者绝不混在一刀里。** 混了出事没法二分定位。

## 3.4 测试怎么改

| 情况 | 允许 |
|---|---|
| **定位方式**：`Path("x.py").read_text()` → `inspect.getsource(对象)` | ✅ 这是 A 阶段的全部内容 |
| 文件名字符串、`spec_from_file_location` 路径 | ✅ |
| **断言的字面量、阈值、期望值、比较方向** | 🚫 **一律不许改。** 断言挂了说明代码行为变了，回去改代码 |
| 删测试 / 加 `skip` / 加 `xfail` | 🚫 一律不许 |
| 为了匹配新文本而改断言字面量 | 🚫 **这等于静默丢守卫**，是 A 阶段最容易犯的错 |

## 3.5 模块级可变全局

`runtime_env` 域持有 7 个被 `global` 改写的模块级变量：

```
_TORCH_MODULE  _MLPOTENTIAL_CLASS  _ORB_PROBED      ← abfe_core:123 处 global 改写
_PYMBAR_MODULE _PYMBAR_PROBED                       ← abfe_core:160
_MDTRAJ_DCD_QUIET_PATCHED                           ← abfe_core:7550
_DEVICE_SETTINGS_CACHE                              ← abfe_core:7840
```

**🚫 不许用 `from X import *` 做 re-export。**
`import *` 复制的是绑定快照：新模块里 `global _PYMBAR_MODULE` 重新绑定后，
`abfe_core._PYMBAR_MODULE` 仍指向 `None`，探测结果永远丢失，**且不报错**。
正确做法：`import runtime_env` + 属性访问，或只 re-export 函数不 re-export 变量。
`test_import_time_side_effects.py` / `test_cli_diagnostics_and_lazy_imports.py`
钉的就是这套惰性探测语义，它们挂了别改测试。

## 3.6 回滚

每一刀的回滚方式必须在开工前就存在：纯搬运靠 md5 留底逐段还原，
改行为靠 `git diff` 反向 apply。**没有回滚路径的刀不开工。**

---

# 第 4 章　路线：A → B → C

```
A 测试守卫解耦   ← 把「全量重构=鲁莽」变成「全量重构=可验证」
B 目标分层定稿   ← 一次性切的方向
C 全量切         ← A 给验证，B 给方向
```

A 和 B **互不依赖，可以并行**。C 必须等两者都完。

---

## A　测试守卫解耦（177 处 → 不再绑文件路径）

**范围：只动 `tests/`。主线一行不改。**
这是全表唯一「全程测试可绿、零物理风险」的阶段——因为它是在加固测试，不是搬代码。
做完之后，文件怎么拆、怎么改名、怎么分包，守卫都不掉。

### A1　整文件文本 → `inspect.getsource(对象)`（主体，C2 119 处）

把 `src = (REPO/"ibs_engine.py").read_text()` 换成
`src = inspect.getsource(ibs_engine.SomeClass.some_method)`，
**断言字面量一字不改**。守卫强度不变，但从「绑文件路径」变成「绑代码对象」。

做法：转换脚本已就位（`refactor/tools/a1_tool.py`，提案在 `refactor/work/a1_proposals/`，
2026-09-19 生成 43 个 diff），落地前人工逐条过。

### A2　包级 AST 扫描（C3 34 处 + C4 11 处 + C1 里的「全仓不得出现」）

这些 `inspect.getsource` 覆盖不了：

- **C3 唯一性**：`src.count("np.savez(handle, **arrays)") == 1`
  语义是「全仓只有一处这样写」。拆包后要改成**遍历整个包的 AST**，
  数匹配节点个数。转完比现在更强（现在只管一个文件）。
- **C4 顺序**：`fn.index("minimizeEnergy") < fn.index("_extend_state_trajectory(")`
  → AST 里比较两个语句节点的位置。
- **C1 全仓禁令**：`assert "abs(e) > 1e5" not in src`
  → 包级 AST 扫描 + 白名单。

先例在 `test_att27_dead_code_removed.py`（已用 `ast.Return` 断言结构）、
`test_no_undefined_globals.py`。照抄范式即可。

### A3　C2 里真正该变行为测试的（**明确押后，不在本轮**）

`assert "raise RuntimeError(" in body` 这类，正解是调用函数验证它真的抛。
但 288 条逐条改是无底洞，**A1 做完它们已经不再阻塞重构**，
本轮到此为止。谁想做谁单独立项。

### A 的验收

```
① tests/ 里 (REPO / "*.py").read_text() 的出现次数 → 0
② spec_from_file_location 指向主线 .py 的 7 处 → 0
③ ./tests/run_offline_tests.sh  0 failed
④ 「收窄后暴露钉错地方」一项已实测为 0（not_found=0 / c1_exposed=0，
   2026-09-19 a1_tool 扫描）——若后续扫描再出现非零，恢复逐条报告的规矩
```

---

## B　目标分层定稿（草案已在下面，等你拍板）

**范围：只产出文档，不动任何代码。**

现在只有现状图，没有目标图。一次性重构里环靠分层由构造解决，
**但前提是层次先定死**。

### B1　分层草案（16 域 → 5 层）

| 层 | 域 | 说明 |
|---|---|---|
| **L0** | `runtime_env` | 依赖探测、惰性 import、设备设置。出边已实测为 0 |
| **L1** | `misc` `result_schema` `orb_surrogate` `rbfe` `outer_lambda_neural` | 独立子系统 / 纯契约 |
| **L2** | `lambda_path` `system_build` `charging` `membrane` `boresch` | 物理与体系构建 |
| **L3** | `mbar_analysis` `windows_sampling` `remd` `stage2_controller` | 采样与分析 |
| **L4** | `pipeline_orchestration` | 编排，只被 CLI 引用 |

**实测这个草案的效果**（`refactor/tools/layer_check.py` 跑出来，2026-09-19；三桶口径）：

```
1336 次跨域引用 = 合法 + 同层 + 上升三桶：
  合法（严格高层 → 低层）   707 次 / 53 边   ← 不用管
  同层                      353 次 / 21 边   ├─ 互为依赖（真 import 环）9 对 / 325 次 ← 仍必须拍板
                                             └─ 单向（真·内聚）        3 条 /  28 次 ← 不用管
  上升违反（低层 → 高层）    276 次 / 23 边   ← 必须拍板
```

**拍板集 = 23 条上升边 + 9 对同层互环 = 32 项 / 601 次**。
（旧版「629 次/44 条」把同层全算违反，作废；同层互环不违反分层，但照样是
ImportError 决策。L2 里 lambda_path/charging/membrane/boresch/system_build
五域互相咬成 8 对，那部分当一个整体决策，不逐边过。）

### B2　第一条决策已经很明显：抽一个 L0 `infra` 域

629 次（同层+上升旧口径）里，贡献最大的名字是：

```
104  [const]  log                    ← 日志句柄
 33  [const]  logger
 29  [def]    NumpyEncoder
 28  [def]    _sha256_file
 21  [def]    _atomic_write_json
 14  [def]    print_once
 12  [def]    _timed
 12  [def]    _atomic_save_npy
─────────────────────────────
253 / 629  =  40%（同层+上升口径；纯上升口径下 infra 名单贡献 180 次，
                     276 → 96，见 layer_check.py 的 infra 抽取模拟）
```

全是**基础设施工具搁浅在高层**。单是 `system_build → pipeline_orchestration`
那条 89 次的反向边，**81 次是常量、其中 72 次是一个 `log`**。

抽一个 L0 `infra`（日志 + 原子写 + 摘要 + 计时 + 编码器 + 常量），
**40% 的引用自动消失（整条消掉 4 条边，其余边削薄）**，而且一行业务逻辑都不用碰。

⚠️ 这里正好撞上主线的重复实现，逐份实测结论如下（**标 🚫 的不许在本阶段合并**）：

| 名字 | 各份位置 | 实测 | 处理 |
|---|---|---|---|
| `_log_print` | `abfe_core:7400`　`ibs_engine:322`　`abfe_pipeline:497` | 三份 md5 全同 `a32d15a7` | ✅ 可合 |
| `_infer_log_level_from_message` | `abfe_core:7392`　`ibs_engine:314`　`abfe_pipeline:489` | 三份 md5 全同 `0a2fa01c` | ✅ 可合 |
| `print_once` + `_ONCE_EMITTED_KEYS` | `abfe_core:7415-7430`　`ibs_engine:337-354` | 代码体逐字相同，仅 docstring 不同 | ✅ 可合（留 `ibs_engine` 那份 docstring） |
| `_split_platform_spec` | `ibs_engine:3180`　`abfe_pipeline:2810` | 代码体逐字相同，仅多一行 docstring | ✅ 可合 |
| `_sha256_file` / `_file_sha256` | `ibs_engine:368`　`runabfe:428`　`apbs_correction:104` | 三份摘要结果相同（差异只在形参名与 `open()`/`Path.open()`） | ✅ 这三份可合 |
| ⟨同上⟩ `abfe_pipeline:1112` | — | **`except OSError: return None`，语义不同** | 🚫 **不许并入。** sha256 进过缓存身份路径，None 与抛错的差别会静默改缓存键 |
| `_atomic_write_json*` | `ibs_engine:1201`　`rbfe_pipeline:205`　`runabfe:3879` | **三份落盘字节都不同**：`sort_keys=True,ensure_ascii=False` / `cls=NumpyEncoder` / tmp 名不带 PID（并发互踩）且失败不清理 | 🚫 **本阶段不合。** 统一任一份的参数都会改 JSON 字节 ⟹ 下游 sha256 变 ⟹ 缓存身份变。单独立项，先算存量代价 |

**⛔ 已撤回的误报**：`_minimum_image_displacement` 那组**不是重复实现**。
主线 `abfe_pipeline:1017`、`ibs_engine:1257`、`ibs_engine:12874` 三份都是薄包装，
全部 `return minimum_image_displacement_nm(...)`，数学只有一份在 `abfe_core`；
`outer_lambda_neural_basis:644` 那份是独立纯 Python 实现，但它**刻意不 import 主线**
（WIRING.md 接线事实第 1 条），是设计不是债。**这一组不要动。**

**还有一条必须先问用户**：`abfe_core:7415` 的 docstring 明写
「`ibs_engine.print_once` 的同源副本……避免 abfe_core 与 ibs_engine 之间新增
import 耦合，这里沿用同一模式」。**这批重复是当初有意为之的**——
抽 L0 `infra` 正好是那个顾虑的正解（不新增横向耦合，而是共同向下依赖），
但推翻一个已做的决定要你点头。

### B3　剩下的拍板集逐条决策：23 条上升边 + 9 对同层互环（32 项 / 601 次）

`deps.py` 会按引用数排序输出，从大到小过。每条只有三个出口：
**接受（调整分层）/ 反转（把被引对象下沉）/ 拆分（把域一分为二）**。
产出一张表，进 `docs/`。

### B4　`pipeline_orchestration` 的巨类是 L4 的内部问题

`ABFEPipeline` 15446 行 + `run_full_pipeline` 3376 行，拆它是 C 的内容，
按 WIRING.md 的巨类子区索引，**先抽无状态纯函数下沉到 L2/L3**。
B 阶段只需要确认：这些纯函数的目标层在哪。

---

## C　全量切

前置：A 完成（守卫不绑路径）+ B 完成（分层定稿）。

到这一步才谈得上「一次性切」：有分层就没有环，有 A 就有全程绿灯信号。
C 的具体刀法等 B 的分层表定了再写——**现在写就是空话**。

唯一可以先说死的：C 依然逐刀走 3.2 的六步流程，
**一次性重构不等于一个巨型 commit**。

---

# 第 5 章　验收口径

| 检查 | 命令 | 通过线 |
|---|---|---|
| 纯搬运无损 | `sed -n 'A,Bp' FILE \| md5sum` 前后对比 | 相同 |
| 分层变好 | `python3 refactor/tools/deps.py` | 违反分层的边数单调下降或持平 |
| 行为不变 | `./tests/run_offline_tests.sh`（远程 kasuga02） | **0 failed** |
| 不是只测语法 | —— | `py_compile` 通过**不算**验证：它不解析名字 |

失败就报失败并贴输出。不允许「本地没跑，应该没问题」。

---

# 附　第一步（2026-09-19 更新：两件都已完成，等落地授权）

**A1 的转换脚本。** 已就位：`refactor/tools/a1_tool.py`（纯 AST，不 import 主线）
+ `refactor/work/a1_proposals/`（43 个 diff + INDEX.md，断言字面量逐字节校验不变）。
只动 `tests/`，不碰主线，全程可绿。落地那刀要单独授权（0.1）。

同时 **B1 已实测**：`refactor/tools/layer_check.py` 三桶口径复现，
拍板集 = 23 条上升边 + 9 对同层互环 = 32 项 / 601 次，逐条拍。
