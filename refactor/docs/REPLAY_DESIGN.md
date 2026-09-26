# 可重放切分体系（cutter + spec + replay）

**2026-09-21 存档。工作暂停，等主线小修告一段落后继续。**

---

## 为什么要它

原来的形态是：`refactor/domains/` 是一份 09-19 主线快照的切分副本，十三刀都切在
它上面。主线一动，副本就是死档——落地时变成**三方合并**，而合并本身没有门覆盖。

实测漂移（2026-09-20 → 21）：主线五个源文件 688 insertions / 197 deletions，
73 个 hunk，其中 **26 个直接落在已经拆过的三个函数体内**。

改成：**把每一刀写成可重放的变换**。主线只做小 bug 修复时，函数的**结构**不变
（不新增/删除巨函数、不改分派链形状），所以按结构定位的脚本仍能找到目标。

```
现在：主线 ──快照──> domains（静态副本）        主线一动 → 死档 → 三方合并
改成：主线 ──重放──> 切分版（任意主线版本）      主线一动 → 重跑 → 得到新的切分版
```

---

## 关键简化：重放不需要域切分

15/16 刀只按「类名 + 方法名」定位，与文件怎么分域**完全无关**：

| 类 | 主线文件 | 刀数 |
|---|---|---|
| `ABFEPipeline` | `abfe_pipeline.py` | 5 |
| `IBSWindowManagerDualLambda` | `ibs_engine.py` | 9 |
| `Stage2RepairController` | `abfe_preoptimizer.py` | 1 |

所以重放**直接对主线三个文件的副本做**，跳过域切分。这很重要：域切分（WIRING 的
行区间逐条抄写）是全流程里最位置相关的一环，主线一漂就全错。现在不需要它了。

`_infra_cut` 是唯一例外（19 个定义跨 6 个域文件搬进 `infra/`），它属于 B 阶段
分层工作，**排除出重放链**。

---

## 三个组件

```
refactor/tools/cutter.py         通用引擎（extract / absorb 两模式）
refactor/cuts/specs/*.py         每刀一份声明式 spec
refactor/tools/check_spec.sh     spec 等价验收
refactor/tools/replay.py         ⏸ 未写：按序驱动 + 每刀过门 + 失败即停
```

### 引擎：两种模式对应两道现成的门

| 模式 | 做什么 | 门 |
|---|---|---|
| `extract` | 语句块原样搬成同类新方法，原位换 keyword 调用 | `verify_extract.py` |
| `absorb` | 散落名字收进数据类对象 | `verify_rename.py` |

**位置无关**：spec 用 `start`/`end` 两个结构谓词定位，不写行号或下标。
引擎提供谓词工具：

```python
assign_attr("dual", "_partition_criterion")   # dual._partition_criterion = ...
assign_name("result")                          # result = ...（含 AnnAssign）
if_test("dual.should_run_stage1")              # if <含该片段>:
call_to("self._partition_windows")             # 独立调用语句
stmt_type(ast.Try)                             # 语句类型
```

锚点命中数 ≠ 1 或次序颠倒 ⟹ **fail-closed 退出**，报「主线结构已变，人工核对后
更新 spec 的锚点，不要改成按下标定位」。原来的下标常量降级成 `slice_expect`，
只作漂移报告：下标变了打印 `[漂移] (26,41) -> (33,48)`，不据它定位。

### absorb 的两个子口径

- `in_place=True` —— 初始化语句留在原处变成 `bind.x = ...`，字段默认 None
- `in_place=False` —— 初始化被删除，初值成为字段默认值（WarmupState 口径）

以及两个正交开关：

- `extend=True` —— 往**已有**数据类追加字段（不新建类、不插构造语句）
- `seed_from_param=[...]` —— 收编的名字里有**原方法形参**时，把它的入参值播种进
  构造点：`dual = PipelineState(stage1_states=stage1_states)`。不播种的话，
  收编后读到的是字段默认 None。

### `proven_conditional`：机械判据管不到的地方，强制写理由

原地收编的安全前提是「字段默认值不可观测」，充分条件是**首写为体顶层无条件直赋**。
引擎逐名机械核验。不满足的名字（首写在嵌套块里，存在绕过它的路径）**必须**在 spec
里给出书面理由，否则 fail-closed：

```python
proven_conditional={
    "expected_vanishing_ranges":
        "条件首写后，由 P-C 段顶层无条件重写覆盖（同一方法内后续必达）",
    "stage1": "条件首写受 should_run_stage1 守卫，而该量在 P-A 无条件赋值……",
}
```

引擎运行时把理由原样打印。**这是本轮最有价值的设计**：当年那三个名字的论证只存在
特化执行器的头注释里，现在变成跑一次就能看见、且不写就过不去的东西。

---

## 验收：引擎产物 vs 历史产物

`refactor/work/*.before.py` 的**留底链**提供 15 组 before/after 对，是现成的回归
测试集。判据（`check_spec.sh`）：

```
① 行尾归一后逐字节相同   证明连注释、空行都没动
② AST 等价              证明逻辑一致
```

**不用裸 `cmp`**：历史上 `_pipeline_state_absorb` 把 40 行 CRLF 写进了一个纯 LF
文件，产物混合行尾——那是当年执行器的缺陷。本引擎按**文件主导行尾**写，不复刻
缺陷，代价就是那几刀要用归一后对账。

---

## 进度（2026-09-23：已完成，本节下方的表是 09-21 的历史快照）

- `replay.py` 已写；19 刀（pipe 5 / ibs 10 / ctl 4）history 全绿、mainline 全链过门。
- 11 刀保留特化执行器（经 `CUT_FILE` 指向目标）而非全部转 spec：形状各异，
  泛化进引擎的代码比执行器本身多；执行器已改为结构锚 + 文件主导行尾 + 类末插入。
- step0 后 4 刀挂起（闭包提升不等价），详见 `work/PROGRESS.md`。

## 进度

| | spec | 验收 |
|---|---|---|
| ✅ | `partition_windows`（extract 样板） | 逐字节 + AST |
| ✅ | `production_state`（absorb / in-place 样板） | 逐字节 + AST |
| ✅ | `pipeline_state`（含 3 条 `proven_conditional`） | 逐字节 + AST |
| ✅ | `pipeline_state2`（extend + seed_from_param） | 逐字节 + AST |
| ⏸ | `warmup_state`（absorb 搬家口径，51 名） | |
| ⏸ | `absorb3`（extend，3 名） | |
| ⏸ | `early_stop_config`、`window_context` | **跨多方法改写，引擎需加 `fns` 复数支持** |
| ⏸ | extract 族 6 份：`scheme_dispatch` / `produce_window` / `commit_frozen_f_k` / `run_warmup_loop` / `summarize_warmup` / `finish_production` / `resume_preoptimized` | |
| ⏸ | `decide_once_step0`（def 提升，特例，保持特化） | |
| ⏸ | `replay.py` | |

**代码量**：特化执行器 226 行 → spec 40 行（extract）；207 行 → 52 行（absorb）。

---

## 引擎在建设过程中被自己的检查抓到的坑

每一条都是 FAIL 暴露的，不是设计出来的：

| 坑 | 症状 | 防御 |
|---|---|---|
| **多行插入文本塞进单个列表元素** | 后续按行号索引全错，组装出 SyntaxError | 一律 `splitlines(keepends=True)` 再进列表。**这条本来就写在 GUIDE §五坑表第 6 条，引擎第一次跑就踩了** |
| **`ast` 的 `col_offset` 是 UTF-8 字节偏移，不是字符偏移** | 中文行上插入点全错；报的 SyntaxError 行号离真凶很远 | 先 `line.encode()[:col].decode()` 转字符偏移。**本仓满屏中文注释与日志，这是必踩的** |
| **两条实现路径的 docstring 缩进口径不一致** | 双向验证 FAIL（字符串值不等） | 行手术与期望树共用同一个渲染函数 |
| **构造语句的子串匹配撞上 docstring** | `dual = PipelineState()` 命中 2 处 | 整句精确匹配（`line.strip() == needle`） |
| **插在文件首行时的空行布局** | 前置/尾随空行与历史产物差一行 | `at == 0` 时去前置、补尾随 |

第二条尤其值得记进 `REFACTORING_GUIDE.md` §五坑表——它对这个仓库是普适的。

---

## 恢复工作时的下一步

1. 引擎加 `fns` 复数支持（`early_stop_config` / `window_context` 跨多方法）
2. 补完 11 份 spec，每份用 `check_spec.sh` 验
3. 写 `replay.py`：固定刀序 + 每刀跑完立刻过门 + 失败即停并报是哪一刀、什么锚
4. 拿**当时的主线**跑一次完整重放 —— 这是这套体系的第一次真考验
5. 重放产物直接对主线，所以**可运行** ⟹ `./tests/run_offline_tests.sh` 第一次能验行为
   （在此之前 `refactor/` 跑不起来，AST 等价是唯一判据）

## 主线改动时要知道的

重放钉在这些结构上，改 bug 不影响，**改名/挪块/拆 if 才会打断**：

| 刀 | 锚 |
|---|---|
| `scheme_dispatch` | `run_full_pipeline` 尾部 `decoupling_scheme == ...` 的 elif 链 |
| `partition_windows` | `_partition_criterion` 赋值 / `if dual.should_run_stage1:` |
| `resume_preoptimized` | `if resume and …preopt1_file:` / `self._partition_windows(...)` 调用 |
| `pipeline_state` ×2 | `_run_dual_lambda_scheme` 的 26 + 2 个局部名 |
| `warmup_state` | 51 个相位 2 初始化名 |
| `produce_window` / `finish_production` | `run_all_windows` 的 `warmup_only` 块、窗口尾清理协议 |
| `early_stop_config` | 9 个 `early_stop_*` 形参 |
| `window_context` | `sampler` / `sim` / `win_sys` / `lc_win` / `lv_win` / `win_sys_xml` |

打断**不会静默**——fail-closed 退出并指名道姓。届时更新 spec 的一行谓词即可。
