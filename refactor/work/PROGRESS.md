# 重构进度落盘（2026-09-20 第三次更新）

**本文是跨对话交接文档：读到它的会话没有前一个对话的上下文。**

## 交接阅读顺序（按序，缺一不可）

1. `refactor/docs/HOW_TO_CONTINUE.md` —— 路线图与**授权边界**（0.1：文档 ≠ 授权，
   每刀要用户单独点名；0.2 绝对禁区；0.3 越界定义）
2. `refactor/docs/CUTTING_PLAN.md` —— 巨函数动刀 v2（当前执行中的计划）
3. 本文 —— 进度与下一步
4. 背景可选：`refactor/docs/WIRING.md`（域切分地图）、`refactor/docs/b3_edge_decisions.md`
   （B 阶段结稿）、`refactor/tools/deps.py` / `layer_check.py`（指标工具）

## ★ 2026-09-24 追主线：19/19 重新全绿

主线 09-23→09-24 漂移：`ibs_engine.py` 169 行（stage2 坐标旁车、BOX-02 盒体积进
checkpoint 指纹、预算途中耗尽走临时生产）、`abfe_pipeline.py` 24 行（attachment 收敛门
开关、REPRO-01 转发）、preoptimizer 0。pipe 5 刀原样过；ibs 线三处 fail-closed，逐条人工核：

| 刀 | 失败 | 核对 | 处置 |
|---|---|---|---|
| produce_window | 搬移 88≠82；行数 1258 超 (1200,1250) | 循环体顶层新增 6 条全在搬移区（坐标旁车 5 + `_save_coords_sidecar` 1） | `stmts={82,88}`、`span` 放到 1290 |
| summarize_warmup | 形参多 `provisional_authorized`/`debug_provisional_on_warmup_failure` | 区内只读；前者全方法唯一写=循环体顶层无条件直赋，后者是形参 | `drift_ok`（同两名 09-23 已在 commit_frozen_f_k 放行） |
| finish_production | 搬移 26≠25；形参多 `production_coords_path` | 收口段新增 1 条；该名 = `_produce_window` 体顶层无条件直赋、`for up` 之前，段内只读 | `stmts={25,26}` + `drift_ok` |

引擎改动：`cutter.check_moved` 的 `stmts` 接受集合（历史值 + 核过的漂移值，与 `drift_ok` 同口径）。
验收：`replay.py history` 19/19 全绿（历史产物对账不变）；`replay.py mainline` 19/19 门 PASS；
三产物 py_compile OK；影子仓库定向跑今天相关的 3 个测试文件（coords_sidecar / repro_budget /
pre_equil_final_state_box）主线 10 passed、影子 10 passed。**全量影子测试未跑。**

未动的挂起项照旧：step0 后 4 刀刀法 (a)/(b)/(c)；44 个源码文本类测试的字面量口径。

## ★ 2026-09-23 现状（下文 09-20/22 各节是历史记录）

**工作方式已换**：不再在 `domains/` 快照上切，而是 `tools/replay.py` 对**当前主线**
三文件整链重放。`domains/` 降为死档（且含下述 4 个历史 bug，别再拿它当产物）。

```
python3 refactor/tools/replay.py history            # 回归：每刀从历史留底出发 → 对账历史产物 + 过门
python3 refactor/tools/replay.py mainline [--shadow] # 主线 → refactor/replay/out/（+ 影子仓库供跑测试）
```

- 刀序唯一权威 = `replay.py` 的 `STEPS`：pipe 5 + ibs 10 + ctl 4 = **19 刀**。
  8 刀走引擎 spec（`cuts/specs/`，本轮补 `finish_production`），其余 11 刀保留特化
  执行器（`CUT_FILE` 指向目标文件）——形状各异，泛化进引擎的代码量大于执行器本身。
- 每刀门 = 原 `.verify.sh` 那组（下标从产物结构现算）+ **新门 `verify_bound.py`**。
- 主线漂移处理：锚点命中漂移 → fail-closed；人工核对后写进 `drift_ok`（spec）/
  `DRIFT_OK`（执行器），**逐名写来源与放行理由**，历史快照上不出现 ⟹ 历史回归不受影响。
  放行判据：①搬移区内只读 ②调用点必然已绑定。
- 主线专属接线：`abfe_pipeline.py` 补 `from dataclasses import dataclass, field`
  （历史切片跑不起来，从没需要过）。数据类一律插在目标类之前（插文件头会挤掉 docstring）。

### 本轮抓到的历史真 bug（旧门全部 PASS，`domains/` 里仍在）

| # | 刀 | 症状（影子仓库实测） | 根因 | 处置 |
|---|---|---|---|---|
| 1 | commit_frozen_f_k / summarize_warmup | `NameError: legacy_repair` | 形参表漏传（09-22 已记） | spec 已修 |
| 2 | scheme_dispatch / resume_preoptimized | single_lambda/2d 路径 `UnboundLocalError: f`；`test_no_undefined_globals` 报 cached/point/raw_path/values | 「调用方可见」按「某处有 Store」算，把**内嵌 def 局部 / 条件绑定**的段内幻影名当形参 `x=x` 传 | 幻影名进 EXEMPT（段内先写后读已逐名验） |
| 3 | window_context | 5 个无关方法 `NameError: wc`（42 处） | 执行器 `ast.walk(klass)` 改写全类；6 道 verify_rename 只看 6 个目标方法 | 改写限定 FNS |
| 4 | step0 plan | 收敛 stage 走 DONE/NO_ACTION 即 `UnboundLocalError: _blocks`（控制器测试 7 处） | 闭包提升：原来调用时才读、且只在用到的分支读；提升后调用点当场求值，而 `_decide_once` 有 3 个 `return plan(...)` 早退在 `_blocks…_noop_led` 赋值之前 | **后 4 刀挂起**（见下） |
| — | run_warmup_loop / summarize_warmup | （潜在）`k` 未必已绑定 | 推导式局部幻影 | 进 exempt |

**结构性结论（第四次同形）**：抽取/提升类的门只验「搬走的代码没变」，**调用点与签名是
新写的代码，不在任何门的射程内**（分派条件 → return 解包 → 形参表 → 调用点实参绑定）。
`verify_bound.py`（保守必然赋值分析；闭包/lambda 内调用一律报；分支条件相关性分析不了时
`--allow 名=理由`）补的是最后这一环。建门先对历史产物自检：bug 2/4 与 `k` 全部抓到。

### ⛔ step0 后 4 刀挂起（等用户拍板刀法）

`_frames_admission` / `_is_noop` / `plan` / `_bounded_rewindow`。纯搬运做不到等价，三个候选：
(a) `plan`/`_bounded_rewindow` 留作内嵌闭包，只提升不捕获晚绑定名的纯函数（= 现状 4 刀）；
(b) 把 `_blocks…_noop_led` 8 条赋值前移到首个 `plan` 调用前——**改行为**，需证 RHS 无依赖无副作用；
(c) 捕获名收进一个上下文对象，早退路径上字段缺省可观测——同样改行为。

### 影子仓库测试（本地快速反馈，非正式验收）

09-23 晚（本机，NFS 争用重；NFS listdir 出重复条目 ⟹ pytest 双收集，下列已去重）：

| | failed（唯一） | 说明 |
|---|---|---|
| 基线影子（主线三文件原样） | 11 | 基线的三文件与其余文件拷贝时刻不一致（同事同时在改盒子 resume），与重构无关 |
| 重放影子 | 55 = 11（同基线）+ **44** | 44 个全部是**源码文本类**测试；**行为失败 0** |

第一轮（修 4 个 bug 前）是 68 failed + 3 errors，其中 `UnboundLocalError: f/_blocks`、
`NameError: wc/legacy_repair` 这类行为失败，本轮全部消失。

44 个源码类的两种形状（抽样 18 个可自动判的：13 / 5）：
- **只需换定位**：literal 在产物里原样存在，只是被搬进了新方法
  （`inspect.getsource(run_full_pipeline)` 取不到 → 改指 `_run_dual_lambda_scheme` 等）= A1 的活。
- **字面量被归组改掉**：`bias_converged` → `warm.bias_converged`、`frozen_mbar_batches` →
  `warm.…`、`lambdas_vdw=lv_win` → `wc.lv_win` 等。**A1 救不了**：3.4 禁止改断言字面量。
  需要用户拍板——要么允许这类测试的字面量随归组同步改（每条逐一核对只改了承载前缀），
  要么归组刀不进主线。

## 授权状态（新会话照此执行，不要自行扩大）

- **已完成**：刀3第0步 8/8；WarmupState 收拢；刀1 A 段；刀a；④勘误；B①②③；
  刀1 复合门（端到端封链 PASS）；**刀2（run_full_pipeline 三路分派抽双方法，
  09-20 用户点名 E）**。
- **以下需用户重新点名才可开工**：
  - **刀2 后续**：_run_dual_lambda_scheme（1988 行）内部细分——五相位边界
    指标已测（见刀2 节），P-C|P-D 需要上游 77、边界普遍肥，预计需要
    PipelineState 类对象；先做与 §2.1 同形的内部相位细测再下刀。
  - 7/8 细分：解锁条件 = n_updates≥1 可达性证明（拍板记录见下）。
  - B 段余下（Config 归组 + WindowContext）；C 三大方法细分；A1 落地 tests/。
- 绝对禁区照 HOW_TO_CONTINUE 0.2。git 用户自己管。

## 环境备忘

- 本机 Python 3.12（f-string 内部是真实 AST 节点，PEP 701——改写工具已依赖此点）。
- 本机**不是 kasuga02**；正式验收 `./tests/run_offline_tests.sh` 在远程
  kasuga02 跑（用户操作），本地 pytest 只作快速反馈、不作数。
- pytest 环境要 mamba activate openmm_dev（tests/run_offline_tests.sh 自带）。

---

## 已完成并验收

### 第一刀：抽 L0 infra（B2，纯搬运）
- 19 个定义块 → `refactor/domains/infra/infra.py`；源文件 6 个。
- 验收：sorted-concat md5 = `37ed8ea5c6b38bb3ea48a9455f4047a1`（逐字节一致）。
- 指标（layer_check.py）：违反引用 276→130（−53%），上升边 23↔23 持平。

### 刀3第0步（_decide_once 内嵌 def 提升，CUTTING_PLAN.md §4.3）：✅ 8/8
留底 `work/step0_*.before.py`。提升顺序 = 被调者先行；
plan 捕获 12（51 处调用点插参，其中 _decide_once 内 **50** 处 `return plan(...)`
+ _bounded_rewindow 体内 1 处——§4.2 的「51 个出口」应为 50，2026-09-20 勘正）、
_bounded_rewindow 捕获 14（4 处调用点）。
终检（09-20）：ast.parse/py_compile OK；复合门（原始留底 + --strip-call）PASS；
`_decide_once` **2779 → 1893 行**，内部零内嵌 def，分派骨架露出。

### 刀1 A 段：`_produce_window` 抽取（CUTTING_PLAN.md §5 序2、§2.2/§2.3）：✅ 2026-09-20
- 执行器 `refactor/_produce_window_cut.py`；留底 `work/produce_window.before.py`。
- **实测修正 §2.2 草图两处**（量过才下刀）：
  1. `if warmup_only:` 块（L3998-4023，内含绑定窗口 for 的 continue + 引用清理）
     **留在调用方**——continue 搬进方法即语法错误；
  2. 窗口尾引用释放协议（probe If ×2、`del sampler`、`del sim.context`、
     `sim = None`、`ibs_wrap/_nan_diagnostics/win_sys = None`、`gc.collect()`，
     L5274-5307）**留在调用方**——源码注释明说这是防「代理悬垂到下一个窗口」
     的生命周期语义；`sim = None` 是名字级写入，搬走会改调用方绑定的释放时机
     （`del sim.context` 是对象级，搬不搬都等价，随协议整块留下）。
- **写逃逸实测为无**（搬移区方法作用域写集 ∩ (清理块读集 ∪ 循环后读集) = ∅）
  → 无返回值，纯调用。循环携带裁决：`e`/`f` 头部各仅 1 处 Load 且都在自身
  绑定结构内（L875 except f-string、L658 `json.load(f)`），安全。
- 变换：搬移区 = 循环体 body[129:211] 共 **82 条语句 / 1228 行**（L4044-5271，
  含生产段头注释）→ 同类方法 `_produce_window(self, *, <30 形参>)`，
  **keyword-only 签名 + keyword 调用点**（杜绝 30 参错序）；46 个方法内无绑定的
  自由名（np/json/IBS_* 常量等）不传，随接线解析。
- 门：verify_extract **`--slice 129:211`** PASS（82 条逐节点等价）；头部/保留区
  内容逐行一致（行尾归一口径）；ast.parse/py_compile OK；**layer_check 1357/107
  分毫不差**。
- 行数：`run_all_windows` **4957 → 3761**；`_produce_window` 1269（签名 34 +
  docstring 7 + 体 1228）；文件 5424 → 5498。

### 违规事故（用户发现，已清）：2026-09-20
- 刀2 分派门自检的 4 个合成样本（b/g/bg/bt.py）落在**仓库根**——自检没切
  `/tmp` 目录。内容核实后删除，仓库根恢复原状。防御规则已进 GUIDE 五、坑表
  （自检一律 /tmp 或 scratchpad 里跑）。

### `_run_stage2_autonomous` 相位测量（纯只读，用户点名）：✅ 2026-09-20
**定位**：ABFEPipeline L8343-9895，1553 行 / 15 形参 / 体 30 条顶层语句。
结构：4 内嵌 def（_disk_signature/_finish/_write_history/_make_ctl）+
**迭代 For L8550-9752（1203 行）**+ 出口处理。控制流：循环内**无方法级
Return、无循环级 break/continue**——出口全走 outcome 状态 + for-else
（ITERATION_CAP_NOT_CONVERGED）。

**For 体相位表**（20 条）：

| 相位 | 语句 | 行区间 | 内容 |
|---|---|---|---|
| P-α 控制器读取 | [0-1] | ctl/view | _make_ctl + ctl.read |
| P-β 尾窗分析 | [2-4] | ~30 行 | _tk/_tail_hi_start |
| P-γ 决策 | [5-7] | plan/act/wins | **ctl.decide(view)** ← 与 _decide_once 衔接 |
| P-δ 重复/升级检测+日志 | [8-13] | 含 84 行 history.append 日志块 | key/esc_key/sig/last_sig |
| P-ε 收口判断 | [14-18] | terminal/seen>=3 升级 | ~150 行 |
| P-ζ 执行 Try | [19] | **873 行** | 动作分派 + 4 恢复 handler |

**P-ζ Try 解剖**：body = **637 行 12 路动作分派 If 链**（RUN_PRODUCTION×2/
PROVISIONAL_PRODUCTION/CONTINUE_WARMUP/RECALIBRATE_FK+PROBE_CANDIDATE_FK/
PROBE_REANCHOR_EPOCH/INSERT_LAMBDA 127 行/SPLIT_TAIL_WINDOW/RELEARN_FK_EPOCH/
IMMUTABLE_REWINDOW/ANALYZE/else）+ 4 except handler 202 行
（route_err 42/fk_err 115/_ens_err 45/exec_err 9）。
分派链与 _decide_once 的 50 个 return plan(...) 同构——控制器线连续性实证。

**三个候选刀口实测**（需要上游 / 写逃逸，序感知+作用域感知口径）：

| 候选 | 需要上游 | 写逃逸 | 裁决 |
|---|---|---|---|
| **Cut A：整 Try 抽 `_execute_stage2_action`** | **18** | **3**（lam/ranges/result 循环携带） | ✅ 刀口正确，但**有两条阻塞，见下节**——「return 三元组协议即可，纯搬运」这个原判**不成立** |
| Cut B：637 行 If 单独 | 19 | 5（含 _pv/_wi，handler 引用） | ❌ 差于 A |
| Cut C：P-β~ε 决策段 | 10 | **18**（_tk/key/sig/plan 等循环携带标量群） | ❌ 需先建迭代状态对象 |

**建议刀序**：Cut A（18 参 + 返回 (lam, ranges, result)）→ 1553 ≈ 700 + 910；
P-ζ 内部再按 12 路分派拆（第三轮，或与 AUDIT-S2-02 的执行侧分派合流）；
Cut C 等迭代状态对象（与 P-D 那批派生值同批处置）。

### ⛔ Cut A 的两条阻塞（2026-09-20 复核实测，开刀前必须处置）

**阻塞 1：`lam` / `ranges` / `result` 必须「既是入参又是出参」，不能只当写逃逸。**

```
lam     Try 内首现 = L8887 Load    For 前 Store=L8393   Try 前 Store=L8579
ranges  Try 内首现 = L8887 Load    For 前 Store=L8394   Try 前 Store=L8579
result  Try 内首现 = L9435 Store   For 前 Store=L8418   Try 前 Store=无
result 的 Store：Try.body 内 3 处，4 个 handler 内 0 处
handler0/1/2：raise=0 return=0（正常走完）    handler3：raise=1（重抛）
```

`lam`/`ranges` 在 Try 内先读后写，显然是入参。**危险的是 `result`**：它在 Try 内
是先写后读，**任何忽略异常流的入流分析都会判它是纯局部**——但 handler 0/1/2
会绕过那 3 个 Store。原代码里无害（循环携带，保留上轮值）；抽成方法后若只当
出参，方法末尾的 `return (lam, ranges, result)` 会在**三条恢复路径上
`UnboundLocalError`**，其中 handler1 是 115 行的 fk 校准失败恢复，是常走路径。

⟹ 执行器的参数表必须把这三名同时算进**入参**，调用点写成
`lam, ranges, result = self._execute_stage2_action(*, lam=lam, ranges=ranges,
result=result, <其余 15 参>)`。「18 上游 / 3 写逃逸」若是两个不相交的集合就漏了这层。

**阻塞 2：return 三元组解包是第三种调用点形态，现有门一概不覆盖。**

`verify_extract` 验搬走的块、`verify_dispatch` 验条件分派——解包不在任何门的
射程内。**这正是刀2「5 路 → 2 路」事故的同一缺口**（见本文件刀2 勘误节）。

开刀前要补一道 return-tuple 门，三条判据都可静态判：
1. 方法体所有**正常出口**（不含 raise）都到达那条 `return (a, b, c)`
   ——实测 Try 的 `orelse=0 / finalbody=0`，故「return 置于 Try 之后的方法体顶层」即满足；
2. 调用点解包的名字与顺序，和 return 元组逐一对应；
3. 元组每个成员都在入参表里（= 阻塞 1）。

按 GUIDE §三，门造完先用**「漏传入参」的合成样本**证明它会 FAIL，再上真刀。

**等用户点名：先补门，再开 Cut A。**

### D 线第二刀：WindowContext 归组（6 窗口对象）：✅ 2026-09-20
- **绑定安全证明**（Config 没有的新维度）：六名首写均在任何读之前（循环体内
  文本序）；phase-1 两个早退 continue（L693/L830）跳过整窗 → **窗内默认值
  不可观测**（每窗重建 `wc = WindowContext()`，不跨窗泄漏）；清理协议随改写
  自动转换（`del sampler`→`del wc.sampler`——del 位点必然已绑定；
  `sim/win_sys = None`→字段写；gc.collect() 不动）。
- 变换（`_window_context_cut.py`，留底 `window_context.before.py`）：
  `@dataclass(slots=True) WindowContext`（6 字段 Any=None，可变——清理协议
  改字段）；构造插循环体顶；**500 处**引用改写 wc.*；5 签名 6→1
  （warmup 24→21、summarize 39→36、commit 23→20、produce 30→25→现在 18、
  finish 21→17）；5 调用点 kwarg 折 1。
- **门建设**（verify_rename 三修）：① `--expand-kwarg` 重写为**全树扫描 +
  before 同形调用唯一匹配**（produce 调用嵌套在 For 子树里，只扫顶层漏；
  折叠调用点的非 bind kwargs 与 before 逐字核对后整体采纳 before kwarg 列表
  ——es 那版只匹配顶层且 bc 侧排除口径漏白名单，两处都被门 FAIL 暴露）；
  ② 各方法白名单 = 其实际所占名字子集（4/4/4/6/5——win_sys(_xml) 并非
  所有方法都收）；③ expand 块移到构造剔除之后（索引对位）。
- 门：六道 verify_rename 全 PASS（run_all_windows ①+①'+②③、其余 ①②③）；
  py_compile OK；**layer_check 1357/107 不变**；复合门扩展到 **10 变换**
  （wc 构造删除 + wc.x 还原；wc=wc 随 inline 消费）；**16 脚本重放全绿**。
- **行数**：`run_all_windows` 1991→**1978**（42 参不变）；`_produce_window`
  480→**471**（**18 参**——从 A 段时的 30 一路折到这里）；`_finish_production`
  833→**829**（17 参）；warmup 804/summarize 731/commit 434；全文件 5795。

### refactor/ 重组（用户提案，Cut A 前空档执行）：✅ 2026-09-20
- **结构**：顶层 52 → 6（README.md + domains/17域 + tools/8工具 + cuts/16执行器
  + _shadow_files×2 + docs/5文档 + work/原scratchpad+a1_proposals）。
  删 refactor.zip 与全部 __pycache__（9 处）。
- **验收全过**：layer_check 输出**逐字一致**（111 行，1357/107，文件 81/域 17）；
  **16/16 .verify.sh 重放全绿**；a1_tool --check 恢复（216 测试文件，C1:35/C2:125）。
- **代码修正 4 处**：layer_check（ROOT 上提 + 过滤改 domains-only + dom=parts[1]）；
  deps（ROOT 上提 + _dom_of() 助手 ×5 处——噪音口径，tools/cuts/work 计为噪音域）；
  compound_gate_knife1（ROOT 上提 + work/ + domains/ 前缀）；
  a1_tool（REPO parents[1]→[2] + OUT_DIR → work/a1_proposals）。
- **教训**：路径重写的 blanket pass（scratchpad/→work/）改了 layer_check 注释，
  导致后续手工替换的源串不匹配而静默空转——**这类替换必须替换后立即 grep 验证**，
  本次是验收 diff 抓的。执行器（cuts/）内的路径已同步重写但**不保证可跑**
  （存档件，重放一律走 work/verify/）。
- **verify.sh 深度**：work/verify 与原 scratchpad/verify 同深，`cd ../../..` 不变。

### D 线第一刀：EarlyStopConfig 归组（§2.3 簇 1/6）：✅ 2026-09-20
- **D 线测量结论**：§2.3 六簇对 run_all_windows（30/41 覆盖）成立；对四个
  内层方法覆盖低（2/24~16/30）——内层长签名大头是**跨相位派生值**不是原始
  入参。D 线正确形状 = 簇对象在 run_all_windows 入口构造、向内穿线，
  各方法按其所占簇成员缩编。
- **变换**（`_early_stop_config_cut.py`，留底 `early_stop_config.before.py`）：
  模块级 `@dataclass(frozen=True, slots=True) EarlyStopConfig`（9 字段无默认
  =必填；只读性执行器核验：三体内零 Store/Del）；run_all_windows 入口构造
  es（kwarg=形参裸名）；三体内引用改写 es.*（30 处）；_produce_window/
  _finish_production 签名 9→1、两调用点 9 kwarg→es=es。run_all_windows
  签名**不动**（外部 API）。
- **门建设**（三件）：verify_rename `--expand-kwarg`（es=es 展开还原）；
  `--in-place` 绑定检查纳入**形参**（此前只扫体语句）；复合门扩展到 **9 变换**
  （es 逆变换须在 inline 前作用于三个体——调用语句会被 inline 消费）。
- **门**：三函数 verify_rename 全 PASS；双向验证拦 3 次执行器错误
  （es 位置×2、ctor 时序）；py_compile OK；**layer_check 1357/107 不变**；
  **15 脚本重放全绿**。
- **行数**：`_produce_window` 496→**480**（30→23 形参）；`_finish_production`
  841→**833**（29→21 形参）；run_all_windows 1991（+构造行，签名不动）。

### 复验脚本体系（用户观察落地）：✅ 2026-09-20
- `work/verify/<刀名>.verify.sh` × **14**：每刀的门命令原样落盘、可直接跑；
  **留底链配对**——每刀的 after = 下一刀的留底（当前文件只作最末刀的 after），
  整链因此可逐刀重放。全部实跑绿（门改动后按指南七全量重验过三次）。
- GUIDE §九 已加清单项；GUIDE 二/A 已加「调用点是门射程外的新代码」（verify_dispatch）。

### 第二波收编 + P-B/P-C 切分（刀2 内部细分）：✅ 2026-09-20
**第二波收编**（`_pipeline_state_absorb2.py`，留底 `pipeline_state2.before.py`）：
- `stage1_states`（**形参→字段**，构造点 `PipelineState(stage1_states=stage1_states)`
  传初值——参数值/重绑值两段流保真；形参暂留签名成 vestigial，D 线统一处置）
  + `opt_res` → PipelineState 28 字段；26 处改写。
- 事故与门修复：① NEW_FIELDS 缩进丢失第三次复发（absorb3 同款，双向验证拦下）
  → 执行器内注释留痕；② verify_rename `--construction` 语义缺口——它假设构造
  语句是本刀新增，第二波的构造是上刀已有、本刀改 kwarg，单侧剔除致条数 53≠52
  → 改为**双侧成对剔除**（kwarg 增量归执行器双向验证，分层不混）。

**P-C 刀**：`_partition_windows(self, *, 5 参)` ← 窗口划分段 15 条/193 行
（e/f 段内自覆盖已裁决：e 无 Store、handler 局部；f with 局部）。
`verify_extract --block fn --slice 26:41` PASS。留底 `partition_windows.before.py`。

**P-B 刀**：`_resume_preoptimized(self, *, 8 参)` ← 预优化 resume 段 8 条/623 行
（f 全部 with 局部幻影；L13592 Return 在内嵌 `def _resolve` 内——嵌套感知检查
放行）。`--slice 18:26` PASS。留底 `resume_preoptimized.before.py`。

**终检**：py_compile OK；**layer_check 1357/107 不变**；verify_dispatch PASS；
**14 脚本重放全绿**。
**行数**：`_run_dual_lambda_scheme` **1989 → 1190**（剩 P-A 准备 + P-D 采样
896 行块 + P-E 收口 + 3 个方法调用）；`_partition_windows` 207 ✅ 达标；
`_resume_preoptimized` 640（临界，第三轮对象）；全文件 15655。

### 刀2 B 段：PipelineState 收编（26 字段）+ 内部边界复测：✅ 2026-09-20
**M① 相位与状态需求测量**（序感知口径——弱规则"读段⊆写段"会漏
optimized_lambdas_2 这类『后段也写』的名字，改为 ∃早期写+后期读）：
- 跨段候选 33 → 扣形参 → **7 个推导式/循环临时名幻影**（a/b/i/item/k/r/x，
  Load 全在自家生成器/循环内——覆盖检查必须认 comprehension 生成器，
  只认 ast.For 会误报）→ **真字段集 26 名**。
- **stage1/stage2/expected_vanishing_ranges 条件首写的全路径绑定证明**：
  `should_run_stage1 = True`、`should_run_stage2 = True` 均为无条件赋值
  （单赋值点，无重写）→ P-D 块必然执行 → 必然绑定；expected_vanishing_ranges
  条件首写后由 L13930 顶层无条件重写覆盖。26/26 默认值不可观测。
- 承载名 `dual`（可用性实测）；构造插 docstring 后第一条（body[1]），
  早于全部 `dual.*` 首用（L13189）。

**M③ 收编**（执行器 `_pipeline_state_absorb.py`，留底 `pipeline_state.before.py`）：
- 模块级 `@dataclass PipelineState`（26 字段，文件头，ABFEPipeline 前）；
  264 处引用改写 `dual.<field>`；构造 `dual = PipelineState()` @ body[1]。
- 门：verify_rename --in-place ① PASS（264 处还原逐节点等价）+ ②③ PASS；
  py_compile OK；**layer_check 1357/107 不变**；verify_dispatch 复跑 PASS。

**M④ 边界复测（收编后）**：
| 边界 | 需要上游 | 写逃逸 |
|---|---|---|
| P-A\|P-B | 47→**13** | stage1_states（形参被 P-B 重绑） |
| P-B\|P-C | 24→**9** | e/f（幻影待裁）+ opt_res（真计算值） |
| P-C\|P-D | 77→**18** | e/f/stage1_states |
| P-D\|P-E | 9→**19**（反向） | f（幻影待裁） |

- **第二波收编对象**：`stage1_states`（方法形参，重绑值跨 P-B→P-D——收编需
  构造点传初值 `PipelineState(stage1_states=...)`）+ `opt_res`。
  收编后 P-A|P-B、P-B|P-C 边界即净，可切 `_run_preopt_resume`（623 行）等。
- 行数：`_run_dual_lambda_scheme` 1988→**1989**（+构造行）；文件 15603。

### 刀2 语义破坏勘误（用户复验发现，已修复）：2026-09-20
- **破坏**：尾块原为 **5 路链**（dual/single/2d_diagonal/2d_geodesic/else
  raise ValueError），我的分派写成 2 路 if/elif 无 else——2d_diagonal/
  2d_geodesic 死码、fail-closed raise 静默化为 None（run_full_pipeline 的
  返回值是下游要用的）。**根因**：verify_extract --orelse 只验「搬走的块
  没变」（PASS 是对的）；调用点是新写代码，不在任何门的射程内——门设计
  的结构性缺口。刀1 未暴露是因为其调用点均为无条件调用。
- **修复**：分派第二支 elif→else（method2 自带全部四条非 dual 路径的测试
  与 raise 兜底）；逐路径核对通过（dual→m1；其余→m2 内部链；无落空）；
  双门 PASS 复跑 + py_compile + layer_check 1357/107 不变。
- **新门**：`verify_dispatch.py`（调用点覆盖等价：前缀 test / 兜底 else 等价 /
  支体 return 形状）——**三情形自检通过**（情形 2 即刀2 事故形态）；
  真刀 PASS。GUIDE 二/A 已补「调用点是门射程外的新代码」条目。
- **教训入册**：抽取条件链时，分派语句必须过 verify_dispatch；
  复合门封链之外，调用点校验是独立的门。

### 刀2（run_full_pipeline 三路分派抽双方法，§3）：✅ 2026-09-20
**相位测量先行**（§3 ⚠️ 要求，纯只读）：run_full_pipeline L12071-15442（3372 行，
56 条顶层语句，16 形参）未漂移。尾块实为**三路 elif 链**（非 §3 草图的两路）：
`if dual_lambda(52条/1952行) / elif single_lambda(16条) / elif 2d_diagonal(306行链)`。
if 体内部五相位（未来细分地图，边界普遍肥）：

| 相位 | 语句 | 行区间 | 与下一边界 |
|---|---|---|---|
| P-A 键/文件准备 | [0-16] | L13063-13134 | 需上游 47、逃逸 stage1_states |
| P-B 预优化 resume | [17-24] | L13134-13756 | 需上游 24、逃逸 4 |
| P-C 窗口划分 | [25-39] | L13781-13973 | 需上游 **77**、逃逸 6 |
| P-D 采样执行 | [40-43] | L13992-14972（含 896 行 should_run_stage2） | 需上游 9、逃逸 f |
| P-E 收口落盘 | [44-51] | L14974-15014 | Return 收尾 |

**变换**：if 体 → `_run_dual_lambda_scheme(self, *, 26 形参)`；**整条 elif 链**
（单条 If 语句，含 single_lambda/2d_diagonal 分派）→
`_run_single_lambda_scheme(self, *, 23 形参)`（首行 elif→if 纯语法改写，AST 同构）；
尾部改 if/elif **return 分派**（55 行）——Return 传播精确性：if 体无条件 Return
收尾、single 侧 3 个 `return final` + 落空隐式 None，调用点 return 包裹后逐路径等价。
5 个方法级内嵌 def 作函数形参传入（闭包随参）；e/exc 按 handler 覆盖分别裁决
（if 体 e 是真形参——L13200 一处真外层读；orelse 侧豁免）。
**参数口径坑**：探针初测漏 `kwargs`——`**kwargs` 绑定不在 args.args 里，
执行器口径（含 vararg/kwarg）为准，两侧名单各 +1。

**门**：verify_extract 新增 `--block fn`（函数体切片）与 `--orelse`（orelse[0]
整条语句为搬移对象，for method2）——method1 `--block if --index 21` PASS（52 条
逐节点等价）；method2 `--orelse` PASS。执行器两次写坏被门拦（method2 误套
12→8 去缩进——elif 链本在方法体层级 8 应逐字搬；门「语句数 0」抓的是上一轮
同类教训的再现）→ 回滚重切两次。

**终检**：ast.parse + py_compile OK；类体无裸 If（写坏特征清零）；尾部分派
if/elif 双 Return 结构核验；**layer_check 1357/107 不变**。
**行数**：run_full_pipeline **3372 → 1037 行**；`_run_dual_lambda_scheme` 1988；
`_run_single_lambda_scheme` 462。留底 `scheme_dispatch.before.py`。

### 刀1 复合门（端到端封链）：✅ 2026-09-20（用户指出方法论缺口后补建）
- **缺口**（用户复验发现）：每刀的门只在自己那一刻成立，后续刀一落无法重放；
  8 次「当时 PASS」≠「当前文件 = 原始 + 且仅 + 8 变换」的端到端证明。刀3 有
  复合门（最原始留底），刀1 链没有。
- **工具**：`refactor/tools/compound_gate_knife1.py`（只读）。重建 = 从当前文件把
  run_all_windows 循环体拼回原始形态：①内联 5 个抽出方法体（_finish_production
  先内联回 _produce_window）②prod 构造删除+prod.x 还原（原地，精确）③
  `warm = WarmupState()` 替换为 51 条原始 init（字段默认值经 ③逆映射重建）+
  warm.x 还原（54 名）。
- **唯一公理**：51 条 init 从散布原位提升到构造点（收拢刀当时单独证过：
  51/51 首写=init、纯字面量无依赖、零先读后写）。比对口径 = 双方剥 51 init 后
  (a)init 集合对账（名序/值/注解）(b)余部逐节点。
- **结果 PASS**：重建 270 条 == 原始 270 条；init 51 名序/值/注解逐一相同
  （含两条 AnnAssign 注解保留到字段）；剥离余部 219 条逐节点相同。
  复合门自检式调试中修掉 4 处自身实现错（内联顺序/过期下标/docstring 内联/
  simple 镜像——与 verify_rename 同款镜像）。
- **B（7/8 细分）拍板：不批准语义变更，维持联合块。** 理由：n_updates==0
  可达性未知 → 变更波及面未知；若路径可达，原 UnboundLocal 是响亮失败
  （可能是潜在 bug），None 流入会把它静默化。解锁正道 = 先证明
  n_updates ≥ 1 的上游保证（预算逻辑可达性调查，单独立项）——证成后
  9 名收编即不可观测，7/8 细分自然解锁。

### 刀1 B 段第 2 步 + 刀d（ProductionState 收编 + [7+8] 联合抽取）：✅ 2026-09-20
**REFACTORING_GUIDE.md 已就位并全程遵循**（四铁律/三门/门自检/数字对不上流程）。

**P① 测量**：6/7 边界 29 真逃逸 − 3 个流幻影临时名（e=handler 局部、gid/e_g=
debug 循环局部——`except as e` 绑定不是 Name 节点，名字级分析看不见；7 侧
Load 全部被自己段内 Store 覆盖）= **26 名**，26/26 首写均为体顶层无条件直赋
→ 原地收编默认值不可观测。`i` 为推导式/for 局部幻影，排除。

**P② verify_rename `--in-place` 模式**（原地收编口径：初始化语句保留为
bind.x= 写入，②改『前有绑定』校验、③改『字段存在性』校验）——
**三情形自检通过**（正确 PASS / 偷改逻辑 FAIL / 偷删字段 FAIL，指南三）。

**P③ 收编**（执行器 `_production_state_absorb.py`，留底 `production_state.before.py`）：
- 模块级 `@dataclass ProductionState`（26 字段，全 `Any = None`，紧跟 WarmupState）；
  `prod = ProductionState()` 构造插入最早首写（原 L2505）前；134 处引用改写。
- 门：双向验证（期望树比对）拦下两次事故——CRLF.join 产生 `\r\r\n` 双回车、
  **AnnAssign `simple` 镜像**（目标 Name→Attribute 后解析器口径 1→0，期望树
  同步重算）；verify_rename 还原方向同镜像（还原回 Name 时 0→1，`_refix_simple`）；
  ① PASS / ②③(原地) PASS；layer_check 1357/107 不变。

**P④ 刀d [7+8] 联合抽取**（执行器 `_finish_production_cut.py`，留底
`finish_production.before.py`）：
- **不单独切 7/8**：两者边界还有 9 个相位 7 内嵌首写名（fmax/state_n 等），
  吸收会把 `n_updates==0` 路径的 UnboundLocal 改成 None 流入相位 8（铁律 4，
  crash 路径语义变更需用户拍板）→ 本轮搬 [7+8] 联合段（零逃逸、语义精确）。
- 25 条语句 / 804 行 → `_finish_production(self, *, 28 形参)`（keyword-only）。
- **门两次抓到写坏**：① `dedent4` 12→8 模板误用——搬移语句本在 8 缩进
  （_produce_window 体层级），去缩进后方法体浅于 docstring，解析器把函数截成
  docstring-only、for-up 浮成类级语句（ast.parse 合法所以执行器自检没拦，
  是 verify_extract 的「语句数 0」抓的）→ 修正为**逐字搬运不去缩进**并回滚重切；
  ② 门锚点修正——搬移区是函数体尾部切片，新增 `--block fn`（配 --slice，
  fn 体下标已剥 docstring 需 −1）。
- 终检：25 条逐节点等价 PASS；py_compile OK；全文件统一 CRLF（5816 行）；
  类体无裸 For（写坏特征清零）；**layer_check 1357/107 不变**。
- 现状：`_produce_window` 1269 → **496 行**（= 相位 6 准备段）；
  `_finish_production` 841 行（体 804，7/8 细分对象）。

### 刀1 B 段第 1 步 + 刀b/刀c（§2.2.1 相位 3/4 抽取）：✅ 2026-09-20
**B① 3 标量语义分析**（`frozen_f_k_snapshot`/`mode`/`ema_mean_p_values`）：
- 首次文本出现均为 Store；首写点在 resume 决策 If 的**分支内**（L2043-2098
  if/else 两路、L2993-3011 两路），**每条路径先写后读 → 字段默认值不可观测**，
  收编语义安全；零内层遮蔽；41 处引用全在 run_all_windows 内。
**B② 收编**（执行器 `_warmup_state_absorb3.py`，留底 `absorb3.before.py`）：
- WarmupState 追加 3 字段（Any=None / str="learning" / Any=None，总 54）；
  41 处引用改写 `warm.<field>`；`_commit_frozen_f_k` 零改动（它以形参读
  mode/ema，调用点 kwarg 值自动变 warm.*——kwarg 名是字符串节点）。
- 门：执行器内置**双向验证**（期望树=原树深拷贝做同一变换+解析法插字段，
  与行手术结果全模块逐节点比对）当场拦下 FIELDS 缩进丢失（字段落到模块级）
  事故；verify_rename 新增 `--keep/--keep-file`（既有 51 字段访问不还原也不算
  stray），① PASS；layer_check 1357/107 不变。
**B③ 相位 3/4 抽取**（执行器 `_warmup_split_cuts.py --phase 3|4`）：
- **测量公式修正**：原 free-vars（loads−binds）会漏「段内先读后写」的名字
  （mode 在相位 3 内 L2111 先读、L2190 后写）。改用**保守公式**：
  参数 = 段内方法作用域 Load ∩ 调用方可见绑定——传入值=调用方当前值，
  与原同作用域读严格等价，代价只是签名变长。
- 收编后两条边界**写逃逸双双清零**（原 2+3 个 → 0）。
- 刀b：相位 3（While，774 行）→ `_run_warmup_loop(self, *, 24 形参)`；
  verify_extract `--slice 99:100` PASS。留底 `run_warmup_loop.before.py`。
- 刀c：相位 4（20 条语句，686 行）→ `_summarize_warmup(self, *, 39 形参)`；
  `--slice 100:120` PASS。留底 `summarize_warmup.before.py`。
  （执行器两处 bug 被自检拦住：PARAMS4 漏 self；相位 4 终止地标误用
  setStepSize，应为 _commit 调用点。）
- **现状**：`run_all_windows` 3381 → **1988 行**（循环体内调用链
  `_run_warmup_loop → _summarize_warmup → _commit_frozen_f_k → 刷新 →
  _produce_window`，顺序与相位序一致）；`_run_warmup_loop` 807 行、
  `_summarize_warmup` 734 行（>400，按 §2.4 属第三轮细分对象）；
  py_compile OK；全文件统一 CRLF（5710 行）；**layer_check 1357/107 不变**。

### 刀1 第二刀·刀a：`_commit_frozen_f_k` 抽取（§2.2.1）：✅ 2026-09-20
- **五边界测量先行**（用户纪律：量完再定刀序），需要上游/写逃逸双口径：

  | 候选边界 | 需要上游 | 写逃逸 | 裁决 |
  |---|---|---|---|
  | 相位5 单独（405行） | 24 | **无** | ✅ 执行（刀a） |
  | 相位3-5 整块（1869行） | 44 | **无** | ✅ 备选（与刀a互斥，已被刀a foreclosure） |
  | 相位3 单独（while 774行） | 24 | frozen_f_k_snapshot、mode | ❌ |
  | 相位4 单独（605行） | 39 | ema_mean_p_values、frozen_f_k_snapshot、mode | ❌ |
  | 相位2-5 整块 | 39 | K、original_dt、raw_probe、raw_span、skip_frozen_revalidation、warm | ❌ |
  | _produce_window 6/7、7/8、8 | 20-41 | 12~30 个 | ❌ 需 ProductionState |

- **WarmupState 之后仍剩 3 个标量跨界**：`frozen_f_k_snapshot`/`mode`/
  `ema_mean_p_values`——当年 51 名单的排除项（嵌套/非字面量初始化）。
  它们是 **B 段收编 WarmupState 的正料**；收编后 3/4、4/5 边界写逃逸自动清零，
  相位3-4/内部细分解锁。相位2-5 整体的 6 个逃逸名是 WindowContext/
  ProductionState 的料。**纯搬运刀的下一批全部以 B 段为前置。**
- 变换：相位 5（`if warm.bias_converged:` 单条，L3570-3974，405 行）→
  `_commit_frozen_f_k(self, *, 23 形参)`，keyword-only + keyword 调用点；
  执行器 `_commit_frozen_f_k_cut.py`（新行一律 CRLF，维持全文件统一 CRLF）。
- 门：verify_extract `--slice 120:121` PASS；头部/保留区内容逐行一致；
  方法体 405 行 == 搬移区去缩进逐行一致；ast.parse/py_compile OK；
  **layer_check 1357/107 分毫不差**。
- 行数：`run_all_windows` **3761 → 3381**；`_commit_frozen_f_k` 437
  （体 405，判据1 ⚠️ 临界，与 §2.4 预判一致）；文件 5498 → 5556。

### 文档勘误（用户待办④）：✅ 2026-09-20
- `HOW_TO_CONTINUE.md` §1 现状行、`b3_edge_decisions.md` 标题下、
  `layer_check.py` docstring 与两处 print 文案：标注 1336/97、707/629/44、
  253/629 均为 **infra 抽取前基线**，抽取后实测 = 总 1357/107，
  合法 913/62、违反 130/23、同层 314/22，非合法 444/45。**指标逻辑零改动**。

### 工具与本轮事故记录（09-20）
- verify_extract.py 新增 `--slice START:END`（块体切片比对；曾先加 `--tail N`，
  被门实测打脸——留尾法下搬移区不是块尾切片——已替换为 `--slice`）。
- **CRLF 教训补强**：本文件首查 `CRLF=False` 是 heredoc 双重转义导致的检查错误；
  该文件主体本就是 CRLF。本轮后**全文件已统一 CRLF**（5498 行 CRLF、0 裸 LF，
  Edit 工具规范化所致，内容零变化；WarmupState 执行器原写入的 67 行 LF 头一并
  归一）。
- `_produce_window_cut.py` 两处实测抓出的 bug（均已修）：`dedent4` 忘写
  `.splitlines(keepends=True)` 对整串逐字符迭代成空操作（ast.parse 自检拦住，
  未写盘）；调用块末行 `")\n"` 再追加 `\n` 产生多余空行（字节比对拦住）。

### WarmupState 收拢（CUTTING_PLAN.md §5 序1、§0.2）：✅ 2026-09-20
- 执行器 `refactor/_warmup_state_cut.py`（fail-closed：名单重算 vs 权威名单逐一
  比对；撞形参/循环外引用/walrus/del/global/match/遮蔽/`warm` 撞名任一即拒写）。
- 留底 `work/warmup_state.before.py`；名单 `work/warmup_state_names.txt`。
- 变换：51 条相位2常量型初始化 → 模块级 `@dataclass WarmupState`（文件头部）；
  循环体原 init 首位插入 `warm = WarmupState()`；体内 **327 处**引用改写为
  `warm.<field>`（其中 **31 处在 f-string 内**，py3.12 真实节点；无 `{x=}` 调试
  说明符，输出文本不变）。承载名 `warm`（§2.2 拆分草图；`state` 已被 OpenMM
  getState 局部量占用，L2969）。
- 门：verify_rename `--block for --index 0 --construction 1` 全绿——
  ①还原 327 处逐节点等价；①'锚点外 13 条语句等价；②③51 条 init 与字段
  初值逐一相同。
- 终检：ast.parse/py_compile OK；`run_all_windows` **5007 → 4957 行**；
  文件 5405 → 5424；**layer_check 17域口径 1357 次/107 边与刀前分毫不差**。
- **名单口径（重要，§0.2 的「51」首次可复现）**：循环体顶层、单名目标
  （Assign/AnnAssign）、相位2区间 [L1204,L2088]、RHS ∈ {纯字面量,
  float('nan'|'inf'), 全大写模块常量名} = 52 条候选，**排除 check_chunk**
  （全循环仅 1 Store+14 Load，是 IBS_WARMUP_FRAME_STRIDE_STEPS 的纯别名、
  非累加器）→ 51。文档的 51 从未落盘，此为唯一有语义依据的重建。
- 注解：AnnAssign 沿用原注解（List[float]、Dict[str, float]）；字面量按值推断
  （bool/int/float/list/dict；模块常量 = int，实测 = 5）；None 初值 10 个字段
  一律 `Any`（诚实未知，未猜）。

### 1365 勘误（09-20，口径修正）
- 旧终检③的 1365 无法复现且无存档输出，判定为旧会话记忆数。
- **17 域口径（layer_check 现行）基准 = 1357 次/107 边**；deps.py 是噪音口径
  （scratchpad 留底与顶层工具都被当域，数字随留底增减漂移，09-20 晚为
  1800/26 域，纯由新增留底副本与 `_warmup_state_cut.py` 引起）。
- layer_check.py 已把 scratchpad 加入过滤（工具修正，指标逻辑未动）。

## 工具链现状（5 门 + 复合门 + 2 体检）

全部在 `refactor/tools/`，一律在**仓库根**执行。

| 工具 | 管什么 | 状态 |
|---|---|---|
| `verify_extract.py` | **纯搬运门**：块整体移位、名字引用不变 | `--block def/fn/for/if/while/with/orelse/defs`、`--index`、`--slice START:END`、`--strip-call NAME=K` |
| `verify_rename.py` | **归组门**：`x` → `bind.x`，逻辑不变 | `--in-place`（初值原地保留）、`--construction N`（构造语句双侧成对剔除）、`--expand-kwarg`（`es=es` 展开还原，全树扫描 + 同形调用唯一匹配）、`--block/--index`（锚到嵌套块，锚外强制等价 ①'）、绑定检查纳入形参、moved 只吸收首个顶层赋值 |
| `verify_dispatch.py` | **条件分派门**：抽取后调用点的进入条件与兜底 | 判据 (a) test 序列是原链同位前缀 (b) 兜底等价（原链有终端 else ⟺ 分派有终端 else）(c) 每支形状恰为 `return self._m(...)` |
| `verify_a1.py` | A1 断言不变性门 | 稳定 |
| `compound_gate_knife1.py` | **刀1 链端到端封链** | 当前 **10 变换**，公理**仅 1 条**（51 init 字面量提升，三条件已独立复核：前移区间零读、init 间零依赖、51/51 顶层无条件） |
| `deps.py` / `layer_check.py` | 域依赖体检 / 分层违反 | **`layer_check` 17 域口径 = 基准 1357 次/107 边**；`deps.py` 是噪音口径，随留底增减漂移，不作基准 |

**门还没覆盖的形态（已知缺口）**：
- ⛔ **return 元组解包**（Cut A 阻塞 2）——待建。
- ⛔ **C 类语义分派**（`AUDIT-S2-02` 第 1 步，50 个 `return plan(...)` 按证据类型重排）——无门，不许做。

**复验链**：`refactor/work/verify/*.verify.sh` ×16，**留底链配对**（每刀 after =
下一刀留底，仅末刀指当前文件）⟹ 整条链任意时刻可逐刀重放，不依赖「当时那一刻」。
每刀落盘必须附 `.verify.sh`（GUIDE §九清单项）。

## 当前量化状态（2026-09-20 复核）

```
复验链 16/16 PASS   |   复合门 PASS（10 变换）   |   layer_check 1357/107（十余刀未动）
主线零改动          |   仓库根干净               |   refactor/ 顶层 6 条目、19M

ABFEPipeline                15614 行 / 99 方法 / ≥400 的 11 个
    1553 _run_stage2_autonomous（未动，Cut A 待开）   1190 _run_dual_lambda_scheme
    1048 _run_dual_lambda_stage（未动）               1037 run_full_pipeline
     664 _run_stage2_with_path_evolution（未动）       640 _resume_preoptimized
     506 pre_equilibrate（未动）  469/462/421/417 …
    数据类：PipelineState(28)

IBSWindowManagerDualLambda   5652 行 / 14 方法 / ≥400 的 6 个
    1978 run_all_windows   829 _finish_production   804 _run_warmup_loop
     731 _summarize_warmup 471 _produce_window(17 参)  434 _commit_frozen_f_k
    数据类：WarmupState(54) EarlyStopConfig(9,frozen+slots) WindowContext(6) ProductionState(26)

Stage2RepairController       5474 行 / 42 方法 / ≥400 的 2 个
    1893 _decide_once（50 个 return plan(...) 分派骨架已露出）  787 _read_single_stage
    数据类：ExitSpec(3)

三大类合计 ≥400 的方法：19 个
达标（≤400）的新单元：_partition_windows(207) 一个
```

**诚实口径**：十三刀减了约 2100 行巨定义体积，但**类总行数未降**（拆方法不减行，
新增数据类与签名还略增）。真正的产出是**门与复验链**——代码体积改善仍在早期。
签名收缩是 §0 判据 2 唯一拿到实质进展处：`_produce_window` 30→17 参、
`_finish_production` 29→16 参。

## 下一步（全部等用户点名）

| 优先 | 事项 | 前置 |
|---|---|---|
| **1** | **return-tuple 门** | 无。Cut A 的硬前置，按 GUIDE §三先自检会 FAIL |
| **2** | **Cut A**：`_run_stage2_autonomous` 整 Try 抽 `_execute_stage2_action` | 门 + 三名既入参又出参（见上节两条阻塞） |
| 3 | `_run_dual_lambda_stage`(1048)、`_run_stage2_with_path_evolution`(664)、`pre_equilibrate`(506) | 各自先做相位测量（未量不下刀） |
| 4 | `_run_warmup_loop`(804) / `_summarize_warmup`(731) / `_finish_production`(829) 细分 | 先量内部相位；剩余签名大头是派生值，可能需新状态对象 |
| 5 | 相位 7/8 细分 | ⛔ 用户已拍板**不批准** `UnboundLocal → None` 语义变更；解锁 = 证明上游保证 `n_updates ≥ 1`（读生产逻辑，**不属本职责范围**，需另立项交出去） |
| 6 | A1 提案落地 `tests/` | 46 diff 影子树全绿（974 passed/0 failed），等用户点名 |
| — | `AUDIT-S2-02` 第 1 步（证据类型分派） | ⛔ **无门，不许做**，另立项 |
| — | WarmupState(54) 补类型标注 | 10 个 `Any`、PipelineState 26 个全 `Any`；要从下游用法反推，独立事项 |

**已完结不再跟进**：L14026 修复（kasuga02 全量 2756 passed / 0 failed）；
文档勘误；1365→1357 口径勘误；refactor/ 重组（52→6 条目）。

---

## ⚠️ 主线漂移（2026-09-20 落盘时发现，影响 C 阶段）

落盘时 `git status` 显示**主线已被别的会话改动**（不是本重构线做的，本线自始至终
只写 `refactor/`）：

```
 M abfe_core.py           +195 行区
 M abfe_pipeline.py       +444 行区
 M abfe_preoptimizer.py     +9 行区
 M ibs_engine.py          +175 行区
 M runabfe.py              +62 行区
 M tests/test_ctl04_rewindow_intent_ledger.py
 M tests/test_stage_diagnostics_persistence.py
?? tests/test_high_medium_fixes_2026_09_20.py
?? tests/test_split_half_and_merged_path_2026_09_20.py
合计 732 insertions / 204 deletions
```

**五个被改的文件全部是 `refactor/domains/` 的快照源头**（WIRING.md 的切分来源）。

### 这意味着什么

- **对已完成的十三刀：无影响。** 所有门比对的都是 `refactor/` 内部的
  before/after 对，`layer_check 1357/107`、复合门 10 变换、16 个 .verify.sh
  全部只在 `refactor/` 树内闭合，与主线现值无关。**本次落盘时实测仍全绿。**
- **对 C 阶段（把重构成果搬回主线）：有影响。** `refactor/domains/` 是
  2026-09-19 的主线快照，现在主线已前进 732/204 行。C 落地时**不能假设
  domains/ 与主线逐字对应**，必须先做一次漂移对账（哪些行区被改、是否落在
  已切过的函数体内）。
- **对 A1（46 个 diff 落地 tests/）：需重验。** 提案是对 09-19 的 tests/ 生成的，
  而 `test_ctl04_rewindow_intent_ledger.py` / `test_stage_diagnostics_persistence.py`
  已被改，且新增两个测试文件。落地前必须重跑影子树（上次 974 passed/0 failed）。

### 处置

- 本重构线**不追这批改动的内容**（生产改动不属本线职责，见用户 09-19 的范围界定）。
- 仅登记事实 + 上述三条影响。**C 阶段开工前第一件事 = 漂移对账**，
  A1 落地前第一件事 = 重跑影子树。
- 漂移量会继续增长（主线在活跃开发）。这是「`refactor/` 是 freeze 点、
  主线一动它就是死档」这条早期判断的实证——见 `docs/HOW_TO_CONTINUE.md` §1。

---

## 可重放切分体系（2026-09-21，进行中 → 暂停，等主线）

**完整设计与坑表见 `refactor/docs/REPLAY_DESIGN.md`。** 这里只记状态。

**起因**：`domains/` 是 09-19 快照的切分副本，主线漂移 688/197 行、26 个 hunk 落在
已拆过的三个函数体内 ⟹ 落地会变成无门覆盖的三方合并。改为**把每刀写成可重放的变换**。

**关键简化**：15/16 刀只按「类名+方法名」定位，与域切分无关 ⟹ 重放直接对主线三个
源文件的副本做，**跳过域切分**（那是全流程最位置相关的一环）。`_infra_cut` 排除
（属 B 阶段分层）。

**已完成**
- `refactor/tools/cutter.py` —— 通用引擎，`extract` / `absorb` 两模式，对应两道现成的门。
  含 `extend`（追加字段到已有数据类）、`seed_from_param`（形参值播种进构造点）、
  `proven_conditional`（首写在嵌套块时**强制写书面理由**，否则 fail-closed）。
- `refactor/tools/check_spec.sh` —— spec 验收：行尾归一后逐字节相同 + AST 等价。
  （不用裸 cmp：历史 `_pipeline_state_absorb` 把 40 行 CRLF 写进纯 LF 文件，
  产物混合行尾是缺陷，引擎按文件主导行尾写、不复刻。）
- 4 份 spec 全部验收通过：`partition_windows`(extract 样板)、`production_state`
  (absorb/in-place 样板)、`pipeline_state`(3 条 proven_conditional)、
  `pipeline_state2`(extend + seed_from_param)。
- 2 个特化执行器的位置锚改造完成（`_partition_windows_cut` / `_resume_preoptimized_cut`
  的 `SLICE` → `_locate()` 结构锚），单测四项全过（留底命中 / 插入无关语句后自动跟随 /
  锚点消失 fail-closed / 前置刀未跑 fail-closed）。

**未完成**
- 引擎缺 `fns` 复数支持（`early_stop_config` / `window_context` 跨多方法改写）。
- 还剩 11 份 spec（含 `warmup_state` 的搬家口径、6 份 extract 族）。
- `replay.py` 未写（固定刀序 + 每刀过门 + 失败即停）。
- **第一次完整重放未做** —— 那才是这套体系的真考验。

**新发现的通用坑（建议进 GUIDE §五坑表）**：`ast` 的 `col_offset` 是 **UTF-8 字节
偏移，不是字符偏移**。本仓满屏中文注释与日志，一行里只要前面有中文，按 col_offset
索引 str 就插到错误位置；表现是莫名其妙的 SyntaxError 且报错行离真凶很远。

**代码量**：特化执行器 226 行 → spec 40 行（extract）；207 → 52（absorb）。

---

## ⚠️ 引擎抓到的历史真 bug：三处方法漏传 `legacy_repair`（2026-09-22）

通用引擎的**参数表对账**（独立重算跨界形参，与 spec 声明比对）抓到：

```
refactor/domains/windows_sampling/window_manager_dual_lambda.py
  _summarize_warmup   L4580  if legacy_repair          ← 形参表 36 个，不含它
                      L4604  legacy_repair
  _commit_frozen_f_k  L3406  if not legacy_repair:     ← 形参表 20 个，不含它
```

四处引用**全是 Load，方法内无任何 Store 绑定**，也不在内层作用域 ⟹ 执行到
这几行必然 `NameError`。`legacy_repair` 是 `run_all_windows` 的形参，抽方法时
漏传了。

**为什么现有的门抓不到**：`verify_extract` 比对的是**搬走的语句块**——块逐节点
相同，它 PASS 是对的。**形参表是签名侧的新代码，不在任何门的射程内。**

这是**同一个结构性缺口的第三次**：

| # | 形态 | 事故 |
|---|---|---|
| 1 | 调用点的**分派条件** | 刀2 的 5 路 → 2 路，三条路径变死码 + fail-closed 的 raise 变静默 None（已修，并建 `verify_dispatch`） |
| 2 | 调用点的 **return 元组解包** | Cut A 阻塞 2（未发生，事前拦下） |
| 3 | **签名的形参表** | 本条（已发生，存活至今） |

⟹ 结论：**抽取类的门只验「搬走的代码没变」，不验「替代它的签名与调用点」。**
通用引擎因为独立重算形参表，成了这个缺口的第一道实际防线。

**处置**：两份 spec（`commit_frozen_f_k` / `summarize_warmup`）的形参表写成
**正确**的（各 +1），因此它们的产物与历史产物**不逐字节相同**，验收只取
AST 等价 + 该单一差异。`refactor/domains/` 里的存量 bug**未修**——修它是一刀
独立的行为改动，需要单独点名。

## 可重放体系进度（2026-09-22）

**8 份 spec 完成**，6 份逐字节 + AST 双绿，2 份如上（产出修好的代码）：

| spec | 模式 | 验收 |
|---|---|---|
| `partition_windows` | extract | ✅ 字节 + AST |
| `produce_window` | extract / loop 容器 / 带前导注释 | ✅ 字节 + AST |
| `run_warmup_loop` | extract / single（整条 While） | ✅ 字节 + AST |
| `summarize_warmup` | extract / start_after 锚 | AST ✅（形参 +1，修 bug） |
| `commit_frozen_f_k` | extract / single + if_test_exact | AST ✅（形参 +1，修 bug） |
| `pipeline_state` | absorb / in-place + 3 条 proven_conditional | ✅ 字节 + AST |
| `pipeline_state2` | absorb / extend + seed_from_param | ✅ 字节 + AST |
| `production_state` | absorb / in-place | ✅ 字节 + AST |

**引擎本轮新增能力**（全部由 FAIL 逼出来）：
- `container="loop"` —— 锚定在方法体唯一循环的体内（`run_all_windows` 那条线四刀都在循环体里）
- `single=True` / `start_after=` —— 搬移区恰为一条语句 / 起点是某调用点的下一条
- `if_test_exact()` —— test 精确相等（`warm.bias_converged` 在循环体出现 6 次，5 次是 `not ... and ...`）
- **两级缩进**：`call_indent`（调用点层）与 `body_indent`（方法体层）分开，loop 容器差 4 必须去缩进，fn 容器相等必须逐字搬（误套模板把文件写坏过）
- `take_leading_comments`（默认 False = 历史口径；`produce_window` 开 True）
- 循环**目标变量**（`for window_idx, (start, end) in …`）算进形参候选
- 方法级 `Return` 检查**排除内嵌作用域**（内嵌 def 的 return 是它自己的）
- 生成文本按**文件行尾**（原来硬编码 `\n`，写进 CRLF 文件就花了）
- `doc_text` —— spec 只存 docstring 内容，缩进与引号由引擎渲染（原来存渲染好的行，repr 转义出过错）
- 新方法插入点**越过类末尾空行**

**未完成**：`warmup_state`（absorb 搬家口径 51 名）、`absorb3`、`early_stop_config` /
`window_context`（**跨多方法，引擎缺 `fns` 复数支持**）、`scheme_dispatch`、
`resume_preoptimized`、`finish_production`、`decide_once_step0`（特例）、`replay.py`。
