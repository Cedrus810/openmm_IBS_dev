# Stage-2 自治控制器：设计、实证与陷阱

2026-09-12。控制器首次独立走完整条 Stage-2（六窗全部 `ANALYSIS_ELIGIBLE` → `DONE`）。

> ⚠️ **2026-09-14 更正：本文原先的头条数字 ΔG_bind = −3.48 ± 0.47 kcal/mol
> （实验 −4.04，1.19σ）已作废，不得再引用。**
>
> 那个数字来自 `cyclod_ligand2/rep1`，而该 run 的 vanishing 项（本文 §10 里那个
> `ΔG_vdw = −49.04`）是**只解了末窗一个窗口的部分和**（`covered_lambda_indices=[17…23]`），
> 被当成完整 ΔG 写进 stage 缓存并被 `final_results` 采信。同 run 自己的全路径中间
> 结果是 **+43.00**，rep2 是 **+44.09**。
>
> 它之所以"对上了实验"，是**两个 ~90 kJ/mol 的错误反号抵消**：部分和让复合物腿
> 低了 +92，stage-1 的分子内库仑湮灭让它低了 −88.7。用 rep1 自己的全路径重算
> → ΔG_bind = **−25.49 kcal/mol**，与 rep2/rep3 一致。
>
> 控制器的**流程**仍然跑通了（这是本文其余部分的价值）；被作废的只是这个 ΔG 数字。
> 两处根因分别记在 `docs/STATUS.md` 与
> `_freeze_ligand_internal_coulomb` 的 docstring。

体系：`cyclod_ligand2/rep1`（环糊精主客体，可溶）。

**接手的人**：§9 代码在哪 · §10 当前状态与复现 · §11 别重新论证的事 ·
§12 规矩。动手前先读 §6 的十条陷阱。

前置文档（读这份之前先读）：
- `PLAN_PATH_REPAIR_2026-09-11.md` —— 设计要求的来源，分支编号（5a/5b/6）出自这里
- `STAGE2_THREE_AXES_COUPLING_2026-09-11.md` —— 三轴耦合
- `STAGE2_AUTONOMOUS_LOOP_STATUS_2026-09-11.md` —— 逐个 bug 的修复流水账

---

## 1. 一句话设计

```
while not DONE:
    evidence = read()          # 只读盘，不算
    action   = decide(evidence)# 唯一决策点
    execute(action)            # 唯一执行点
```

**一个 stage 只许有一个控制器。** 这不是风格偏好 —— 两套机制各判各的会直接
打架（见 §5.2 实证）。

## 2. 三个真终态，其余全是路由

只有这三个允许终止循环：

| 终态 | 含义 |
|---|---|
| `DONE` | 全路径证据齐备且达标 |
| `GLOBAL_BUDGET_EXHAUSTED` | **所有**窗口预算都耗尽（单窗耗尽不算） |
| `NO_FEASIBLE_ACTION` | 归因成功但动作都不可行 |

`LOCAL_VALIDATION_CAP` / `INSUFFICIENT_DATA` / `CUMULATIVE_FK_MISALIGNMENT` /
`SKIPPED_WINDOW` / `HALT_BUDGET` **全是路由信号** —— 被主循环消费、换个动作继续，
**不得炸出流水线**。

> ### 🔑 [2026-09-13 修正，协议 v2] `DONE` 不许当"跑完了"用
>
> 分支 8 原先是 `"DONE" if has_stage_result else "ANALYZE"` —— **只要 stage 结果
> 文件存在就返回 `DONE`+`exit=DONE`**，哪怕结果里明写 `converged=False`。
> 这与上表直接冲突：`DONE` 的定义是「全路径证据齐备且**达标**」。
>
> 后果不只是措辞：主循环 `_run_stage2_autonomous` **只看 `plan["terminal"]` 就 break**，
> 不看 `evidence_status` —— 所以判据没过时自治诊断会**提前结束**。
> （最终结果没有因此被错误发布：下游另有默认拒绝 `converged=False` 的质量门。
> 但循环白停了。）
>
> **三维停止标准仍然成立，但要分清哪一维承载什么**：窗口采样「执行完」记
> `execution_status=COMPLETE` 是对的，**那不能顺带把动作也记成 `DONE`**。
>
> 现在的口径：
>
> | 情况 | 动作 | 出口 |
> |---|---|---|
> | 判据通过 | `DONE` | `DONE` |
> | 判据没过 + 调用方显式 `allow_untrusted` | `DONE` | `DONE_UNTRUSTED`（发布策略，`evidence_status` 仍不是 `CONVERGED`） |
> | 判据没过 + 支撑/偏斜类门失败 | `SPLIT_TAIL_WINDOW` / `INSERT_LAMBDA` | 无（路由） |
> | 判据没过 + 样本量/精度类门失败 | `RUN_PRODUCTION` | 无（路由） |
> | 判据没过 + 归因不出来 / 动作都不可行 | **`NO_ACTION`**（新词） | `NO_FEASIBLE_ACTION` |
>
> 新动作 `NO_ACTION` 是必要的：借 `DONE` 会让 `plan()` 把 `execution_status`
> 算成 `COMPLETE`（它按 `action == "DONE"` 判）。终态在主循环里**先于**执行器
> 分发被 break，所以执行器不需要认识它。
>
> ⚠️ **不许改成 `ANALYZE`** —— 分析已经跑过了，再跑一次是空转
> （而且 ANALYZE 每轮都重写结果文件，连停滞保护都未必兜得住）。
>
> 归因怎么读：`converged` 是**五个条件的合取**，结果 payload 里每一项都带实测值
> 和阈值，所以"为什么没过"是读得出来的，见
> `abfe_preoptimizer.stage_quality_gate_failures()`。对症分类的出处是本文
> §3（低支撑是「尚不可测」⟹ 加采样）与 §5.1（加帧治不了偏斜 ⟹ 缩跨度）；
> **读不出来的一律标 `unattributed`，不许猜一个动作出来。**
>
> 回归用例：`tests/test_stage2_repair_controller.py::test_stage_not_converged_is_never_done`
> （六种失败形状参数化）。

⚠️ 异常也一样。`IBSValidationBudgetIndeterminateError`（没测出来）和
`IBSWarmupConvergenceError`（f_k 压不平）必须被接住并路由；只有
`IBSFrozenCalibrationValidationError`（f_k 被**统计驳回**）是有功效的否决 ——
它封存该候选、最多换一次 Epoch，仍然不终止整个 Stage-2。

## 3. 四个量的角色，不许混用

| 量 | 角色 |
|---|---|
| `n_decorr` | **求解器资格**：够不够格尝试。不是验收量 |
| `N_eff,k / g_k` | **主验收量**。门 = 10 |
| `top1% weight` | 否决**警报**（重尾/单帧支配） |
| occupancy / coverage ESS | **只诊断 f_k 训练**，永不决定 ΔG 是否有效 |

判据分档：`min(N_eff/g) < 1` → `HARD_INSUFFICIENT`；`1–10` → `INSUFFICIENT_DATA`；
`≥ 10` → `ANALYSIS_ELIGIBLE`（**不是** `VALID_PASS`）。

**低支撑永远不是 FAIL，是「尚不可测」。**

⚠️ 日志里 `worst_target_state_local` 来自**去相关**计算（对 g 最差的态），
跟 `min N_eff/g` 的最差态**不是同一个量**。印在一起会把人带偏（本人中过一次，
据此误判了 win3）。

## 4. 窗口三态 + earliest-unresolved

```
ELIGIBLE / PROBLEM / UNKNOWN
earliest = 第一个 PROBLEM  ⟹ 只路由它，下游一律阻塞
```

- **缺证据 ≠ 有问题**：`UNKNOWN` 的正确动作是「产出证据」（ANALYZE），不是重标定。
- **但「证据被我们自己作废」≠「还不知道」**：布局一变（插 λ / 拆末窗），
  旧证据描述的是另一个窗口几何、已被剔除 —— 那是**确定的未解决**，必须判
  `PROBLEM` 挡住下游。不这么做就会「跳过前面的窗口去跑后面的」。

## 5. 实证：win4 的修复链（这是这套设计成立的证据）

| 动作 | min N_eff/g | g |
|---|---|---|
| 初始 500 帧 | 3.44 | 15.1 |
| 加帧到 1000 | 2.29 | 49.2 |
| 换 Epoch | 2.01 | 28.5 |
| 再换 Epoch | 1.62 | **168.8** |
| **插 λ**（跨度 0.0739→0.0441） | 6.39 | 49.5 |
| **再插**（→0.0220） | **38.99** ✅ | 8.97 |

**加帧和换 f_k 三次都把它推得更糟，缩跨度两次解决。** 而且这个转向是控制器
**自己**做的 —— 判据是「`min N_eff/g` 没有随采样上升 ⟹ 同分布加帧已被本窗口
自己的数据证伪」。

### 5.1 为什么加帧治不了

同一个偏斜分布里加帧，N 涨 g 也涨，**比值不动甚至倒退**。物理上 win4 是
**窗口内不遍历**：不同时间块占据不同 λ 区域，连续采样再多也摊不平。
对症动作是缩跨度（插 λ）或换 f_k（重加权），不是更多帧。

### 5.2 model B 的代价是真的

插 λ 时**索引区间固定**、多出来的态全落末窗。所以修 win4 的代价直接落在 win5：

```
win4  1.62 → 38.99      win5  25.93 → 2.67
```

末窗是溢出槽，它的上界是**可拆上限** `2*hi−1`（不是 `hi`）—— 超过它两个子窗
不可能都落在 `[lo,hi]`（`p+q−1=K`），溢出槽从此是死胡同。

⚠️ 4/5 配置下 **K=6 是死区**：超单窗上限 5，又够不到可拆下限 7。出路是补 1 到
K=7，唯一拆法 4+4（`[17,18,19,X] + [X,20,21,22]`，共享**新插的** X）。
想要「补 2 且俩 4 不相交」不行 —— 接缝上没有共享节点，边 X→Y 没人负责，
ΔG 会少一项。

---

## 6. 陷阱清单（每一条都在真机上咬过）

### 6.1 ⚠️⚠️ 同一个不变量的 N 份实现 —— 最贵的一个

λ 身份判定曾经有**四份**：

| 位置 | 原口径 |
|---|---|
| resume 缓存门 | `np.allclose(atol=1e-9)`（默认 rtol=1e-5 ⟹ 实际 ~3e-6） |
| `load_ibs_window_outputs_from_dir` | 浮点**精确相等** |
| production manifest | 浮点**精确相等** |
| 控制器过期过滤 | `abs(x−y) ≤ 1e-9` |

而 λ 落盘时被 `lambda_path_versions._q()` 量化到 8 位、采样用的是**未量化**值，
天然差 **5e-9** —— 正好卡在这几个阈值中间。后果：

> 采样侧说「λ 匹配，跳过重采」，分析侧说「λ 不匹配」抛 ValueError
> ⟹ 那个窗口**永远采不了也永远读不了**，每次启动必崩。

**修法**：
1. 前向 —— `append_version` 回写的 λ 才是采样权威（不再制造新错位）；
2. 存量 —— 所有 λ 比较统一**归一到量化栅格**再精确比较，
   `LAMBDA_GRID_DECIMALS` 从 `lambda_path_versions` 导入，**不另立常量**
   （两份常量各自漂移正是成因）。

这不是放宽判据：λ 量化到 8 位 ⟹ 真实差异必 ≥ 1e-8，量化噪声 ≤ 5e-9。

### 6.2 段号不是「新旧」的代理

`read_aggregated` 曾按「段号最大且有 convergence」取证据。那在「段只往前开、
基准段从不重跑」时成立；但循环会往基准段写、布局变更时整段重跑，
**基准段反而常常最新**。实测：

```
win4 在 base:        43.25 ✅ 16:36（最新）
win4 在 vanishing_5:  1.62 ❌ 15:51（废弃 Epoch）
聚合认的 →           1.62        ⟹ 继续"修"一个已经好了的窗口
```

正解不是比 mtime（脆弱），是**语义身份**：每份证据带着它产出时用的那套 λ，
对不上当前布局就丢弃（丢弃后留**占位记录**，见 §4）。

### 6.3 非变异动作不能用「盘变没变」判进展

`PROBE_CANDIDATE_FK` 按定义不改盘 ⟹「盘上状态未变」对它**永远成立** ⟹
必然触发停滞保护 ⟹ 连探三次后 `NO_FEASIBLE_ACTION` 退出。
而它**已经给出结论了**，只是 `diag` 拿到就丢。

正解：探针结论落盘（`stage2_fk_recalibration_probe.json`），
`decide()` 消费它、探过就不再重选。

### 6.4 中间结果只活在内存里 = 结构性死锁

`decide()` 要读 `cumulative_fk_residual_production`（在 stage 结果里），
而 stage 结果原本只在**整个 stage 跑完**才落盘 —— 但循环正因为拿不到它而永远
跑不完。`DONE` 在结构上不可达。

正解：每轮 ANALYZE 之后立刻落盘到
`stage2_vanishing_autonomous_inprogress.json`。
⚠️ 文件名**刻意不是** `stage2_vanishing.json` —— 那个是 stage 缓存的**完成标记**，
写了会让下次 resume 以为 stage 2 已经跑完。

### 6.5 窗口子集跑不产出 stage 级裁决

`only_window_indices` 非空时，「全局 TMBAR」本来就不是完整路径的结论，
它失败**不构成致命错误**。实测链条：win4 判 `HARD_INSUFFICIENT` ⟹ 求解器跳过它
⟹ 这次只跑了 win4 一个窗口 ⟹ 全局 TMBAR 一个可用窗口都没有 ⟹
`no_local_tmbar_results` ⟹ 炸穿循环。

### 6.6 部分段是常态

修复动作限定窗口之后，段 N 里只有被修的那几个窗口。loader 对**单个段**要求
全窗口齐全是对的（缺首/末窗会产出**截断的 ΔG 却仍报 converged=True**），
但那个不变量属于**合并之后的全体**。所以逐段**显式声明**缺窗
（`excluded_local_windows`，不靠文件缺失隐式决定），再断言「每个期望窗口至少被
某一个段覆盖」。

### 6.7 末窗豁免只活在 path-version 层

插 λ 允许末窗涨到 `2*hi−1`；但执行时的权威 `window_ranges` 校验曾对**所有**窗口
一律套 `hi`，于是合法的溢出布局被拒 —— 而且拒在**执行时**、离写入点十万八千里。
那个报错自己的理由也只管下界（"K<lo 压不平占据"），上界从来没有过论证。

### 6.8 段是「相加」不是「取代」

采纳新段等于把旧段的帧全扔掉（实测 w1/w2/w3 里 2/3 到 4/5 的 ESS）。
多段必须按窗口**合并**求解（`_load_ibs_window_outputs_merged` +
`multi_segment_analysis` 在最终帧集上自洽塌缩）。
⚠️ 自治循环最初**没接这个** —— 它修好了窗口，却报出没修过的那个答案。

### 6.9 别拿 hash 做自产产物的身份

主线在快速变动，哈希里放什么一改，所有已封存记录全部失配。而且这是本仓库
**已复发四次**的坑。规则：「只有用户输入才配做身份」。正解不是删掉身份，
是改成**语义身份**（如 f_k 向量本身，mean-center 后比距离）。

### 6.10 跨 f_k epoch 比数字 = 比了个寂寞

本人栽过：拿「段2 的新 f_k 结果 10.52」减「基准段的旧 f_k 结果 4.39」，
然后说成「加帧加坏了」。两条剖面形状完全不同（平 vs 陡降），那是 f_k 的差别。
**同 f_k 内的对比才有意义。**

反过来也有一条已记录的偏差：**更短的序列会让新段在所有 g/ESS 类诊断上系统性
显得更好**（实测段 1 的 w3 全长 1000 帧 g=202.9，截到 500 帧变成 13.9/31.0）。
比之前先对齐帧数。

---

## 7. 已知缺口

> **2026-09-13 更新**：本节原标题是「未修，留给后人」，现在五条里**已修两条**
> （第 2、第 4），逐条标在原文上。**已修的不删、原文保留** —— 它们各自留下了一条
> 仍然有效的判断（见每条的标注）。剩下的三条是第 1、3、5。

1. **累计 f_k 残差门对多段窗口结构性不适用。** 多段窗口的帧采自两份不同偏置，
   `_load_ibs_window_outputs_merged` 因此显式 `base.pop("f_k")`。
   于是循环每用换 Epoch 修好一个窗口，那个窗口就**永久失去**这项证据
   （本跑 win0–3 已全丢）。逐段残差不与合并后的 ΔF 直接望远镜相消，
   需要单独定口径。
2. ~~**`ibs_engine.py` 续验路径上验证要求随预算膨胀**~~ —— **2026-09-13 已修**
   （`S2-B`）。原式 `minimum_complete_validation_frames = max(200, budget/stride)`
   已去掉预算那一支，只剩统计目标 `required_consecutive_bias_updates * 20`。
   ⚠️ 原文写的「可达性判据的第 2 档因此失效」**是过期的**：可达性预检的 T 早已
   改读 `decorrelated_frames_required`（去相关下限 10），不读这个量；本仓自第一个
   commit 起这个量就**只进报告、不当门**。所以修它不改变任何判定 —— 修的理由是
   这个随预算浮动的数**会骗读它的人**（T 一度就被错取成它，gcrit 算小 20 倍），
   而且两次 run 的报告没法横向比。守卫：
   `tests/test_fk_relearn_and_reachability.py::test_completeness_requirement_is_decoupled_from_budget`
   （按源码 AST 断言右侧不得引用任何预算量，已用旧公式变异验证会红）。
3. **边际增长判据需要至少两段历史**，只有一段的窗口用不上。
4. ~~**代码仍散在三个文件**~~ —— **2026-09-12 已关闭**（`S2-D`，整段进
   [archive/TODO_closed_2026-09-12.md](archive/TODO_closed_2026-09-12.md)）。
   ⚠️ 原文写的「设计要求是**包在一起**」**是被否掉的口径**：收拢的判据不是
   "都塞进一个文件"，而是**决策同源的进 `abfe_preoptimizer`、写盘的留
   `abfe_pipeline`**（"执行器"= 写盘，由 `test_controller_never_writes_anything`
   钉住）。逐符号现状见 §9.4；`ibs_engine` 那三个是纯函数/常量，**本来就该在引擎侧、
   别搬**（§9.5）。
5. 旧的修复机制（path_evolution 修复分支 / production rescue / rescue 后重标定）
   目前是**关掉**不是删掉。等自治这条再跑通几次再删。

## 7.5 跨腿构象一致性门：WARN 级，不阻断（2026-09-12 裁决）

`combine_binding_free_energy` 里的 P0-12a/§3.0 门原先**硬抛 ValueError**，
把两条腿合不起来。本体系实测被它拦下：

```
复合物腿 max_internal_heavy_distance_nm [p5,p95] = [1.111, 1.231] nm
溶剂腿                                            = [0.682, 1.107] nm
差 0.004 nm 没重叠
```

**裁决：这类装置上构象差是必然产物，不是采样不足。** 溶剂腿里配体没有任何东西
撑着、vdW 一关就塌；复合物腿里主体腔把它卡住、塌不了。拿它去硬拦 ΔG_bind
等于用装置固有的性质否决整条热力学循环。

改成：

- **最多 WARN 级，不阻断汇总。** 判据一字未改，只改「炸不炸」。
- **但不得当作「完全跳出」的判据。** 结果里**永远**写
  `cross_leg_conformer_gate`（`PASSED` / `WARN` / `NOT_EVALUATED`），
  WARN 时带完整 report。WARN **不等于已处理** —— 它的含义是
  「这个 ΔG_bind 里有一项构象自由能差没被计入」。
- 想硬阻断的调用方显式传 `strict_cross_leg_conformer=True`。

⚠️ 报错原文里的 `实测 0.657–0.672 nm，σ=0.005` 是**硬编码在字符串里的旧例子**
（`abfe_core.py` / `abfe_pipeline.py` 各一处），不是当次测量值。本人据此误判过一次
「溶剂腿被困在窄 basin」—— 实际这次溶剂腿区间宽 0.425 nm，比复合物腿的 0.120 还宽
3.5 倍。**引用报错里的数字前先确认它是不是算出来的。**

## 7.6 后分析 / A/B 对比

为了跟 **outer-λ 增强采样**对照，逐跑的中间量已经统一落盘：

- `Stage2RepairController.comparison_manifest()` —— **纯聚合**（不新算任何东西），
  把身份（λ 表 / 窗口划分 / 协议版本 / 采样段）、逐窗验收量、路径级 ΔG±σ、
  **代价**（总步数 / 段数 / 自治轮数）、控制器动作拍平到一份，落在
  `<run>/checkpoints/stage2_comparison_manifest.json`。
- 自治历史 `stage2_autonomous_history.json` 的每一轮都带**决策那一刻的盘面快照**
  （逐窗 min N_eff/g、verdict、去相关帧数、生产步数、剩余预算、λ 跨度、脱轨状态
  + 当时的 `path_version` / `window_ranges`）。只看终态是看不出差异在哪一步
  产生的。⚠️ 2026-09-12 之后的跑才有。

后分析脚本（**单独成文件，因为这是后分析不是控制器逻辑**）：

```bash
python stage2_ab_report.py <run_dir>                    # 单跑演化报告
python stage2_ab_report.py --ab <baseline> <candidate>  # A/B 对比
python stage2_ab_report.py <run_dir> --json             # 机读
```

A/B 三块必看，缺一块结论就是空的：

1. **身份可比性** —— λ 表 / window_ranges / path_version 对不对得上。
   **λ 表不同就别逐窗相减**，两条臂走的不是同一条路径，只有路径级 ΔG 可比。
2. **验收量** —— 路径 ΔG 差折算成几个 σ；逐窗 min N_eff/g 的 B/A 比值与 verdict 迁移。
3. **代价** —— 生产步数 / 采样段数 / 自治轮数。
   **outer 臂每步更贵，只比 ΔG 不比代价等于没比。**

本次 baseline（2026-09-12，cyclod_ligand2/rep1）：

```
path v4 | 24 态 | K=[5,5,4,4,4,7] | ΔG=−49.04 ± 0.99 kJ/mol
逐窗 min N_eff/g = 19.56 / 49.42 / 63.96 / 10.52 / 18.75 / 10.39
代价 = 1750000 生产步 | 6 个采样段 | 5 轮自治
```

## 8. 验收口径

**不是**绿测试、不是漂亮的 trace。是：

> resume 当前 run，人不碰它，它自己处理有问题的窗口、跑完全路径、产出完整
> Stage-2 结果。

2026-09-12 17:22 首次达成。

---

# 交接

## 9. 代码在哪

⚠️ 分在三个文件，**这是有意的**（收拢判据见 §9.4）：决策同源的进
`abfe_preoptimizer`、写盘的留 `abfe_pipeline`、纯判据/常量留 `ibs_engine`。
§7 第 4 条原先写的「设计要求是包在一起、没做到」已于 2026-09-12 被否掉，别再照那句搬。

> ### 🗑️ [2026-09-13] 本节不再写行号
>
> 原先每张表都有一列「行」。实测 2026-09-13：§9.1 **九个行号全错**
> （`decide()` 标 2378、实际 2517），§9.3 错两个，§9.2 除错位外还有
> **四个符号根本不在那个文件里了**（被 `S2-D` 搬去 `abfe_preoptimizer`，
> 与 §9.4 自相矛盾）。
>
> 行号注定会漂，而第一列本来就是**可 grep 的符号名** —— 那一列不提供任何
> 符号名给不了的信息，只提供出错的机会。所以删掉。
> （本仓 TODO 的记录规矩本来就允许「`文件:行号` **或函数名**」。）

### 9.1 `abfe_preoptimizer.py` —— 决策（`class Stage2RepairController`）

| 符号 | 职责 |
|---|---|
| `ACTIONS` | 动作白名单。⚠️ 含 `NO_ACTION`（2026-09-13 新增）：它不是动作，是「没有动作」，只能配终态出口 |
| `EXITS` / `TERMINAL_EXITS` | 全部结局 / 其中真能终止的那几个；其余一律路由。两份清单与 `decide()` 实际发出的词由 `tests/test_stage2_controller_vocabulary.py` 按 AST 对账 |
| `decide()` | **唯一决策点**。分支顺序即优先级 |
| `stage_quality_gate_failures()` | 模块级纯函数（2026-09-13 新增）：stage 判据没过时**哪几道门没过**，逐条带实测值/阈值/对症类别/最差窗口。fail-closed，读不出来标 `unattributed`、不猜动作 |
| `read_aggregated()` | 多段聚合 + **λ 身份过期过滤** |
| `read()` | 聚合模式的入口 |
| `for_physical_stage()` | 构造（按物理 stage 聚合全部段） |
| `replay()` | 只读重放，不执行 |
| `comparison_manifest()` / `write_comparison_manifest()` | A/B 清单（纯聚合）/ 落盘 |

### 9.2 `abfe_pipeline.py` —— 执行

| 符号 | 职责 |
|---|---|
| `_run_stage2_autonomous()` | **主循环**，全部动作的执行器都在里面。⚠️ 它**只看 `plan["terminal"]` 就 break**，不看 `evidence_status` —— §2 那条修正的由来 |
| `_solve_merged_segments_if_any()` | 多段按窗口合并求解（求解不是决策，故留在 pipeline） |
| `_legalize_tail_window()` | 末窗死区 + 拆窗（启动与插 λ 后共用）。**它写盘**，所以是执行器 |
| `_recalibrate_f_k_and_resample_segment()` | 换 Epoch（`probe_only=True` 时只判不动手） |

⚠️ `_relearn_epoch_required_steps` / `_segment_dirs_for_evidence` /
`_lambdas_from_version_record` / `_existing_segment_names` 这四个**已于 2026-09-12
迁入 `abfe_preoptimizer` 模块级**（`S2-D`），本表 2026-09-13 前一直漏改。
`_latest_segment_dirs` **已删除**（被 `segment_dirs_for_evidence` 取代）。详见 §9.4。

### 9.3 `ibs_engine.py` —— 纯判据 / 常量

| 符号 | 职责 |
|---|---|
| `IBS_LOCAL_MBAR_GATE_MIN_FRAMES` | 去相关帧数下限 = 10，**可达性判据的 T** |
| `LAMBDA_GRID_DECIMALS` | λ 比较栅格，从 `lambda_path_versions` 导入（**不另立一份**） |
| `validation_reachability_verdict()` | 验证可达性四档 |
| `sealed_candidate_matches()` | 候选是否实质同一份（mean-center 后比距离） |
| `window_self_support_check()` | 逐窗自检，`min N_eff/g` 的来源 |
| `solve_stage_integrated()` 里的 `converged` | stage 级判据，**五个条件的合取**；每一项的实测值与阈值都进结果 payload，`stage_quality_gate_failures()` 就是读它们 |

⚠️ `minimum_complete_validation_frames`（原始帧完整性目标）与上面那个 T
（去相关下限）是**两个量**，别合并：混掉会把 gcrit 算小 20 倍。
它**只进报告、不当门**，且 2026-09-13 起不再随预算浮动（`S2-B`）。

`stage2_ab_report.py` —— 后分析，**刻意独立**（是后分析不是控制器逻辑）。

### 9.4 收拢的结论（2026-09-12 已执行）

`abfe_pipeline` 里那批 helper 已迁入 `abfe_preoptimizer` 模块级，**四个**：
`relearn_epoch_required_steps` / `segment_dirs_for_evidence` /
`lambdas_from_version_record` / `existing_segment_names`。
边界理由写在 `abfe_preoptimizer` 迁入处的注释块里：**决策同源的进来，执行器不进来**
（"执行器" = 写盘的那些，由 `test_controller_never_writes_anything` 钉住）。

**三个**留在 `abfe_pipeline`，都是有意的：

| 符号 | 结论 | 理由 |
|---|---|---|
| `_solve_merged_segments_if_any` | **留在 pipeline** | 它跑的是合并求解。设计里控制器「只读不算」，求解不是决策。为满足归档规则给它加个回调参数，是为形式增加间接层 |
| `_legalize_tail_window` | **留在 pipeline** | 它会 `append_version` 和 `record_tail_repartition_version` —— **写盘**。它用到的纯函数（插 λ / 尾段重分 / 版本记录）都在 preopt，但"合法化"这个**动作**是执行器的事。⚠️ 早先本节和迁入处注释都把它列进"已迁入"，那是错的，2026-09-12 更正 |
| `_latest_segment_dirs` | **已删除** | 不是"没人调所以清理"，是被 `segment_dirs_for_evidence` **取代**：真机实证「段号最大的段」可能根本没有目标窗口的帧，正确语义是「**有这个窗口数据的**最新段」。留着等于把一个已修的崩溃摆在下一个人手边。要历史见 git |

### 9.5 以后再搬的话

- `_run_stage2_autonomous` 主循环本身是**执行器**，留在 pipeline 合理。
- `ibs_engine` 那三个是纯函数/常量，本来就该在引擎侧，**别搬**。
- ⚠️ 搬之前先跑 `tests/test_stage2_autonomous_segment_routing.py` 和
  `tests/test_fk_relearn_and_reachability.py` 存基线 —— 它们有若干条是
  **按源码文本/AST 断言**的（分支顺序、是否传 `probe_only`、handler 里有没有裸
  `raise`），位移会动到它们。

## 10. 当前状态

**run**：`/home/ruigengji/abfe-benchmark/openmm_IBS/runs/cyclod_ligand2/rep1`

```
Stage-2  六窗全部 ANALYSIS_ELIGIBLE → DONE     ΔG_vdw = −49.04 ± 0.99 kJ/mol
复合物腿 ΔG_cplx = 541.937 ± 1.303 kJ/mol   （Boresch 已含）
溶剂腿   ΔG_solv = 527.315 ± 1.477 kJ/mol
ΔG_bind = −14.62 ± 1.97 kJ/mol = −3.48 ± 0.47 kcal/mol
实验 −4.04 ⟹ 1.19σ
```

> ⚠️ **2026-09-14：上面这一屏的 ΔG 全部作废**（见本文开头的更正）。
> `ΔG_vdw = −49.04` 是只覆盖 λ 17→23 的部分和，不是这条路径的完整 vanishing；
> 完整值是 +43.00（本 run 自己的中间结果）/ +44.09（rep2）。
> 留着这一屏是为了让后来的人能对上日志，**不是**可引用的结果。

⚠️ 日志里打的 `578.35` 是解耦三项的**裸和**（4.12 + 623.27 − 49.04），
还没扣 Boresch 限制释放的 36.41。合并要用 `total_delta_G_complex_kJ_mol = 541.937`。

**盘上 6 个采样段**（`vanishing`, `vanishing_2..6`）。`vanishing_3` 是空段 ——
`PROBE_CANDIDATE_FK` 增殖 bug 的产物，已修，留着不碍事。

**验收怎么复现**：

```bash
python stage2_ab_report.py /home/.../runs/cyclod_ligand2/rep1
```

## 11. 别重新论证的事

以下每条都在真机上花过时间，**结论已定**：

1. **末窗不套 `_hi`。** 它是溢出槽，上界是可拆上限 `2*hi−1`。
   那个报错自己的理由只管下界。
2. **「补 2 且拆成俩 4 不相交」不行。** 共享边界态是 `p+q−1=K` 的由来；
   不共享则接缝那条边没人负责，ΔG 少一项。K=6 → 补 1 → K=7 → 唯一拆法 4+4。
3. **加帧治不了偏斜。** 同分布加帧 N 涨 g 也涨。对症动作是换 f_k 或缩跨度。
4. **跨腿构象门只到 WARN 级。** 构象差是这类装置的必然产物
   （溶剂腿里 vdW 一关配体就塌，复合物腿里主体腔卡住它）。
   但 WARN **不等于没事**，标记必须留在结果里。
5. **不做「没救了」的提前外推。** 周期**内**按 n_eff 外推判死已被否决；
   可达性预检只在周期**用尽之后**决定要不要开下一个周期。
6. **别拿 hash 做自产产物的身份。** 用语义身份。
7. **occupancy / coverage ESS 永不决定 ΔG 是否有效**，只诊断 f_k 训练。
8. **低支撑不是 FAIL**，是「尚不可测」。

## 12. 给下一个人的规矩

1. **先读 §6 的十条陷阱再动手。** 每一条都是真机咬出来的，重新踩一遍很贵。
2. **一个不变量只能有一份实现。** 本次最贵的 bug 就是 λ 身份有四份口径。
   加判据之前先 grep 有没有人已经判过同一件事。
3. **报错里的数字先确认是不是算出来的。** `实测 0.657–0.672 nm` 是硬编码的旧例子，
   本人据此误判过一次。
4. **跨 f_k epoch / 跨 λ 版本的数字不能直接相减。** 剖面形状不同；
   而且更短的序列在 g/ESS 类诊断上会系统性显得更好。
5. **一个 stage 只许一个控制器。** 旧机制现在是**关掉**不是删掉
   （`stage2_autonomous_controller=True` 时把 rescue 三件套一并关闭，带 assert）。
   等自治这条再跑通几次再删。
6. **改动后先跑那两份测试**（§9.4），它们锁的是分支顺序和语义，不是数值。
7. **验收口径是 §8** —— 不是绿测试，是「人不碰它，自己跑完」。
